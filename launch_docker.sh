#!/bin/bash

# Enhanced Docker launcher with flexible path configuration and remote support

echo "MRI AI Service Docker Launcher"
echo "================================"

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input PATH     Input directory path (required if not in config)"
    echo "  -o, --output PATH    Output directory path (required if not in config)"
    echo "  -t, --template PATH  Template file path (optional)"
    echo "  -c, --config FILE    Config file to use (default: config/config.yaml)"
    echo "  -p, --port PORT      Port to expose (default: 5001)"
    echo "  -d, --detach         Run in background (detached mode)"
    echo "  -r, --remote         Use remote paths via SSH (as configured in config.yaml)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Use config/config.yaml"
    echo "  $0 -i /data/input -o /data/output            # Specify paths directly"
    echo "  $0 -c /path/to/custom/config.yaml           # Use custom config file"
    echo "  $0 -r                                        # Use remote paths from config"
    echo "  $0 -r -d                                     # Remote paths in background"
    echo ""
}

# Default values
USE_ENV_PATHS=false
USE_REMOTE_PATHS=false
INPUT_PATH=""
OUTPUT_PATH=""
TEMPLATE_PATH=""
CONFIG_FILE="config/config.yaml"
PORT="5001"
DETACHED=false
MOUNT_DIR="/tmp/mri_remote_mounts"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_PATH="$2"
            USE_ENV_PATHS=true
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            USE_ENV_PATHS=true
            shift 2
            ;;
        -t|--template)
            TEMPLATE_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--detach)
            DETACHED=true
            shift
            ;;
        -r|--remote)
            USE_REMOTE_PATHS=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to cleanup remote mounts
cleanup_remote_mounts() {
    echo "Cleaning up remote mounts..."
    # Kill any remaining sshfs processes for these mounts
    pkill -f "sshfs.*$MOUNT_DIR" 2>/dev/null
    
    # Unmount with retry
    for dir in input output template; do
        if [ -d "$MOUNT_DIR/$dir" ]; then
            # Try fusermount first
            fusermount -u "$MOUNT_DIR/$dir" 2>/dev/null || \
            # Then try umount
            umount "$MOUNT_DIR/$dir" 2>/dev/null || \
            # Force unmount as last resort
            fusermount -uz "$MOUNT_DIR/$dir" 2>/dev/null || true
        fi
    done
    
    # Remove mount directory if empty
    if [ -d "$MOUNT_DIR" ]; then
        rm -rf "$MOUNT_DIR" 2>/dev/null || true
    fi
}

# Set up trap to cleanup on exit
trap cleanup_remote_mounts EXIT

# If no direct paths specified, try to extract from config file
if [ "$USE_ENV_PATHS" = false ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file $CONFIG_FILE not found!"
        echo "Please specify paths directly with -i and -o, or provide a valid config file."
        show_usage
        exit 1
    fi

    echo "Extracting paths from config file: $CONFIG_FILE"
    
    # Extract paths and SSH info from config file
    if command -v python3 &> /dev/null; then
        python3 -c "
import yaml
import sys
import os

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract paths
    use_remote = '$USE_REMOTE_PATHS' == 'true'
    
    if use_remote:
        # For remote paths, we'll construct them based on server config
        remote_paths = config.get('remote_paths', {})
        host_paths = config.get('host_paths', {})
        
        # If remote_paths section exists, use it; otherwise use host_paths
        if remote_paths:
            input_path = remote_paths.get('raw_input_dir', '')
            output_path = remote_paths.get('output_base_dir', '')
            template_path = remote_paths.get('template_path', '')
        else:
            # Use host_paths as remote paths
            input_path = host_paths.get('raw_input_dir', '')
            output_path = host_paths.get('output_base_dir', '')
            template_path = host_paths.get('template_path', '')
        
        # Get SSH connection info
        server_config = config.get('server_mriqc', {})
        ssh_user = server_config.get('ssh_user', '')
        ssh_host = server_config.get('ssh_host', '')
        
        print(f'REMOTE_INPUT_PATH={input_path}')
        print(f'REMOTE_OUTPUT_PATH={output_path}')
        print(f'REMOTE_TEMPLATE_PATH={template_path}')
        print(f'SSH_USER={ssh_user}')
        print(f'SSH_HOST={ssh_host}')
    else:
        # Local paths
        host_paths = config.get('host_paths', {})
        if host_paths:
            input_path = host_paths.get('raw_input_dir', '')
            output_path = host_paths.get('output_base_dir', '')
            template_path = host_paths.get('template_path', '')
        else:
            paths = config.get('paths', {})
            input_path = paths.get('raw_input_dir', '')
            output_path = paths.get('output_base_dir', '')
            template_path = paths.get('template_path', '')
        
        print(f'INPUT_PATH={input_path}')
        print(f'OUTPUT_PATH={output_path}')
        print(f'TEMPLATE_PATH={template_path}')
    
except Exception as e:
    print(f'ERROR: Could not parse config file: {e}', file=sys.stderr)
    sys.exit(1)
" > /tmp/docker_paths.env

        if [ -f /tmp/docker_paths.env ] && ! grep -q "ERROR:" /tmp/docker_paths.env; then
            source /tmp/docker_paths.env
            rm /tmp/docker_paths.env
        else
            echo "Error: Could not extract paths from config file"
            cat /tmp/docker_paths.env 2>/dev/null
            rm -f /tmp/docker_paths.env
            exit 1
        fi
    else
        echo "Error: Python3 is required to parse config file"
        echo "Please install Python3 or specify paths directly with -i and -o options"
        exit 1
    fi
fi

# Handle remote path mounting if needed
if [ "$USE_REMOTE_PATHS" = true ]; then
    # Clean up any stale mounts first
    cleanup_remote_mounts 2>/dev/null
    
    # Check if sshfs is installed
    if ! command -v sshfs &> /dev/null; then
        echo "Error: sshfs is required for remote paths but not installed"
        echo "Please install: sudo apt-get install sshfs (Ubuntu/Debian) or equivalent"
        exit 1
    fi
    
    # Validate SSH configuration
    if [ -z "$SSH_USER" ] || [ -z "$SSH_HOST" ]; then
        echo "Error: SSH configuration missing in config file (server_mriqc.ssh_user/ssh_host)"
        exit 1
    fi
    
    echo ""
    echo "Setting up remote mounts..."
    echo "  SSH Host: $SSH_USER@$SSH_HOST"
    
    # Create mount directories
    mkdir -p "$MOUNT_DIR/input" "$MOUNT_DIR/output"
    
    # Test SSH connection
    echo "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_USER@$SSH_HOST" echo "SSH OK" >/dev/null 2>&1; then
        echo "Error: Cannot connect to $SSH_USER@$SSH_HOST"
        echo "Please ensure:"
        echo "  1. SSH key authentication is set up"
        echo "  2. The server is accessible"
        echo "  3. Your SSH key is loaded (ssh-add)"
        exit 1
    fi
    
    # Check if user is in fuse group (recommended)
    if ! groups | grep -q "\bfuse\b"; then
        echo "Warning: You're not in the 'fuse' group. This might cause permission issues."
        echo "Consider running: sudo usermod -a -G fuse $USER && newgrp fuse"
    fi
    
    # Mount remote directories with allow_other option
    echo "Mounting remote input directory..."
    if [ -n "$REMOTE_INPUT_PATH" ]; then
        # Use allow_other to let Docker access the mount
        sshfs -o allow_other,default_permissions,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,uid=$(id -u),gid=$(id -g) \
              "$SSH_USER@$SSH_HOST:$REMOTE_INPUT_PATH" "$MOUNT_DIR/input"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to mount remote input directory"
            echo "If you see 'option allow_other only allowed if...' error, run:"
            echo "  sudo sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf"
            exit 1
        fi
        INPUT_PATH="$MOUNT_DIR/input"
    fi
    
    echo "Mounting remote output directory..."
    if [ -n "$REMOTE_OUTPUT_PATH" ]; then
        # Create remote output directory if it doesn't exist
        ssh "$SSH_USER@$SSH_HOST" "mkdir -p '$REMOTE_OUTPUT_PATH'" 2>/dev/null
        
        sshfs -o allow_other,default_permissions,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,uid=$(id -u),gid=$(id -g) \
              "$SSH_USER@$SSH_HOST:$REMOTE_OUTPUT_PATH" "$MOUNT_DIR/output"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to mount remote output directory"
            echo "If you see 'option allow_other only allowed if...' error, run:"
            echo "  sudo sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf"
            exit 1
        fi
        OUTPUT_PATH="$MOUNT_DIR/output"
    fi
    
    # Mount template directory if specified
    if [ -n "$REMOTE_TEMPLATE_PATH" ]; then
        TEMPLATE_DIR=$(dirname "$REMOTE_TEMPLATE_PATH")
        TEMPLATE_FILE=$(basename "$REMOTE_TEMPLATE_PATH")
        mkdir -p "$MOUNT_DIR/template"
        
        echo "Mounting remote template directory..."
        sshfs -o allow_other,default_permissions,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,uid=$(id -u),gid=$(id -g) \
              "$SSH_USER@$SSH_HOST:$TEMPLATE_DIR" "$MOUNT_DIR/template"
        if [ $? -eq 0 ]; then
            TEMPLATE_PATH="$MOUNT_DIR/template/$TEMPLATE_FILE"
        else
            echo "Warning: Failed to mount remote template directory, will use default"
            TEMPLATE_PATH=""
        fi
    fi
    
    echo "✅ Remote mounts established successfully"
fi

# Validate required paths
if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Error: Both input and output paths are required"
    echo ""
    echo "Either:"
    echo "1. Specify them directly: $0 -i /path/to/input -o /path/to/output"
    echo "2. Or set them in your config file under 'host_paths' section"
    echo "3. Or use remote paths: $0 -r"
    echo ""
    show_usage
    exit 1
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Input path:    $INPUT_PATH $([ "$USE_REMOTE_PATHS" = true ] && echo "[REMOTE]" || echo "[LOCAL]")"
echo "  Output path:   $OUTPUT_PATH $([ "$USE_REMOTE_PATHS" = true ] && echo "[REMOTE]" || echo "[LOCAL]")"
echo "  Template path: ${TEMPLATE_PATH:-'Using default template'}"
echo "  Config file:   $CONFIG_FILE"
echo "  Port:          $PORT"
echo "  Mode:          $([ "$DETACHED" = true ] && echo "Background (detached)" || echo "Interactive")"

# Validate input path exists
if [ ! -d "$INPUT_PATH" ]; then
    echo ""
    echo "Warning: Input path $INPUT_PATH does not exist or is not accessible!"
    if [ "$DETACHED" = false ]; then
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Running in detached mode, continuing anyway..."
    fi
fi

# Create output directory
echo "Creating output directory if needed..."
mkdir -p "$OUTPUT_PATH"

# Stop existing container
if [ "$(docker ps -aq -f name=mri-ai-service)" ]; then
    echo "Stopping existing mri-ai-service container..."
    docker stop mri-ai-service 2>/dev/null
    docker rm mri-ai-service 2>/dev/null
fi

# Build image if needed
if [ "$(docker images -q mri-ai-service:latest 2> /dev/null)" == "" ]; then
    echo "Building Docker image (this may take 15-20 minutes)..."
    docker build -t mri-ai-service:latest .
    
    if [ $? -ne 0 ]; then
        echo "Error: Docker build failed!"
        exit 1
    fi
fi

# Prepare Docker run command
DOCKER_ARGS=()

# Basic container settings
if [ "$DETACHED" = true ]; then
    DOCKER_ARGS+=("-d")
else
    DOCKER_ARGS+=("-it")
fi

DOCKER_ARGS+=("--rm")
DOCKER_ARGS+=("-p" "${PORT}:5001")
DOCKER_ARGS+=("--name" "mri-ai-service")

# Mount directories
DOCKER_ARGS+=("-v" "${INPUT_PATH}:/app/data/input:ro")
DOCKER_ARGS+=("-v" "${OUTPUT_PATH}:/app/data/output:rw")
DOCKER_ARGS+=("-v" "$(pwd)/logs:/app/logs:rw")
DOCKER_ARGS+=("-v" "$(pwd)/scripts:/app/scripts:ro")
DOCKER_ARGS+=("-v" "$(pwd)/pipeline:/app/pipeline:ro")

# Mount config file
if [ -f "$CONFIG_FILE" ]; then
    DOCKER_ARGS+=("-v" "$(realpath "$CONFIG_FILE"):/app/config/config.yaml:ro")
else
    echo "Warning: Config file not found, using environment variables only"
fi

# Mount template directory or specific template file
if [ -n "$TEMPLATE_PATH" ] && [ -f "$TEMPLATE_PATH" ]; then
    echo "Using custom template: $TEMPLATE_PATH"
    DOCKER_ARGS+=("-v" "${TEMPLATE_PATH}:/app/mni_templates/custom_template.nii:ro")
    DOCKER_ARGS+=("-e" "MRI_TEMPLATE_PATH=/app/mni_templates/custom_template.nii")
elif [ -d "$(pwd)/mni_templates" ]; then
    DOCKER_ARGS+=("-v" "$(pwd)/mni_templates:/app/mni_templates:ro")
fi

# Set environment variables
DOCKER_ARGS+=("-e" "MRI_INPUT_PATH=/app/data/input")
DOCKER_ARGS+=("-e" "MRI_OUTPUT_PATH=/app/data/output")

# Add image name
DOCKER_ARGS+=("mri-ai-service:latest")

# Show the command
echo ""
echo "Docker command:"
echo "docker run ${DOCKER_ARGS[@]}"
echo ""

# Function to monitor remote mounts
monitor_remote_mounts() {
    while true; do
        sleep 30
        if [ "$USE_REMOTE_PATHS" = true ]; then
            # Check if mounts are still active
            if ! mountpoint -q "$MOUNT_DIR/input" 2>/dev/null || \
               ! mountpoint -q "$MOUNT_DIR/output" 2>/dev/null; then
                echo "⚠️  Warning: Remote mount disconnected, attempting to remount..."
                cleanup_remote_mounts
                exit 1
            fi
        fi
    done
}

# Start mount monitor in background if using remote paths
if [ "$USE_REMOTE_PATHS" = true ] && [ "$DETACHED" = false ]; then
    monitor_remote_mounts &
    MONITOR_PID=$!
fi

# Start the service
if [ "$DETACHED" = true ]; then
    echo "Starting MRI AI Service in background..."
    docker run "${DOCKER_ARGS[@]}"
    
    if [ $? -eq 0 ]; then
        echo "✅ Service started successfully in background!"
        echo "🌐 Access at: http://localhost:${PORT}"
        echo ""
        echo "Useful commands:"
        echo "  docker logs -f mri-ai-service     # View logs"
        echo "  docker stop mri-ai-service       # Stop service"
        echo "  docker exec -it mri-ai-service bash  # Enter container"
        if [ "$USE_REMOTE_PATHS" = true ]; then
            echo ""
            echo "⚠️  Note: Keep this terminal open or run with 'nohup' to maintain remote mounts"
        fi
        echo ""
        echo "Check status:"
        sleep 2
        if docker ps | grep -q mri-ai-service; then
            echo "✅ Container is running"
        else
            echo "❌ Container failed to start, check logs:"
            docker logs mri-ai-service
        fi
    else
        echo "❌ Failed to start service"
        exit 1
    fi
else
    echo "Starting MRI AI Service (interactive mode)..."
    echo "🌐 Service will be available at: http://localhost:${PORT}"
    echo "📋 Press Ctrl+C to stop the service"
    if [ "$USE_REMOTE_PATHS" = true ]; then
        echo "📁 Using remote paths via SSH"
    fi
    echo ""
    
    docker run "${DOCKER_ARGS[@]}"
    
    # Kill monitor if running
    [ -n "$MONITOR_PID" ] && kill $MONITOR_PID 2>/dev/null
    
    echo ""
    echo "Service stopped."
fi