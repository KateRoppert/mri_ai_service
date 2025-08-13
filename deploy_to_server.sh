#!/bin/bash

# Improved deployment script with better error handling and multiple transfer methods

if [ $# -eq 0 ]; then
    echo "Usage: $0 username@server-address [target-directory] [method]"
    echo "Methods:"
    echo "  rsync     - Use rsync (recommended for large files)"
    echo "  scp       - Use scp (default)"
    echo "  build     - Build on server (no image transfer)"
    echo ""
    echo "Examples:"
    echo "  $0 john@192.168.1.100                              # Deploy to ~/mri_ai_service using scp"
    echo "  $0 john@192.168.1.100 /opt/mri-service rsync       # Deploy to /opt using rsync"
    echo "  $0 john@192.168.1.100 /opt/mri-service build       # Deploy and build on server"
    exit 1
fi

SERVER=$1
TARGET_DIR=${2:-"~/mri_ai_service"}
METHOD=${3:-"scp"}

echo "Deploying MRI AI Service to: $SERVER"
echo "Target directory: $TARGET_DIR"
echo "Transfer method: $METHOD"

# Check if Docker image exists
if [ "$(docker images -q mri-ai-service:latest 2> /dev/null)" == "" ]; then
    echo "Error: Docker image mri-ai-service:latest not found!"
    echo "Please build the image first: docker build -t mri-ai-service:latest ."
    exit 1
fi

# Function to check available space on server
check_server_space() {
    echo "Checking available space on server..."
    ssh $SERVER "df -h /tmp && df -h $TARGET_DIR 2>/dev/null || df -h \$(dirname $TARGET_DIR)"
}

# Method 1: Build on server (recommended for large images)
deploy_with_build() {
    echo "Method: Build on Server (no image transfer needed)"
    echo "=================================================="
    
    # Step 1: Transfer only source code
    echo "Step 1/2: Transferring project files..."
    rsync -avz --progress \
        --exclude='venv/' \
        --exclude='*.tar.gz' \
        --exclude='data/' \
        --exclude='output/' \
        --exclude='logs/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        . $SERVER:$TARGET_DIR/
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer project files"
        exit 1
    fi
    
    # Step 2: Build on server
    echo "Step 2/2: Building image on server..."
    ssh $SERVER << ENDSSH
        cd $TARGET_DIR
        echo "Building Docker image on server..."
        docker build -t mri-ai-service:latest .
        
        if [ \$? -eq 0 ]; then
            echo "✓ Image built successfully on server"
        else
            echo "✗ Failed to build image on server"
            exit 1
        fi
ENDSSH
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build on server"
        exit 1
    fi
}

# Method 2: Transfer with rsync (more reliable for large files)
deploy_with_rsync() {
    echo "Method: Transfer with rsync"
    echo "=========================="
    
    # Step 1: Save Docker image
    echo "Step 1/4: Saving Docker image..."
    docker save mri-ai-service:latest | gzip > mri-ai-service.tar.gz
    if [ $? -ne 0 ]; then
        echo "Error: Failed to save Docker image"
        exit 1
    fi
    echo "Image saved: $(du -h mri-ai-service.tar.gz | cut -f1)"
    
    # Check server space
    check_server_space
    
    # Step 2: Transfer image using rsync (more reliable)
    echo "Step 2/4: Transferring image to server with rsync..."
    rsync -avz --progress --partial --inplace mri-ai-service.tar.gz $SERVER:/tmp/
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer image with rsync"
        echo "Try using 'build' method instead: $0 $SERVER $TARGET_DIR build"
        exit 1
    fi
    
    # Step 3: Transfer project files
    echo "Step 3/4: Transferring project files..."
    rsync -avz --progress \
        --exclude='venv/' \
        --exclude='*.tar.gz' \
        --exclude='data/' \
        --exclude='output/' \
        --exclude='logs/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        . $SERVER:$TARGET_DIR/
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer project files"
        exit 1
    fi
    
    # Step 4: Load image on server
    load_image_on_server
}

# Method 3: Transfer with scp (original method)
deploy_with_scp() {
    echo "Method: Transfer with scp"
    echo "========================"
    
    # Step 1: Save Docker image
    echo "Step 1/4: Saving Docker image..."
    docker save mri-ai-service:latest | gzip > mri-ai-service.tar.gz
    if [ $? -ne 0 ]; then
        echo "Error: Failed to save Docker image"
        exit 1
    fi
    echo "Image saved: $(du -h mri-ai-service.tar.gz | cut -f1)"
    
    # Check server space
    check_server_space
    
    # Step 2: Transfer image with scp
    echo "Step 2/4: Transferring image to server with scp..."
    scp mri-ai-service.tar.gz $SERVER:/tmp/
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer image with scp"
        echo "Try using rsync method: $0 $SERVER $TARGET_DIR rsync"
        echo "Or build on server: $0 $SERVER $TARGET_DIR build"
        exit 1
    fi
    
    # Step 3: Transfer project files
    echo "Step 3/4: Transferring project files..."
    rsync -avz --progress \
        --exclude='venv/' \
        --exclude='*.tar.gz' \
        --exclude='data/' \
        --exclude='output/' \
        --exclude='logs/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        . $SERVER:$TARGET_DIR/
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer project files"
        exit 1
    fi
    
    # Step 4: Load image on server
    load_image_on_server
}

# Function to load image on server
load_image_on_server() {
    echo "Step 4/4: Setting up service on server..."
    ssh $SERVER << ENDSSH
        # Load Docker image
        echo "Loading Docker image..."
        gunzip -c /tmp/mri-ai-service.tar.gz | docker load
        if [ \$? -ne 0 ]; then
            echo "Error: Failed to load Docker image"
            exit 1
        fi
        
        # Clean up transferred image
        rm /tmp/mri-ai-service.tar.gz
        echo "✓ Docker image loaded and temporary file cleaned up"
ENDSSH
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to load image on server"
        exit 1
    fi
}

# Function to setup server scripts
setup_server_scripts() {
    ssh $SERVER << ENDSSH
        # Create and go to project directory
        mkdir -p $TARGET_DIR
        cd $TARGET_DIR
        
        # Create necessary directories
        mkdir -p data logs output
        mkdir -p data/{input,output}
        
        # Create server launch script
        cat > launch_server.sh << EOF
#!/bin/bash

echo "Starting MRI AI Service on Server..."

# Server paths - edit these as needed
INPUT_PATH="$TARGET_DIR/data/input"
OUTPUT_PATH="$TARGET_DIR/data/output"

# Create directories
mkdir -p "\\\$INPUT_PATH" "\\\$OUTPUT_PATH"

# Stop existing container
docker stop mri-ai-service 2>/dev/null
docker rm mri-ai-service 2>/dev/null

# Run container
docker run -d \\\\
    --name mri-ai-service \\\\
    --restart unless-stopped \\\\
    -p 5001:5001 \\\\
    -v "\\\$INPUT_PATH":/app/data/input:ro \\\\
    -v "\\\$OUTPUT_PATH":/app/data/output:rw \\\\
    -v "\\\$(pwd)/config":/app/config:ro \\\\
    -v "\\\$(pwd)/mni_templates":/app/mni_templates:ro \\\\
    -v "\\\$(pwd)/logs":/app/logs:rw \\\\
    mri-ai-service:latest

if [ \\\$? -eq 0 ]; then
    echo "✓ Service started successfully!"
    echo "Access at: http://\\\$(hostname -I | awk '{print \\\$1}'):5001"
    echo "Check logs with: docker logs -f mri-ai-service"
    echo "Stop with: docker stop mri-ai-service"
else
    echo "✗ Failed to start service"
    exit 1
fi
EOF
        
        chmod +x launch_server.sh
        
        # Create server-specific config
        cat > config/server-config.yaml << EOF
# SERVER CONFIGURATION
# Edit these paths for your server setup

host_paths:
  raw_input_dir: "$TARGET_DIR/data/input"
  output_base_dir: "$TARGET_DIR/data/output"
  template_path: "$TARGET_DIR/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii"

# Copy the rest from your main config.yaml and adjust as needed
EOF
        
        # Create simple management script
        cat > manage_service.sh << 'EOF'
#!/bin/bash
SERVICE_NAME="mri-ai-service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"

case $1 in
    start)
        echo "Starting service..."
        docker stop $SERVICE_NAME 2>/dev/null
        docker rm $SERVICE_NAME 2>/dev/null
        docker run -d --name $SERVICE_NAME --restart unless-stopped -p 5001:5001 \
            -v "$DATA_DIR/input":/app/data/input:ro \
            -v "$DATA_DIR/output":/app/data/output:rw \
            -v "$PROJECT_DIR/config":/app/config:ro \
            -v "$PROJECT_DIR/mni_templates":/app/mni_templates:ro \
            -v "$PROJECT_DIR/logs":/app/logs:rw \
            mri-ai-service:latest
        echo "✓ Service started"
        ;;
    stop)
        docker stop $SERVICE_NAME && docker rm $SERVICE_NAME 2>/dev/null
        echo "✓ Service stopped"
        ;;
    status)
        if docker ps | grep -q $SERVICE_NAME; then
            echo "✓ Service is RUNNING"
            docker ps | grep $SERVICE_NAME
        else
            echo "✗ Service is NOT RUNNING"
        fi
        ;;
    logs)
        docker logs -f $SERVICE_NAME
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        ;;
esac
EOF
        
        chmod +x manage_service.sh
        
        # Start the service
        echo "Starting the service..."
        ./launch_server.sh
ENDSSH
}

# Main deployment logic
case $METHOD in
    build)
        deploy_with_build
        setup_server_scripts
        ;;
    rsync)
        deploy_with_rsync
        setup_server_scripts
        ;;
    scp)
        deploy_with_scp
        setup_server_scripts
        ;;
    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Available methods: scp, rsync, build"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. SSH to your server: ssh $SERVER"
    echo "2. Go to project directory: cd $TARGET_DIR"
    echo "3. Check service status: ./manage_service.sh status"
    echo "4. Copy your data to: $TARGET_DIR/data/input/"
    echo "5. Access service at: http://your-server-ip:5001"
    echo ""
    echo "Service management commands:"
    echo "  $TARGET_DIR/manage_service.sh start     # Start service"
    echo "  $TARGET_DIR/manage_service.sh stop      # Stop service"
    echo "  $TARGET_DIR/manage_service.sh status    # Check status"
    echo "  $TARGET_DIR/manage_service.sh logs      # View logs"
else
    echo "❌ Deployment failed. Check the error messages above."
    exit 1
fi

# Clean up local files
if [ -f "mri-ai-service.tar.gz" ]; then
    rm mri-ai-service.tar.gz
    echo "Cleaned up local temporary files."
fi