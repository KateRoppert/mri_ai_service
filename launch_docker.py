#!/usr/bin/env python3
"""
Docker launcher for MRI AI Service
This script reads the config.yaml, extracts paths, and launches Docker with proper volume mounts
"""

import os
import yaml
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import socket
import time
import urllib.request

class DockerLauncher:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.container_name = "mri-ai-service"
        self.image_name = "mri-ai-service:latest"
        self.path_mappings = {}
        self.executable_mappings = {}
        
    def load_config(self):
        """Load the original config.yaml"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_paths(self, config):
        """Extract all paths that need to be mounted"""
        paths_to_mount = set()
        
        # Extract paths from 'paths' section
        if 'paths' in config:
            paths = config['paths']
            if 'raw_input_dir' in paths:
                paths_to_mount.add(paths['raw_input_dir'])
            if 'output_base_dir' in paths:
                paths_to_mount.add(paths['output_base_dir'])
            if 'template_path' in paths:
                # Only add if it's a local path, not the default Docker path
                if not paths['template_path'].startswith('/app/'):
                    paths_to_mount.add(paths['template_path'])
        
        # Extract executables paths
        if 'executables' in config:
            executables = config['executables']
            for key, value in executables.items():
                if isinstance(value, str) and value.startswith('/') and not value.startswith('http'):
                    # Skip Docker internal paths
                    if not value.startswith('/usr/local/bin/bids-validator'):
                        paths_to_mount.add(value)
        
        # Extract any template paths from preprocessing
        if 'steps' in config and 'preprocessing' in config['steps']:
            preprocessing = config['steps']['preprocessing']
            if 'registration' in preprocessing and 'template_path' in preprocessing['registration']:
                template_path = preprocessing['registration']['template_path']
                if not template_path.startswith('/app/'):
                    paths_to_mount.add(template_path)
        
        return paths_to_mount
    
    def create_mount_mappings(self, paths):
        """Create consistent mappings between host and container paths"""
        for host_path in paths:
            host_path = os.path.abspath(host_path)
            
            # Determine if it's a file or directory
            if os.path.isfile(host_path):
                # For files, mount the parent directory
                parent_dir = os.path.dirname(host_path)
                container_parent = f"/mnt/host{parent_dir}"
                self.path_mappings[parent_dir] = container_parent
            else:
                # For directories, mount as is
                container_path = f"/mnt/host{host_path}"
                self.path_mappings[host_path] = container_path
    
    def create_container_config(self, original_config):
        """Create a modified config with container paths"""
        config = original_config.copy()
        
        # Update paths section
        if 'paths' in config:
            paths = config['paths']
            for key in ['raw_input_dir', 'output_base_dir', 'template_path']:
                if key in paths and not paths[key].startswith('/app/'):
                    host_path = os.path.abspath(paths[key])
                    if os.path.isfile(host_path):
                        parent_dir = os.path.dirname(host_path)
                        filename = os.path.basename(host_path)
                        container_parent = self.path_mappings.get(parent_dir, parent_dir)
                        paths[key] = os.path.join(container_parent, filename)
                    else:
                        paths[key] = self.path_mappings.get(host_path, host_path)
        
        # Update executables section
        if 'executables' in config:
            executables = config['executables']
            for key, value in executables.items():
                if isinstance(value, str) and value.startswith('/') and not value.startswith('http'):
                    # Skip Docker internal paths
                    if not value.startswith('/usr/local/bin/bids-validator'):
                        host_path = os.path.abspath(value)
                        if os.path.isfile(host_path):
                            parent_dir = os.path.dirname(host_path)
                            filename = os.path.basename(host_path)
                            container_parent = self.path_mappings.get(parent_dir, parent_dir)
                            executables[key] = os.path.join(container_parent, filename)
        
        # Update preprocessing registration template path
        if 'steps' in config and 'preprocessing' in config['steps']:
            preprocessing = config['steps']['preprocessing']
            if 'registration' in preprocessing and 'template_path' in preprocessing['registration']:
                template_path = preprocessing['registration']['template_path']
                if not template_path.startswith('/app/'):
                    host_path = os.path.abspath(template_path)
                    if os.path.isfile(host_path):
                        parent_dir = os.path.dirname(host_path)
                        filename = os.path.basename(host_path)
                        container_parent = self.path_mappings.get(parent_dir, parent_dir)
                        preprocessing['registration']['template_path'] = os.path.join(container_parent, filename)
        
        return config
    
    def check_port_availability(self, port=5001):
        """Check if port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return True
        except OSError:
            return False
    
    def find_container_using_port(self, port=5001):
        """Find which container is using the specified port"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}\t{{.Ports}}'],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f':{port}->' in line or f'0.0.0.0:{port}->' in line:
                    container_name = line.split('\t')[0]
                    return container_name
        except:
            pass
        return None
    
    def stop_existing_container(self):
        """Stop and remove existing container"""
        print(f"Checking for existing '{self.container_name}' container...")
        
        # Check if container exists
        result = subprocess.run(
            ['docker', 'ps', '-a', '--format', '{{.Names}}'],
            capture_output=True, text=True
        )
        
        if self.container_name in result.stdout:
            print(f"Stopping existing '{self.container_name}' container...")
            subprocess.run(['docker', 'stop', self.container_name], capture_output=True)
            subprocess.run(['docker', 'rm', self.container_name], capture_output=True)
            print("Existing container stopped and removed.")
    
    def run_docker_direct(self):
        """Run docker directly (simplified version that works)"""
        
        # Check port availability first
        if not self.check_port_availability(5001):
            container_using_port = self.find_container_using_port(5001)
            if container_using_port:
                print(f"\nWARNING: Port 5001 is already in use by container '{container_using_port}'")
                if container_using_port == self.container_name:
                    print("Stopping the existing container...")
                else:
                    print(f"Please stop the container '{container_using_port}' first:")
                    print(f"  docker stop {container_using_port}")
                    sys.exit(1)
            else:
                print("\nERROR: Port 5001 is already in use by another process.")
                print("Please free the port and try again.")
                sys.exit(1)
        
        # Stop and remove any existing container
        self.stop_existing_container()
        
        # Build docker run command
        cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '-p', '5001:5001',
            '--restart', 'unless-stopped'
        ]
        
        # Add environment variables
        cmd.extend([
            '-e', 'FLASK_ENV=development',
            '-e', 'FLASK_DEBUG=1',
            '-e', 'PYTHONPATH=/app'
        ])
        
        # Get current directory (project root)
        project_root = os.getcwd()
        
        # Add standard volumes using the same approach that works manually
        cmd.extend([
            '-v', f"{project_root}/webapp:/app/webapp",
            '-v', f"{project_root}/scripts:/app/scripts",
            '-v', f"{project_root}/pipeline:/app/pipeline",
            '-v', f"{project_root}/mni_templates:/app/mni_templates"
        ])
        
        # Create directories if they don't exist
        for dir_name in ['tests', 'data', 'logs', 'output']:
            dir_path = os.path.join(project_root, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            cmd.extend(['-v', f"{dir_path}:/app/{dir_name}:rw"])
        
        # Mount the entire config directory instead of just the modified file
        # This ensures the app can find the config
        cmd.extend(['-v', f"{project_root}/config:/app/config:rw"])
        
        # Add dynamic path mappings for data directories
        for host_path, container_path in self.path_mappings.items():
            cmd.extend(['-v', f"{host_path}:{container_path}:rw"])
        
        # Add image name
        cmd.append(self.image_name)
        
        # Debug: show the command being run
        print("\nDebug: Docker command:")
        print(" ".join(cmd[:10]) + " \\")  # First part
        for i in range(10, len(cmd)-1, 2):
            if i < len(cmd)-1:
                print(f"  {cmd[i]} {cmd[i+1] if i+1 < len(cmd)-1 else ''} \\")
        print(f"  {cmd[-1]}")
        print()
        
        # Run the container
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: Failed to start container:")
            print(result.stderr)
            # Try to get more info
            print("\nTrying to get more information...")
            subprocess.run(['docker', 'logs', self.container_name])
            sys.exit(1)
        
        # Give container a moment to start
        time.sleep(2)
        
        # Check if container is actually running
        result = subprocess.run(['docker', 'ps', '--filter', f'name={self.container_name}'], 
                              capture_output=True, text=True)
        
        if self.container_name not in result.stdout:
            print("\nERROR: Container started but immediately stopped.")
            print("Checking logs:")
            subprocess.run(['docker', 'logs', self.container_name])
            sys.exit(1)
            
        print(f"\n✓ Container '{self.container_name}' started successfully!")
        print(f"✓ Web interface available at: http://localhost:5001")
        
        # Test if port is actually accessible
        time.sleep(3)  # Give Flask more time to start
        try:
            import urllib.request
            response = urllib.request.urlopen('http://localhost:5001', timeout=5)
            print("✓ Web interface is responding!")
        except Exception as e:
            print("\n⚠ Warning: Web interface not responding yet.")
            print("  The application may still be starting up.")
            print("  Try refreshing http://localhost:5001 in a few seconds.")
            print("\nChecking container logs:")
            subprocess.run(['docker', 'logs', '--tail', '20', self.container_name])
    
    def launch(self):
        """Main launch method"""
        print("MRI AI Service Docker Launcher")
        print("==============================")
        
        # Check if we're in the right directory
        if not os.path.exists('webapp/app.py'):
            print("\nERROR: Cannot find webapp/app.py")
            print("Please run this script from the project root directory.")
            print(f"Current directory: {os.getcwd()}")
            sys.exit(1)
        
        # Check if Docker image exists
        result = subprocess.run(['docker', 'images', '-q', self.image_name], 
                              capture_output=True, text=True)
        
        if not result.stdout.strip():
            print(f"\nERROR: Docker image '{self.image_name}' not found.")
            print("Building Docker image...")
            build_result = subprocess.run(['docker', 'build', '-t', self.image_name, '.'])
            if build_result.returncode != 0:
                print("Failed to build Docker image.")
                sys.exit(1)
        
        # Load original config
        print(f"Loading configuration from {self.config_path}...")
        original_config = self.load_config()
        
        # Extract paths that need mounting
        print("Extracting paths to mount...")
        paths_to_mount = self.extract_paths(original_config)
        
        # Validate paths exist
        print("Validating paths...")
        for path in paths_to_mount:
            if not os.path.exists(path):
                print(f"WARNING: Path does not exist: {path}")
                # Don't exit, just warn
        
        # Create mount mappings
        print("Creating mount mappings...")
        self.create_mount_mappings(paths_to_mount)
        
        # Don't modify the original config - the app should handle path translation
        # This keeps things simpler and doesn't require app modifications
        
        # Use docker run directly (simpler and more reliable)
        print("Starting Docker container...")
        self.run_docker_direct()
        
        if self.path_mappings:
            print("\nPath mappings:")
            print("-" * 50)
            for host, container in sorted(self.path_mappings.items()):
                print(f"{host} -> {container}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Launch MRI AI Service in Docker')
    parser.add_argument('--config', default='config/config.yaml', 
                        help='Path to config.yaml (default: config/config.yaml)')
    parser.add_argument('--stop', action='store_true',
                        help='Stop the running container')
    parser.add_argument('--status', action='store_true',
                        help='Check container status')
    parser.add_argument('--logs', action='store_true',
                        help='Show container logs')
    
    args = parser.parse_args()
    
    launcher = DockerLauncher(args.config)
    
    if args.stop:
        print("Stopping MRI AI Service container...")
        result = subprocess.run(['docker', 'stop', launcher.container_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            subprocess.run(['docker', 'rm', launcher.container_name], capture_output=True)
            print("✓ Container stopped and removed.")
        else:
            print("Container was not running.")
        sys.exit(0)
    
    if args.status:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'], 
                              capture_output=True, text=True)
        if launcher.container_name in result.stdout:
            print("Container status:")
            print(result.stdout)
        else:
            print(f"Container '{launcher.container_name}' is not running.")
        sys.exit(0)
    
    if args.logs:
        subprocess.run(['docker', 'logs', '--tail', '50', launcher.container_name])
        sys.exit(0)
    
    launcher.launch()

if __name__ == '__main__':
    main()