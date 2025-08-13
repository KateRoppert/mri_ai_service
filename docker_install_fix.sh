#!/bin/bash
# docker_install_fix.sh - Fix Docker installation conflicts

echo "Fixing Docker installation conflicts..."

# Method 1: Clean up existing Docker packages
echo "Step 1: Removing conflicting Docker packages..."
sudo apt remove --purge docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc containerd.io -y

# Clean up apt cache
sudo apt autoremove -y
sudo apt autoclean

# Method 2: Install Docker from official Docker repository
echo "Step 2: Installing Docker from official repository..."

# Update package index
sudo apt update

# Install prerequisites
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again
sudo apt update

# Install Docker Engine
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Test Docker installation
echo "Step 3: Testing Docker installation..."
sudo docker run hello-world

if [ $? -eq 0 ]; then
    echo "Docker installed successfully!"
    
    # Add current user to docker group
    echo "Step 4: Adding user to docker group..."
    sudo usermod -aG docker $USER
    
    echo "Docker installation complete!"
    echo "Please logout and login again, or run: newgrp docker"
    echo "Then you can run Docker commands without sudo"
else
    echo "Docker installation failed. Trying alternative method..."
    
    # Alternative method: Install from snap
    echo "Trying snap installation..."
    sudo snap install docker
    
    if [ $? -eq 0 ]; then
        echo "Docker installed via snap successfully!"
        echo "Note: With snap, Docker commands might need sudo"
    else
        echo "All installation methods failed. Please check your system configuration."
    fi
fi

echo "Installation script completed."