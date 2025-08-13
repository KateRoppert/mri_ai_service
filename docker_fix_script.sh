#!/bin/bash

echo "=== Docker Troubleshooting Script ==="
echo

# Step 1: Check Docker service status
echo "Step 1: Checking Docker service status..."
sudo systemctl status docker.service
echo

# Step 2: Check Docker daemon logs
echo "Step 2: Checking Docker daemon logs..."
sudo journalctl -xeu docker.service --no-pager -n 20
echo

# Step 3: Try to start Docker service manually
echo "Step 3: Attempting to start Docker service..."
sudo systemctl start docker
echo "Docker service start result: $?"
echo

# Step 4: Enable Docker to start on boot
echo "Step 4: Enabling Docker to start on boot..."
sudo systemctl enable docker
echo

# Step 5: Check if Docker daemon is running
echo "Step 5: Checking if Docker daemon is now running..."
sudo systemctl is-active docker
echo

# Step 6: Test Docker with hello-world
echo "Step 6: Testing Docker with hello-world..."
sudo docker run hello-world
echo

# Step 7: Add user to docker group (optional)
echo "Step 7: Adding current user to docker group..."
sudo usermod -aG docker $USER
echo "User $USER added to docker group. You'll need to log out and back in for this to take effect."
echo

# Step 8: Alternative - try using snap docker
echo "Step 8: If regular Docker still fails, try snap docker..."
echo "Command to try: sudo snap run docker run hello-world"
echo

echo "=== Troubleshooting complete ==="
echo "If Docker still doesn't work, please share the output of this script."