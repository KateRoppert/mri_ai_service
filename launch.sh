#!/bin/bash
# MRI AI Service Docker Launcher for Linux/Mac
# This script launches the Docker container with automatic path mapping

echo "MRI AI Service Docker Launcher"
echo "=============================="
echo

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is not running or not installed."
    echo "Please start Docker and try again."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.7+ and try again."
    exit 1
fi

# Check for PyYAML
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip3 install pyyaml
fi

# Make the launcher executable if it isn't already
chmod +x launch_docker.py

# Run the launcher
python3 launch_docker.py "$@"