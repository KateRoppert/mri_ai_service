#!/bin/bash
# Diagnostic script for MRI AI Service Docker setup

echo "MRI AI Service - Diagnostic Check"
echo "================================="
echo

# Check current directory
echo "1. Current directory:"
pwd
echo

# Check if required directories exist
echo "2. Project structure:"
for dir in webapp scripts pipeline config mni_templates; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ exists"
    else
        echo "  ✗ $dir/ MISSING"
    fi
done
echo

# Check if key files exist
echo "3. Key files:"
if [ -f "webapp/app.py" ]; then
    echo "  ✓ webapp/app.py exists"
else
    echo "  ✗ webapp/app.py MISSING"
fi

if [ -f "Dockerfile" ]; then
    echo "  ✓ Dockerfile exists"
else
    echo "  ✗ Dockerfile MISSING"
fi

if [ -f "config/config.yaml" ]; then
    echo "  ✓ config/config.yaml exists"
else
    echo "  ✗ config/config.yaml MISSING"
fi
echo

# Check Docker
echo "4. Docker status:"
if docker info >/dev/null 2>&1; then
    echo "  ✓ Docker is running"
else
    echo "  ✗ Docker is not running"
fi

# Check if image exists
echo
echo "5. Docker image:"
if docker images | grep -q "mri-ai-service"; then
    echo "  ✓ mri-ai-service image exists"
    docker images | grep mri-ai-service
else
    echo "  ✗ mri-ai-service image NOT FOUND"
    echo "  Run: docker build -t mri-ai-service:latest ."
fi

# Check if container exists
echo
echo "6. Container status:"
if docker ps -a | grep -q "mri-ai-service"; then
    echo "  Container found:"
    docker ps -a | grep mri-ai-service
else
    echo "  No container found"
fi

# Test mount paths
echo
echo "7. Testing volume mount (if container exists):"
if docker ps -a | grep -q "mri-ai-service"; then
    echo "  Checking /app/webapp in container:"
    docker run --rm -v "$(pwd)/webapp:/app/webapp" mri-ai-service:latest ls -la /app/webapp/ 2>&1 | head -5
fi

echo
echo "================================="
echo "Diagnostic check complete"