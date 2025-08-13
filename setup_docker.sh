#!/bin/bash
# setup_docker.sh - Initial setup script

echo "Setting up MRI AI Service Docker environment..."

# 2. Check if health endpoint already exists
if ! grep -q "@app.route('/health')" webapp/app.py; then
    echo "Adding health check endpoint to webapp/app.py..."
    cat >> webapp/app.py << 'EOF'

@app.route('/health')
def health_check():
    from datetime import datetime
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
EOF
fi

# 3. Create necessary directories
echo "Creating necessary directories..."
mkdir -p data logs output

# 4. Create a .dockerignore file
echo "Creating .dockerignore file..."
cat > .dockerignore << 'EOF'
venv/
__pycache__/
*.pyc
.git/
.gitignore
*.log
.DS_Store
.idea/
.vscode/
*.swp
*.swo
tests/
docs/
README.md
EOF

# 5. Ensure Flask app runs on the correct host
echo "Updating Flask app to run on 0.0.0.0..."
if [ -f "webapp/app.py" ]; then
    # Check if app.run() exists and update it
    if grep -q "app.run(" webapp/app.py; then
        sed -i.bak "s/app.run(.*)/app.run(host='0.0.0.0', port=5001, debug=False)/" webapp/app.py
    else
        # Add app.run() at the end of the file
        echo -e "\nif __name__ == '__main__':\n    app.run(host='0.0.0.0', port=5001, debug=False)" >> webapp/app.py
    fi
fi

# 6. Build Docker image
echo "Building Docker image (this may take 15-20 minutes for the first build)..."
docker build -t mri-ai-service:latest . --no-cache

# 7. Check if build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    
    # 8. Test the build
    echo "Testing the build..."
    docker run --rm mri-ai-service:latest python3 -c "import flask, nibabel, numpy; print('Core imports successful')"
    
    echo -e "\n=== Setup complete! ==="
    echo "You can now use:"
    echo "  docker-compose up -d           # for development with live code reloading"
    echo "  docker run -p 5001:5001 -v \$(pwd)/data:/app/data -v \$(pwd)/output:/app/output mri-ai-service:latest  # for testing"
    echo ""
    echo "To check if the service is running:"
    echo "  curl http://localhost:5001/health"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f        # for docker-compose"
    echo "  docker logs <container_id>    # for docker run"
else
    echo "Error: Docker build failed. Please check the error messages above."
    exit 1
fi