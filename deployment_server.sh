# deployment_server.sh - Server deployment script
echo "
#!/bin/bash
# deployment_server.sh - Run this on your server

echo 'Deploying MRI AI Service on server...'

# Pull latest code
git pull origin main

# Build production image
docker build -t mri-ai-service:production .

# Stop existing container
docker stop mri-ai-service || true
docker rm mri-ai-service || true

# Run new container
docker run -d \\
  --name mri-ai-service \\
  --restart unless-stopped \\
  -p 5001:5001 \\
  -v \$(pwd)/data:/app/data \\
  -v \$(pwd)/logs:/app/logs \\
  -v \$(pwd)/output:/app/output \\
  mri-ai-service:production

echo 'Service deployed successfully!'
echo 'Check status: docker ps'
echo 'View logs: docker logs mri-ai-service'
" > deployment_server.sh

chmod +x setup_docker.sh deployment_server.sh