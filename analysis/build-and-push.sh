#!/bin/bash

# Build and Push Script for AI-Powered Data Analysis System
# This script builds the Docker image with amd64 architecture and pushes it to Docker Hub

set -e

# Configuration
IMAGE_NAME="ai-data-analysis"
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-your-dockerhub-username}"
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="${DOCKER_HUB_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "🚀 Building Docker image for AI-Powered Data Analysis System"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Platform: linux/amd64"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image with amd64 architecture
echo "🔨 Building Docker image..."
docker build --platform linux/amd64 -t ${FULL_IMAGE_NAME} .

echo "✅ Docker image built successfully!"

# Test the image locally
echo "🧪 Testing the image locally..."
docker run --rm -d --name test-analysis -p 3041:3040 ${FULL_IMAGE_NAME}

# Wait for the container to start
echo "⏳ Waiting for container to start..."
sleep 10

# Test health endpoint
if curl -f http://localhost:3041/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    docker stop test-analysis
    exit 1
fi

# Stop test container
docker stop test-analysis
echo "✅ Local test completed successfully!"

# Ask for confirmation before pushing
echo ""
read -p "🤔 Do you want to push the image to Docker Hub? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Pushing to Docker Hub..."
    
    # Login to Docker Hub (if not already logged in)
    if ! docker info | grep -q "Username:"; then
        echo "🔐 Please log in to Docker Hub:"
        docker login
    fi
    
    # Push the image
    docker push ${FULL_IMAGE_NAME}
    
    echo "✅ Image pushed successfully to Docker Hub!"
    echo "📦 Image: ${FULL_IMAGE_NAME}"
    echo ""
    echo "🎉 You can now pull and run the image with:"
    echo "   docker run -p 3040:3040 -e OPENAI_API_KEY=your_key ${FULL_IMAGE_NAME}"
    echo ""
    echo "🔗 Or use docker-compose:"
    echo "   OPENAI_API_KEY=your_key docker-compose up"
else
    echo "⏭️  Skipping push to Docker Hub"
fi

echo ""
echo "🏁 Build process completed!" 