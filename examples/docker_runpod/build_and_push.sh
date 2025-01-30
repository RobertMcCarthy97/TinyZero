#!/bin/bash

# Configuration
DOCKER_USERNAME="rmcc11"  # Change this to your Docker Hub username
IMAGE_NAME="tinyzero"
TAG="latest"

# Derived variables
FULL_IMAGE_NAME="$DOCKER_USERNAME/$IMAGE_NAME:$TAG"

# Print configuration
echo "Building and pushing Docker image with following configuration:"
echo "Docker Username: $DOCKER_USERNAME"
echo "Image Name: $IMAGE_NAME"
echo "Tag: $TAG"
echo "Full Image Name: $FULL_IMAGE_NAME"
echo

# Confirm with user
read -p "Continue with these settings? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborting..."
    exit 1
fi

# Build the image
echo "Building Docker image..."
docker build -t $FULL_IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login
if [ $? -ne 0 ]; then
    echo "Error: Docker login failed"
    exit 1
fi

# Push the image
echo "Pushing image to Docker Hub..."
docker push $FULL_IMAGE_NAME
if [ $? -ne 0 ]; then
    echo "Error: Docker push failed"
    exit 1
fi

echo "Successfully built and pushed $FULL_IMAGE_NAME"