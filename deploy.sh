#!/bin/bash

echo "🚀 Starting Deployment of Supreme System V5..."

# 1. Pull latest changes (if using git)
# git pull origin main

# 2. Build Docker Image
echo "🔨 Building Docker Image..."
docker-compose build

# 3. Start Services
echo "⚡ Starting Services..."
docker-compose up -d

# 4. Verify Status
echo "✅ Checking Status..."
docker-compose ps

echo "🎉 Deployment Complete! Logs available at 'docker-compose logs -f'"
