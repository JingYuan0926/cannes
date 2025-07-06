#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting AI Data Analysis Pipeline${NC}"
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Please create one from .env.example${NC}"
    echo "   cp .env.example .env"
    echo "   # Then edit .env with your OpenAI API key"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Pull latest images
echo -e "${BLUE}📥 Pulling latest images from Docker Hub...${NC}"
docker-compose pull

# Start services
echo -e "${BLUE}🏃 Starting all services...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check service health
echo -e "${BLUE}🏥 Checking service health...${NC}"
echo "=================================="

services=("etl:3030" "preprocessing:3031" "eda:3035" "analysis:3040")

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s -f "http://localhost:$port/" > /dev/null 2>&1 || curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name service (port $port) - HEALTHY${NC}"
    else
        echo -e "${RED}❌ $name service (port $port) - DOWN${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎉 Pipeline started successfully!${NC}"
echo "=================================="
echo "📊 Services running on:"
echo "   • ETL Service:          http://localhost:3030"
echo "   • Preprocessing:        http://localhost:3031"
echo "   • EDA Service:          http://localhost:3035"
echo "   • Analysis Service:     http://localhost:3040"
echo ""
echo "🔧 Management commands:"
echo "   • View logs:            docker-compose logs -f"
echo "   • Stop services:        docker-compose down"
echo "   • Restart services:     docker-compose restart"
echo "   • Check status:         docker-compose ps"
echo ""
echo -e "${BLUE}🌐 Open your web application and start analyzing data!${NC}" 