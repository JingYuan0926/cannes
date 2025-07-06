#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Data Analysis Pipeline Deployment${NC}"
echo "========================================"

# Function to deploy locally
deploy_local() {
    echo -e "${BLUE}📦 Deploying locally with Colima...${NC}"
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        echo -e "${YELLOW}⚠️  .env file not found. Please create one from .env.example${NC}"
        echo "   cp .env.example .env"
        echo "   # Then edit .env with your OpenAI API key"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker/Colima.${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🏃 Starting local services...${NC}"
    docker-compose -f compose.yaml up -d
    
    echo -e "${GREEN}✅ Local deployment complete!${NC}"
    echo "Services running on:"
    echo "   • ETL Service:          http://localhost:3030"
    echo "   • Preprocessing:        http://localhost:3031"
    echo "   • EDA Service:          http://localhost:3035"
    echo "   • Analysis Service:     http://localhost:3040"
}

# Function to deploy to ROFL
deploy_rofl() {
    echo -e "${BLUE}🔒 Deploying to ROFL TEE...${NC}"
    
    # Check if oasis CLI is available
    if ! command -v oasis &> /dev/null; then
        echo -e "${RED}❌ Oasis CLI not found. Please install it first.${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🏗️  Building ROFL application...${NC}"
    oasis rofl build
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ROFL deployment complete!${NC}"
        echo "Your AI pipeline is now running in a Trusted Execution Environment!"
    else
        echo -e "${RED}❌ ROFL deployment failed. Check the logs above.${NC}"
        exit 1
    fi
}

# Function to stop services
stop_services() {
    echo -e "${BLUE}🛑 Stopping services...${NC}"
    docker-compose -f compose.yaml down
    echo -e "${GREEN}✅ Services stopped.${NC}"
}

# Main menu
case "$1" in
    local)
        deploy_local
        ;;
    rofl)
        deploy_rofl
        ;;
    stop)
        stop_services
        ;;
    *)
        echo "Usage: $0 {local|rofl|stop}"
        echo ""
        echo "Commands:"
        echo "  local  - Deploy locally with Docker/Colima"
        echo "  rofl   - Deploy to ROFL Trusted Execution Environment"
        echo "  stop   - Stop local services"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh local   # Start local development"
        echo "  ./deploy.sh rofl    # Deploy to ROFL TEE"
        echo "  ./deploy.sh stop    # Stop local services"
        exit 1
        ;;
esac 