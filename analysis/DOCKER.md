# Docker Deployment Guide

This guide explains how to build, run, and deploy the AI-Powered Data Analysis System using Docker with amd64 architecture.

## 🐳 Docker Files Overview

- **`Dockerfile`** - Multi-stage Docker build configuration
- **`docker-compose.yml`** - Docker Compose configuration for easy deployment
- **`.dockerignore`** - Excludes unnecessary files from Docker build context
- **`build-and-push.sh`** - Automated build and push script

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Start the service:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Health check: http://localhost:3040/health
   - API documentation: http://localhost:3040/

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build --platform linux/amd64 -t ai-data-analysis .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name ai-analysis \
     -p 3040:3040 \
     -e OPENAI_API_KEY="your-openai-api-key-here" \
     ai-data-analysis
   ```

## 🔧 Build and Push to Docker Hub

### Using the automated script:

1. **Set your Docker Hub username:**
   ```bash
   export DOCKER_HUB_USERNAME="your-dockerhub-username"
   ```

2. **Run the build script:**
   ```bash
   ./build-and-push.sh
   ```

### Manual build and push:

1. **Build with platform specification:**
   ```bash
   docker build --platform linux/amd64 -t your-username/ai-data-analysis:latest .
   ```

2. **Test locally:**
   ```bash
   docker run --rm -p 3040:3040 -e OPENAI_API_KEY="your-key" your-username/ai-data-analysis:latest
   ```

3. **Push to Docker Hub:**
   ```bash
   docker login
   docker push your-username/ai-data-analysis:latest
   ```

## 📦 Docker Image Details

### Base Image
- **Platform:** linux/amd64
- **Base:** python:3.11-slim
- **Size:** ~1.2GB (optimized with multi-stage build)

### Installed Dependencies
- Python 3.11
- Flask web framework
- Machine learning libraries (scikit-learn, xgboost, lightgbm)
- Data processing libraries (pandas, numpy, scipy)
- Visualization libraries (plotly, matplotlib, seaborn)
- OpenAI API client

### Exposed Ports
- **3040** - Main application port

### Health Check
- **Endpoint:** `/health`
- **Interval:** 30 seconds
- **Timeout:** 10 seconds
- **Retries:** 3

## 🔐 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI features | - | ✅ |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4` | ❌ |
| `OPENAI_TEMPERATURE` | AI creativity level (0.0-1.0) | `0.7` | ❌ |
| `FLASK_ENV` | Flask environment | `production` | ❌ |
| `HOST` | Server host | `0.0.0.0` | ❌ |
| `PORT` | Server port | `3040` | ❌ |
| `DEBUG` | Debug mode | `False` | ❌ |

## 💾 Volume Mounts

### Recommended volumes:
```bash
docker run -v ./data:/app/data -v ./results:/app/results ai-data-analysis
```

- **`/app/data`** - Input data files
- **`/app/results`** - Analysis results storage
- **`/app/uploads`** - Temporary file uploads

## 🚀 Production Deployment

### Using Docker Compose with environment file:

1. **Create `.env` file:**
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   OPENAI_MODEL=gpt-4
   OPENAI_TEMPERATURE=0.7
   ```

2. **Deploy:**
   ```bash
   docker-compose up -d
   ```

### Using Docker Swarm:

1. **Initialize swarm:**
   ```bash
   docker swarm init
   ```

2. **Deploy stack:**
   ```bash
   docker stack deploy -c docker-compose.yml ai-analysis
   ```

### Using Kubernetes:

1. **Create deployment:**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: ai-analysis
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: ai-analysis
     template:
       metadata:
         labels:
           app: ai-analysis
       spec:
         containers:
         - name: ai-analysis
           image: your-username/ai-data-analysis:latest
           ports:
           - containerPort: 3040
           env:
           - name: OPENAI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: openai-secret
                 key: api-key
   ```

## 🔍 Monitoring and Logging

### Health Checks
```bash
# Check container health
docker ps

# Check logs
docker logs ai-analysis

# Follow logs
docker logs -f ai-analysis
```

### Application Metrics
- **Health endpoint:** `GET /health`
- **Status endpoint:** `GET /`
- **Analysis list:** `GET /list-analyses`

## 🛠️ Troubleshooting

### Common Issues:

1. **OpenAI API key not set:**
   ```bash
   # Check environment variables
   docker exec ai-analysis env | grep OPENAI
   ```

2. **Port already in use:**
   ```bash
   # Use different port
   docker run -p 3041:3040 ai-data-analysis
   ```

3. **Memory issues:**
   ```bash
   # Increase memory limit
   docker run --memory=4g ai-data-analysis
   ```

4. **Build fails on Apple Silicon:**
   ```bash
   # Force amd64 platform
   docker build --platform linux/amd64 -t ai-data-analysis .
   ```

### Debug Mode:
```bash
docker run -e DEBUG=True -e FLASK_ENV=development ai-data-analysis
```

## 📊 Performance Tuning

### Resource Limits:
```yaml
# docker-compose.yml
services:
  analysis:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

### Scaling:
```bash
# Scale with docker-compose
docker-compose up -d --scale analysis=3

# Use load balancer (nginx, traefik, etc.)
```

## 🔄 Updates and Maintenance

### Update process:
1. Pull latest image
2. Stop current container
3. Start new container
4. Verify health

```bash
# Automated update script
docker pull your-username/ai-data-analysis:latest
docker-compose down
docker-compose up -d
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/) 