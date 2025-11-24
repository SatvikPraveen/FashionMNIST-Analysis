# Deployment Guide for FashionMNIST-Analysis

This guide provides instructions for deploying the FashionMNIST-Analysis project using Docker and other deployment options.

## Table of Contents

1. [Quick Start with Docker](#quick-start-with-docker)
2. [Docker Compose Deployment](#docker-compose-deployment)
3. [Building and Running Containers](#building-and-running-containers)
4. [API Endpoints](#api-endpoints)
5. [Dashboard Access](#dashboard-access)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start with Docker

### Prerequisites

- Docker >= 20.10
- Docker Compose >= 1.29
- 4GB+ RAM available
- GPU (CUDA) support is optional but recommended

### Step 1: Build Docker Image

```bash
# Navigate to project directory
cd FashionMNIST-Analysis

# Build the Docker image
docker build -f docker/Dockerfile -t fashionmnist:latest .
```

### Step 2: Run with Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

### Step 3: Access Services

- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **Gradio App**: http://localhost:7860
- **Jupyter Lab**: http://localhost:8888

---

## Docker Compose Deployment

The `docker-compose.yml` file orchestrates multiple services:

### Services

1. **API Service** (FastAPI)

   - Port: 8000
   - Auto-reloads on code changes
   - Health check enabled

2. **Dashboard Service** (Streamlit)

   - Port: 8501
   - Depends on API service

3. **Gradio Service** (Gradio)

   - Port: 7860
   - Lightweight alternative UI

4. **Jupyter Service** (Jupyter Lab)
   - Port: 8888
   - For development and experimentation

### Volumes

- `./models` - Model weights storage
- `./data` - Input data
- `./results` - Output results
- `./logs` - Application logs
- `./config.yaml` - Configuration file

---

## Building and Running Containers

### Build Image

```bash
docker build -f docker/Dockerfile -t fashionmnist:latest .
```

### Run API Server Only

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name fashionmnist-api \
  fashionmnist:latest
```

### Run with GPU Support

```bash
docker run --gpus all \
  -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name fashionmnist-api \
  fashionmnist:latest
```

### View Container Logs

```bash
docker logs -f fashionmnist-api
```

### Stop Container

```bash
docker stop fashionmnist-api
docker rm fashionmnist-api
```

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### Initialize Model

```bash
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/best_model_weights/best_model_weights.pth",
    "config_path": "./config.yaml",
    "transfer_learning": false
  }'
```

### Single Image Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -F "top_k=3"
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "top_k=1"
```

### Uncertainty Estimation

```bash
curl -X POST http://localhost:8000/predict/uncertainty \
  -F "file=@image.jpg" \
  -F "num_samples=10"
```

### Interactive API Docs

Navigate to `http://localhost:8000/docs` for Swagger UI documentation.

---

## Dashboard Access

### Streamlit Dashboard

```bash
# Local access
http://localhost:8501

# Features:
# - Single image prediction
# - Batch prediction
# - Model comparison
# - Explainability (Grad-CAM)
```

### Gradio App

```bash
# Local access
http://localhost:7860

# Features:
# - Simple web interface
# - Model selection
# - Real-time predictions
```

### Jupyter Lab

```bash
# Local access
http://localhost:8888

# Use for:
# - Development and experimentation
# - Running notebooks
# - Training new models
```

---

## Production Deployment

### Using Gunicorn (for FastAPI)

```bash
docker run -d \
  -p 8000:8000 \
  -e WORKERS=4 \
  fashionmnist:latest \
  gunicorn src.api_server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Using Kubernetes

Create a deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fashionmnist-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fashionmnist-api
  template:
    metadata:
      labels:
        app: fashionmnist-api
    spec:
      containers:
        - name: api
          image: fashionmnist:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
```

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy service
docker service create \
  --name fashionmnist-api \
  -p 8000:8000 \
  --replicas 3 \
  fashionmnist:latest
```

### Environment Variables

Set configuration via environment:

```bash
docker run -e WORKERS=4 \
           -e LOG_LEVEL=INFO \
           -e DEVICE=cuda \
           fashionmnist:latest
```

---

## Troubleshooting

### Issue: Container fails to start

**Solution:**

```bash
# Check logs
docker logs fashionmnist-api

# Check image exists
docker images | grep fashionmnist

# Rebuild image
docker build -f docker/Dockerfile -t fashionmnist:latest .
```

### Issue: Port already in use

**Solution:**

```bash
# Check what's using the port
lsof -i :8000

# Use different port
docker run -p 8001:8000 fashionmnist:latest
```

### Issue: Out of memory

**Solution:**

```bash
# Limit memory usage
docker run -m 2g fashionmnist:latest

# Or increase Docker memory limit in Docker Desktop settings
```

### Issue: GPU not detected

**Solution:**

```bash
# Verify GPU support
docker run --gpus all fashionmnist:latest nvidia-smi

# Check NVIDIA Docker plugin
nvidia-docker version
```

### Issue: Slow predictions

**Solution:**

```bash
# Use GPU if available
docker run --gpus all fashionmnist:latest

# Increase number of workers
docker run -e WORKERS=8 fashionmnist:latest
```

---

## Performance Tuning

### CPU Optimization

```bash
docker run --cpus="2" \
           --cpuset-cpus="0,1" \
           fashionmnist:latest
```

### Memory Optimization

```bash
docker run -m 4g \
           --memory-swap 4g \
           fashionmnist:latest
```

### GPU Configuration

```bash
# Enable specific GPU
docker run --gpus '"device=0"' fashionmnist:latest

# Enable all GPUs
docker run --gpus all fashionmnist:latest
```

---

## Monitoring

### View Container Metrics

```bash
docker stats fashionmnist-api
```

### Check Container Logs

```bash
# Real-time logs
docker logs -f fashionmnist-api

# Last 100 lines
docker logs --tail 100 fashionmnist-api

# With timestamps
docker logs -f --timestamps fashionmnist-api
```

### Access Application Logs

```bash
# Inside container
docker exec fashionmnist-api tail -f /app/logs/app.log
```

---

## Security Considerations

### Run as Non-Root

```bash
# In Dockerfile:
USER appuser

# Or at runtime:
docker run --user 1000:1000 fashionmnist:latest
```

### Network Isolation

```bash
docker network create fashionmnist-net
docker run --network fashionmnist-net fashionmnist:latest
```

### Secret Management

```bash
# Use Docker secrets (in swarm mode)
echo "my_secret_key" | docker secret create db_password -

# Reference in compose file:
secrets:
  db_password:
    external: true
```

---

## Cleanup

### Stop All Services

```bash
docker-compose -f docker/docker-compose.yml down
```

### Remove Images

```bash
docker rmi fashionmnist:latest
```

### Remove Dangling Resources

```bash
docker system prune -a
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Streamlit Deployment](https://docs.streamlit.io/library/deploy)

---

For issues or questions, please refer to the main [README.md](../README.md) or create an issue in the repository.
