# Docker Setup for FashionMNIST-Analysis

This directory contains Docker configuration for running the FashionMNIST-Analysis project in containers.

## Prerequisites

- Docker Desktop (for macOS: https://docs.docker.com/desktop/install/mac-install/)
- Docker Compose (included with Docker Desktop)

## Services

The Docker setup provides 4 services:

1. **FastAPI API Server** (`api`) - Port 8000
   - REST API for model inference
   - Health check endpoint: http://localhost:8000/health
   - API docs: http://localhost:8000/docs

2. **Streamlit Dashboard** (`dashboard`) - Port 8501
   - Interactive web dashboard for analysis
   - Access at: http://localhost:8501

3. **Gradio App** (`gradio`) - Port 7860
   - Lightweight inference UI
   - Access at: http://localhost:7860

4. **Jupyter Lab** (`jupyter`) - Port 8888 (Optional)
   - Development environment with notebooks
   - Access at: http://localhost:8888
   - No token required (for development only)

## Quick Start

### Run All Services

```bash
cd docker
docker-compose up
```

### Run Specific Service

```bash
# API only
docker-compose up api

# Dashboard only
docker-compose up dashboard

# Gradio only
docker-compose up gradio

# Jupyter only
docker-compose up jupyter
```

### Run in Background (Detached Mode)

```bash
docker-compose up -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
```

### Stop Services

```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Build from Scratch

If you need to rebuild the images (after code changes):

```bash
docker-compose build
docker-compose up
```

Or rebuild and run in one command:

```bash
docker-compose up --build
```

## Volume Mounts

The services mount the following directories from your host:
- `../models` → `/app/models` - Pre-trained models
- `../data` → `/app/data` - Dataset files
- `../results` → `/app/results` - Training results
- `../logs` → `/app/logs` - Application logs
- `../config.yaml` → `/app/config.yaml` - Configuration

This allows you to:
- Use pre-trained models without copying into containers
- Access training results from host machine
- Update config without rebuilding images

## Architecture

### Multi-stage Build
The Dockerfile uses a multi-stage build to optimize image size:
1. **Builder stage**: Installs dependencies
2. **Runtime stage**: Copies only what's needed to run

### Health Checks
All services include health checks for monitoring:
- API service checks `/health` endpoint
- Services marked unhealthy will auto-restart

## Troubleshooting

### Port Already in Use

If you see "port already in use" errors:

```bash
# Find what's using the port
lsof -i :8000  # or :8501, :7860, :8888

# Change port mapping in docker-compose.yml
ports:
  - "8001:8000"  # Maps host 8001 to container 8000
```

### Container Won't Start

```bash
# Check logs
docker-compose logs [service-name]

# Check container status
docker-compose ps

# Restart specific service
docker-compose restart [service-name]
```

### Rebuild After Code Changes

```bash
# Force rebuild without cache
docker-compose build --no-cache

# Remove all containers and rebuild
docker-compose down
docker-compose up --build
```

### Permission Issues (Linux)

On Linux, if you encounter permission issues:

```bash
# Run with sudo
sudo docker-compose up

# Or add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

## Development Workflow

### With Live Reload

The API service uses `--reload` flag for development:
- Changes to code are automatically detected
- No need to rebuild container

### Without Jupyter

If you don't need Jupyter:

```bash
docker-compose up api dashboard gradio
```

Or comment out the jupyter service in docker-compose.yml

## Production Deployment

For production:

1. Remove `--reload` from API command
2. Set proper environment variables
3. Use secrets for sensitive data
4. Configure proper logging
5. Set up reverse proxy (nginx/traefik)
6. Enable HTTPS

## Network

All services run on the `fashionmnist-network` bridge network, allowing them to communicate using service names:
- API accessible from dashboard as `http://api:8000`
- Inter-service communication uses container names

## Resource Limits

To set memory/CPU limits, add to each service:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

## Next Steps

- Access API docs: http://localhost:8000/docs
- Try Streamlit dashboard: http://localhost:8501
- Use Gradio for quick tests: http://localhost:7860
- Develop in Jupyter: http://localhost:8888
