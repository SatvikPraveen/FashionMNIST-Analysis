# Docker

Quick reference. The full, tested guide (API usage, volumes, GPU build,
security, troubleshooting) is [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).

| Service | Port | What it is |
|---|---|---|
| `api` | 8000 | FastAPI server; interactive docs at `/docs` |
| `dashboard` | 8501 | Streamlit dashboard |
| `gradio` | 7860 | Gradio demo |
| `jupyter` | 127.0.0.1:8888 | Jupyter Lab for development; requires `JUPYTER_TOKEN` |

```bash
# one image for all services (CPU PyTorch, about 5.3 GB)
docker build -f docker/Dockerfile -t fashionmnist:latest .

# the stack, from this folder
cd docker
docker compose up -d api dashboard gradio
docker compose ps
docker compose logs -f api
docker compose down

# load the shipped model and classify an image
curl -X POST "http://localhost:8000/initialize?model_path=/app/models/best_model_weights/best_model_weights.pth&config_path=/app/config.yaml"
curl -F "file=@shirt.png" http://localhost:8000/predict
```

Compose mounts `models/`, `data/`, `results/`, `logs/` and `config.yaml`
from the repository, so trained weights are never baked into the image.
