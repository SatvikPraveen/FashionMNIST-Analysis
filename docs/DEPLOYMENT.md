# Deployment

How to run the trained model as a service: a single API container, or the
full stack with the Streamlit dashboard, the Gradio demo and Jupyter.

Everything on this page was run end to end on 2026-09-27 (Docker 29.7,
Apple Silicon, CPU image): the image builds, the API loads the shipped
TinyVGG and classifies 186 of 200 official test images correctly (93%,
matching its test accuracy), and all four compose services start and answer.
The GPU build is the one part not tested here; see [GPU image](#gpu-image).

## Contents

1. [What gets deployed](#what-gets-deployed)
2. [Build the image](#build-the-image)
3. [Run the API](#run-the-api)
4. [Run the full stack](#run-the-full-stack)
5. [Use the API](#use-the-api)
6. [Models and volumes](#models-and-volumes)
7. [GPU image](#gpu-image)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

## What gets deployed

| Service | Port | Command | Purpose |
|---|---|---|---|
| `api` | 8000 | `uvicorn src.serving.api:app` | REST API: load a checkpoint, classify images |
| `dashboard` | 8501 | `streamlit run apps/streamlit_dashboard.py` | single and batch prediction, model comparison |
| `gradio` | 7860 | `python apps/gradio_app.py` | minimal upload-and-classify demo |
| `jupyter` | 8888 (localhost only) | `jupyter lab` | development; needs a token |

All four use the same image, built from `docker/Dockerfile`. Every service
accepts images of any size, colour or grayscale, converts them to the 28 × 28
grayscale input the models were trained on, and inverts photos with a light
background to match Fashion-MNIST.

## Build the image

```bash
docker build -f docker/Dockerfile -t fashionmnist:latest .
```

The build installs the CPU-only PyTorch wheels and then `requirements.txt`.
The resulting image is about 5.3 GB; with the default PyTorch wheels, which
bundle CUDA libraries, it was 12.3 GB. The build context excludes `data/`,
`models/`, `results/` and `docs/` (see `.dockerignore`), so trained weights
are mounted at run time rather than baked in.

## Run the API

```bash
docker run -d --name fashionmnist-api -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/config.yaml:/app/config.yaml" \
  fashionmnist:latest

curl http://localhost:8000/health
```

The container has a built-in health check (`docker ps` shows `healthy`). The
API starts without a model; load one with `/initialize` (next sections).

## Run the full stack

```bash
cd docker
docker compose up -d api dashboard gradio      # add `jupyter` for development
docker compose ps
docker compose logs -f api
docker compose down
```

| Open | URL |
|---|---|
| API and interactive docs | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |
| Gradio demo | http://localhost:7860 |
| Jupyter Lab | http://127.0.0.1:8888 (see [Security](#security)) |

Compose mounts `models/`, `data/`, `results/`, `logs/` and `config.yaml` from
the repository into every service, and restarts services unless stopped. The
`api` service runs with `--reload` so code changes apply immediately; drop it
for production. After changing dependencies, rebuild with
`docker compose build`.

## Use the API

```bash
# load the shipped best model (architecture read from best_model_info.json)
curl -X POST "http://localhost:8000/initialize?model_path=/app/models/best_model_weights/best_model_weights.pth&config_path=/app/config.yaml"

curl http://localhost:8000/model/info
curl -F "file=@shirt.png" http://localhost:8000/predict
curl -F "files=@a.png" -F "files=@b.png" http://localhost:8000/predict/batch
```

`/predict` returns the predicted class, its name, the confidence, the top 3
classes and all ten probabilities. Paths passed to `/initialize` are paths
inside the container. The full list of endpoints is in the
[feature guide](FEATURES.md#inference-and-serving), and the request and
response schemas at `/docs`.

## Models and volumes

`/initialize` can load any checkpoint written by `src/cli/train.py`: the
architecture comes from the `_spec.json` file next to the weights, or from
`best_model_info.json` for the model in `models/best_model_weights/`. To serve
a model you trained, keep its weights and spec file under `models/` and pass
the in-container path, for example
`/app/models/all_models/resnet/resnet_best.pth`.

The dashboard and Gradio app list the shipped best model and the three custom
CNNs. A CNN without trained weights in `models/all_models/<name>/` is shown as
untrained rather than silently guessing.

## GPU image

Not tested here (no NVIDIA GPU on the machine used). Build with a CUDA
PyTorch index and run with the NVIDIA container toolkit installed:

```bash
docker build -f docker/Dockerfile -t fashionmnist:gpu \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
docker run --gpus all -p 8000:8000 -v "$PWD/models:/app/models" \
  -v "$PWD/config.yaml:/app/config.yaml" fashionmnist:gpu
```

Set `model.device` in `config.yaml` to `cuda` or `auto`. For these small models
a CPU serves a prediction in milliseconds, so a GPU mainly matters for the
timm backbones at 224 px.

## Security

- **Jupyter** listens on 127.0.0.1 only and requires a token. Set one before
  starting it, then open the URL printed in the logs:

  ```bash
  export JUPYTER_TOKEN=$(openssl rand -hex 16)
  docker compose up -d jupyter
  docker compose logs jupyter | grep token
  ```

  It mounts the whole repository and can run any code, so never publish its
  port beyond localhost.
- **The API, dashboard and Gradio** have no authentication and listen on all
  interfaces. Put them behind a reverse proxy with authentication, or change
  their port mappings to `127.0.0.1:<port>:<port>`, before exposing them to a
  network you do not trust.
- **`/initialize`** loads whatever file path it is given. Only expose it to
  trusted callers; checkpoints are loaded with `weights_only=True`, which
  refuses arbitrary pickled code.

## Troubleshooting

**`/initialize` returns `libGL.so.1: cannot open shared object file`.** The
image was built from an older Dockerfile. Rebuild; the current one installs
the system libraries OpenCV needs.

**`/initialize` returns `[Errno 5] Input/output error` on macOS.** The file is
in a folder synced by iCloud (for example `Documents`) and has been offloaded
to the cloud; containers cannot trigger the download. Download it
(`brctl download <path>`), or better, keep the project outside iCloud-synced
folders or turn off "Optimise Mac Storage". Docker Desktop may keep reporting
the error for that mount until it is restarted.

**A port is already in use.** Change the left-hand side of the port mapping,
for example `-p 8001:8000`, and find the other user with `lsof -i :8000`.

**The container is killed while loading a large backbone.** Raise the memory
limit in Docker Desktop, or pass `-m 4g` to `docker run`.

**Code changes do not show up.** Rebuild the image. In the compose stack the
`api` service reloads Python changes automatically; the others need
`docker compose restart <service>`.
