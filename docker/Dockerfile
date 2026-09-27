# Multi-stage Dockerfile for FashionMNIST-Analysis

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies. PyTorch comes from the CPU-only index first:
# the default wheels bundle several GB of CUDA libraries a CPU serving image
# never uses. Build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
# (or another CUDA index) for a GPU image.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies (libgl1 and libglib2.0-0 are needed by OpenCV,
# which is pulled in by the Grad-CAM and augmentation packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/models /app/data /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command: FastAPI server
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose ports
EXPOSE 8000 7860 8501
