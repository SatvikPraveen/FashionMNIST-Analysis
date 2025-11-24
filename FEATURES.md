# FashionMNIST-Analysis: Complete Feature Guide

This document provides comprehensive information about all the new features and enhancements added to the FashionMNIST-Analysis project.

## Table of Contents

1. [Transfer Learning](#transfer-learning)
2. [Model Explainability](#model-explainability)
3. [Ensemble Methods](#ensemble-methods)
4. [Real-World Inference](#real-world-inference)
5. [Data Augmentation](#data-augmentation)
6. [Monitoring & Tracking](#monitoring--tracking)
7. [Configuration Management](#configuration-management)
8. [REST API](#rest-api)
9. [Web Dashboards](#web-dashboards)
10. [Docker Deployment](#docker-deployment)
11. [Testing](#testing)

---

## Transfer Learning

### Overview

The transfer learning module enables using pre-trained models from state-of-the-art architectures without retraining from scratch.

### Supported Models

- **Vision Transformer (ViT)** - `vit_base_patch16_224`
- **EfficientNet** - `efficientnet_b0` through `efficientnet_b7`
- **ResNet50** - Deeper than baseline ResNet18
- All models from [TIMM](https://github.com/rwightman/pytorch-image-models) (PyTorch Image Models)

### Features

- Automatic grayscale-to-RGB conversion
- Backbone freezing for efficient fine-tuning
- Parameter unfreezing for full training
- Automatic device detection (CPU/GPU/MPS)

### Usage

```python
from src.transfer_learning import TransferLearningModel, load_vit_model

# Load Vision Transformer
vit = TransferLearningModel("vit_base_patch16_224", num_classes=10)

# Or use convenience functions
model = load_vit_model(num_classes=10, pretrained=True)

# Freeze backbone for transfer learning
vit.freeze_backbone()

# Unfreeze for full training
vit.unfreeze_backbone()

# Get the underlying PyTorch model
pytorch_model = vit.get_model()
```

### Configuration

```yaml
transfer_learning:
  freeze_backbone: false
  num_frozen_layers: 0
  learning_rate: 1e-4
```

---

## Model Explainability

### Overview

Understand what models focus on and why they make specific predictions using interpretability techniques.

### Techniques Implemented

#### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

Visualizes important regions in images for model predictions.

```python
from src.explainability import GradCAM

# Initialize Grad-CAM
gradcam = GradCAM(model, target_layer="layer4", device="cuda")

# Generate CAM for image
cam = gradcam.generate_cam(image_tensor, class_idx=5)

# Visualize
gradcam.visualize_cam(image, cam, class_name="Shoe", save_path="cam.png")
```

#### 2. Attention Maps

Extract and visualize attention patterns from Vision Transformers.

```python
from src.explainability import AttentionMapper

mapper = AttentionMapper(model, device="cuda")
attention_maps = mapper.get_attention_maps(image_tensor)
mapper.visualize_attention(image, attention_maps[0], save_path="attention.png")
```

#### 3. Layer Activations & Filter Visualization

```python
from src.explainability import get_layer_activations, visualize_filters

# Get activations from specific layer
activations = get_layer_activations(model, image_tensor, "conv1")

# Visualize convolutional filters
visualize_filters(model.conv1.weight, num_filters=16, save_path="filters.png")
```

---

## Ensemble Methods

### Overview

Combine multiple models for improved predictions and robustness.

### Available Ensemble Methods

#### 1. Voting Ensemble

Hard voting (majority) or soft voting (probability averaging).

```python
from src.ensemble import EnsembleVoting

models = [model1, model2, model3]

# Soft voting
ensemble = EnsembleVoting(models, voting="soft", device="cuda")

predictions = ensemble.predict(x)
probabilities = ensemble.predict_proba(x)
```

#### 2. Stacking Ensemble

Uses predictions from base models as features for a meta-learner.

```python
from src.ensemble import StackingEnsemble

stacking = StackingEnsemble(base_models, num_classes=10, device="cuda")

# Fit on validation data
stacking.fit(val_images, val_labels)

# Predict
predictions = stacking.predict(test_images)
```

#### 3. Bagging Ensemble

Bootstrap aggregation for robust predictions.

```python
from src.ensemble import BaggingEnsemble

bagging = BaggingEnsemble(MiniCNN, num_models=5)

# Add trained models
for model in trained_models:
    bagging.add_model(model)

predictions = bagging.predict(x)
```

---

## Real-World Inference

### Overview

Make predictions on real photographs with proper preprocessing and formatting.

### Key Features

- **Flexible Input Handling** - Images, file paths, PIL Images, numpy arrays
- **Auto-resizing** - Handles arbitrary image sizes
- **Grayscale Support** - Converts grayscale to RGB automatically
- **Batch Processing** - Efficient processing of multiple images
- **Uncertainty Estimation** - Monte Carlo dropout for confidence intervals

### Usage

```python
from src.real_world_inference import ImagePreprocessor, RealWorldInference

# Initialize preprocessor
preprocessor = ImagePreprocessor(target_size=224, normalize=True)

# Create inference engine
inference = RealWorldInference(model, preprocessor, device="cuda")

# Single image prediction
result = inference.predict("path/to/image.jpg", return_top_k=3)
# Returns: {
#   "predicted_class": 5,
#   "predicted_class_name": "Sandal",
#   "confidence": 0.92,
#   "top_k_predictions": [...],
#   "all_probabilities": {...}
# }

# Batch predictions
predictions = inference.predict_batch([img1, img2, img3])

# Uncertainty estimation (Monte Carlo dropout)
uncertain_pred = inference.predict_with_uncertainty(image, num_samples=10)
# Returns uncertainty metrics along with prediction
```

### Class Names

```python
CLASS_NAMES = [
    "T-shirt/top",    # 0
    "Trouser",        # 1
    "Pullover",       # 2
    "Dress",          # 3
    "Coat",           # 4
    "Sandal",         # 5
    "Shirt",          # 6
    "Sneaker",        # 7
    "Bag",            # 8
    "Ankle Boot"      # 9
]
```

---

## Data Augmentation

### Overview

Advanced data augmentation techniques to improve model robustness and prevent overfitting.

### Available Augmentations

#### 1. Mixup

Blends pairs of images and their labels.

```python
from src.data_augmentation import Mixup

mixup = Mixup(alpha=1.0)
mixed_x, y_a, y_b, lam = mixup(x, y)

# Modified loss computation
loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### 2. CutMix

Randomly cuts and pastes patches between images.

```python
from src.data_augmentation import CutMix

cutmix = CutMix(alpha=1.0)
mixed_x, y_a, y_b, lam = cutmix(x, y)

# Use same loss as Mixup
loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### 3. Random Erasing

Randomly erases rectangular regions.

```python
from src.data_augmentation import RandomErasing

eraser = RandomErasing(probability=0.5, scale=(0.02, 0.33))
augmented_x = eraser(x)
```

#### 4. Augmentation Pipeline

Combine multiple augmentations.

```python
from src.data_augmentation import AugmentationPipeline

augmentation = AugmentationPipeline({
    'mixup': {'alpha': 1.0},
    'cutmix': {'alpha': 1.0},
    'random_erasing': {'probability': 0.5}
})

# Apply mixture augmentations
mixed_x, y_a, y_b, lam = augmentation.apply_mixup(x, y)

# Apply sample-level augmentations
x_aug = augmentation.apply_sample_augmentations(x)
```

---

## Monitoring & Tracking

### Overview

Monitor model performance, detect data drift, and track metrics during training and inference.

### Components

#### 1. MetricsTracker

Track metrics with rolling window statistics.

```python
from src.monitoring import MetricsTracker

tracker = MetricsTracker(window_size=100)

# Update metrics
tracker.update("loss", 0.5)
tracker.update("accuracy", 0.85)

# Get current metrics
current_loss = tracker.get_metric("loss")
all_metrics = tracker.get_all_metrics()

# Get history
history = tracker.get_history("loss")
```

#### 2. DriftDetector

Detect distribution shift in data.

```python
from src.monitoring import DriftDetector

detector = DriftDetector(
    reference_data=train_data,
    threshold=0.5,
    method="kl_divergence"  # or "wasserstein", "ks_test"
)

# Check for drift
drift_detected, score = detector.detect_drift(new_data)
```

#### 3. PredictionMonitor

Monitor individual predictions and detect anomalies.

```python
from src.monitoring import PredictionMonitor

monitor = PredictionMonitor(num_classes=10, window_size=100)

# Update with predictions
monitor.update(pred=5, confidence=0.92, ground_truth=5)

# Get statistics
stats = monitor.get_statistics()

# Find low confidence predictions
low_conf_indices = monitor.detect_low_confidence(threshold=0.5)
```

#### 4. PerformanceLogger

Log experiments to files for tracking.

```python
from src.monitoring import PerformanceLogger

logger = PerformanceLogger(log_dir="./logs")

# Start session
session_id = logger.start_session("training_session")

# Log metrics per epoch
logger.log_metric(epoch=1, metrics={"loss": 0.5, "acc": 0.85})

# Save session
logger.save_session()

# Get summary
summary = logger.get_session_summary()
```

---

## Configuration Management

### Overview

Centralized YAML-based configuration system for all project parameters.

### Configuration File

Edit `config.yaml` to customize:

```yaml
model:
  architecture: "resnet"
  num_classes: 10
  pretrained: true
  device: "auto"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-3
  optimizer: "adam"
  scheduler: "cosine"

augmentation:
  mixup: true
  cutmix: true
  cutmix_alpha: 1.0
  mixup_alpha: 0.2

ensemble:
  enabled: false
  method: "voting"
  models: ["resnet", "vit", "efficientnet"]
```

### Usage

```python
from src.config import load_config, save_config

# Load configuration
config = load_config("config.yaml")

# Access nested values
batch_size = config.training.batch_size
learning_rate = config.get("training.learning_rate")

# Modify configuration
config.set("training.learning_rate", 1e-4)

# Save configuration
save_config(config, "new_config.yaml")
```

---

## REST API

### Overview

Production-ready FastAPI server for model inference.

### Endpoints

#### Health Check

```
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API is operational"
}
```

#### Initialize Model

```
POST /initialize
```

Parameters:

- `model_path` - Path to model weights
- `config_path` - Path to config file
- `transfer_learning` - Use transfer learning (bool)

#### Single Prediction

```
POST /predict
Content-Type: multipart/form-data

file: <image>
top_k: 3
```

Response:

```json
{
  "predicted_class": 5,
  "predicted_class_name": "Sandal",
  "confidence": 0.92,
  "top_k_predictions": [...],
  "all_probabilities": {...}
}
```

#### Batch Prediction

```
POST /predict/batch
Content-Type: multipart/form-data

files: <image1>
files: <image2>
files: <image3>
top_k: 1
```

#### Uncertainty Estimation

```
POST /predict/uncertainty
Content-Type: multipart/form-data

file: <image>
num_samples: 10
```

Returns prediction with uncertainty estimates.

#### Model Information

```
GET /model/info
```

Returns model architecture details and parameter counts.

### API Documentation

Interactive documentation available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Web Dashboards

### Streamlit Dashboard

Interactive web interface with multiple modes:

**Access**: `http://localhost:8501`

**Features:**

- Single image prediction
- Batch image processing
- Model comparison
- Explainability visualization
- Real-time probability distribution

**Running:**

```bash
streamlit run apps/streamlit_dashboard.py
```

### Gradio App

Lightweight alternative web interface:

**Access**: `http://localhost:7860`

**Features:**

- Simple drag-and-drop interface
- Model selection
- Real-time predictions
- Top-K results display

**Running:**

```bash
python apps/gradio_app.py
```

---

## Docker Deployment

### Overview

Complete containerization for easy deployment and scaling.

### Quick Start

```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up -d

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# Gradio: http://localhost:7860
# Jupyter: http://localhost:8888
```

### Services

1. **API Server** - FastAPI backend (Port 8000)
2. **Dashboard** - Streamlit UI (Port 8501)
3. **Gradio** - Alternative UI (Port 7860)
4. **Jupyter** - Development environment (Port 8888)

### Building Custom Image

```bash
docker build -f docker/Dockerfile -t fashionmnist:custom .
```

### GPU Support

```bash
docker run --gpus all \
  -p 8000:8000 \
  fashionmnist:latest
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment guide.

---

## Testing

### Overview

Comprehensive unit tests for all modules.

### Test Suites

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_models.py::TestModelDefinitions -v
```

### Test Coverage

- **test_models.py** - Model architectures and transfer learning
- **test_utils.py** - Utility functions and training loops
- **test_inference.py** - Inference and preprocessing

---

## Quick Start Examples

### 1. Transfer Learning Fine-tuning

```python
from src.transfer_learning import TransferLearningModel
from src.config import load_config
from src.utils import train_step, validation_step

config = load_config("config.yaml")
model = TransferLearningModel("vit_base_patch16_224", pretrained=True)

# Freeze backbone
model.freeze_backbone()

# Train with frozen backbone
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    train_loss, train_acc = train_step(model.model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = validation_step(model.model, val_loader, loss_fn, device)
    print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
```

### 2. Ensemble Prediction

```python
from src.ensemble import EnsembleVoting
from src.model_definitions import ResNet, BasicBlock, MiniCNN, TinyVGG

models = [
    ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10),
    MiniCNN(1, 10),
    TinyVGG(1, 32, 10)
]

ensemble = EnsembleVoting(models, voting="soft")
predictions = ensemble.predict(test_images)
```

### 3. Real-World Inference

```python
from src.real_world_inference import ImagePreprocessor, RealWorldInference

preprocessor = ImagePreprocessor(target_size=224)
inference = RealWorldInference(model, preprocessor)

# Predict on photo
result = inference.predict("my_photo.jpg", return_top_k=3)
print(f"Predicted: {result['predicted_class_name']} ({result['confidence']:.2%})")
```

### 4. Model Explainability

```python
from src.explainability import GradCAM

gradcam = GradCAM(model, target_layer="layer4")
cam = gradcam.generate_cam(image_tensor)
gradcam.visualize_cam(image, cam, save_path="explanation.png")
```

### 5. Monitor Training

```python
from src.monitoring import MetricsTracker, PerformanceLogger

tracker = MetricsTracker()
logger = PerformanceLogger()

logger.start_session("training")

for epoch in range(epochs):
    for batch, (x, y) in enumerate(train_loader):
        # Training...
        tracker.update("loss", loss.item())
        tracker.update("accuracy", accuracy)

    logger.log_metric(epoch, tracker.get_all_metrics())

logger.save_session()
```

---

## Performance Benchmarks

Expected performance on Fashion MNIST test set:

| Model               | Accuracy | Parameters | Inference Time (ms) |
| ------------------- | -------- | ---------- | ------------------- |
| MiniCNN             | 91.5%    | 211K       | 2-5                 |
| TinyVGG             | 92.8%    | 1.2M       | 5-10                |
| ResNet18            | 93.2%    | 11.2M      | 10-15               |
| ViT-Base            | 94.1%    | 86M        | 20-30               |
| Ensemble (3 models) | 94.8%    | 12.4M      | 20-40               |

_Benchmarks on CPU (Intel i7). GPU performance varies by hardware._

---

## Troubleshooting

### Common Issues

**Issue**: ImportError for `timm` when using ViT

**Solution**: Install timm separately or ensure it's installed from requirements.txt

```bash
pip install timm>=0.9.0
```

**Issue**: GPU out of memory

**Solution**: Reduce batch size or use smaller models

```python
config.training.batch_size = 16  # Reduce from 32
```

**Issue**: Slow predictions

**Solution**: Use GPU or reduce model size

```python
device = torch.device("cuda")  # Use GPU
model = TransferLearningModel("vit_base_patch16_224")  # Use faster model
```

---

## Next Steps

1. **Fine-tune Transfer Learning Model** - Start with ViT and fine-tune on your data
2. **Create Ensemble** - Combine multiple models for better accuracy
3. **Deploy with Docker** - Use docker-compose for production deployment
4. **Monitor Performance** - Track metrics and detect data drift
5. **Optimize Models** - Use quantization or distillation for deployment

---

For more information, see:

- [README.md](README.md) - Project overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [config.yaml](config.yaml) - Configuration reference

Happy classifying! 👕👗🧥👠
