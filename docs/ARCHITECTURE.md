# FashionMNIST-Analysis Architecture

> **Comprehensive guide to the project structure, design patterns, and data flow**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Data Pipeline](#data-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Module Dependencies](#module-dependencies)
8. [Design Patterns](#design-patterns)

---

## Project Overview

FashionMNIST-Analysis is a production-ready deep learning project for fashion item classification with:
- **Modular architecture** separating data, models, training, and inference
- **Configuration-driven** design using YAML
- **Device-agnostic** training (CPU, CUDA, MPS for Apple Silicon)
- **Advanced augmentation** including Mixup, CutMix, and torchvision transforms
- **Multiple interfaces**: CLI scripts, Jupyter notebooks, web apps (Gradio/Streamlit), REST API

---

## Directory Structure

```
FashionMNIST-Analysis/
│
├── 📁 data/                      # Data storage (gitignored)
│   ├── raw/                      # Raw downloaded datasets
│   ├── processed/                # Preprocessed CSV files
│   └── FashionMNIST/             # PyTorch dataset cache
│
├── 📁 models/                    # Saved model weights
│   ├── all_models/               # All trained model checkpoints
│   └── best_model_weights/       # Best performing model
│
├── 📁 checkpoints/               # Training checkpoints (gitignored)
│   └── *.pth                     # Intermediate training states
│
├── 📁 figures/                   # Generated visualizations
│   ├── EDA_plots/                # Exploratory data analysis
│   ├── modeling_plots/           # Training curves, losses
│   ├── evaluation_plots/         # Confusion matrices, predictions
│   └── Traditional_ML_Algo_plots/
│
├── 📁 results/                   # Training/evaluation results
│   ├── fine_tuning_results/      # Hyperparameter search results
│   └── Traditional_ML_Algo_results/
│
├── 📁 logs/                      # Application logs (gitignored)
│   └── *.log                     # Training and inference logs
│
├── 📁 src/                       # Core Python modules
│   ├── config.py                 # Configuration management
│   ├── dataset.py                # PyTorch Dataset classes
│   ├── data_augmentation.py      # Augmentation techniques
│   ├── model_definitions.py      # Neural network architectures
│   ├── utils.py                  # Training utilities
│   ├── evaluation.py             # Metrics and evaluation
│   ├── transfer_learning.py      # Pretrained model support
│   ├── ensemble.py               # Ensemble methods
│   ├── explainability.py         # Grad-CAM, attention maps
│   ├── real_world_inference.py   # Production inference
│   ├── monitoring.py             # MLflow, WandB integration
│   └── api_server.py             # FastAPI REST server
│
├── 📁 scripts/                   # Executable training scripts
│   ├── prepare_data.py           # Data download & preprocessing
│   ├── train.py                  # Main training pipeline
│   ├── finetune.py               # Hyperparameter tuning
│   └── main.py                   # Model evaluation (legacy)
│
├── 📁 notebooks/                 # Jupyter notebooks
│   ├── training_demo.ipynb       # Quick start training demo
│   ├── DataPreparation.ipynb     # Data exploration
│   ├── modeling.ipynb            # Model training (legacy)
│   ├── finetuning.ipynb          # Fine-tuning (legacy)
│   ├── evaluate_best_model.ipynb # Evaluation
│   └── Traditional_ML_Algo.ipynb # Classical ML methods
│
├── 📁 apps/                      # Web applications
│   ├── gradio_app.py             # Gradio demo UI
│   └── streamlit_dashboard.py    # Comprehensive dashboard
│
├── 📁 tests/                     # Unit and integration tests
│   ├── test_models.py            # Model architecture tests
│   ├── test_utils.py             # Utility function tests
│   └── test_inference.py         # Inference pipeline tests
│
├── 📁 docs/                      # Documentation
│   ├── ARCHITECTURE.md           # This file
│   ├── USAGE_GUIDE.md            # User guide
│   ├── PROJECT_REVAMP_PLAN.md    # Development plan
│   ├── FEATURES.md               # Feature documentation
│   ├── DEPLOYMENT.md             # Deployment guide
│   ├── IMPLEMENTATION_SUMMARY.md # Implementation notes
│   ├── CODE_OF_CONDUCT.md
│   └── CONTRIBUTING.md
│
├── 📁 docker/                    # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📁 eda/                       # Exploratory Data Analysis
│   └── EDA.ipynb
│
├── 📄 config.yaml                # Main configuration file
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project overview
├── 📄 LICENSE
├── 📄 setup_project.py           # Project setup script
└── 📄 .gitignore
```

---

## Core Components

### 1. **Configuration Module** (`src/config.py`)

**Purpose**: Centralized configuration management

```python
from src.config import load_config

config = load_config("config.yaml")
batch_size = config.training.batch_size
device = config.model.device
```

**Features**:
- YAML-based configuration
- Dot-notation access
- Type-safe attribute access
- Environment-specific overrides

---

### 2. **Dataset Module** (`src/dataset.py`)

**Purpose**: PyTorch Dataset implementations with augmentation support

```python
from src.dataset import FashionMNISTDataset, get_augmentation_transforms

transforms = get_augmentation_transforms(config, train=True)
dataset = FashionMNISTDataset(csv_path, transform=transforms)
```

**Classes**:
- `FashionMNISTDataset`: CSV-based dataset
- `FashionMNISTFromTorch`: Direct PyTorch dataset wrapper
- `get_augmentation_transforms()`: Augmentation factory

---

### 3. **Data Augmentation** (`src/data_augmentation.py`)

**Purpose**: Advanced augmentation techniques

**Techniques**:
- **Mixup**: Blends pairs of images
- **CutMix**: Cuts and pastes image regions
- **RandomErasing**: Random rectangular masking
- **AugmentationPipeline**: Orchestrates multiple augmentations

**Integration**:
```python
from src.data_augmentation import AugmentationPipeline

aug_pipeline = AugmentationPipeline({
    'mixup': {'alpha': 1.0},
    'cutmix': {'alpha': 1.0}
})

# In training loop
if config.augmentation.mixup:
    x, y_a, y_b, lam = aug_pipeline.apply_mixup(x, y)
    loss = aug_pipeline.mixup.mixup_criterion(criterion, pred, y_a, y_b, lam)
```

---

### 4. **Model Definitions** (`src/model_definitions.py`)

**Purpose**: Neural network architectures

**Models**:
- **MiniCNN**: Lightweight 2-layer CNN (~150K parameters)
- **TinyVGG**: VGG-inspired architecture (~550K parameters)
- **ResNet**: ResNet-18 with BasicBlocks (~11M parameters)

**Features**:
- Batch normalization
- Dropout regularization
- Residual connections (ResNet)
- Flexible output dimensions

---

### 5. **Transfer Learning** (`src/transfer_learning.py`)

**Purpose**: Pretrained model integration

**Supported Models**:
- Vision Transformer (ViT)
- EfficientNet (B0-B7)
- ResNet50
- All TIMM models

**Features**:
- Automatic grayscale→RGB conversion
- Backbone freezing/unfreezing
- Device-aware loading

---

### 6. **Training Utilities** (`src/utils.py`)

**Purpose**: Training loop helpers

**Functions**:
- `train_step()`: Single epoch training with augmentation support
- `validation_step()`: Validation loop
- Device detection and management

---

## Data Pipeline

### Flow Diagram

```
┌─────────────────────────┐
│  FashionMNIST Dataset   │
│   (torchvision)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  prepare_data.py        │
│  - Download dataset     │
│  - Split: 80/10/10      │
│  - Save as CSV          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  data/processed/        │
│  - train.csv            │
│  - val.csv              │
│  - test.csv             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  FashionMNISTDataset    │
│  (src/dataset.py)       │
│  - Loads CSV            │
│  - Applies transforms   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  DataLoader             │
│  - Batching             │
│  - Shuffling            │
│  - Multi-processing     │
└─────────────────────────┘
```

### Augmentation Integration

```
Image Batch
     │
     ├─→ Basic Transforms (resize, normalize)
     │
     ├─→ Mixup/CutMix (if enabled)
     │     └─→ Blends entire batch
     │
     └─→ Sample augmentations (RandomErasing)
           └─→ Per-image transformations
```

---

## Training Pipeline

### High-Level Flow

```
┌──────────────┐
│  config.yaml │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  train.py                │
│  1. Load config          │
│  2. Setup device         │
│  3. Create datasets      │
│  4. Initialize model     │
│  5. Setup optimizer      │
│  6. Training loop        │
│  7. Save checkpoints     │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Checkpoints/            │
│  - epoch_*.pth           │
│  - best_model.pth        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  models/best_model_weights/│
│  - final_model.pth       │
└──────────────────────────┘
```

### Training Loop Detail

```python
for epoch in range(epochs):
    # Training phase
    for batch, (X, y) in enumerate(train_loader):
        # Device transfer
        X, y = X.to(device), y.to(device)
        
        # Apply mixing augmentation if enabled
        if config.augmentation.mixup:
            X, y_a, y_b, lam = mixup(X, y)
            
        # Forward pass
        pred = model(X)
        
        # Compute loss (mixed or standard)
        loss = criterion(pred, y) if not mixup else mixup_criterion(...)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    with torch.inference_mode():
        val_loss, val_acc = validation_step(model, val_loader, criterion, device)
    
    # Checkpoint saving
    if val_acc > best_val_acc:
        save_checkpoint(model, optimizer, epoch, val_acc)
    
    # Early stopping check
    if not improved_for(patience_epochs):
        break
```

---

## Inference Pipeline

### Real-World Inference Flow

```
User Image (any size, any format)
        │
        ▼
┌────────────────────────────┐
│  ImagePreprocessor         │
│  - Resize to 224x224       │
│  - Grayscale → RGB         │
│  - Normalize [0,1]         │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Model.forward()           │
│  - Feature extraction      │
│  - Classification head     │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│  RealWorldInference        │
│  - Softmax probabilities   │
│  - Top-K predictions       │
│  - Confidence thresholding │
└────────┬───────────────────┘
         │
         ▼
    Prediction Result
```

---

## Module Dependencies

```
┌─────────────┐
│ config.yaml │
└──────┬──────┘
       │
       ├─→ config.py ─┐
       │              │
       ▼              ▼
   dataset.py ←─ data_augmentation.py
       │              │
       ▼              ▼
   DataLoader    AugmentationPipeline
       │              │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  train.py    │
       └──────┬───────┘
              │
              ├─→ model_definitions.py
              ├─→ transfer_learning.py
              ├─→ utils.py (train_step/val_step)
              ├─→ evaluation.py
              └─→ monitoring.py
```

---

## Design Patterns

### 1. **Configuration-Driven Design**
- All hyperparameters in `config.yaml`
- Environment-specific overrides supported
- Type-safe access via `Config` class

### 2. **Factory Pattern**
- `get_augmentation_transforms()`: Creates transform pipelines
- `create_model()`: Model instantiation
- `get_optimizer()`: Optimizer factory

### 3. **Strategy Pattern**
- `AugmentationPipeline`: Composable augmentation strategies
- `EnsembleVoting`: Pluggable voting strategies

### 4. **Observer Pattern**
- `MetricsTracker`: Logs metrics to MLflow/WandB
- Checkpoint callbacks during training

### 5. **Separation of Concerns**
- **Data**: `dataset.py`, `data_augmentation.py`
- **Models**: `model_definitions.py`, `transfer_learning.py`
- **Training**: `train.py`, `utils.py`
- **Evaluation**: `evaluation.py`, `explainability.py`
- **Deployment**: `api_server.py`, `apps/`

---

## Device Management

### Multi-Platform Support

```python
# Automatic device detection
if config.model.device == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")
```

### Best Practices
- Always transfer data **before** augmentation
- Use `torch.inference_mode()` for evaluation
- Enable `pin_memory=True` for DataLoader on GPU
- Set `num_workers` based on CPU cores

---

## Extension Points

### Adding New Models
1. Define architecture in `src/model_definitions.py`
2. Add to model factory in `train.py`
3. Update `config.yaml` with new architecture name

### Adding New Augmentations
1. Implement class in `src/data_augmentation.py`
2. Add to `AugmentationPipeline`
3. Update `config.yaml` with parameters

### Adding New Datasets
1. Create Dataset class in `src/dataset.py`
2. Implement `__getitem__` and `__len__`
3. Update `prepare_data.py` for preprocessing

---

## Performance Considerations

### Memory Optimization
- Use `DataLoader` with `num_workers > 0` for parallel loading
- Enable `pin_memory=True` on GPU
- Batch size: 32-64 for 28x28 images

### Training Speed
- **MPS (Apple Silicon)**: 2-3x faster than CPU
- **CUDA (GPU)**: 10-50x faster than CPU
- Mixed precision training: `torch.cuda.amp` (CUDA only)

### Storage
- Raw data: ~82 MB (FashionMNIST)
- Processed CSV: ~148 MB
- Model weights: ~50 MB per checkpoint
- Use checkpoints sparingly or implement checkpoint rotation

---

## Summary

This architecture provides:
- ✅ **Modularity**: Each component has a single responsibility
- ✅ **Configurability**: YAML-driven hyperparameters
- ✅ **Extensibility**: Easy to add models, augmentations, datasets
- ✅ **Scalability**: Supports distributed training frameworks
- ✅ **Maintainability**: Clear separation between research (notebooks) and production (scripts)
- ✅ **Reproducibility**: Seed management, config versioning, git tracking

For usage examples, see [`docs/USAGE_GUIDE.md`](USAGE_GUIDE.md).
