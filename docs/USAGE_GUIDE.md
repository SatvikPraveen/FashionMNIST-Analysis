# Usage Guide - FashionMNIST-Analysis

Complete guide for using the new training pipeline with data augmentation and device support.

---

## 🚀 Quick Start

### 1. Prepare Data

```bash
# Download and prepare FashionMNIST dataset
python scripts/prepare_data.py --output-dir data_preparation

# Options:
python scripts/prepare_data.py \
    --data-dir ./data \
    --output-dir ./data_preparation \
    --train-split 0.8 \
    --seed 42
```

### 2. Train Models  

```bash
# Train all models (MiniCNN, TinyVGG, ResNet)
python scripts/train.py --model all

# Train specific model
python scripts/train.py --model resnet

# Use CSV data instead of torchvision
python scripts/train.py --model minicnn --use-csv \
    --train-csv data_preparation/fashion_mnist_train.csv \
    --val-csv data_preparation/fashion_mnist_val.csv \
    --test-csv data_preparation/fashion_mnist_test.csv
```

### 3. Fine-tune with Hyperparameter Search

```bash
# Fine-tune ResNet with grid search
python scripts/finetune.py --model resnet \
    --pretrained models/all_models/resnet/resnet_best.pth \
    --learning-rates 1e-5 5e-6 \
    --batch-sizes 32 64 \
    --patience-values 2 3

# Quick test with single set of hyperparameters
python scripts/finetune.py --model minicnn \
    --learning-rates 1e-4 \
    --batch-sizes 32 \
    --patience-values 2
```

### 4. Evaluate Best Model

```bash
# Evaluate model on test set
python main.py \
    --model_path models/all_models/resnet/resnet_best.pth \
    --test_csv data_preparation/fashion_mnist_test.csv \
    --test_dir results/evaluation
```

---

## 📋 Command Reference

### prepare_data.py

```bash
python scripts/prepare_data.py [options]

Options:
  --data-dir DIR          Raw data directory (default: ./data)
  --output-dir DIR        Output CSV directory (default: ./data_preparation)
  --train-split FLOAT     Train/val split ratio (default: 0.8)
  --no-csv                Skip CSV conversion (only download)
  --seed INT              Random seed (default: 42)
```

### train.py

```bash
python scripts/train.py [options]

Options:
  --config FILE           Config file path (default: config.yaml)
  --model {minicnn,tinyvgg,resnet,all}
                          Model to train (default: all)
  --output-dir DIR        Output directory (default: ./models/all_models)
  --use-csv               Use CSV datasets instead of torchvision
 --train-csv FILE        Path to training CSV (if --use-csv)
  --val-csv FILE          Path to validation CSV (if --use-csv)
  --test-csv FILE         Path to test CSV (if --use-csv)
  --force-cpu             Force CPU usage (disable GPU/MPS)
```

### finetune.py

```bash
python scripts/finetune.py [options]

Options:
  --config FILE           Config file path (default: config.yaml)
  --model {minicnn,tinyvgg,resnet}
                          Model to fine-tune (required)
  --output-dir DIR        Results directory (default: ./results/fine_tuning_results)
  --pretrained FILE       Path to pretrained weights (optional)
  --learning-rates LR...  Learning rates to try (default: 1e-5 5e-6)
  --batch-sizes BS...     Batch sizes to try (default: 32 64)
  --patience-values P...  Patience values to try (default: 2 3)
  --force-cpu             Force CPU usage
```

---

## ⚙️ Configuration (config.yaml)

All training parameters are centralized in `config.yaml`:

### Training Configuration

```yaml
training:
  epochs: 50                    # Number of training epochs
  batch_size: 32                # Batch size
  learning_rate: 1e-3           # Initial learning rate
  weight_decay: 1e-4            # L2 regularization
  optimizer: "adam"             # adam or sgd
  scheduler: "cosine"           # cosine, step, exponential, none
  early_stopping_patience: 5    # Early stopping patience
```

### Data Augmentation

```yaml
augmentation:
  enabled: true                 # Enable/disable augmentation
  strategy: "advanced"          # basic, advanced, custom
  rotation: 15                  # Random rotation degrees
  zoom: 0.2                     # Random zoom factor
  horizontal_flip: true         # Random horizontal flip
  vertical_flip: false          # Random vertical flip
  cutmix: true                  # Enable CutMix
  cutmix_alpha: 1.0             # CutMix alpha parameter
  mixup: true                   # Enable Mixup
  mixup_alpha: 0.2              # Mixup alpha parameter
```

---

## 🖥️ Device Support

The pipeline automatically detects the best available device:

| Priority | Device | Description |
|----------|--------|-------------|
| 1 | **CUDA** | NVIDIA GPU (highest performance) |
| 2 | **MPS** | Apple Silicon M1/M2/M3 (MacBook Pro) |
| 3 | **CPU** | Fallback for all systems |

### Force CPU Usage

```bash
python scripts/train.py --model resnet --force-cpu
```

---

## 📊 Output Structure

After training, you'll have:

```
models/all_models/
├── minicnn/
│   ├── minicnn_best.pth           # Best MiniCNN model weights
│   └── minicnn_history.json       # Training history
├── tinyvgg/
│   ├── tinyvgg_best.pth           # Best TinyVGG model weights
│   └── tinyvgg_history.json       # Training history
└── resnet/
    ├── resnet_best.pth            # Best ResNet model weights
    └── resnet_history.json        # Training history

results/fine_tuning_results/
├── resnet_bs32_lr1e-05_pat2.pth
├── resnet_bs32_lr1e-05_pat3.pth
├── ...
└ resnet_finetuning_results.json  # All results summary

results/evaluation/
├── predictions_vector.csv
├── evaluation_metrics.csv
├── Best_ResNet_confusion_matrix.png
└── prediction_visualization.png
```

---

## 💡 Usage Examples

### Example 1: Full Training Pipeline

```bash
# 1. Prepare data
python scripts/prepare_data.py

# 2. Train all models (will take ~30-60 minutes)
python scripts/train.py --model all

# 3. Fine-tune best model
python scripts/finetune.py --model resnet \
    --pretrained models/all_models/resnet/resnet_best.pth

# 4. Evaluate
python main.py \
    --model_path models/all_models/resnet/resnet_best.pth \
    --test_csv data_preparation/fashion_mnist_test.csv
```

### Example 2: Quick Test (5 minutes)

```bash
# Modify config.yaml: Set epochs: 2

# Train single model
python scripts/train.py --model minicnn

# Check results
cat models/all_models/minicnn/minicnn_history.json
```

### Example 3: Use Pre-existing CSV Data

```bash
# Train using existing CSV files
python scripts/train.py --model resnet --use-csv \
    --train-csv path/to/train.csv \
    --val-csv path/to/val.csv \
    --test-csv path/to/test.csv
```

---

## 🔧 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Training too slow on CPU

**Solution**: Use smaller model or reduce epochs:
```bash
python scripts/train.py --model minicnn  # Faster than ResNet
```

Or modify `config.yaml`:
```yaml
training:
  epochs: 10  # Reduce from 50
```

### Issue: MPS not detected on MacBook Pro

**Solution**: Ensure PyTorch 2.0+ is installed:
```bash
python -c "import torch; print(torch.__version__)"
pip install --upgrade torch torchvision
```

### Issue: Import errors

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📈 Expected Results

With default configuration (50 epochs, augmentation enabled):

| Model | Val Accuracy | Test Accuracy | Parameters | Training Time (MPS) |
|-------|-------------|---------------|------------|---------------------|
| MiniCNN | ~88-90% | ~87-89% | 106K | ~10 min |
| TinyVGG | ~89-91% | ~88-90% | 125K | ~12 min |
| ResNet | ~92-94% | ~91-93% | 6.5M | ~25 min |

After fine-tuning:
- **ResNet**: ~93-95% test accuracy
- **TinyVGG**: ~90-92% test accuracy
- **MiniCNN**: ~89-91% test accuracy

---

## 🎓 Tips for Best Results

1. **Use data augmentation**: Keep `augmentation.enabled: true` in config.yaml
2. **Early stopping**: Prevents overfitting, saves training time
3. **Learning rate**: Start with 1e-3, reduce to 1e-5 for fine-tuning
4. **Batch size**: 32 is a good default, try 64 if you have more memory
5. **Device**: Use MPS (Apple Silicon) or CUDA (NVIDIA) for 5-10x speedup

---

## 🚀 Next Steps

1. **Experiment with augmentation**: Modify `config.yaml` augmentation settings
2. **Try transfer learning**: Use `scripts/` with Vision Transformer or EfficientNet
3. **Ensemble models**: Combine predictions from multiple models
4. **Deploy**: Use Gradio/Streamlit apps in `apps/` folder

---

**Questions?** Check the main README or open an issue on GitHub!
