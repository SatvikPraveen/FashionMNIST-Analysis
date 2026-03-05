#!/usr/bin/env python3
"""
Main training script for FashionMNIST-Analysis.

End-to-end training pipeline with:
- Config-driven training
- Data augmentation (Mixup, CutMix, transforms)
- Multi-device support (CUDA/MPS/CPU)
- Model checkpointing
- Logging and metrics tracking
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import load_config
from src.data.dataset import create_dataloaders, get_default_transforms
from src.data.augmentation import AugmentationPipeline, Mixup, CutMix
from src.models.architectures import MiniCNN, TinyVGG, ResNet, BasicBlock
from src.training.utils import (
    get_device,
    print_device_info,
    count_parameters,
    model_summary,
    train_step,
    validation_step,
    test_step
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping on validation accuracy (maximize)."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change in val_acc to count as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_acc: float, epoch: int) -> bool:
        """
        Check if should stop training (monitors val_acc, higher is better).

        Returns:
            True if should stop, False otherwise
        """
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_epoch = epoch
            if self.verbose:
                logger.info(f"📊 Initial validation accuracy: {val_acc:.4f}")
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"⏸️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                return True
        else:
            improvement = val_acc - self.best_acc
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"📈 Val accuracy improved by {improvement:.4f}")

        return False


def get_model(model_name: str, num_classes: int = 10, in_channels: int = 1) -> nn.Module:
    """
    Get model by name.
    
    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == "minicnn":
        return MiniCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "tinyvgg":
        return TinyVGG(in_channels=in_channels, hidden_units=64, num_classes=num_classes)
    elif model_name == "resnet":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: minicnn, tinyvgg, resnet")


def train_epoch_with_augmentation(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    augmentation_pipeline: Optional[AugmentationPipeline] = None,
    use_mixup_cutmix: bool = False
) -> Tuple[float, float]:
    """
    Train for one epoch with optional augmentation.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        augmentation_pipeline: Augmentation pipeline
        use_mixup_cutmix: Whether to use Mixup/CutMix
        
    Returns:
        Tuple of (train_loss, train_acc)
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Apply sample-level augmentations (rotation, flips, erasing, etc.)
        if augmentation_pipeline:
            X = augmentation_pipeline.apply_sample_augmentations(X)

        # Apply Mixup or CutMix with 30% probability (less label noise)
        if use_mixup_cutmix and augmentation_pipeline:
            if np.random.rand() < 0.3:
                # Randomly choose between Mixup and CutMix
                if np.random.rand() < 0.5 and augmentation_pipeline.mixup_enabled:
                    X, y_a, y_b, lam = augmentation_pipeline.apply_mixup(X, y)
                    y_pred = model(X)
                    loss = Mixup.mixup_criterion(loss_fn, y_pred, y_a, y_b, lam)
                elif augmentation_pipeline.cutmix_enabled:
                    X, y_a, y_b, lam = augmentation_pipeline.apply_cutmix(X, y)
                    y_pred = model(X)
                    loss = CutMix.cutmix_criterion(loss_fn, y_pred, y_a, y_b, lam)
                else:
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
            else:
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
        else:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
        
        train_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: dict,
    device: torch.device,
    output_dir: str,
    model_name: str
) -> dict:
    """
    Complete training pipeline for a model.
    
    Returns:
        Dictionary with training history
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {model_name}")
    logger.info(f"{'='*60}")
    
    # Model to device
    model = model.to(device)
    
    # Print model summary
    logger.info(f"\n📊 Model Parameters: {count_parameters(model):,}")
    
    # Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    
    # Ensure numeric values
    learning_rate = float(config.training.learning_rate)
    weight_decay = float(config.training.weight_decay)
    
    if config.training.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
    # Setup learning rate scheduler
    scheduler = None
    epochs = int(config.training.epochs)
    
    if config.training.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif config.training.scheduler.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    
    # Setup augmentation pipeline
    augmentation_pipeline = None
    use_augmentation = config.augmentation.enabled
    
    if use_augmentation:
        aug_config = {}

        if config.augmentation.mixup:
            aug_config['mixup'] = {'alpha': config.augmentation.mixup_alpha}

        if config.augmentation.cutmix:
            aug_config['cutmix'] = {'alpha': config.augmentation.cutmix_alpha}

        # --- Torchvision geometric + color transforms ---
        tv_cfg = {}

        if config.augmentation.rotation > 0:
            tv_cfg['rotation'] = config.augmentation.rotation

        if config.augmentation.horizontal_flip:
            tv_cfg['horizontal_flip'] = True

        if config.augmentation.vertical_flip:
            tv_cfg['vertical_flip'] = True

        if getattr(config.augmentation, 'random_crop', False):
            tv_cfg['random_crop'] = {
                'size': 28,
                'padding': int(getattr(config.augmentation, 'random_crop_padding', 4))
            }

        if getattr(config.augmentation, 'random_affine', False):
            zoom = float(getattr(config.augmentation, 'zoom', 0.1))
            tv_cfg['random_affine'] = {
                'degrees': 0,
                'translate': (0.1, 0.1),
                'scale': (1.0 - zoom / 2, 1.0 + zoom / 2)
            }

        if getattr(config.augmentation, 'color_jitter', False):
            tv_cfg['color_jitter'] = {
                'brightness': float(getattr(config.augmentation, 'brightness', 0.2)),
                'contrast':   float(getattr(config.augmentation, 'contrast',   0.2)),
                'saturation': 0.0,   # grayscale — no effect
                'hue':        0.0,
            }

        if tv_cfg:
            aug_config['torchvision'] = tv_cfg

        # --- Sample-level noise / occlusion ---
        if getattr(config.augmentation, 'random_erasing', False):
            aug_config['random_erasing'] = {
                'probability': float(getattr(config.augmentation, 'random_erasing_prob', 0.3))
            }

        if getattr(config.augmentation, 'gaussian_noise', False):
            aug_config['gaussian_noise'] = {
                'std': float(getattr(config.augmentation, 'gaussian_noise_std', 0.04))
            }

        augmentation_pipeline = AugmentationPipeline(aug_config)
        logger.info(f"✅ Augmentation enabled: {list(aug_config.keys())}")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        min_delta=0.001,
        verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Training loop
    logger.info(f"\n🚀 Starting training for {epochs} epochs...")
    logger.info(f"   Batch size: {int(config.training.batch_size)}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Optimizer: {config.training.optimizer}")
    logger.info(f"   Device: {device}\n")

    best_val_acc = 0.0
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    best_model_path = os.path.join(model_output_dir, f"{model_name}_best.pth")

    for epoch in range(epochs):
        # Train with augmentation
        train_loss, train_acc = train_epoch_with_augmentation(
            model, train_loader, loss_fn, optimizer, device,
            augmentation_pipeline, use_mixup_cutmix=use_augmentation
        )

        # Validation
        val_loss, val_acc = validation_step(model, val_loader, loss_fn, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best model (track by val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"   💾 Saved best model (val_acc: {val_acc:.4f})")

        # Early stopping check (monitors val_acc)
        if early_stopping(val_acc, epoch):
            logger.info(f"🛑 Early stopping at epoch {epoch+1}")
            break

        # Learning rate scheduler step
        if scheduler:
            scheduler.step()

    # Test evaluation
    if test_loader:
        logger.info("\n📊 Evaluating on test set...")
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc
        logger.info(f"   Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    logger.info(f"\n✅ Training complete for {model_name}")
    logger.info(f"   Best val acc: {best_val_acc:.4f}")
    logger.info(f"   Model saved: {best_model_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train FashionMNIST models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["minicnn", "tinyvgg", "resnet", "all"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/all_models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Use CSV datasets instead of torchvision"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="./data_preparation/fashion_mnist_train.csv",
        help="Path to training CSV"
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="./data_preparation/fashion_mnist_val.csv",
        help="Path to validation CSV"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="./data_preparation/fashion_mnist_test.csv",
        help="Path to test CSV"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage"
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Get device
    device = print_device_info() if not args.force_cpu else get_device(force_cpu=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    logger.info("\n📦 Loading datasets...")
    if args.use_csv:
        logger.info(f"   Using CSV files from {os.path.dirname(args.train_csv)}")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            batch_size=config.training.batch_size,
            num_workers=0
        )
    else:
        logger.info("   Using torchvision FashionMNIST dataset")
        train_loader, val_loader, test_loader = create_dataloaders(
            use_torchvision=True,
            batch_size=config.training.batch_size,
            num_workers=0
        )
    
    logger.info(f"✅ Datasets loaded:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    # Determine which models to train
    if args.model == "all":
        models_to_train = ["minicnn", "tinyvgg", "resnet"]
    else:
        models_to_train = [args.model]
    
    # Training results
    all_results = {}
    
    # Train each model
    for model_name in models_to_train:
        model = get_model(model_name, num_classes=10)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            output_dir=args.output_dir,
            model_name=model_name
        )
        
        all_results[model_name] = history
        
        # Save history
        history_path = os.path.join(args.output_dir, model_name, f"{model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"   History saved: {history_path}\n")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")

    for model_name, history in all_results.items():
        best_val_acc = max(history['val_acc'])
        test_acc = history.get('test_acc', 0)
        logger.info(f"{model_name.upper():10s} | Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")

    logger.info(f"{'='*60}\n")

    # ── Best model selection ───────────────────────────────────────────────────
    # Pick the model with the highest test accuracy and copy its weights to
    # models/best_model_weights/best_model_weights.pth
    best_name = max(
        all_results,
        key=lambda m: all_results[m].get('test_acc', 0)
    )
    best_test_acc  = all_results[best_name].get('test_acc', 0)
    best_val_acc   = max(all_results[best_name]['val_acc'])
    best_src_path  = os.path.join(args.output_dir, best_name, f"{best_name}_best.pth")
    best_dest_dir  = os.path.join(os.path.dirname(args.output_dir), "best_model_weights")
    best_dest_path = os.path.join(best_dest_dir, "best_model_weights.pth")

    os.makedirs(best_dest_dir, exist_ok=True)
    import shutil
    shutil.copy2(best_src_path, best_dest_path)

    best_info = {
        "model_name":   best_name,
        "best_val_acc": best_val_acc,
        "test_acc":     best_test_acc,
        "source_path":  best_src_path,
        "saved_to":     best_dest_path,
    }
    info_path = os.path.join(best_dest_dir, "best_model_info.json")
    with open(info_path, 'w') as f:
        json.dump(best_info, f, indent=2)

    logger.info(f"🏆 Best model: {best_name.upper()}  "
                f"(val_acc={best_val_acc:.4f}, test_acc={best_test_acc:.4f})")
    logger.info(f"   Saved to: {best_dest_path}")
    logger.info(f"   Info:     {info_path}")
    logger.info(f"{'='*60}\n")
    logger.info("🎉 All training complete!")


if __name__ == "__main__":
    main()
