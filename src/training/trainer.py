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
import time
import contextlib
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
from src.data.augmentation import AugmentationPipeline, Mixup, CutMix, FMNIST_BACKGROUND
from src.models.architectures import MiniCNN, TinyVGG, ResNet, BasicBlock
from src.models.registry import (
    build_from_spec, model_spec_from_config, resolve_name, spec_path_for, CUSTOM_MODELS
)
from src.training.reproducibility import set_seed
from src.training.experiment import ExperimentLogger
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


def get_model(model_name: str, num_classes: int = 10, in_channels: int = 1,
              config=None) -> nn.Module:
    """
    Build a model by name via the registry.

    ``minicnn`` / ``tinyvgg`` / ``resnet`` are the custom CNNs. Anything else
    (``resnet18``, ``efficientnet_b0``, ``vit_tiny``, ``timm:<id>`` ...) is a
    timm backbone; ``config`` supplies pretrained / image_size / freeze flags.

    Args:
        model_name (str): Model architecture name or alias
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels (1 for grayscale)
        config: Project Config; when None, timm models are random-init

    Returns:
        Model instance (also carries ``model.spec`` for checkpoint metadata)
    """
    if config is not None:
        spec = model_spec_from_config(model_name, config, num_classes=num_classes)
    else:
        from src.models.registry import ModelSpec
        spec = ModelSpec(name=resolve_name(model_name), num_classes=num_classes,
                         in_channels=in_channels)
    model = build_from_spec(spec)
    model.spec = spec
    return model


def make_autocast(device: torch.device, enabled: bool):
    """
    Return (autocast_context_factory, GradScaler-or-None, dtype-name).

    * CUDA: bf16 autocast when the GPU supports it (no scaler needed),
      otherwise fp16 autocast + GradScaler.
    * CPU: bf16 autocast (functional, mainly for tests).
    * MPS: autocast is not reliable across torch versions -> disabled.
    """
    if not enabled:
        return (lambda: contextlib.nullcontext()), None, "fp32"
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None, "bf16"
        scaler = torch.amp.GradScaler("cuda")
        return (lambda: torch.autocast("cuda", dtype=torch.float16)), scaler, "fp16"
    if device.type == "cpu":
        return (lambda: torch.autocast("cpu", dtype=torch.bfloat16)), None, "bf16"
    logger.warning(f"AMP requested but not supported on device '{device.type}'; using fp32")
    return (lambda: contextlib.nullcontext()), None, "fp32"


def train_epoch_with_augmentation(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    augmentation_pipeline: Optional[AugmentationPipeline] = None,
    use_mixup_cutmix: bool = False,
    autocast_ctx=None,
    scaler=None,
    grad_clip: Optional[float] = None,
    mix_prob: float = 0.3,
) -> Tuple[float, float]:
    """
    Train for one epoch with optional augmentation, AMP and grad clipping.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        augmentation_pipeline: Augmentation pipeline
        use_mixup_cutmix: Whether to use Mixup/CutMix
        autocast_ctx: zero-arg callable returning an autocast context (see make_autocast)
        scaler: torch.amp.GradScaler for fp16, else None
        grad_clip: max gradient norm, or None
        mix_prob: probability of applying Mixup/CutMix to a batch
        
    Returns:
        Tuple of (train_loss, train_acc), both sample-weighted
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    if autocast_ctx is None:
        autocast_ctx = contextlib.nullcontext
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Apply sample-level augmentations (rotation, flips, erasing, etc.)
        if augmentation_pipeline:
            X = augmentation_pipeline.apply_sample_augmentations(X)

        mixed = False
        if use_mixup_cutmix and augmentation_pipeline and np.random.rand() < mix_prob:
            if np.random.rand() < 0.5 and augmentation_pipeline.mixup_enabled:
                X, y_a, y_b, lam = augmentation_pipeline.apply_mixup(X, y)
                criterion = Mixup.mixup_criterion
                mixed = True
            elif augmentation_pipeline.cutmix_enabled:
                X, y_a, y_b, lam = augmentation_pipeline.apply_cutmix(X, y)
                criterion = CutMix.cutmix_criterion
                mixed = True

        with autocast_ctx():
            y_pred = model(X)
            if mixed:
                loss = criterion(loss_fn, y_pred, y_a, y_b, lam)
            else:
                loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Sample-weighted accumulation (accuracy is measured against the
        # un-mixed labels, which is the usual convention under Mixup/CutMix)
        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        total_samples += n
    
    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def build_augmentation_pipeline(config) -> Optional[AugmentationPipeline]:
    """Translate the ``augmentation`` config section into an AugmentationPipeline."""
    # Setup augmentation pipeline
    augmentation_pipeline = None
    use_augmentation = bool(config.augmentation.enabled)
    
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
            # Pixels exposed by padding / rotation must look like the black
            # background, i.e. its *normalised* value, not 0 (= mid-grey).
            fill = config.get('augmentation.fill', 'background')
            tv_cfg['fill'] = FMNIST_BACKGROUND if fill == 'background' else float(fill)
            tv_cfg['legacy_batch_mode'] = bool(config.get('augmentation.legacy_batch_mode', False))
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
    
    return augmentation_pipeline


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module of a torch.compile'd model."""
    return getattr(model, "_orig_mod", model)


def build_optimizer(model: nn.Module, config) -> optim.Optimizer:
    lr = float(config.training.learning_rate)
    wd = float(config.training.weight_decay)
    name = str(config.training.optimizer).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd,
                         nesterov=bool(config.get("training.nesterov", True)))
    raise ValueError(f"Unknown optimizer '{name}' (adam, adamw, sgd)")


def build_scheduler(optimizer: optim.Optimizer, config, epochs: int):
    name = str(config.training.scheduler).lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if name == "onecycle":
        return None  # handled per-step; not used here
    return None


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: dict,
    device: torch.device,
    output_dir: str,
    model_name: str,
    run_name: Optional[str] = None,
    resume: bool = False,
    experiment: Optional[ExperimentLogger] = None,
) -> dict:
    """
    Complete training pipeline for a model.

    Writes into ``<output_dir>/<model_name>/``:
        ``<model_name>_best.pth``       best-val weights (plus ``_spec.json``)
        ``<model_name>_last.pt``        full resume state (model/opt/sched/epoch)
        ``metrics.jsonl``, ``run.json`` experiment log (see experiment.py)

    Args:
        run_name: Name for the experiment logger (default: model_name).
        resume: If True and ``<model_name>_last.pt`` exists, continue from it.
            The LR scheduler is rebuilt for the current ``training.epochs``
            and fast-forwarded to the resumed epoch.
        experiment: An existing ExperimentLogger to reuse; else one is created.
    
    Returns:
        Dictionary with training history
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {model_name}")
    logger.info(f"{'='*60}")

    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    best_model_path = os.path.join(model_output_dir, f"{model_name}_best.pth")
    last_state_path = os.path.join(model_output_dir, f"{model_name}_last.pt")

    # Model to device (+ optional torch.compile on CUDA)
    model = model.to(device)
    use_compile = bool(config.get("training.compile", False)) and device.type == "cuda"
    if use_compile:
        try:
            model = torch.compile(model)
            logger.info("⚡ torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing eagerly: {e}")
    raw_model = _unwrap(model)

    logger.info(f"\n📊 Model Parameters: {count_parameters(raw_model):,}")

    # Loss / optimizer / scheduler
    label_smoothing = float(config.get("training.label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    learning_rate = float(config.training.learning_rate)
    optimizer = build_optimizer(model, config)
    epochs = int(config.training.epochs)
    scheduler = build_scheduler(optimizer, config, epochs)

    # Mixed precision / grad clipping
    autocast_ctx, scaler, amp_dtype = make_autocast(device, bool(config.get("training.amp", False)))
    grad_clip = config.get("training.grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip else None

    # Augmentation
    use_augmentation = bool(config.augmentation.enabled)
    augmentation_pipeline = build_augmentation_pipeline(config) if use_augmentation else None
    mix_prob = float(config.get("augmentation.mix_prob", 0.3))

    # Early stopping
    early_stopping = EarlyStopping(
        patience=int(config.training.early_stopping_patience),
        min_delta=0.001,
        verbose=True
    )

    # Training history
    history = {
        'model_name': model_name,
        'seed': config.get('training.seed', None),
        'amp': amp_dtype,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'epoch_time_sec': [],
    }
    best_val_acc = 0.0
    start_epoch = 0

    # Resume ------------------------------------------------------------ #
    if resume and os.path.exists(last_state_path):
        state = torch.load(last_state_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # The scheduler is rebuilt for the *current* epoch budget and
        # fast-forwarded, rather than restored: restoring would keep the old
        # horizon (e.g. cosine T_max), which drives the LR to 0 if the run
        # is resumed with a larger --epochs.
        # Schedulers derive step 0 from the optimizer's *current* lr, so put
        # it back to the initial value before rebuilding and fast-forwarding.
        for g in optimizer.param_groups:
            g["lr"] = g.get("initial_lr", learning_rate)
        scheduler = build_scheduler(optimizer, config, epochs)
        for _ in range(state["epoch"] + 1):
            if scheduler is not None:
                scheduler.step()
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best_val_acc = state["best_val_acc"]
        start_epoch = state["epoch"] + 1
        es = state.get("early_stopping", {})
        early_stopping.best_acc = es.get("best_acc")
        early_stopping.counter = es.get("counter", 0)
        early_stopping.best_epoch = es.get("best_epoch", 0)
        logger.info(f"⏯️  Resumed from {last_state_path} at epoch {start_epoch} "
                    f"(best_val_acc={best_val_acc:.4f})")
    elif resume:
        logger.info("No resume state found; starting from scratch")

    # Experiment logger -------------------------------------------------- #
    own_experiment = experiment is None
    if own_experiment:
        mon = config.get("monitoring", {}) or {}
        experiment = ExperimentLogger(
            run_dir=model_output_dir,
            run_name=run_name or model_name,
            config=config,
            device=device,
            use_mlflow=bool(mon.get("mlflow_tracking", False)),
            use_wandb=bool(mon.get("wandb_enabled", False)),
            mlflow_tracking_uri=mon.get("mlflow_tracking_uri"),
            experiment_name=mon.get("experiment_name", "fashion-mnist"),
            wandb_project=mon.get("wandb_project", "fashion-mnist"),
        )
    spec = getattr(raw_model, "spec", None)
    experiment.log_params({
        "model": model_name,
        "model_spec": spec.to_dict() if spec is not None else None,
        "seed": config.get("training.seed", None),
        "epochs": epochs,
        "batch_size": int(config.training.batch_size),
        "learning_rate": learning_rate,
        "weight_decay": float(config.training.weight_decay),
        "optimizer": str(config.training.optimizer),
        "scheduler": str(config.training.scheduler),
        "label_smoothing": label_smoothing,
        "grad_clip": grad_clip,
        "amp": amp_dtype,
        "compile": use_compile,
        "augmentation": config.augmentation.to_dict() if use_augmentation else {"enabled": False},
        "num_parameters": count_parameters(raw_model),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset) if test_loader else 0,
    })

    logger.info(f"\n🚀 Starting training for {epochs} epochs...")
    logger.info(f"   Batch size: {int(config.training.batch_size)}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Optimizer: {config.training.optimizer} | Scheduler: {config.training.scheduler}")
    logger.info(f"   AMP: {amp_dtype} | Grad clip: {grad_clip} | Label smoothing: {label_smoothing}")
    logger.info(f"   Augmentation: {'on' if use_augmentation else 'off'}")
    logger.info(f"   Device: {device}\n")

    status = "finished"
    try:
        for epoch in range(start_epoch, epochs):
            t_epoch = time.time()
            train_loss, train_acc = train_epoch_with_augmentation(
                model, train_loader, loss_fn, optimizer, device,
                augmentation_pipeline, use_mixup_cutmix=use_augmentation,
                autocast_ctx=autocast_ctx, scaler=scaler, grad_clip=grad_clip,
                mix_prob=mix_prob,
            )
            val_loss, val_acc = validation_step(model, val_loader, loss_fn, device)
            epoch_time = time.time() - t_epoch
            lr_now = optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(lr_now)
            history['epoch_time_sec'].append(epoch_time)
            experiment.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": lr_now, "epoch_time_sec": epoch_time,
            }, step=epoch + 1)

            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {lr_now:.6f} | {epoch_time:.1f}s"
            )

            # Save best model (track by val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(raw_model.state_dict(), best_model_path)
                if spec is not None:
                    spec.save(spec_path_for(best_model_path))
                logger.info(f"   💾 Saved best model (val_acc: {val_acc:.4f})")

            stop = early_stopping(val_acc, epoch)

            if scheduler is not None:
                scheduler.step()

            # Full state for pre-emption-safe resume (written every epoch)
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "history": history,
                "early_stopping": {"best_acc": early_stopping.best_acc,
                                   "counter": early_stopping.counter,
                                   "best_epoch": early_stopping.best_epoch},
            }, last_state_path)

            if stop:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # Test evaluation (best-val checkpoint, evaluated exactly once)
        summary = {"best_val_acc": best_val_acc,
                   "epochs_trained": len(history['train_loss']),
                   "train_time_sec": float(sum(history['epoch_time_sec']))}
        if test_loader:
            logger.info("\n📊 Evaluating on test set...")
            raw_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
            history['test_loss'] = test_loss
            history['test_acc'] = test_acc
            summary.update(test_loss=test_loss, test_acc=test_acc)
            logger.info(f"   Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        history['best_val_acc'] = best_val_acc
    except BaseException:
        status = "failed"
        raise
    finally:
        if own_experiment:
            experiment.finish(summary if status == "finished" else None, status=status)
        experiment.log_artifact(best_model_path)

    logger.info(f"\n✅ Training complete for {model_name}")
    logger.info(f"   Best val acc: {best_val_acc:.4f}")
    logger.info(f"   Model saved: {best_model_path}")

    return history


def apply_overrides(config, overrides) -> None:
    """Apply ``key.path=value`` overrides (values parsed as YAML) to a Config."""
    import yaml
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, _, raw = item.partition("=")
        config.set(key.strip(), yaml.safe_load(raw))


def main(argv=None):
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
        nargs="+",
        default=["all"],
        help=("Model(s) to train. Custom CNNs: minicnn, tinyvgg, resnet; "
              "'all' = those three. Any timm id or alias also works: "
              "resnet18, resnet50, efficientnet_b0, convnext_tiny, vit_tiny, "
              "deit_small, timm:<id>. (default: all)")
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Use ImageNet weights for timm models (overrides model.pretrained)"
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Random-init timm models"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input resolution for timm backbones (overrides model.image_size)"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the classifier head of a timm backbone"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides training.seed in config; default 42)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic algorithms (slower; for bit-exact reproduction)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (overrides data.num_workers in config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size from config"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override training.learning_rate from config"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (bf16 on supporting GPUs, else fp16+GradScaler)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (CUDA only)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output-dir>/<model>/<model>_last.pt if present"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment run name (default: <model>_seed<seed>)"
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Mirror metrics to MLflow (monitoring.mlflow_tracking)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Mirror metrics to Weights & Biases (monitoring.wandb_enabled)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=("Override any config key, e.g. --set training.label_smoothing=0.1 "
              "--set augmentation.mixup=false (repeatable; values parsed as YAML)")
    )
    parser.add_argument(
        "--skip-best-selection",
        action="store_true",
        help="Do not copy the best model to models/best_model_weights/ (sweeps)"
    )
    
    args = parser.parse_args(argv)
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    apply_overrides(config, args.overrides)

    # Seed everything before any model / data code runs
    seed = args.seed if args.seed is not None else int(config.get('training.seed', 42))
    config.set('training.seed', seed)
    deterministic = args.deterministic or bool(config.get('training.deterministic', False))
    set_seed(seed, deterministic=deterministic)

    num_workers = args.num_workers if args.num_workers is not None else int(config.get('data.num_workers', 0))

    # Training overrides from the CLI
    if args.epochs is not None:
        config.set('training.epochs', int(args.epochs))
    if args.batch_size is not None:
        config.set('training.batch_size', int(args.batch_size))
    if args.lr is not None:
        config.set('training.learning_rate', float(args.lr))

    if args.amp:
        config.set('training.amp', True)
    if args.compile:
        config.set('training.compile', True)
    if args.mlflow:
        config.set('monitoring.mlflow_tracking', True)
    if args.wandb:
        config.set('monitoring.wandb_enabled', True)

    # Model-family overrides from the CLI
    if args.pretrained is not None:
        config.set('model.pretrained', bool(args.pretrained))
    if args.image_size is not None:
        config.set('model.image_size', int(args.image_size))
    if args.freeze_backbone:
        config.set('transfer_learning.freeze_backbone', True)
    
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
            num_workers=num_workers,
            seed=seed
        )
    else:
        logger.info("   Using torchvision FashionMNIST dataset")
        train_loader, val_loader, test_loader = create_dataloaders(
            use_torchvision=True,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            seed=seed
        )
    
    logger.info(f"✅ Datasets loaded:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    # Determine which models to train ('all' expands to the custom CNNs)
    models_to_train = []
    for name in args.model:
        if name.lower() == "all":
            models_to_train.extend(CUSTOM_MODELS)
        else:
            models_to_train.append(resolve_name(name))
    models_to_train = list(dict.fromkeys(models_to_train))  # dedupe, keep order
    
    # Training results
    all_results = {}
    
    # Train each model
    for model_name in models_to_train:
        model = get_model(model_name, num_classes=10, config=config)
        
        run_name = args.run_name or f"{model_name}_seed{seed}"
        if len(models_to_train) > 1 and args.run_name:
            run_name = f"{args.run_name}_{model_name}"
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            output_dir=args.output_dir,
            model_name=model_name,
            run_name=run_name,
            resume=args.resume,
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

    if args.skip_best_selection:
        logger.info("🎉 All training complete! (best-model selection skipped)")
        return all_results

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
    best_spec_src = spec_path_for(best_src_path)
    best_spec = None
    if os.path.exists(best_spec_src):
        shutil.copy2(best_spec_src, spec_path_for(best_dest_path))
        with open(best_spec_src) as f:
            best_spec = json.load(f)

    best_info = {
        "model_name":   best_name,
        "model_spec":   best_spec,
        "seed":         seed,
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
    return all_results


if __name__ == "__main__":
    main()
