"""
Training utility functions for FashionMNIST-Analysis.

This module provides:
    - Device detection (CUDA / MPS / CPU) with automatic priority selection.
    - Model introspection helpers (parameter counts, summary).
    - Core per-epoch training, validation, and test step functions that are
      shared by both the full trainer and the hyperparameter tuner.

Usage:
    from src.training.utils import get_device, train_step, validation_step, test_step
"""

import torch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DEVICE DETECTION AND UTILITIES
# ============================================================================

_CPU_CONV_CHECKED = False


def ensure_cpu_conv_backend() -> bool:
    """
    Probe a tiny CPU convolution and disable oneDNN (mkldnn) if it fails.

    Some virtualised login nodes (e.g. KVM head nodes) expose a CPU
    without AVX/SSE4 flags; oneDNN then raises "could not create a
    primitive" for every conv. Falling back to the native ATen kernels is
    slower but correct, and only matters for CPU runs such as tests.

    Returns:
        True if oneDNN was disabled by this call.
    """
    global _CPU_CONV_CHECKED
    if _CPU_CONV_CHECKED:
        return False
    _CPU_CONV_CHECKED = True
    if not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled:
        return False
    try:
        # Mirror a real first layer: a 1-in/1-out 8x8 probe passes on the
        # affected hosts while multi-output-channel convs fail.
        torch.nn.functional.conv2d(torch.zeros(2, 1, 28, 28), torch.zeros(16, 1, 3, 3), padding=1)
        return False
    except RuntimeError as e:
        if "primitive" not in str(e):
            raise
        torch.backends.mkldnn.enabled = False
        logger.warning("oneDNN CPU convolutions unavailable on this host "
                       f"({e}); disabled torch.backends.mkldnn")
        return True


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Automatically detect the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Args:
        force_cpu (bool): Force CPU usage even if GPU is available
        
    Returns:
        torch.device: The selected device
    """
    if force_cpu:
        device = torch.device("cpu")
        ensure_cpu_conv_backend()
        logger.info("🖥️  Device: CPU (forced)")
        return device
    
    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Device: CUDA - {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Device: MPS (Apple Silicon)")
        return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    ensure_cpu_conv_backend()
    logger.info("🖥️  Device: CPU")
    return device


def print_device_info():
    """Print detailed device information."""
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: Yes")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        print(f"❌ CUDA Available: No")
    
    # MPS
    if hasattr(torch.backends, "mps"):
        print(f"✅ MPS Available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print(f"   MPS Built: {torch.backends.mps.is_built()}")
    else:
        print(f"❌ MPS Available: No")
    
    # Selected device
    device = get_device()
    print(f"\n🎯 Selected Device: {device}")
    print("="*60 + "\n")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: torch.nn.Module, input_size: Tuple[int, ...] = None):
    """Print model summary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    if input_size:
        try:
            from torchinfo import summary
            print("\nDetailed Summary:")
            summary(model, input_size=input_size)
        except ImportError:
            logger.warning("torchinfo not available for detailed summary")
    
    print("="*60 + "\n")


# ============================================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================================

# Sample-weighted metric accumulation
# ----------------------------------
# Every step below accumulates *counts* (correct predictions, summed loss)
# and divides by the number of samples at the end. The previous version
# averaged per-batch accuracies, which over-weights a partial final batch:
# on the 10,000-image test set with batch_size=32 the last batch of 16
# images counted as much as a full batch of 32, so the recorded number was
# (sum of batch accuracies) / 313 rather than correct / 10000.


def _run_eval_epoch(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    """Shared eval loop returning sample-weighted (loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            n = y.size(0)
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += n

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch (no augmentation).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        (train_loss, train_accuracy), both averaged over samples, e.g.
        (0.1112, 0.8743)
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += (y_pred.argmax(dim=1) == y).sum().item()
        total_samples += n

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def validation_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    """Validates a PyTorch model for a single epoch.

    Returns:
        (val_loss, val_accuracy), both averaged over samples.
    """
    return _run_eval_epoch(model, dataloader, loss_fn, device)


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model on a held-out set.

    Returns:
        (test_loss, test_accuracy), both averaged over samples, e.g.
        (0.0223, 0.8985)
    """
    return _run_eval_epoch(model, dataloader, loss_fn, device)
