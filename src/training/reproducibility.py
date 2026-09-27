"""
Reproducibility helpers for FashionMNIST-Analysis.

Every experiment should call :func:`set_seed` once at start-up so that
model initialisation, data shuffling, augmentation sampling and the
train/val split are all determined by a single integer. Multi-seed
studies then simply vary that integer.

Usage:
    from src.training.reproducibility import set_seed, seed_worker, make_generator

    set_seed(42)
    loader = DataLoader(ds, shuffle=True, worker_init_fn=seed_worker,
                        generator=make_generator(42))
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> int:
    """
    Seed every RNG the training pipeline touches.

    Args:
        seed: The seed to use.
        deterministic: If True, also force deterministic cuDNN kernels and
            ``torch.use_deterministic_algorithms``. This costs speed and some
            ops will raise if no deterministic implementation exists, so it is
            off by default; bit-exact reproduction studies can turn it on.

    Returns:
        The seed that was set (handy for logging / recording in results).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required by use_deterministic_algorithms for CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    logger.info(f"🔒 Seed set to {seed} (deterministic={deterministic})")
    return seed


def seed_worker(worker_id: int) -> None:
    """
    ``worker_init_fn`` for DataLoader so each worker's numpy / random state
    derives from the torch seed instead of being identical across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_indices(n: int, val_fraction: float, seed: int):
    """
    The train/validation split used everywhere in the project.

    Returns ``(train_indices, val_indices)`` into a dataset of length ``n``.
    It reproduces ``torch.utils.data.random_split(ds, [n_train, n_val],
    generator=make_generator(seed))`` exactly (one ``randperm``, train first),
    so runs made before this helper existed keep their splits, and the CSV
    files from ``prepare_data`` hold the same validation images as training
    with the same seed.
    """
    n_train = int((1.0 - val_fraction) * n)
    perm = torch.randperm(n, generator=make_generator(seed)).tolist()
    return perm[:n_train], perm[n_train:]


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Return a CPU ``torch.Generator`` seeded with ``seed`` (or None)."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
