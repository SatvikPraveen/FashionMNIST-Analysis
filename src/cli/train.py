#!/usr/bin/env python3
"""
CLI entry-point for model training.

Delegates to src.training.trainer, which runs the full training pipeline:
data loading, augmentation (Mixup/CutMix), early stopping, checkpointing,
and metric logging.

Usage (from project root):
    python src/cli/train.py --model minicnn
    python src/cli/train.py --model all
    python src/cli/train.py --model resnet --force-cpu

Run `python src/cli/train.py --help` for full options.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.trainer import main

if __name__ == "__main__":
    main()
