#!/usr/bin/env python3
"""
CLI entry-point for dataset preparation.

Delegates to src.data.preparation, which downloads Fashion MNIST via
torchvision, splits it into train/validation/test sets, and writes the
results as flat CSV files under data/processed/.

Usage (from project root):
    python src/cli/prepare_data.py
    python src/cli/prepare_data.py --data-dir data/raw --output-dir data/processed

Run `python src/cli/prepare_data.py --help` for full options.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preparation import main

if __name__ == "__main__":
    main()
