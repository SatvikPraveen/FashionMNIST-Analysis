#!/usr/bin/env python3
"""
CLI entry-point for hyperparameter tuning.

Delegates to src.training.tuner, which performs a grid search over
learning rate, batch size, and early-stopping patience, saving per-run
JSON results to results/fine_tuning_results/.

Usage (from project root):
    python src/cli/finetune.py --model minicnn
    python src/cli/finetune.py --model all

Run `python src/cli/finetune.py --help` for full options.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.tuner import main

if __name__ == "__main__":
    main()
