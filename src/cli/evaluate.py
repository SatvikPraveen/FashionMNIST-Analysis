#!/usr/bin/env python3
"""
CLI entry-point for model evaluation.

Delegates to src.evaluation.evaluate, which loads a pre-trained model,
runs it over a test CSV, and saves metrics, a confusion matrix, and a
prediction-visualisation grid.

Usage (from project root):
    python src/cli/evaluate.py \\
        --model_path models/best_model_weights/best_model_weights.pth \\
        --test_csv   data/processed/test.csv

Run `python src/cli/evaluate.py --help` for full options.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluate import main

if __name__ == "__main__":
    main()
