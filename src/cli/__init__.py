"""
CLI entry-points for FashionMNIST-Analysis.

Each script in this package is a thin wrapper that adds the project root
to sys.path and delegates to the corresponding module in src/.

Scripts:
    train.py         - End-to-end model training
    finetune.py      - Hyperparameter grid search
    prepare_data.py  - Dataset download and CSV generation
    evaluate.py      - Model evaluation and metrics export
"""

__all__ = []
