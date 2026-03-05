#!/usr/bin/env python3
"""
CLI wrapper for hyperparameter tuning script.
Provides backward compatibility for scripts/finetune.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and run the tuning module
from src.training.tuner import main

if __name__ == "__main__":
    main()
