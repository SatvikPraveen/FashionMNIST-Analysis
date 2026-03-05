#!/usr/bin/env python3
"""
CLI wrapper for training script.
Provides backward compatibility for scripts/train.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and run the training module
from src.training.trainer import main

if __name__ == "__main__":
    main()
