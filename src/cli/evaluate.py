#!/usr/bin/env python3
"""
CLI wrapper for model evaluation script.
Provides backward compatibility for scripts/main.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and run the evaluation module
from src.evaluation.evaluate import main

if __name__ == "__main__":
    # The original evaluate.py doesn't have argparse, it runs directly
    # So we'll just import and run the code
    import src.evaluation.evaluate
