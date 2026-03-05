#!/usr/bin/env python3
"""
CLI wrapper for data preparation script.
Provides backward compatibility for scripts/prepare_data.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and run the data preparation module
from src.data.preparation import main

if __name__ == "__main__":
    main()
