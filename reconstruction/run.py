#!/usr/bin/env python3
"""
Entry point for Code Reconstruction task (RQ5).
Runs OCR inference and evaluation pipeline.
"""

import sys
import os

# Ensure the current directory is in sys.path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.main import main

if __name__ == "__main__":
    main()
