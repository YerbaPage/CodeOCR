#!/usr/bin/env python3
"""
Refactored entry point for the Gemini CodeOCR pipeline.
The monolithic logic has been moved to the `gemini_pipeline` package.
"""

import sys
import os

# Ensure the current directory is in sys.path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.main import main

if __name__ == "__main__":
    main()
