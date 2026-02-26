#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Add src/ to Python path so the 'denoising' package can be imported (temporary workaround)
sys.path.insert(0, str(SRC))

from denoising.training.launch import main

if __name__ == "__main__":
    main()