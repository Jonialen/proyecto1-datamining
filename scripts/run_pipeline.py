#!/usr/bin/env python3
"""Legacy compat: llama al CLI."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import app

if __name__ == "__main__":
    app()
