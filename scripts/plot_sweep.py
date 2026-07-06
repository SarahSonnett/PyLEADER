#!/usr/bin/env python
"""Source-tree shim; real logic in pyleader.cli.plot_sweep (installed entry point: see pyproject.toml)."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyleader.cli.plot_sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
