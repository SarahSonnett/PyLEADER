#!/usr/bin/env python
"""Source-tree shim; real logic in pyleader.cli.download_models (installed entry point: see pyproject.toml)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyleader.cli.download_models import main

if __name__ == "__main__":
    raise SystemExit(main())
