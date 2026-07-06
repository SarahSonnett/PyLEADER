#!/usr/bin/env python
"""Plot a synthetic-sweep summary from an existing sweep_stats.csv.

Produces the 2-panel recovered-vs-assigned figure without re-running the sweep.

Example::

    python scripts/plot_sweep.py ~/synthetic_sweep/sweep_stats.csv
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyleader.synthetic.sweep_plots import plot_sweep  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot a synthetic-sweep summary from sweep_stats.csv.")
    p.add_argument("csv", help="path to sweep_stats.csv")
    p.add_argument("-o", "--out", default=None, help="output PNG (default: sweep_summary.png next to the CSV)")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "sweep_summary.png")
    plot_sweep(args.csv, out, show=args.show)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
