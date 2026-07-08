#!/usr/bin/env python
"""Plot a bias-map summary from an existing bias_map_stats.csv.

Produces the 2-panel recovered-vs-assigned figure without re-running the bias map.

Example::

    python scripts/plot_bias_map.py ~/bias_map/bias_map_stats.csv
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.bias_map_plots import plot_bias_map  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot a bias-map summary from bias_map_stats.csv.")
    p.add_argument("csv", help="path to bias_map_stats.csv")
    p.add_argument("-o", "--out", default=None, help="output PNG (default: bias_map_summary.png next to the CSV)")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "bias_map_summary.png")
    plot_bias_map(args.csv, out, show=args.show)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
