#!/usr/bin/env python
"""Sweep synthetic validation over a grid of assigned (p_peak, b_peak) values.

Runs one synthetic validation per (p_peak, b_peak) combination ("trial"), each
into its own subdirectory, and writes a combined ``sweep_stats.csv`` with one
row per trial: the assigned peaks and the min/max/mean/median of the assigned
vs. recovered p and beta distributions (beta in degrees).

Example::

    python scripts/sweep_synthetic.py \
        --p-peaks 0.4 0.5 0.6 --b-peaks 0.2 0.4 0.8 1.2 \
        --ndraws 1000 --seed 0 --outdir ~/Projects/PyLEADER/synthetic_sweep
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pyleader.synthetic.config import SyntheticConfig  # noqa: E402
from pyleader.synthetic.population import run_synthetic  # noqa: E402
from pyleader.synthetic.stats import stats_row  # noqa: E402
from pyleader.synthetic.sweep_plots import plot_sweep  # noqa: E402

# Flattened stat columns, in a stable order (matches stats.stats_row keys).
_STAT_COLS = [f"{q}_{kind}_{stat}"
              for q in ("p", "beta")
              for kind in ("assigned", "recovered")
              for stat in ("min", "max", "mean", "median")]
_COLUMNS = (["trial", "seed", "p_peak", "b_peak_rad", "b_peak_deg",
             "p_recovered_peak", "beta_recovered_peak_deg", "relerr"] + _STAT_COLS)


def build_parser() -> argparse.ArgumentParser:
    d = SyntheticConfig()
    p = argparse.ArgumentParser(description="Sweep synthetic validation over a (p_peak, b_peak) grid.")
    p.add_argument("--p-peaks", type=float, nargs="+", required=True, help="assigned p peaks")
    p.add_argument("--b-peaks", type=float, nargs="+", required=True, help="assigned beta peaks (radians)")
    p.add_argument("--ndraws", type=int, default=d.Ndraws, help="objects per trial")
    p.add_argument("--scattering", choices=("ls_lambert", "hapke"), default=d.scattering)
    p.add_argument("--damit-dir", default=d.damit_dir)
    p.add_argument("--geometry-dir", default=d.geometry_dir)
    p.add_argument("--damit-list", default=d.damit_list)
    p.add_argument("--base-dir", default=d.base_dir)
    p.add_argument("--outdir", required=True, help="parent directory for the sweep")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed")
    p.add_argument("--nseeds", type=int, default=1, help="seeds (realizations) per grid point")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    grid = list(itertools.product(args.p_peaks, args.b_peaks))
    print(f"Sweep: {len(grid)} trials ({len(args.p_peaks)} p x {len(args.b_peaks)} beta), "
          f"{args.ndraws} draws each")

    print(f"{args.nseeds} seed(s) per grid point -> {len(grid) * args.nseeds} runs total")

    rows = []
    run_idx = 0
    for i, (p_peak, b_peak) in enumerate(grid):
        base = os.path.join(args.outdir, f"trial{i:03d}_p{p_peak:.2f}_b{b_peak:.2f}")
        print(f"\n=== trial {i}: p_peak={p_peak}, b_peak={b_peak} rad "
              f"({np.rad2deg(b_peak):.1f} deg)")
        for s in range(args.nseeds):
            subdir = base if args.nseeds == 1 else os.path.join(base, f"seed{s}")
            cfg = SyntheticConfig(
                p_peak=p_peak, b_peak=b_peak, Ndraws=args.ndraws, scattering=args.scattering,
                damit_dir=args.damit_dir, geometry_dir=args.geometry_dir,
                damit_list=args.damit_list, base_dir=args.base_dir, outdir=subdir,
            )
            # Only draw per-run figures for the first seed of each grid point.
            res = run_synthetic(cfg, seed=args.seed + run_idx, make_plots=(s == 0))
            run_idx += 1

            row = {
                "trial": i, "seed": s, "p_peak": p_peak,
                "b_peak_rad": b_peak, "b_peak_deg": np.rad2deg(b_peak),
                "p_recovered_peak": res.P[np.argmax(res.Pmargin)],
                "beta_recovered_peak_deg": np.rad2deg(res.BETA[np.argmax(res.Bmargin)]),
                "relerr": res.inversion.relerr,
            }
            row.update(stats_row(res.stats))
            rows.append(row)

    csv_path = os.path.join(args.outdir, "sweep_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in _COLUMNS})

    summary_png = os.path.join(args.outdir, "sweep_summary.png")
    plot_sweep(csv_path, summary_png)
    print(f"\nSweep complete.\n  per-trial stats: {csv_path}\n  summary plot:    {summary_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
