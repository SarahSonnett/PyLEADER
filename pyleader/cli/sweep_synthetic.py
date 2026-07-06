#!/usr/bin/env python
"""Sweep synthetic validation over a grid of assigned (p_peak, b_peak) values.

Runs one synthetic validation per (p_peak, b_peak) combination ("trial"), each
into its own subdirectory, and writes a combined ``sweep_stats.csv`` with one
row per trial × seed: the assigned peaks and the min/max/mean/median of the
assigned vs. recovered p and beta distributions (beta in degrees), plus a
``sweep_summary.png``.

Example::

    python scripts/sweep_synthetic.py \
        --p-peaks 0.4 0.5 0.6 --b-peaks 0.2 0.4 0.8 1.2 \
        --ndraws 1000 --nseeds 3 --seed 0 --outdir ~/synthetic_sweep
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.config import SyntheticConfig  # noqa: E402
from pyleader.synthetic.sweep import run_sweep  # noqa: E402


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
    base_cfg = SyntheticConfig(
        Ndraws=args.ndraws, scattering=args.scattering, damit_dir=args.damit_dir,
        geometry_dir=args.geometry_dir, damit_list=args.damit_list, base_dir=args.base_dir,
    )
    csv_path = run_sweep(base_cfg, args.p_peaks, args.b_peaks,
                         nseeds=args.nseeds, seed=args.seed, outdir=args.outdir)
    print(f"\nSweep complete.\n  per-trial stats: {csv_path}"
          f"\n  summary plot:    {os.path.join(args.outdir, 'sweep_summary.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
