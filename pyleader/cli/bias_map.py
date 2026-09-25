#!/usr/bin/env python
"""Determine the bias map over a grid of assigned (p_peak, b_peak) values.

Runs one synthetic validation per (p_peak, b_peak) combination ("trial"), each
into its own subdirectory, and writes a combined ``bias_map_stats.csv`` with one
row per trial × seed: the assigned peaks and the min/max/mean/median of the
assigned vs. recovered p and beta distributions (beta in degrees), plus a
``bias_map_summary.png``.

Example::

    python scripts/bias_map.py \
        --p-peaks 0.4 0.5 0.6 --b-peaks 10 30 50 75 \
        --ndraws 1000 --nseeds 3 --seed 0 --outdir ~/bias_map

(``--b-peaks`` are in degrees; they are converted to radians internally.)
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.config import SyntheticConfig  # noqa: E402
from pyleader.synthetic.bias_map import run_bias_map  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    d = SyntheticConfig()
    p = argparse.ArgumentParser(description="Determine the bias map over a (p_peak, b_peak) grid.")
    p.add_argument("--p-peaks", type=float, nargs="+", required=True, help="assigned p peaks")
    p.add_argument("--b-peaks", type=float, nargs="+", required=True,
                   help="assigned beta peaks in DEGREES (0 < beta < 90)")
    p.add_argument("--ndraws", type=int, default=d.Ndraws, help="objects per trial")
    p.add_argument("--scattering", choices=("ls_lambert", "hapke"), default=d.scattering)
    p.add_argument("--damit-dir", default=d.damit_dir)
    p.add_argument("--geometry-dir", default=d.geometry_dir)
    p.add_argument("--damit-list", default=d.damit_list)
    p.add_argument("--base-dir", default=d.base_dir)
    p.add_argument("--outdir", required=True, help="parent directory for the bias map")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed")
    p.add_argument("--nseeds", type=int, default=1, help="seeds (realizations) per grid point")
    p.add_argument("--noise-model", choices=("empirical", "flat"), default="empirical",
                   help="synthetic photometric noise: 'empirical' (default) fits the "
                        "geometry files' own flux-fluxerr relation; 'flat' is the "
                        "original 1%% Gaussian")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    noise = None
    if args.noise_model == "empirical":
        import glob as _glob
        from pyleader.pipeline import fit_population_noise
        geom = sorted(_glob.glob(os.path.join(args.geometry_dir, "*.obs")))
        geom = [g for g in geom if not os.path.basename(g).startswith("Nofilter")]
        os.makedirs(args.outdir, exist_ok=True)
        noise = fit_population_noise(geom, docdir=args.outdir)
    base_cfg = SyntheticConfig(
        Ndraws=args.ndraws, scattering=args.scattering, damit_dir=args.damit_dir,
        geometry_dir=args.geometry_dir, damit_list=args.damit_list, base_dir=args.base_dir,
        noise_model=noise,
    )
    # CLI takes beta in degrees; internal configs/math use radians.
    b_peaks_rad = [math.radians(b) for b in args.b_peaks]
    csv_path = run_bias_map(base_cfg, args.p_peaks, b_peaks_rad,
                         nseeds=args.nseeds, seed=args.seed, outdir=args.outdir)
    print(f"\nBias map complete.\n  per-run stats:  {csv_path}"
          f"\n  summary plot:   {os.path.join(args.outdir, 'bias_map_summary.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
