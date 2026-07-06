#!/usr/bin/env python
"""CLI for the synthetic LEADER validation run.

Replaces ``leader_synth_main_WISE.m``. Builds a synthetic population with
assigned shape/spin peaks, recovers them with the LEADER inversion, and writes
recovered-vs-assigned validation plots.

Needs DAMIT shape models (see ``--download``) and WISE geometry .obs files.

Examples::

    # download the models listed in asteroideja.txt, then run
    python scripts/run_synthetic.py --download --p-peak 0.5 --b-peak 0.4 --seed 0

    # quick run against already-downloaded models
    python scripts/run_synthetic.py --p-peak 0.6 --b-peak 0.3 --ndraws 200 --seed 1
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.config import SyntheticConfig  # noqa: E402
from pyleader.synthetic.population import run_synthetic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    d = SyntheticConfig()
    p = argparse.ArgumentParser(description="Run a synthetic LEADER validation experiment.")
    p.add_argument("--p-peak", type=float, default=None, help="assigned shape-elongation peak (random if unset)")
    p.add_argument("--b-peak", type=float, default=None, help="assigned spin-latitude peak in radians (random if unset)")
    p.add_argument("--ndraws", type=int, default=d.Ndraws, help="number of synthetic objects")
    p.add_argument("--wanted", type=int, default=d.wanted, help="min points per apparition")
    p.add_argument("--date-tol", type=float, default=d.date_tol, help="max JD gap within an apparition")
    p.add_argument("--phase-angle-limit", type=float, default=d.phase_angle_limit, help="max solar phase angle (deg)")
    p.add_argument("--noise-level", type=float, default=d.noise_level, help="fractional Gaussian noise on L")
    p.add_argument("--scattering", choices=("ls_lambert", "hapke"), default=d.scattering, help="scattering law")
    p.add_argument("--trot-min-hr", type=float, default=d.trot_min_hr, help="min rotation period (hours)")
    p.add_argument("--trot-max-hr", type=float, default=d.trot_max_hr, help="max rotation period (hours)")
    p.add_argument("--no-degrees", dest="convert2degrees", action="store_false", default=d.convert2degrees)
    p.add_argument("--damit-list", default=d.damit_list, help="asteroideja.txt model listing")
    p.add_argument("--damit-dir", default=d.damit_dir, help="directory of DAMIT model files")
    p.add_argument("--geometry-dir", default=d.geometry_dir, help="directory of WISE .obs geometry files")
    p.add_argument("--base-dir", default=d.base_dir, help="root working directory")
    p.add_argument("--outdir", default=None, help="output directory (default: auto by peaks)")
    p.add_argument("--download", action="store_true", help="download DAMIT models in the listing first")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--show", action="store_true", help="display plots as well as saving")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SyntheticConfig(
        p_peak=args.p_peak, b_peak=args.b_peak, Ndraws=args.ndraws, wanted=args.wanted,
        date_tol=args.date_tol, phase_angle_limit=args.phase_angle_limit,
        noise_level=args.noise_level, scattering=args.scattering,
        trot_min_hr=args.trot_min_hr, trot_max_hr=args.trot_max_hr,
        convert2degrees=args.convert2degrees, damit_list=args.damit_list,
        damit_dir=args.damit_dir, geometry_dir=args.geometry_dir,
        base_dir=args.base_dir, outdir=args.outdir,
    )

    if args.download:
        from pyleader.synthetic.damit import download_damit_models, parse_model_list
        numbers = parse_model_list(cfg.damit_list)
        download_damit_models(numbers, cfg.damit_dir)

    run_synthetic(cfg, seed=args.seed, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
