#!/usr/bin/env python
"""CLI entry point for the LEADER shape/spin analysis.

Replaces the ``LEADER_python_final`` / ``_bg`` / ``_forcedN`` notebooks.
All values default to the original ``LEADER_python_final`` top cell, so::

    python scripts/run_analysis.py

reproduces that notebook's configuration.  Override any field on the command
line, e.g. a background population with forced-N subsampling::

    python scripts/run_analysis.py --famid BG_PB_Ctypes --population background
    python scripts/run_analysis.py --forced-n --wanted 11 --famid 4

Examples for a quick test run against existing data::

    python scripts/run_analysis.py --ntrials 2 --ndraws 50 --overwrite --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

# Use a non-interactive backend by default so the script runs headless.
os.environ.setdefault("MPLBACKEND", "Agg")

# Allow running from a source checkout without installing the package.

from pyleader.config import AnalysisConfig  # noqa: E402
from pyleader.analysis import run_analysis  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    d = AnalysisConfig()  # defaults
    p = argparse.ArgumentParser(description="Run the LEADER shape/spin inversion experiment.")
    p.add_argument("--famid", default=d.famid, help="family / population identifier")
    p.add_argument("--cat", default=d.cat, help="catalog used to generate the .obs files")
    p.add_argument("--filterpriority", default=d.filterpriority, help="photometry filter to analyze")
    p.add_argument("--diam-low", type=float, default=d.diam_low, help="lower diameter limit (km)")
    p.add_argument("--diam-high", type=float, default=d.diam_high, help="upper diameter limit (km)")
    p.add_argument("--phase-angle-limit", type=float, default=d.phase_angle_limit, help="max solar phase angle (deg)")
    p.add_argument("--ndraws", type=int, default=d.Ndraws, help="random draws per trial")
    p.add_argument("--ntrials", type=int, default=d.Ntrials, help="number of trials")
    p.add_argument("--date-tol", type=float, default=d.date_tol, help="max JD gap within an apparition")
    p.add_argument("--wanted", type=int, default=d.wanted, help="min data points per object per epoch")
    p.add_argument("--overwrite", action="store_true", default=d.overwrite, help="overwrite existing output")
    p.add_argument("--no-degrees", dest="convert2degrees", action="store_false", default=d.convert2degrees,
                   help="report/plot beta in radians instead of degrees")
    p.add_argument("--neowise-fle", default=d.neowise_fle, help="NEOWISE catalog file (name or absolute path)")
    p.add_argument("--population", dest="population_kind", choices=("family", "background"),
                   default=d.population_kind, help="directory-naming scheme")
    p.add_argument("--forced-n", dest="forced_n", action="store_true", default=d.forced_n,
                   help="subsample each object to `wanted` amplitudes (forcedN mode)")
    p.add_argument("--base-dir", default=d.base_dir, help="root working directory for inputs/outputs")
    p.add_argument("--obsdir", default=None,
                   help="read .obs files from this exact directory (bypasses the naming convention)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible draws")
    p.add_argument("--show", action="store_true", help="display plots interactively as well as saving")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = AnalysisConfig(
        famid=args.famid,
        cat=args.cat,
        filterpriority=args.filterpriority,
        diam_low=args.diam_low,
        diam_high=args.diam_high,
        phase_angle_limit=args.phase_angle_limit,
        Ndraws=args.ndraws,
        Ntrials=args.ntrials,
        date_tol=args.date_tol,
        wanted=args.wanted,
        overwrite=args.overwrite,
        convert2degrees=args.convert2degrees,
        neowise_fle=args.neowise_fle,
        population_kind=args.population_kind,
        forced_n=args.forced_n,
        base_dir=args.base_dir,
        obsdir=args.obsdir,
    )
    print(f"Reading .obs files from: {cfg.datadir}")
    outdir = run_analysis(cfg, seed=args.seed, show=args.show)
    print(f"Done. Output written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
