#!/usr/bin/env python
"""Run the delta-basis campaign for a population's probabilistic correction.

Builds near-delta synthetic populations on an assigned ``(p_peak, b_peak)``
grid, observed at the population's own geometry — the forward-model samples the
posterior correction (and the response-matrix unfolding) are built from.
Resumable: completed grid points are skipped, so an interrupted campaign can
simply be re-run. Runs execute in parallel (one process per core by default).

Examples::

    # basis for family 1128 (default 8x8 grid x 4 seeds; ~10-15 min pooled)
    python scripts/basis_runs.py 1128 --diam-low 1 --diam-high 100

    # as one of 8 job-array chunks
    python scripts/basis_runs.py 1128 --task 3/8

    # arbitrary geometry directory instead of a population
    python scripts/basis_runs.py custom --geometry-dir /path/to/obs --outdir /path/to/basis
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

from pyleader.pipeline import PopulationConfig  # noqa: E402
from pyleader.synthetic.basis import run_basis  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the delta-basis campaign for a population.")
    p.add_argument("pop_id", help="family id (e.g. 1128), background id (BG_*), or a label when using --geometry-dir")
    p.add_argument("--grid-np", type=int, default=8, help="number of p grid points (default 8)")
    p.add_argument("--grid-nb", type=int, default=8, help="number of beta grid points (default 8)")
    p.add_argument("--p-range", type=float, nargs=2, default=[0.30, 0.80], metavar=("LO", "HI"),
                   help="p grid range (default 0.30 0.80)")
    p.add_argument("--b-range", type=float, nargs=2, default=[6.0, 84.0], metavar=("LO", "HI"),
                   help="beta grid range in DEGREES (default 6 84)")
    p.add_argument("--nseeds", type=int, default=4, help="realizations per grid point (default 4)")
    p.add_argument("--ndraws", type=int, default=1000, help="synthetic objects per run (default 1000)")
    p.add_argument("--scattering", choices=("ls_lambert", "hapke"), default="ls_lambert")
    # population/geometry selection (matched tolerances, like the pipeline)
    p.add_argument("--diam-low", type=float, default=3.0)
    p.add_argument("--diam-high", type=float, default=5.0)
    p.add_argument("--phase-angle-limit", type=float, default=40.0)
    p.add_argument("--date-tol", type=float, default=60.0)
    p.add_argument("--wanted", type=int, default=5)
    p.add_argument("--population", dest="population_kind", choices=("family", "background"), default=None)
    p.add_argument("--obsdir", default=None, help="population .obs directory (bypasses naming convention)")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--geometry-dir", default=None,
                   help="use this exact .obs directory as the geometry source instead of the population lookup")
    # execution
    p.add_argument("--outdir", default=None,
                   help="basis directory (default: '<analysis outdir>_basis' next to the analysis)")
    p.add_argument("--nproc", type=int, default=None, help="worker processes (default: 8, capped at cores - 2)")
    p.add_argument("--task", default=None, metavar="k/N", help="run only the k-th of N chunks")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    pc = PopulationConfig(
        pop_id=a.pop_id, population_kind=a.population_kind,
        diam_low=a.diam_low, diam_high=a.diam_high,
        phase_angle_limit=a.phase_angle_limit, date_tol=a.date_tol, wanted=a.wanted,
        sweep_ndraws=a.ndraws, scattering=a.scattering,
        base_dir=a.base_dir, obsdir=a.obsdir,
    )

    if a.geometry_dir is not None:
        import glob as _glob
        geom = sorted(_glob.glob(os.path.join(a.geometry_dir, "*.obs")))
        geom = [g for g in geom if not os.path.basename(g).startswith("Nofilter")]
        outdir = a.outdir
        if outdir is None:
            raise SystemExit("--outdir is required when using --geometry-dir")
    else:
        from pyleader.analysis import diameter_matched_files
        geom = diameter_matched_files(pc.analysis_config())
        outdir = a.outdir or f"{pc.analysis_config().outdir}_basis"

    if not geom:
        raise SystemExit("No .obs geometry files found for the basis.")
    print(f"[{a.pop_id}] basis geometry: {len(geom)} .obs files")

    base_cfg = pc.synthetic_base(geom)
    p_grid = np.linspace(a.p_range[0], a.p_range[1], a.grid_np)
    b_grid = np.deg2rad(np.linspace(a.b_range[0], a.b_range[1], a.grid_nb))

    run_basis(base_cfg, p_grid, b_grid, nseeds=a.nseeds, seed=a.seed,
              outdir=outdir, nproc=a.nproc, task=a.task)
    print(f"Basis directory: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
