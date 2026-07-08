#!/usr/bin/env python
"""End-to-end per-population pipeline CLI.

Input a family or background population ID; the pipeline runs LEADER, derives a
correction from that population's own observing geometry, and applies it.

By default it assumes the population's ``.obs`` data directory already exists
(the analysis path). Use ``--build`` to first query NEOWISE/IPAC and write the
``.obs`` files (needs ``astropy``/``sunpy``/``requests`` + internet).

Examples::

    # analyze an existing family dataset end-to-end
    python scripts/run_population.py 1128 --diam-low 1 --diam-high 100 \
        --ntrials 5 --ndraws 200 --p-peaks 0.4 0.6 --b-peaks 0.3 0.9

    # a background population, building .obs first
    python scripts/run_population.py BG_IB_Ctypes --build
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.pipeline import PopulationConfig, run_population  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    d = PopulationConfig(pop_id="_")
    p = argparse.ArgumentParser(description="Run the end-to-end per-population LEADER pipeline.")
    p.add_argument("pop_id", help="family id (e.g. 1128) or background id (e.g. BG_IB_Ctypes)")
    p.add_argument("--population", dest="population_kind", choices=("family", "background"),
                   default=None, help="override population kind (inferred from the id by default)")
    p.add_argument("--cat", default=d.cat)
    p.add_argument("--filterpriority", default=d.filterpriority)
    p.add_argument("--diam-low", type=float, default=d.diam_low)
    p.add_argument("--diam-high", type=float, default=d.diam_high)
    p.add_argument("--ntrials", type=int, default=d.Ntrials)
    p.add_argument("--ndraws", type=int, default=d.Ndraws)
    p.add_argument("--phase-angle-limit", type=float, default=d.phase_angle_limit)
    p.add_argument("--date-tol", type=float, default=d.date_tol)
    p.add_argument("--wanted", type=int, default=d.wanted)
    p.add_argument("--p-peaks", type=float, nargs="+", default=list(d.p_peaks))
    p.add_argument("--b-peaks", type=float, nargs="+", default=[10.0, 30.0, 50.0, 75.0],
                   help="assigned beta peaks in DEGREES (0 < beta < 90)")
    p.add_argument("--sweep-ndraws", type=int, default=d.sweep_ndraws)
    p.add_argument("--nseeds", type=int, default=d.nseeds)
    p.add_argument("--scattering", choices=("ls_lambert", "hapke"), default=d.scattering)
    p.add_argument("--correction-stat", choices=("peak", "mean", "median"), default=d.correction_stat)
    p.add_argument("--correction-method", choices=("quadratic", "posterior", "both"),
                   default=d.correction_method,
                   help="quadratic (bias-map fit), posterior (credible intervals from a delta "
                        "basis), or both (default)")
    p.add_argument("--basis-dir", default=None,
                   help="delta-basis directory (default '<analysis outdir>_basis'; auto-built/"
                        "resumed when missing)")
    p.add_argument("--basis-nseeds", type=int, default=d.basis_nseeds,
                   help="realizations per basis grid point (default 4)")
    p.add_argument("--basis-nproc", type=int, default=None,
                   help="parallel workers for the basis (default: 8, capped at cores - 2)")
    p.add_argument("--posterior-stat", choices=("peak", "median", "both"), default=d.posterior_stat,
                   help="recovered statistic the posterior inverts; 'both' (default) also "
                        "reports a peak-vs-median consistency check")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--obsdir", default=None,
                   help="read/write .obs from this exact directory (bypasses the naming convention); "
                        "the correction sweep's geometry follows it")
    p.add_argument("--build", action="store_true", help="query NEOWISE and write .obs first")
    p.add_argument("--refresh-models", action="store_true",
                   help="re-download the latest DAMIT versions of the asteroideja.txt models first")
    p.add_argument("--seed", type=int, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cfg = PopulationConfig(
        pop_id=a.pop_id, population_kind=a.population_kind, cat=a.cat,
        filterpriority=a.filterpriority, diam_low=a.diam_low, diam_high=a.diam_high,
        Ntrials=a.ntrials, Ndraws=a.ndraws, phase_angle_limit=a.phase_angle_limit,
        date_tol=a.date_tol, wanted=a.wanted, p_peaks=tuple(a.p_peaks),
        # CLI takes beta in degrees; the config/API level uses radians.
        b_peaks=tuple(math.radians(b) for b in a.b_peaks),
        sweep_ndraws=a.sweep_ndraws, nseeds=a.nseeds, scattering=a.scattering,
        correction_stat=a.correction_stat, correction_method=a.correction_method,
        basis_dir=a.basis_dir, basis_nseeds=a.basis_nseeds, basis_nproc=a.basis_nproc,
        posterior_stat=a.posterior_stat, base_dir=a.base_dir, obsdir=a.obsdir,
    )
    res = run_population(cfg, do_build=a.build, refresh_models=a.refresh_models, seed=a.seed)
    print(f"\nDone. Report + correction in: {res.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
