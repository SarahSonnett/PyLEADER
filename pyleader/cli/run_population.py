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
    p.add_argument("--b-peaks", type=float, nargs="+", default=list(d.b_peaks))
    p.add_argument("--sweep-ndraws", type=int, default=d.sweep_ndraws)
    p.add_argument("--nseeds", type=int, default=d.nseeds)
    p.add_argument("--scattering", choices=("ls_lambert", "hapke"), default=d.scattering)
    p.add_argument("--correction-stat", choices=("peak", "mean", "median"), default=d.correction_stat)
    p.add_argument("--base-dir", default=None)
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
        date_tol=a.date_tol, wanted=a.wanted, p_peaks=tuple(a.p_peaks), b_peaks=tuple(a.b_peaks),
        sweep_ndraws=a.sweep_ndraws, nseeds=a.nseeds, scattering=a.scattering,
        correction_stat=a.correction_stat, base_dir=a.base_dir,
    )
    res = run_population(cfg, do_build=a.build, refresh_models=a.refresh_models, seed=a.seed)
    print(f"\nDone. Report + correction in: {res.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
