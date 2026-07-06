"""Run a synthetic sweep over a (p_peak, b_peak) grid and tabulate the stats.

Shared by ``scripts/sweep_synthetic.py`` and the per-population pipeline.
"""

from __future__ import annotations

import csv
import itertools
import os
from dataclasses import replace

import numpy as np

from .config import SyntheticConfig
from .population import run_synthetic
from .stats import stats_row
from .sweep_plots import plot_sweep

_STAT_COLS = [f"{q}_{kind}_{stat}"
              for q in ("p", "beta")
              for kind in ("assigned", "recovered")
              for stat in ("min", "max", "mean", "median")]
COLUMNS = (["trial", "seed", "p_peak", "b_peak_rad", "b_peak_deg",
            "p_recovered_peak", "beta_recovered_peak_deg", "relerr"] + _STAT_COLS)


def run_sweep(base_cfg: SyntheticConfig, p_peaks, b_peaks, *,
              nseeds: int = 1, seed: int = 0, outdir: str, verbose: bool = True) -> str:
    """Run the grid × seeds sweep; write ``sweep_stats.csv`` + ``sweep_summary.png``.

    ``base_cfg`` supplies everything except ``p_peak``/``b_peak``/``outdir`` (which
    vary per run) — including the geometry source (``geometry_dir`` or
    ``geometry_files``), ``Ndraws``, scattering, and the matched tolerances.
    Returns the path to ``sweep_stats.csv``.
    """
    os.makedirs(outdir, exist_ok=True)
    grid = list(itertools.product(p_peaks, b_peaks))
    if verbose:
        print(f"Sweep: {len(grid)} grid points x {nseeds} seed(s) = {len(grid) * nseeds} runs")

    rows = []
    run_idx = 0
    for i, (p_peak, b_peak) in enumerate(grid):
        base = os.path.join(outdir, f"trial{i:03d}_p{p_peak:.2f}_b{b_peak:.2f}")
        if verbose:
            print(f"\n=== trial {i}: p_peak={p_peak}, b_peak={b_peak} rad "
                  f"({np.rad2deg(b_peak):.1f} deg)")
        for s in range(nseeds):
            subdir = base if nseeds == 1 else os.path.join(base, f"seed{s}")
            cfg = replace(base_cfg, p_peak=p_peak, b_peak=b_peak, outdir=subdir)
            res = run_synthetic(cfg, seed=seed + run_idx, make_plots=(s == 0))
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

    csv_path = os.path.join(outdir, "sweep_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in COLUMNS})

    plot_sweep(csv_path, os.path.join(outdir, "sweep_summary.png"))
    return csv_path
