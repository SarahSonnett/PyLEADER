#!/usr/bin/env python
"""Fit and record a bias-correction function from a synthetic sweep.

Reads a ``sweep_stats.csv`` (from ``scripts/sweep_synthetic.py``), fits the
recovered->true correction surfaces for p and beta, writes the coefficients to
``correction_function.json``, and saves a predicted-vs-actual diagnostic plot.

Apply the result to real LEADER output with::

    from pyleader.synthetic.correction import load_correction, apply_correction
    p_true, beta_true = apply_correction(p_recovered, beta_recovered_deg, load_correction("correction_function.json"))

Example::

    python scripts/fit_correction.py ~/synthetic_sweep/sweep_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from pyleader.synthetic.correction import (  # noqa: E402
    apply_correction, fit_from_csv, save_correction,
)


def _diagnostic_plot(csv_path, coeffs, out_png, stat):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    p_rec = np.array([float(r[f"p_recovered_{stat}"]) for r in rows])
    b_rec = np.array([float(r[f"beta_recovered_{stat}"]) for r in rows])
    p_true = np.array([float(r[f"p_assigned_{stat}"]) for r in rows])
    b_true = np.array([float(r[f"beta_assigned_{stat}"]) for r in rows])
    p_fit, b_fit = apply_correction(p_rec, b_rec, coeffs)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    d = coeffs["diagnostics"]
    for ax, true, fit, lbl, r2 in (
        (a1, p_true, p_fit, "p", d["r2_p"]),
        (a2, b_true, b_fit, "β (deg)", d["r2_b"]),
    ):
        lo, hi = min(true.min(), fit.min()), max(true.max(), fit.max())
        ax.plot([lo, hi], [lo, hi], "k--", label="1:1")
        ax.scatter(true, fit, s=25, alpha=0.8)
        ax.set_xlabel(f"true {lbl}")
        ax.set_ylabel(f"corrected {lbl}")
        ax.set_title(f"{lbl}: corrected vs true  (R²={r2:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Correction fit ({stat}-based, recovered→true)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fit a bias-correction function from a sweep CSV.")
    p.add_argument("csv", help="path to sweep_stats.csv")
    p.add_argument("--stat", default="mean", choices=("mean", "median"), help="statistic to correct")
    p.add_argument("-o", "--out", default=None, help="output JSON (default next to CSV)")
    args = p.parse_args(argv)

    outdir = os.path.dirname(os.path.abspath(args.csv))
    out_json = args.out or os.path.join(outdir, "correction_function.json")
    out_png = os.path.join(outdir, "correction_fit.png")

    coeffs = fit_from_csv(args.csv, stat=args.stat)
    save_correction(coeffs, out_json)
    _diagnostic_plot(args.csv, coeffs, out_png, args.stat)

    d = coeffs["diagnostics"]
    print(f"Fitted recovered->true correction ({args.stat}-based, n={d['n']}):")
    print(f"  p    : R²={d['r2_p']:.3f}  RMSE={d['rmse_p']:.3f}")
    print(f"  beta : R²={d['r2_b']:.3f}  RMSE={d['rmse_b']:.2f} deg  "
          f"(recovered beta only spans {d['b_rec_range'][0]:.0f}-{d['b_rec_range'][1]:.0f} deg)")
    print(f"  wrote {out_json}")
    print(f"  wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
