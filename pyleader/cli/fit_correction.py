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
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.correction import (  # noqa: E402
    fit_from_csv, plot_correction_fit, save_correction,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fit a bias-correction function from a sweep CSV.")
    p.add_argument("csv", help="path to sweep_stats.csv")
    p.add_argument("--stat", default="mean", choices=("mean", "median", "peak"),
                   help="statistic to correct (peak matches LEADER's pmax/betamax)")
    p.add_argument("-o", "--out", default=None, help="output JSON (default next to CSV)")
    args = p.parse_args(argv)

    outdir = os.path.dirname(os.path.abspath(args.csv))
    out_json = args.out or os.path.join(outdir, "correction_function.json")
    out_png = os.path.join(outdir, "correction_fit.png")

    coeffs = fit_from_csv(args.csv, stat=args.stat)
    save_correction(coeffs, out_json)
    plot_correction_fit(args.csv, coeffs, out_png)

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
