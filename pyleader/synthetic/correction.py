"""Fit and apply a bias-correction function from a synthetic sweep.

The synthetic validation shows that LEADER's recovered (p, beta) distributions
are biased relative to the assigned truth. This module fits a correction that
maps what LEADER *recovers* back to the *true* value, so it can be applied to
LEADER results on real (non-synthetic) data:

    (p_true, beta_true) ~= C(p_recovered, beta_recovered)

Each quantity is modeled as a 2D quadratic surface in (p_rec, beta_rec):

    q_true = c0 + c1 p + c2 b + c3 p^2 + c4 p b + c5 b^2

fit by least squares to the per-(grid-point, seed) means in a sweep CSV. Beta is
in degrees throughout.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os

import numpy as np

TERMS = ["1", "p", "b", "p^2", "p*b", "b^2"]

# Canonical correction shipped with the package (fit from a 20x3 synthetic
# sweep; regenerate with scripts/fit_correction.py on your own sweep).
_DEFAULT_CORRECTION = os.path.join(os.path.dirname(__file__), "data", "correction_function.json")


def _design(p, b):
    """Quadratic design matrix for inputs p and b (degrees)."""
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.column_stack([np.ones_like(p), p, b, p * p, p * b, b * b])


def _fit_one(X, y):
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coeffs
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    return coeffs, r2, rmse


def fit_correction(p_rec, b_rec, p_true, b_true, *, stat="mean", source=None) -> dict:
    """Fit the recovered->true correction surfaces for p and beta (degrees).

    Returns a JSON-serializable dict of coefficients + fit diagnostics.
    """
    p_rec = np.asarray(p_rec, float)
    b_rec = np.asarray(b_rec, float)
    X = _design(p_rec, b_rec)

    cp, r2p, rmsep = _fit_one(X, np.asarray(p_true, float))
    cb, r2b, rmseb = _fit_one(X, np.asarray(b_true, float))

    return {
        "direction": "recovered -> true (apply to LEADER results on real data)",
        "stat": stat,
        "beta_units": "degrees",
        "terms": TERMS,
        "coeffs_p_true": cp.tolist(),
        "coeffs_b_true": cb.tolist(),
        "diagnostics": {
            "n": int(len(p_rec)),
            "r2_p": r2p, "rmse_p": rmsep,
            "r2_b": r2b, "rmse_b": rmseb,
            "p_rec_range": [float(p_rec.min()), float(p_rec.max())],
            "b_rec_range": [float(b_rec.min()), float(b_rec.max())],
        },
        "caveat": (
            "The recovered beta range is compressed relative to the true range, so "
            "the beta correction is poorly constrained and extrapolates outside "
            "b_rec_range; treat corrected beta as indicative, not precise."
        ),
        "source_csv": source,
        "generated_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def apply_correction(p_rec, b_rec, coeffs: dict):
    """Apply a fitted correction: return estimated ``(p_true, beta_true_deg)``."""
    X = _design(p_rec, b_rec)
    p_true = X @ np.asarray(coeffs["coeffs_p_true"], float)
    b_true = X @ np.asarray(coeffs["coeffs_b_true"], float)
    # keep results physical
    p_true = np.clip(p_true, 0.0, 1.0)
    b_true = np.clip(b_true, 0.0, 90.0)
    return p_true, b_true


def fit_from_csv(csv_path: str, *, stat: str = "mean") -> dict:
    """Fit a correction from a ``sweep_stats.csv`` using the chosen statistic."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    p_rec = [float(r[f"p_recovered_{stat}"]) for r in rows]
    b_rec = [float(r[f"beta_recovered_{stat}"]) for r in rows]
    p_true = [float(r[f"p_assigned_{stat}"]) for r in rows]
    b_true = [float(r[f"beta_assigned_{stat}"]) for r in rows]
    return fit_correction(p_rec, b_rec, p_true, b_true, stat=stat, source=csv_path)


def save_correction(coeffs: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(coeffs, f, indent=2)


def load_correction(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def default_correction() -> dict:
    """Load the canonical correction shipped with the package.

    Regenerate it for your own sweep with ``scripts/fit_correction.py`` and pass
    the resulting JSON to :func:`load_correction` instead.
    """
    return load_correction(_DEFAULT_CORRECTION)
