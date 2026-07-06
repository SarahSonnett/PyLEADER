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


def _n_terms(n_points: int) -> int:
    """Polynomial terms to fit given the sample count (avoid overfitting).

    Quadratic (6) needs a comfortable margin; drop to linear (3) or a constant
    offset (1) for small sweeps so the correction never interpolates noise.
    """
    if n_points >= 12:
        return 6
    if n_points >= 6:
        return 3
    return 1


def _fit_one(X, y, k):
    """Least-squares fit using the first ``k`` design terms; return full coeffs."""
    coeffs_k, *_ = np.linalg.lstsq(X[:, :k], y, rcond=None)
    coeffs = np.zeros(X.shape[1])
    coeffs[:k] = coeffs_k
    resid = y - X[:, :k] @ coeffs_k
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
    k = _n_terms(len(p_rec))

    cp, r2p, rmsep = _fit_one(X, np.asarray(p_true, float), k)
    cb, r2b, rmseb = _fit_one(X, np.asarray(b_true, float), k)

    return {
        "direction": "recovered -> true (apply to LEADER results on real data)",
        "stat": stat,
        "beta_units": "degrees",
        "terms": TERMS,
        "n_terms": k,
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
    """Fit a correction from a ``sweep_stats.csv`` using the chosen statistic.

    ``stat`` is ``"mean"`` or ``"median"`` (distribution statistics), or
    ``"peak"`` — the distribution peak (recovered peak vs the assigned peak),
    which matches what the LEADER analysis reports per trial (pmax/betamax).
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if stat == "peak":
        p_rec = [float(r["p_recovered_peak"]) for r in rows]
        b_rec = [float(r["beta_recovered_peak_deg"]) for r in rows]
        p_true = [float(r["p_peak"]) for r in rows]
        b_true = [float(r["b_peak_deg"]) for r in rows]
    else:
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


def plot_correction_fit(csv_path: str, coeffs: dict, out_png: str) -> None:
    """Diagnostic: corrected-vs-true scatter for p and beta, with 1:1 lines."""
    import csv as _csv
    import numpy as _np
    import matplotlib.pyplot as plt

    stat = coeffs.get("stat", "mean")
    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if stat == "peak":
        p_rec = _np.array([float(r["p_recovered_peak"]) for r in rows])
        b_rec = _np.array([float(r["beta_recovered_peak_deg"]) for r in rows])
        p_true = _np.array([float(r["p_peak"]) for r in rows])
        b_true = _np.array([float(r["b_peak_deg"]) for r in rows])
    else:
        p_rec = _np.array([float(r[f"p_recovered_{stat}"]) for r in rows])
        b_rec = _np.array([float(r[f"beta_recovered_{stat}"]) for r in rows])
        p_true = _np.array([float(r[f"p_assigned_{stat}"]) for r in rows])
        b_true = _np.array([float(r[f"beta_assigned_{stat}"]) for r in rows])
    p_fit, b_fit = apply_correction(p_rec, b_rec, coeffs)

    d = coeffs["diagnostics"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, true, fit, lbl, r2 in ((a1, p_true, p_fit, "p", d["r2_p"]),
                                   (a2, b_true, b_fit, "β (deg)", d["r2_b"])):
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
