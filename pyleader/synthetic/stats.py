"""Summary statistics for assigned vs. recovered (p, beta) distributions."""

from __future__ import annotations

import numpy as np


def distribution_stats(values, weights=None) -> dict:
    """Return ``{min, max, mean, median}`` for a distribution.

    * ``weights=None`` -> raw sample statistics (used for the *assigned* truth,
      where we have the individual object values).
    * ``weights`` given -> weighted statistics over a grid (used for the
      *recovered* distribution function). ``min``/``max`` are the support
      (grid values carrying non-zero weight); ``mean`` is the weighted mean and
      ``median`` the weighted 50th percentile.
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }

    weights = np.asarray(weights, dtype=float)
    nz = weights > 0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return {
        "min": float(values[nz].min()) if nz.any() else float("nan"),
        "max": float(values[nz].max()) if nz.any() else float("nan"),
        "mean": float(np.sum(values * weights) / np.sum(weights)),
        "median": float(np.interp(0.5 * cw[-1], cw, v)),
    }


def compute_stats(p_true, beta_true, P, Pmargin, BETA, Bmargin) -> dict:
    """Assemble assigned/recovered stats for p and beta (beta in degrees)."""
    return {
        "p": {
            "assigned": distribution_stats(p_true),
            "recovered": distribution_stats(P, Pmargin),
        },
        "beta_deg": {
            "assigned": distribution_stats(np.rad2deg(beta_true)),
            "recovered": distribution_stats(np.rad2deg(BETA), Bmargin),
        },
    }


_QUANTITIES = (("p", "p"), ("beta_deg", "beta (deg)"))
_KINDS = ("assigned", "recovered")


def write_stats_file(path: str, stats: dict, label: str = "") -> None:
    """Write a human-readable min/max/mean/median table for one run."""
    with open(path, "w") as f:
        if label:
            f.write(f"# {label}\n")
        f.write("quantity        kind        min       max      mean    median\n")
        for key, disp in _QUANTITIES:
            for kind in _KINDS:
                s = stats[key][kind]
                f.write("%-14s  %-9s  %8.4f  %8.4f  %8.4f  %8.4f\n"
                        % (disp, kind, s["min"], s["max"], s["mean"], s["median"]))


def stats_row(stats: dict) -> dict:
    """Flatten stats to ``{p_assigned_mean: ..., beta_recovered_median: ...}`` for a CSV row."""
    row = {}
    for key, _ in _QUANTITIES:
        name = "p" if key == "p" else "beta"
        for kind in _KINDS:
            for stat, val in stats[key][kind].items():
                row[f"{name}_{kind}_{stat}"] = val
    return row
