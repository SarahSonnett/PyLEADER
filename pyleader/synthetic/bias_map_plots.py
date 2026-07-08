"""Summarize a bias map: recovered vs. assigned as a 2-panel figure.

Reads a ``bias_map_stats.csv`` (from ``scripts/bias_map.py``) and plots, per
trial, the recovered vs. assigned distribution means as a function of the two
assigned input parameters:

* top panel    — mean ``p`` vs assigned ``p_peak`` (one series per ``b_peak``)
* bottom panel — mean ``beta`` vs assigned ``b_peak`` (one series per ``p_peak``)

Recovered points carry error bars = ±1σ across the seeds run at each grid point;
the black dashed line is the assigned truth (the target the recovery should hit).
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def _load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else np.nan) for k, v in r.items()})
    return rows


def _agg(rows, key_field, series_field, y_recovered, y_assigned):
    """Aggregate over seeds: return series -> (x, y_rec_mean, y_rec_std, y_assigned)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r[series_field], r[key_field])].append(r)

    series_vals = sorted({r[series_field] for r in rows})
    x_vals = sorted({r[key_field] for r in rows})

    out = {}
    for s in series_vals:
        xs, yr, ye, ya = [], [], [], []
        for x in x_vals:
            grp = groups.get((s, x))
            if not grp:
                continue
            rec = np.array([g[y_recovered] for g in grp])
            xs.append(x)
            yr.append(rec.mean())
            ye.append(rec.std())
            ya.append(np.mean([g[y_assigned] for g in grp]))
        out[s] = (np.array(xs), np.array(yr), np.array(ye), np.array(ya))
    return out, x_vals


def plot_bias_map(csv_path: str, out_png: str | None = None, *, show: bool = False):
    """Render the 2-panel recovered-vs-assigned summary for a bias-map CSV."""
    rows = _load(csv_path)
    nseeds = len({r["seed"] for r in rows}) if "seed" in rows[0] else 1

    fig, (axp, axb) = plt.subplots(2, 1, figsize=(8, 9))

    # --- top: p vs p_peak, one series per b_peak ---
    p_series, p_x = _agg(rows, "p_peak", "b_peak_deg", "p_recovered_mean", "p_assigned_mean")
    for bdeg, (xs, yr, ye, _ya) in p_series.items():
        axp.errorbar(xs, yr, yerr=ye, marker="o", capsize=3, label=f"recovered, β_peak={bdeg:.0f}°")
    # assigned reference (mean over all b at each p_peak)
    asgn_p = [np.mean([r["p_assigned_mean"] for r in rows if r["p_peak"] == x]) for x in p_x]
    axp.plot(p_x, asgn_p, "k--", marker="s", label="assigned (true)")
    axp.set_xlabel("assigned  p_peak")
    axp.set_ylabel("recovered mean p")
    axp.set_title("Shape elongation p: recovered vs assigned")
    axp.grid(True, alpha=0.3)
    axp.legend(fontsize=8)

    # --- bottom: beta vs b_peak, one series per p_peak ---
    b_series, b_x = _agg(rows, "b_peak_deg", "p_peak", "beta_recovered_mean", "beta_assigned_mean")
    for pk, (xs, yr, ye, _ya) in b_series.items():
        axb.errorbar(xs, yr, yerr=ye, marker="o", capsize=3, label=f"recovered, p_peak={pk:.2f}")
    asgn_b = [np.mean([r["beta_assigned_mean"] for r in rows if r["b_peak_deg"] == x]) for x in b_x]
    axb.plot(b_x, asgn_b, "k--", marker="s", label="assigned (true)")
    axb.set_xlabel("assigned  β_peak (deg)")
    axb.set_ylabel("recovered mean β (deg)")
    axb.set_title(r"Spin latitude $\beta$: recovered vs assigned")
    axb.grid(True, alpha=0.3)
    axb.legend(fontsize=8)

    fig.suptitle(f"Bias map summary (error bars = ±1σ over {nseeds} seed"
                 f"{'s' if nseeds != 1 else ''})")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_png
