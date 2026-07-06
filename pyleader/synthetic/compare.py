"""Compare two recovered (p, beta) distributions.

Ports ``KS_comparison.m`` (the L1/L2/L-inf distances between two populations'
marginal CDFs) and the wiring role of ``ast_comparison_WISE.m``. Comparing a
*recovered* synthetic population against its *assigned* truth (or two different
populations) is the basis for a correction function on real-data results.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt


def _cdf(grid, margin, hi):
    """Return padded grid ``[0, grid, hi]`` and its normalized CDF ``[0, ..., 1]``."""
    margin = np.asarray(margin, dtype=float)
    margin = margin / margin.sum()
    g = np.concatenate([[0.0], np.asarray(grid, dtype=float), [hi]])
    c = np.concatenate([[0.0], np.cumsum(margin), [1.0]])
    return g, c


def _distances(c1, c2i):
    """L1/4, L2, and 2*L-inf differences between two CDFs (as in KS_comparison.m)."""
    d = c1 - c2i
    return np.array([np.linalg.norm(d, 1) / 4, np.linalg.norm(d), 2 * np.linalg.norm(d, np.inf)])


def ks_comparison(P1, Pm1, B1, Bm1, P2, Pm2, B2, Bm2, *,
                  outdir=None, labels=("pop 1", "pop 2"), show=False):
    """Compare the p and beta marginal CDFs of two populations.

    Returns ``(DvalueP, DvalueB, results)`` where each ``Dvalue`` is
    ``[L1/4, L2, 2*Linf]``. ``B*`` grids are spin latitudes in radians.
    """
    # --- shape elongation p (domain [0, 1]) ---
    P1g, CP1 = _cdf(P1, Pm1, 1.0)
    P2g, CP2 = _cdf(P2, Pm2, 1.0)
    CP2i = np.interp(P1g, P2g, CP2)
    DvalueP = _distances(CP1, CP2i)

    # --- spin latitude beta (domain [0, pi/2]) ---
    B1g, CB1 = _cdf(B1, Bm1, np.pi / 2)
    B2g, CB2 = _cdf(B2, Bm2, np.pi / 2)
    CB2i = np.interp(B1g, B2g, CB2)
    DvalueB = _distances(CB1, CB2i)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        _plot_pair(P1g, CP1, CP2i, "p", DvalueP, f"{outdir}/Comparison_p.png", labels, show)
        _plot_pair(B1g, CB1, CB2i, r"$\beta$ (rad)", DvalueB, f"{outdir}/Comparison_beta.png", labels, show)

    results = {
        "p": dict(grid=P1g, cdf1=CP1, cdf2_interp=CP2i),
        "beta": dict(grid=B1g, cdf1=CB1, cdf2_interp=CB2i),
        "DvalueP": DvalueP, "DvalueB": DvalueB,
    }
    return DvalueP, DvalueB, results


def _plot_pair(grid, c1, c2i, xlabel, dval, path, labels, show):
    plt.figure()
    plt.plot(grid, c1, "b-", lw=3, label=labels[0])
    plt.plot(grid, c2i, "r-.", lw=3, label=f"{labels[1]} (interp)")
    plt.xlabel(xlabel)
    plt.ylabel(f"CDF of {xlabel}")
    plt.title(f"D(L1)={dval[0]:.4f}, D(L2)={dval[1]:.4f}, D(Linf)={dval[2]:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show() if show else None
    plt.close()


def compare_populations(npz1: str, npz2: str, outdir: str, *,
                        labels=("pop 1", "pop 2"), show=False):
    """Compare two saved synthetic runs (``synthetic_result.npz`` files).

    Returns ``(DvalueP, DvalueB, results)``.
    """
    from .population import SyntheticResult

    a = SyntheticResult.load_marginals(npz1)
    b = SyntheticResult.load_marginals(npz2)
    return ks_comparison(
        a["P"], a["Pmargin"], a["BETA"], a["Bmargin"],
        b["P"], b["Pmargin"], b["BETA"], b["Bmargin"],
        outdir=outdir, labels=labels, show=show,
    )
