"""Plots for a synthetic validation run.

Produces the solution diagnostics plus the key validation figures: the
*recovered* marginal distributions of ``p`` and ``beta`` overlaid on the
*assigned* (true) ones. Flat output in ``outdir`` (no per-trial subdirs).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..inversion import InversionResult


def _normalize(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return v / s if s > 0 else v


def synthetic_plots(result: InversionResult, p_true, beta_true, outdir, *,
                    convert2degrees=True, show=False):
    """Write validation + solution plots for one synthetic run into ``outdir``."""
    P = result.P
    BETA = result.BETA                       # radians
    W = result.W
    Pmargin = np.sum(W, axis=1)
    Bmargin = np.sum(W, axis=0)
    beta_axis = np.rad2deg(BETA) if convert2degrees else BETA
    beta_true_plot = np.rad2deg(beta_true) if convert2degrees else beta_true
    beta_label = r"$\beta$ (deg)" if convert2degrees else r"$\beta$"

    # --- 1. CDF(A) fit ---
    plt.figure()
    plt.plot(result.Asort, result.CDFA, "bo", ms=3, label="CDF of A")
    plt.plot(result.Asort, result.M @ result.W_back, "rx", ms=3, label=r"$\sum w_{ij}F_{ij}$")
    plt.grid(True)
    plt.xlabel("A")
    plt.title(f"Relative error of the fit: {result.relerr:.5f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/RelativeError.png", dpi=300)
    plt.show() if show else None
    plt.close()

    # --- 2. Occupation-number contour ---
    plt.figure()
    cp = plt.contourf(P, beta_axis, W.T, levels=100, cmap="viridis")
    plt.colorbar(cp)
    plt.xlabel("p")
    plt.ylabel(beta_label)
    plt.title("Recovered occupation weights")
    plt.tight_layout()
    plt.savefig(f"{outdir}/OccupationNumbers_w_contour.png", dpi=300)
    plt.show() if show else None
    plt.close()

    # --- 3. Recovered vs assigned marginal of p ---
    plt.figure()
    plt.hist(p_true, bins=15, range=(0, 1), density=True, alpha=0.5,
             color="green", label="assigned (true)")
    width = P[1] - P[0] if len(P) > 1 else 0.05
    plt.bar(P, _normalize(Pmargin) / width, width=width, alpha=0.6,
            color="tab:blue", label="recovered")
    plt.xlabel("p")
    plt.ylabel("normalized DF")
    plt.title("Shape elongation p: recovered vs assigned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/Margin_p_recovered_vs_true.png", dpi=300)
    plt.show() if show else None
    plt.close()

    # --- 4. Recovered vs assigned marginal of beta ---
    plt.figure()
    hi = 90 if convert2degrees else np.pi / 2
    plt.hist(beta_true_plot, bins=16, range=(0, hi), density=True, alpha=0.5,
             color="green", label="assigned (true)")
    bwidth = beta_axis[1] - beta_axis[0] if len(beta_axis) > 1 else hi / 16
    plt.bar(beta_axis, _normalize(Bmargin) / bwidth, width=bwidth, alpha=0.6,
            color="tab:blue", label="recovered")
    plt.xlabel(beta_label)
    plt.ylabel("normalized DF")
    plt.title(r"Spin latitude $\beta$: recovered vs assigned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/Margin_beta_recovered_vs_true.png", dpi=300)
    plt.show() if show else None
    plt.close()

    # --- 5. Smoothed joint distribution (as in leader_postprocess_WISE) ---
    pind, bind = np.unravel_index(np.argmax(W), W.shape)
    W_after = np.empty_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_after[i, j] = W[i, j] / ((abs(pind - i) + abs(bind - j) + 1))
    plt.figure()
    cp = plt.contourf(P, beta_axis, W_after.T, levels=100, cmap="viridis")
    plt.colorbar(cp)
    plt.xlabel("p")
    plt.ylabel(beta_label)
    plt.title("Smoothed joint distribution f(p, β)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Solutions_smoothed.png", dpi=300)
    plt.show() if show else None
    plt.close()
