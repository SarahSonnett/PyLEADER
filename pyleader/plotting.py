"""Plotting routines for LEADER results.

Ported from the ``leader_plots`` and ``plot_alltrials`` cells.  All state that
the notebook pulled from globals (the inversion result, ``outdir``, ``trial``,
config) is now passed in explicitly.

FIX (#4): the notebook imported ``scipy.interpolate.interp2d`` (removed in
modern SciPy) and ``matplotlib.mlab`` (removed in modern Matplotlib).  The only
use of ``interp2d`` was an unreachable branch (it required globals ``p``/``beta``
that were never defined), so it has been dropped along with the dead imports.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy.stats import norm

from .config import AnalysisConfig
from .inversion import InversionResult


def leader_plots(
    result: InversionResult,
    cfg: AnalysisConfig,
    outdir: str,
    trial: int,
    *,
    show: bool = False,
) -> None:
    """Write the per-trial diagnostic plots and the marginal-DF text file.

    ``result.BETA`` may already be in degrees (the driver converts it when
    ``cfg.convert2degrees`` is set), matching the notebook's behaviour.
    """
    Asort, CDFA = result.Asort, result.CDFA
    M, W_back, W = result.M, result.W_back, result.W
    P, BETA = result.P, result.BETA
    pmax, betamax, relerr = result.pmax, result.betamax, result.relerr
    tdir = f"{outdir}/Trial{trial + 1}"

    # --- 1. CDF(A) and the fit ---
    plt.figure()
    plt.plot(Asort, CDFA, "bo", label="CDF of A")
    plt.plot(Asort, M @ W_back, "rx", label=r"$\sum w_{ij} F_{ij}$")
    plt.grid(True)
    plt.xlabel("A")
    plt.title(f"Relative error of the fit: {relerr:.5f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{tdir}/RelativeError_trial{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- 2. Surface plot of occupation numbers W ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    P_grid, BETA_grid = np.meshgrid(P, BETA, indexing="ij")
    ax.plot_surface(P_grid, BETA_grid, W, cmap="viridis", edgecolor="none")
    ax.plot([pmax], [betamax], [np.max(W)], "rx", markersize=10, label="Peak")
    ax.set_xlim(0, 1)
    if cfg.convert2degrees:
        ax.set_ylim(0, 90)
        ax.set_ylabel(r"$\beta$" + " (deg)")
    else:
        ax.set_ylim(0, np.pi / 2)
        ax.set_ylabel(r"$\beta$")
    ax.set_xlabel("p")
    ax.set_zlabel("w")
    ax.set_title("Occupation numbers (w)")
    plt.tight_layout()
    plt.savefig(f"{tdir}/OccupationNumbers_w_trial{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- 3. Contour plot ---
    plt.figure()
    cp = plt.contourf(P, BETA, W.T, levels=100, cmap="viridis")
    plt.colorbar(cp)
    plt.xlabel("p")
    plt.ylabel(r"$\beta$")
    plt.title("Contour plot of occupation weights")
    plt.tight_layout()
    plt.savefig(f"{tdir}/OccupationNumbers_w_contour_trial{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- 4. Marginal distributions ---
    Pmargin = np.sum(W, axis=1)   # sum over beta
    Bmargin = np.sum(W, axis=0)   # sum over p

    plt.figure()
    plt.bar(P, Pmargin, width=0.04)
    plt.xlabel("p")
    plt.ylabel("w")
    plt.ylim(0, max(np.max(Pmargin), 0.2))
    plt.title("Marginal DF of p")
    plt.tight_layout()
    plt.savefig(f"{tdir}/Margin_p_trial{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    plt.figure()
    if cfg.convert2degrees:
        plt.bar(BETA, Bmargin, width=4.0)
    else:
        plt.bar(BETA, Bmargin, width=0.05)
    plt.ylim(0, max(np.max(Bmargin), 0.2))
    plt.xlabel(r"$\beta$")
    plt.ylabel("w")
    plt.title("Marginal DF of β")
    plt.tight_layout()
    plt.savefig(f"{tdir}/Margin_beta_trial{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- 5. Diameter distribution of the population drawn ---
    drawn_file = f"{tdir}/ObjectsDrawn_famid{cfg.famid}trial{trial + 1}.txt"
    diam_used = np.genfromtxt(drawn_file, unpack=True, usecols=(2), dtype=float, skip_header=1)
    diam_used = np.asarray(diam_used)
    binsize = int(cfg.Ndraws / 20.0)
    plt.hist(diam_used, binsize)
    plt.xlabel("Diameter (km)")
    plt.ylabel("Number of times this diameter \nwas drawn from the sample")
    plt.title(f"Sample used for Trial {trial + 1}")
    plt.tight_layout()
    plt.savefig(f"{tdir}/ObjectsDrawn_famid{cfg.famid}trial{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()

    # --- marginal DF text output ---
    with open(f"{tdir}/MarginalDF_p_beta_trial{trial + 1}.txt", "w+") as outfile:
        outfile.write("p   DF_p   Beta   DF_beta\n")
        for i in range(len(Pmargin)):
            outfile.write(
                "%1.5f  %1.5f  %1.5f  %1.5f\n" % (P[i], Pmargin[i], BETA[i], Bmargin[i])
            )


def plot_alltrials(dist: np.ndarray, ttle: str, pltname: str, outdir: str, *, show: bool = False) -> None:
    """Histogram + Gaussian fit across all trials (notebook ``plot_alltrials``)."""
    dist_nn = dist[~np.isnan(dist)]

    mu, sigma = norm.fit(dist_nn)

    plt.hist(dist_nn, 20, density=1, facecolor="green", alpha=0.75)

    x = np.linspace(min(dist_nn), max(dist_nn), 100)
    y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, "r--", linewidth=2)

    plt.xlabel(ttle)
    plt.ylabel("Probability")
    plt.title(r"$\mathrm{Peak, Width\ of\ Gaussian\ fit: }\ \mu=%.3f,\ \sigma=%.3f$" % (mu, sigma))
    plt.grid(True)
    plt.savefig(f"{outdir}/{pltname}.png", dpi=300)
    if show:
        plt.show()
    plt.close()
