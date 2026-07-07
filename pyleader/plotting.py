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

import glob
import os

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


def _population_counts(outdir: str, famid: str):
    """Median (N_obj, AvgPointsPerObj) across trials, from the summary file."""
    summary = f"{outdir}/SummaryAnalysis_Famid{famid}.txt"
    try:
        nobj, npoints = np.genfromtxt(summary, unpack=True, usecols=(4, 5), dtype=float, skip_header=1)
        return float(np.median(np.atleast_1d(nobj))), float(np.median(np.atleast_1d(npoints)))
    except (OSError, ValueError):
        return 0.0, 0.0


def plot_population_df(outdir: str, cfg: AnalysisConfig, *, show: bool = False) -> bool:
    """Population-level marginal DFs of ``p`` and ``β``, aggregated over all trials.

    Reproduces the ``DF_p_all.png`` / ``DF_b_all.png`` (+ ``.txt``) figures from the
    post-analysis ``Analyze_LEADER_results`` notebook: every trial's marginal DF is
    overplotted faintly, with the across-trial median ± 1σ drawn on top. Reads the
    per-trial ``MarginalDF_p_beta_trial*.txt`` files written by :func:`leader_plots`.
    Returns ``True`` if figures were produced.
    """
    files = sorted(glob.glob(os.path.join(outdir, "Trial*", "MarginalDF_p_beta_trial*.txt")))
    if not files:
        return False

    P, DFP, B, DFB = [], [], [], []
    for f in files:
        pf, dfpf, bf, dfbf = np.genfromtxt(f, unpack=True, skip_header=1, dtype=float)
        P.append(pf); DFP.append(dfpf); B.append(bf); DFB.append(dfbf)
    P, DFP, B, DFB = np.array(P), np.array(DFP), np.array(B), np.array(DFB)

    nobj, npoints = _population_counts(outdir, cfg.famid)
    title = (f"{cfg.famid}, {int(cfg.diam_low)}" + r"$\leq$" + " D (km) < "
             + f"{int(cfg.diam_high)},\n{int(nobj)} objects, "
             + f"{round(npoints, 2)} data points per object")

    for quant, grid, dfs, faint, medc, medfmt, xlabel, png, txt in (
        ("p", P, DFP, "0.65", "k", "-", "b:a axis ratio", "DF_p_all", "DF_p_all.txt"),
        ("b", B, DFB, "lightskyblue", "b", "-.", "Spin pole polar angle (degrees)", "DF_b_all", "DF_b_all.txt"),
    ):
        med = np.median(grid, axis=0)
        med_df = np.median(dfs, axis=0)
        err = np.std(dfs, axis=0)

        plt.figure()
        for i in range(len(grid)):
            plt.plot(grid[i], dfs[i], color=faint, linestyle="-", lw=0.2,
                     alpha=(0.9 if quant == "b" else 1.0))
        plt.errorbar(med, med_df, yerr=err, color=medc, fmt=medfmt, capsize=0)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density function")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{png}.png", dpi=300)
        if show:
            plt.show()
        plt.close()

        with open(f"{outdir}/{txt}", "w+") as outfile:
            outfile.write(f"{int(nobj)} {round(npoints, 2)}\n")
            for i in range(len(med)):
                outfile.write("%1.2f  %1.3f  %1.3f\n" % (med[i], med_df[i], err[i]))

    return True
