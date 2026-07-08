"""Main LEADER analysis driver.

This is the ``MAIN PART OF THE CODE`` cell of the analysis notebooks, refactored
into a single function.  The three notebook variants (``final``, ``final_bg``,
``forcedN``) are all expressed through :class:`~pyleader.config.AnalysisConfig`
flags, so this one driver reproduces all of them.
"""

from __future__ import annotations

import datetime
import os
import random
import shutil

import numpy as np

from .config import AnalysisConfig, require_neowise
from .inversion import leader_invert
from .lightcurve import lcg_read_WISE
from .naming import convert_to_mpecname
from .plotting import leader_plots, plot_alltrials, plot_population_df
from .postprocess import leader_postprocess_WISE


def _load_neowise_diameters(cfg: AnalysisConfig):
    """Return ``(name_mpced_n, diam_n)`` from the NEOWISE catalog."""
    require_neowise(cfg.neowise_path)
    name_mpced_n_full = np.genfromtxt(cfg.neowise_path, unpack=True, usecols=(2), delimiter=",", dtype=str)
    name_mpced_n = np.asarray(
        [name_mpced_n_full[i][1:-1].replace(" ", "") for i in range(len(name_mpced_n_full))]
    )
    diam_n = np.genfromtxt(cfg.neowise_path, unpack=True, usecols=(11), delimiter=",", dtype=float)
    return name_mpced_n, diam_n


def diameter_matched_files(cfg: AnalysisConfig, name_mpced_n=None, diam_n=None):
    """List the population's ``.obs`` files whose NEOWISE diameter is in range.

    Shared by :func:`run_analysis` and the per-population pipeline (so the
    synthetic correction samples geometry from exactly the analyzed objects).
    """
    if name_mpced_n is None or diam_n is None:
        name_mpced_n, diam_n = _load_neowise_diameters(cfg)

    lcg_files = [os.path.join(cfg.datadir, f) for f in os.listdir(cfg.datadir)
                 if f.endswith(".obs") and not f.startswith("Nofilter")]

    matched = []
    for path in lcg_files:
        objname = path.split("/")[-1].split(".")[0]
        objname_mpec = convert_to_mpecname(objname)
        diammatch = np.mean(np.asarray(diam_n.compress((name_mpced_n == objname_mpec).flat), dtype=float))
        if cfg.diam_low <= diammatch <= cfg.diam_high:
            matched.append(path)
    return matched


def run_analysis(cfg: AnalysisConfig, *, seed: int | None = None, show: bool = False) -> str:
    """Run the full LEADER inversion experiment and return the output directory.

    Parameters
    ----------
    cfg:
        Run configuration (sample selection, statistics, variant flags, paths).
    seed:
        Optional RNG seed for reproducible draws (notebooks were unseeded).
    show:
        If ``True``, display plots interactively in addition to saving them.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    datadir = cfg.datadir
    outdir = cfg.outdir
    famid = cfg.famid

    name_mpced_n, diam_n = _load_neowise_diameters(cfg)
    lcg_files_diammatch = diameter_matched_files(cfg, name_mpced_n, diam_n)

    # --- prepare output directory ---
    try:
        os.mkdir(outdir)
    except OSError:
        shutil.rmtree(outdir)
        os.mkdir(outdir)

    summary_path = f"{outdir}/{cfg.summary_name}"
    if cfg.overwrite:
        wfile = open(summary_path, "w+")
        wfile.write(
            "Trial  Pmax    Betamax    Relerr  N_obj  AvgPointsPerObj  N_apparitions_total\n"
        )
        wfile.flush()
    else:
        wfile = open(summary_path, "a")

    Ndraws = cfg.Ndraws

    # Full run log in the output directory; the terminal shows a progress bar only.
    log = open(f"{outdir}/analysis.log", "w")
    log.write(f"# LEADER analysis — {famid} ({cfg.population_kind})\n")
    log.write(f"# started {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    log.write(f"# cat={cfg.cat} filter={cfg.filterpriority} "
              f"diam=[{cfg.diam_low},{cfg.diam_high}] km Ntrials={cfg.Ntrials} "
              f"Ndraws={Ndraws} seed={seed}\n")
    log.write(f"# datadir={datadir}\n# outdir={outdir}\n\n")
    log.write("Trial  Pmax      Betamax(deg)  Relerr    N_obj\n")
    log.flush()

    for trial in range(cfg.Ntrials):
        print(f"\r[{famid}] LEADER analysis: trial {trial + 1}/{cfg.Ntrials} "
              f"({100.0 * (trial + 1) / cfg.Ntrials:4.0f}%)", end="", flush=True)

        trialdir = f"{outdir}/Trial{trial + 1}"
        if cfg.overwrite:
            try:
                os.mkdir(trialdir)
            except OSError:
                shutil.rmtree(trialdir)
                os.mkdir(trialdir)
        else:
            try:
                os.mkdir(trialdir)
            except OSError:
                continue

        A_tot = []
        Npoints_avg = []
        Napparlist = np.zeros((Ndraws, 1))
        Objects_drawn = []

        objfile = open(f"{trialdir}/ObjectsDrawn_famid{famid}trial{trial + 1}.txt", "w+")
        objfile.write("Filepath   Diameter \n")
        objfile.flush()

        for draw in range(Ndraws):
            # pick an object whose mean NEOWISE diameter is within bounds
            fname = random.choice(lcg_files_diammatch)
            objname = fname.split("/")[-1].split(".")[0]
            objname_mpec = convert_to_mpecname(objname)
            diammatch = np.mean(
                np.asarray(diam_n.compress((name_mpced_n == objname_mpec).flat), dtype=float)
            )

            Nppo, Nappar, A = lcg_read_WISE(fname, cfg)

            Napparlist[draw] = Nappar
            Npoints_avg.append(Nppo)
            A_tot += list(A)

            Objects_drawn.append(fname.split("/")[-1].split(".")[0])
            objfile.write(fname.split(".")[0] + "  " + str(int(Nppo)) + "  " + str(round(diammatch, 2)) + "\n")
            objfile.flush()

        objfile.close()

        A_tot = np.asarray(A_tot)
        Npoints_avg = np.asarray(Npoints_avg)
        Asort = np.sort(A_tot)
        CDFA = np.linspace(1 / len(Asort), 1, len(Asort))

        result = leader_invert(Asort, CDFA, verbose=False)

        Nobjs = len(set(Objects_drawn))

        # NOTE: betamax is written in radians*(180/pi) BEFORE the (optional)
        # in-place degree conversion below, matching the notebook ordering.
        wfile.write(
            "%s  %1.5f  %2.5f  %1.4f  %5i  %3i  %5i\n"
            % (
                trial + 1,
                result.pmax,
                result.betamax * (180.0 / np.pi),
                result.relerr,
                Nobjs,
                np.median(Npoints_avg),
                np.sum(Napparlist),
            )
        )
        wfile.flush()
        log.write("%5s  %.5f  %10.4f  %.4f  %5i\n"
                  % (trial + 1, result.pmax, result.betamax * (180.0 / np.pi),
                     result.relerr, Nobjs))
        log.flush()

        if cfg.convert2degrees:
            result.BETA = np.rad2deg(result.BETA)
            result.BETA_Gr = np.rad2deg(result.BETA_Gr)
            result.betamax = np.rad2deg(result.betamax)

        leader_plots(result, cfg, outdir, trial, show=show)
        leader_postprocess_WISE(result, outdir, trial, show=show, verbose=False)

    wfile.close()
    print()  # finish the progress-bar line
    log.write(f"\n# completed {cfg.Ntrials} trials at "
              f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
    log.close()

    # --- summary plots over all trials ---
    trials, pmax_all, betamax_all, relerr_all = np.genfromtxt(
        summary_path, unpack=True, dtype=float, usecols=(0, 1, 2, 3)
    )
    plot_alltrials(betamax_all, "Peak of " + r"$\beta$" + " distribution",
                   f"Summary_betamax_Famid{famid}_{cfg.diam_tag}", outdir, show=show)
    plot_alltrials(pmax_all, "Peak of p distribution",
                   f"Summary_pmax_Famid{famid}_{cfg.diam_tag}", outdir, show=show)

    # population-level marginal DFs of p and beta (DF_p_all/DF_b_all .png + .txt)
    plot_population_df(outdir, cfg, show=show)

    return outdir
