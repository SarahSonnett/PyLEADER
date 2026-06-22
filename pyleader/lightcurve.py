"""Reading and phase-correcting WISE/NEOWISE ``.obs`` light-curve files.

Ported from the ``lcg_read_WISE`` and ``leader_phasecorr`` cells of the LEADER
analysis notebooks.  The notebook versions communicated through globals
(``phase_angle_limit``, ``date_tol``, ``wanted``); here those come from an
:class:`~pyleader.config.AnalysisConfig`.

Bug fixes relative to the notebooks are marked ``# FIX:`` (see the plan):
  * phase-correction returns a meaningful "combined" flag instead of the value
    that the notebook mis-unpacked as ``Nappar``;
  * apparition slices now include their last point (off-by-one);
  * forcedN subsampling is implemented cleanly.
"""

from __future__ import annotations

import random

import numpy as np

from .config import AnalysisConfig


def _normr(mat: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix (equivalent to MATLAB's ``normr``)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1  # prevent division by zero
    return mat / norms


def leader_phasecorr(Nappar, pointsperapp, dates, ang_back, L_back, cfg: AnalysisConfig):
    """Phase-correct each apparition and, if warranted, combine them.

    Returns ``(L_back, combined)`` where ``combined`` is ``True`` when the
    apparitions were merged into a single effective apparition.

    FIX (#1): the notebook returned ``single_apparition`` here and the caller
    unpacked it as ``Nappar`` (a bool), silently disabling the combine logic.
    We return an explicit ``combined`` flag and have the caller honor it.
    """
    L_back = np.asarray(L_back, dtype=float)

    # Phase-correction thresholds
    crit_change1 = 1
    crit_change2 = 2
    crit_low = 8

    for i in range(Nappar):
        ind = int(np.sum(pointsperapp[:i]))
        inde = int(ind + pointsperapp[i])  # FIX (#2): include the last point
        temp = ang_back[ind:inde]

        if len(temp) > 1:
            expcorr = (np.min(temp) < crit_low) and (np.max(temp) - np.min(temp) > crit_change1)
            lincorr = (np.min(temp) >= crit_low) and (np.max(temp) - np.min(temp) > crit_change2)

            if expcorr:
                # Phase correction with exponential fit
                AA = np.vstack([np.ones_like(temp), temp]).T
                yy = np.log(L_back[ind:inde])
                xx, _, _, _ = np.linalg.lstsq(AA, yy, rcond=None)

                if xx[1] > 0:
                    XX0, XX1 = 1.0, 0.0
                else:
                    XX0, XX1 = np.exp(xx[0]), xx[1]

                L_back[ind:inde] = L_back[ind:inde] / (XX0 * np.exp(XX1 * temp))

            elif lincorr:
                # Phase correction with linear fit
                AA = np.vstack([temp, np.ones_like(temp)]).T
                yy = L_back[ind:inde]
                xx, _, _, _ = np.linalg.lstsq(AA, yy, rcond=None)

                if xx[0] > 0 or xx[1] < 0:
                    XX = [0.0, 1.0]
                else:
                    XX = xx

                L_back[ind:inde] = L_back[ind:inde] / (XX[0] * temp + XX[1])

    # Determine if all measurements span a single apparition window.
    single_apparition = False
    if len(dates) > 0:
        single_apparition = (dates[-1] - dates[0]) <= cfg.date_tol

    # Combine sets into a single apparition if some are undersized.
    combined = False
    if Nappar > 1 and np.any(np.asarray(pointsperapp) < cfg.wanted) and single_apparition:
        # Scale every set to match the first set's mean intensity.
        avg_scale = np.mean(L_back[: pointsperapp[0]])
        for i in range(1, Nappar):
            ind = int(np.sum(pointsperapp[:i]))
            inde = int(ind + pointsperapp[i])  # FIX (#2): include the last point
            avg_temp = np.mean(L_back[ind:inde])
            scale_factor = avg_scale / avg_temp
            L_back[ind:inde] = scale_factor * L_back[ind:inde]
        combined = True

    return L_back, combined


def lcg_read_WISE(fname: str, cfg: AnalysisConfig):
    """Read a single ``.obs`` file and return its amplitude statistics.

    Returns ``(Npoints_avg, Nappar, A)``: the mean number of points per usable
    apparition, the number of apparitions found, and the array of derived
    amplitudes ``A`` (one per apparition with at least ``cfg.wanted`` points).
    """
    with open(fname, "r") as fid:
        lines = [line.rstrip() for line in fid]

    nblocks = int(lines[0])

    dates = np.zeros(nblocks)
    e_sun = np.zeros((nblocks, 3))
    e_earth = np.zeros((nblocks, 3))
    flux = [[None for _ in range(4)] for _ in range(nblocks)]
    fluxerr = [[None for _ in range(4)] for _ in range(nblocks)]

    ilinestart = 1
    for i in range(nblocks):
        if lines[ilinestart]:
            dates[i] = float(lines[ilinestart].split()[0])
            nfilters = int(lines[ilinestart].split()[1])
            e_sun[i, :] = lines[ilinestart + 1].split()
            e_earth[i, :] = lines[ilinestart + 2].split()
            for j in range(nfilters):
                filter_index = int(lines[ilinestart + 3 + j].split()[3])
                flux[i][filter_index] = float(lines[ilinestart + 3 + j].split()[1])
                fluxerr[i][filter_index] = float(lines[ilinestart + 3 + j].split()[2])
            ilinestart += 5 + nfilters
        else:
            ilinestart += 1

    # Construct per-filter flux vectors
    flux1, flux2, flux3, flux4 = [], [], [], []
    flux1e, flux2e, flux3e, flux4e = [], [], [], []
    for i in range(nblocks):
        if flux[i][0] is not None:
            flux1.append(flux[i][0])
            flux1e.append(fluxerr[i][0])
        if flux[i][1] is not None:
            flux2.append(flux[i][1])
            flux2e.append(fluxerr[i][1])
        if flux[i][2] is not None:
            flux3.append(flux[i][2])
            flux3e.append(fluxerr[i][2])
        if flux[i][3] is not None:
            flux4.append(flux[i][3])
            flux4e.append(fluxerr[i][3])

    # Total error per filter; discard empty filters
    flux_tot_err = [
        np.linalg.norm(flux1e) if flux1e else 0,
        np.linalg.norm(flux2e) if flux2e else 0,
        np.linalg.norm(flux3e) if flux3e else 0,
        np.linalg.norm(flux4e) if flux4e else 0,
    ]
    flux_tot_err = [err if err != 0 else np.inf for err in flux_tot_err]

    # Best (lowest total error) filter
    bestf = flux_tot_err.index(min(flux_tot_err))

    # Intensity series from the best filter
    L_big = np.asarray([flux[i][bestf] for i in range(nblocks) if flux[i][bestf] is not None])
    indeksit = np.asarray([i for i in range(nblocks) if flux[i][bestf] is not None])

    e_sun = e_sun[indeksit, :]
    e_earth = e_earth[indeksit, :]
    dates = dates[indeksit]

    e_sun = _normr(e_sun)
    e_earth = _normr(e_earth)

    # Remove measurements with solar phase angle > limit
    ang_tol = np.deg2rad(cfg.phase_angle_limit)
    largeanglepoints = np.zeros(len(L_big), dtype=bool)
    ang = np.zeros(len(L_big))
    for i in range(len(L_big)):
        ang[i] = np.arccos(np.clip(np.dot(e_sun[i, :], e_earth[i, :]), -1.0, 1.0))
        if ang[i] > ang_tol:
            largeanglepoints[i] = True

    dates = dates[~largeanglepoints]
    L_big = L_big[~largeanglepoints]
    e_sun = e_sun[~largeanglepoints]
    e_earth = e_earth[~largeanglepoints]
    ang = ang[~largeanglepoints]

    # Split the series into apparitions (epochs)
    dates_back, ang_back, temp_angle, L_back, pointsperapp = [], [], [], [], []
    Nappar = 0
    i = 0
    while i < len(L_big) - 1:
        L = [L_big[i]]
        for j in range(i + 1, len(L_big)):
            if dates[j] - dates[i] <= cfg.date_tol:
                L.append(L_big[j])
                if j == len(L_big) - 1:
                    i_old = i
                    i = j
            else:
                i_old = i
                i = j
                break

        dates_back.extend(dates[i_old:i_old + len(L)])
        ang_back.extend(ang[i_old:i_old + len(L)])
        L_back.extend(L)
        pointsperapp.append(len(L))
        Nappar += 1

        temp = ang[i_old:i_old + len(L)]
        temp_angle.append(np.max(temp) - np.min(temp))

    if len(dates_back) > 0:
        dates_back = dates_back - dates_back[0]
        ang_back = np.rad2deg(ang_back)

    combined = False
    if len(L_back) > 0:
        L_back, combined = leader_phasecorr(Nappar, pointsperapp, dates, ang_back, L_back, cfg)

    # Compute amplitude A for each usable apparition.
    # FIX (#1): honor the `combined` flag from phase correction; FIX (#2): the
    # apparition slices below include their last point.
    if combined:
        sets = [np.asarray(L_back, dtype=float)]
    else:
        sets = []
        for i in range(Nappar):
            ind = int(np.sum(pointsperapp[:i]))
            inde = int(ind + pointsperapp[i])
            sets.append(np.asarray(L_back[ind:inde], dtype=float))

    A = []
    Npoints = []
    for L in sets:
        # FIX (#3): forcedN — subsample to exactly `wanted` points before
        # computing eta (the notebook's main-loop subsampling referenced
        # undefined variables; this implements the intent cleanly).
        if cfg.forced_n and len(L) >= cfg.wanted:
            iselect = random.sample(range(len(L)), cfg.wanted)
            L = L[iselect]

        if len(L) >= cfg.wanted:
            Npoints.append(len(L))
            L2 = np.asarray(L) ** 2.0
            eta = np.std(L2) / np.mean(L2)
            A_val = np.sqrt(1 - 1 / ((1 / (np.sqrt(8) * eta)) + 0.5))
            A.append(A_val)

    A = np.array(A)
    Npoints = np.asarray(Npoints)
    Npoints_avg = float(np.mean(Npoints)) if len(Npoints) > 0 else 0

    # Remove complex / non-finite amplitudes
    A = A[np.isreal(A)]
    A = A[np.isfinite(A)]

    return Npoints_avg, Nappar, A
