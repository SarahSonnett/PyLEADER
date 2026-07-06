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
from .obsio import read_obs


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
    obs = read_obs(fname)
    nblocks = obs.n
    dates = obs.dates.copy()
    e_sun = obs.e_sun.copy()
    e_earth = obs.e_earth.copy()
    flux = obs.flux
    fluxerr = obs.fluxerr

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

    # Split the series into apparitions, phase-correct, and reduce to amplitudes.
    # (Shared with the synthetic pipeline via the helpers below.)
    L_back, pointsperapp, ang_back, dates_back, Nappar = split_apparitions(
        L_big, dates, ang, cfg.date_tol
    )

    combined = False
    if len(L_back) > 0:
        L_back, combined = leader_phasecorr(Nappar, pointsperapp, dates, ang_back, L_back, cfg)

    sets = apparition_sets(L_back, pointsperapp, Nappar, combined)
    A, Npoints = amplitudes_from_sets(sets, cfg.wanted, forced_n=cfg.forced_n)

    Npoints_avg = float(np.mean(Npoints)) if len(Npoints) > 0 else 0

    return Npoints_avg, Nappar, A


def split_apparitions(L_big, dates, ang, date_tol):
    """Group a phase-angle-filtered intensity series into apparitions (epochs).

    Two consecutive points belong to the same apparition while their date gap
    (measured from the apparition's first point) stays within ``date_tol``.
    Returns ``(L_back, pointsperapp, ang_back_deg, dates_back, Nappar)`` where
    ``ang_back`` is converted to degrees and ``dates_back`` is offset to start
    at zero, matching the original ``lcg_read_WISE`` bookkeeping.
    """
    dates_back, ang_back, L_back, pointsperapp = [], [], [], []
    Nappar = 0
    i = 0
    while i < len(L_big) - 1:
        L = [L_big[i]]
        for j in range(i + 1, len(L_big)):
            if dates[j] - dates[i] <= date_tol:
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

    if len(dates_back) > 0:
        dates_back = dates_back - dates_back[0]
        ang_back = np.rad2deg(ang_back)

    return L_back, pointsperapp, ang_back, dates_back, Nappar


def apparition_sets(L_back, pointsperapp, Nappar, combined):
    """Slice ``L_back`` into per-apparition intensity arrays.

    FIX (#1): honors the ``combined`` flag from phase correction (the notebook
    mis-unpacked it). FIX (#2): slices include their last point (off-by-one).
    """
    if combined:
        return [np.asarray(L_back, dtype=float)]
    sets = []
    for i in range(Nappar):
        ind = int(np.sum(pointsperapp[:i]))
        inde = int(ind + pointsperapp[i])
        sets.append(np.asarray(L_back[ind:inde], dtype=float))
    return sets


def amplitudes_from_sets(sets, wanted, *, forced_n=False):
    """Reduce per-apparition intensities to LEADER amplitudes ``A``.

    For each apparition with at least ``wanted`` points, compute the brightness
    dispersion ``eta = std(L^2)/mean(L^2)`` and the amplitude
    ``A = sqrt(1 - 1/(1/(sqrt(8) eta) + 1/2))``. Returns ``(A, Npoints)`` with
    complex/non-finite amplitudes removed.
    """
    A = []
    Npoints = []
    for L in sets:
        # FIX (#3): forcedN — subsample to exactly `wanted` points before eta.
        if forced_n and len(L) >= wanted:
            iselect = random.sample(range(len(L)), wanted)
            L = L[iselect]

        if len(L) >= wanted:
            Npoints.append(len(L))
            L2 = np.asarray(L) ** 2.0
            eta = np.std(L2) / np.mean(L2)
            # Argument can go negative for near-spherical/low-variation cases;
            # the resulting nan is dropped below, so silence the sqrt warning.
            with np.errstate(invalid="ignore"):
                A_val = np.sqrt(1 - 1 / ((1 / (np.sqrt(8) * eta)) + 0.5))
            A.append(A_val)

    A = np.array(A)
    Npoints = np.asarray(Npoints)

    # Remove complex / non-finite amplitudes
    A = A[np.isreal(A)]
    A = A[np.isfinite(A)]
    return A, Npoints
