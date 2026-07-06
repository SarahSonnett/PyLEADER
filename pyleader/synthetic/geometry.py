"""Observing geometry for synthetic light curves.

Ports ``transform_mat.m`` (inertial -> body rotation matrix) and
``lcg_read_synth_WISE.m`` (read a real WISE ``.obs`` file for its Sun/observer
geometry, which is borrowed to illuminate the synthetic shape).
"""

from __future__ import annotations

import numpy as np

from ..obsio import read_obs


def transform_mat(phi0, omega, t, t0, beta, lam):
    """Rotation matrix from the inertial frame to the asteroid body frame.

    Recommended inputs: ``phi0 = 0``, ``omega = 2*pi/period``, ``t0 = 0``.
    ``beta`` is the spin latitude and ``lam`` the pole longitude.
    """
    arg1 = phi0 + omega * (t - t0)
    RZ1 = np.array([[np.cos(arg1), np.sin(arg1), 0.0],
                    [-np.sin(arg1), np.cos(arg1), 0.0],
                    [0.0, 0.0, 1.0]])
    RY1 = np.array([[np.cos(beta), 0.0, -np.sin(beta)],
                    [0.0, 1.0, 0.0],
                    [np.sin(beta), 0.0, np.cos(beta)]])
    RZ2 = np.array([[np.cos(lam), np.sin(lam), 0.0],
                    [-np.sin(lam), np.cos(lam), 0.0],
                    [0.0, 0.0, 1.0]])
    return RZ1 @ RY1 @ RZ2


def _normr(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


def read_synth_geometry(path: str, phase_angle_limit: float = 30.0):
    """Read a WISE ``.obs`` file for its observing geometry.

    Returns ``(dates, e_sun, e_earth, ang)`` after selecting the best filter,
    normalizing the Sun/observer directions, and dropping epochs with solar
    phase angle above ``phase_angle_limit`` (degrees). Only the geometry is
    used downstream; the synthetic brightness is computed from the shape model.
    """
    obs = read_obs(path)
    nblocks = obs.n
    dates = obs.dates
    e_sun = obs.e_sun
    e_earth = obs.e_earth
    flux, fluxerr = obs.flux, obs.fluxerr

    # per-filter error to choose the best filter
    errs = []
    for k in range(4):
        e = [fluxerr[i][k] for i in range(nblocks) if fluxerr[i][k] is not None]
        errs.append(np.linalg.norm(e) if e else 0)
    errs = [e if e != 0 else np.inf for e in errs]
    bestf = errs.index(min(errs))

    keep = [i for i in range(nblocks) if flux[i][bestf] is not None]
    keep = np.asarray(keep, dtype=int)
    L_big = np.asarray([flux[i][bestf] for i in keep])
    e_sun = _normr(e_sun[keep, :])
    e_earth = _normr(e_earth[keep, :])
    dates = dates[keep]

    # drop large phase angles
    ang_tol = np.deg2rad(phase_angle_limit)
    ang = np.arccos(np.clip(np.sum(e_sun * e_earth, axis=1), -1.0, 1.0))
    good = ang <= ang_tol

    return dates[good], e_sun[good], e_earth[good], ang[good]
