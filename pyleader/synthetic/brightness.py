"""Synthetic brightness -> amplitude for one asteroid.

Ports ``leader_brightness_synth_WISE.m``. Given a shape model's facet normals
and areas plus a borrowed real observing geometry, it renders a synthetic
brightness series under an assigned spin, then reduces it to LEADER amplitudes
``A`` using the same apparition-splitting / phase-correction / eta machinery as
the real-data reader.
"""

from __future__ import annotations

import numpy as np

from ..lightcurve import (
    apparition_sets,
    amplitudes_from_sets,
    leader_phasecorr,
    split_apparitions,
)
from .geometry import transform_mat
from .scattering import hapke_bright, ls_lambert


def synthetic_amplitudes(normals, areas, dates, e_sun, e_earth, ang, beta, cfg):
    """Render a synthetic light curve for an assigned spin and reduce it to ``A``.

    Nuisance parameters (pole longitude ``lambda`` and rotation period ``Trot``)
    are drawn internally as in the MATLAB code; the assigned spin latitude
    ``beta`` is passed in (and recorded by the caller). Returns ``(A, Nappar)``.
    """
    normals = np.asarray(normals, dtype=float)
    areas = np.asarray(areas, dtype=float)

    lam = 2 * np.pi * np.random.rand()
    # rotation period, hours -> days
    trot = (cfg.trot_min_hr + (cfg.trot_max_hr - cfg.trot_min_hr) * np.random.rand()) / 24.0
    omega_spin = 2 * np.pi / trot

    L = np.zeros(len(dates))
    for i in range(len(dates)):
        ROTM = transform_mat(0.0, omega_spin, dates[i], 0.0, beta, lam)
        omega = ROTM @ e_earth[i, :]
        omega0 = ROTM @ e_sun[i, :]

        mu = normals @ omega
        mu0 = normals @ omega0
        visible = (mu > 0) & (mu0 > 0)

        if cfg.scattering == "hapke":
            total = 0.0
            idx = np.where(visible)[0]
            for j in idx:
                val = hapke_bright(omega, omega0, mu[j], mu0[j], cfg.hapke_param, cfg.hapke_rough)
                if np.isfinite(val):
                    total += areas[j] * val
            L[i] = total
        else:  # ls_lambert (default)
            L[i] = np.sum(visible * areas * ls_lambert(mu, mu0))

    # Add fractional Gaussian noise
    L = L + cfg.noise_level * np.mean(L) * np.random.randn(len(L))

    # Reduce to amplitudes via the shared machinery
    L_back, pointsperapp, ang_back, _dates_back, Nappar = split_apparitions(
        L, dates, ang, cfg.date_tol
    )
    combined = False
    if len(L_back) > 0:
        L_back, combined = leader_phasecorr(Nappar, pointsperapp, dates, ang_back, L_back, cfg)
    sets = apparition_sets(L_back, pointsperapp, Nappar, combined)
    A, _ = amplitudes_from_sets(sets, cfg.wanted, forced_n=False)

    return A, Nappar
