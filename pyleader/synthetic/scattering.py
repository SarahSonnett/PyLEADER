"""Surface scattering laws for the synthetic brightness model.

``ls_lambert`` is the combined Lommel-Seeliger + Lambert law actually used by
the MATLAB ``leader_brightness_synth_WISE.m``. ``hapke_bright`` ports the (there
commented-out) Hapke law and is available as an optional model.
"""

from __future__ import annotations

import numpy as np


def ls_lambert(mu: np.ndarray, mu0: np.ndarray) -> np.ndarray:
    """Combined Lommel-Seeliger + Lambert single-particle scattering per facet.

    ``L = mu*mu0/(mu+mu0) + 0.1*mu*mu0`` (the Lambert term weighted by 0.1),
    evaluated elementwise. The ``mu+mu0 == 0`` case returns 0 for that facet.
    """
    mu = np.asarray(mu, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    denom = mu + mu0
    ls = np.where(denom != 0, mu * mu0 / np.where(denom != 0, denom, 1.0), 0.0)
    return ls + 0.1 * mu * mu0


def hapke_bright(E, E0, mu, mu0, p, th):
    """Hapke scattering brightness for a single facet.

    Port of ``hapke_bright.m`` (Viikinkoski & Kaasalainen). ``E``/``E0`` are the
    (body-frame) view and Sun unit directions; ``mu``/``mu0`` the facet
    emission/incidence cosines; ``p = (albedo, h, S0, g)``; ``th`` the roughness
    angle in degrees.
    """
    tth = np.tan(np.pi / 180.0 * th)
    cth = 1.0 / np.sqrt(1.0 + np.pi * tth)
    cal = float(np.dot(E, E0))
    alpha = np.arccos(cal)

    sh, mueiefi, mu0eiefi = _shadow(mu, mu0, tth, cal)
    mu0new = cth * mu0eiefi
    munew = cth * mueiefi
    dnom = munew + mu0new
    sls = mu * mu0new / dnom
    fh = _hapke(p, munew, mu0new, alpha)
    return fh * sls * sh


def _hapke(p, mu, mu0, alpha):
    alb, h, S0, g = p
    ta = np.tan(0.5 * alpha)
    ca = np.cos(alpha)
    B0 = S0 * (1 + g) ** 2 / (alb * (1 - g))
    B = B0 / (1 + ta / h)                       # opposition surge
    fhg = (1 - g ** 2) / ((1 + g ** 2 + 2 * g * ca) ** 1.5)  # particle phase function
    sqalb = np.sqrt(1 - alb)
    chmu = (1 + 2 * mu) / (1 + 2 * mu * sqalb)   # H(alb, mu)
    chmu0 = (1 + 2 * mu0) / (1 + 2 * mu0 * sqalb)  # H(alb, mu0)
    fm = chmu * chmu0 - 1                         # multiple scattering
    return (1 + B) * fhg + fm


def _shadow(mu, mu0, tth, cal):
    inc = np.arccos(mu0)
    em = np.arccos(mu)
    if abs(inc) < 1e-6:
        inc = 1e-6
    if abs(em) < 1e-6:
        em = 1e-6

    innerfi = (cal - mu * mu0) / (np.sin(inc) * np.sin(em))
    fi = np.arccos(np.clip(innerfi, -1.0, 1.0))
    f = np.exp(-2 * np.tan(0.5 * fi))
    E1e = np.exp(-2 / (tth * np.tan(em) * np.pi))
    E1i = np.exp(-2 / (tth * np.tan(inc) * np.pi))
    E2e = np.exp(-1 / (tth ** 2 * np.tan(em) ** 2 * np.pi))
    E2i = np.exp(-1 / (tth ** 2 * np.tan(inc) ** 2 * np.pi))

    if inc < em:
        top = np.sin(0.5 * fi) ** 2 * E2i
        bot = 2 - E1e - (fi / np.pi) * E1i
        mu0ei0pi = mu0 + np.sin(inc) * tth * E2i / (2 - E1i)
        mue0e0 = mu + np.sin(em) * tth * E2e / (2 - E1e)
        mueiefi = mu + np.sin(em) * tth * (E2e - top) / bot
        coef = mu0 / mu0ei0pi
        sh = mueiefi * coef / (mue0e0 * (1 - f * (1 - coef)))
        mu0eiefi = mu0 + np.sin(inc) * tth * (np.cos(fi) * E2e + top) / bot
    else:
        top = np.sin(0.5 * fi) ** 2 * E2e
        bot = 2 - E1i - (fi / np.pi) * E1e
        mu0ei00 = mu0 + np.sin(inc) * tth * E2i / (2 - E1i)
        mue0epi = mu + np.sin(em) * tth * E2e / (2 - E1e)
        mueiefi = mu + np.sin(em) * tth * (np.cos(fi) * E2i + top) / bot
        coef = mu0 / mu0ei00
        sh = mueiefi * coef / (mue0epi * (1 - f * (1 - coef)))
        mu0eiefi = mu0 + np.sin(inc) * tth * (E2i - top) / bot

    return sh, mueiefi, mu0eiefi
