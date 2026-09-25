"""Empirical photometric-noise model for synthetic populations.

The original LEADER release adds flat 1% Gaussian noise to every synthetic
brightness — but real NEOWISE photometry is strongly heteroscedastic: the
relative uncertainty grows toward faint fluxes, and many objects are far from
1% photometry. This module fits, **once per population**, a polynomial to the
flux–uncertainty relation measured in the population's own ``.obs`` files:

    log10(sigma_F / F) = c_0 + c_1 * log10(F) + ... + c_deg * log10(F)^deg

The fit is documented (JSON coefficients + a diagnostic figure) and then
evaluated per epoch on the synthetic fluxes, so fainter objects — and fainter
rotational phases of one object — receive proportionally larger noise.

Anchoring model units to physical flux: each synthetic object borrows a real
object's observing geometry, so its model brightness is scaled to that object's
mean measured flux before the relation is evaluated (see
``brightness.synthetic_amplitudes``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..obsio import read_obs
from .geometry import select_best_filter


@dataclass
class NoiseModel:
    """Fitted flux -> relative-uncertainty relation (log10–log10 polynomial).

    ``coeffs`` are ``np.polyval`` coefficients (highest power first) of
    ``log10(relerr)`` as a function of ``log10(flux)`` — they describe the
    **catalog** fluxerr relation. ``white_fraction`` is the measured fraction
    of that budget that behaves as *independent per-epoch* (white) noise: the
    catalog fluxerr is a total error budget including calibration terms that
    do not fluctuate point-to-point, and only the white part belongs in the
    synthetic scatter (see :func:`measure_white_fraction`). ``relerr()``
    returns the calibrated (scaled) value. Evaluation clips ``log10(flux)``
    to the fitted domain (no extrapolation) and caps the result at
    ``relerr_cap``.
    """

    coeffs: tuple                 # polynomial coefficients, highest power first
    log10_flux_domain: tuple      # (lo, hi) fitted domain of log10(flux)
    scatter_dex: float            # rms residual of the fit, dex
    npoints: int                  # measurements used in the fit
    nfiles: int                   # .obs files contributing
    relerr_cap: float = 1.0       # ceiling on the evaluated relative error
    white_fraction: float = 1.0   # effective white-noise fraction of the catalog fluxerr

    def relerr(self, flux) -> np.ndarray:
        """Effective white-noise sigma_F/F at the given flux(es) (calibrated)."""
        f = np.maximum(np.asarray(flux, dtype=float), 1e-30)
        lf = np.clip(np.log10(f), *self.log10_flux_domain)
        cat = 10.0 ** np.polyval(self.coeffs, lf)
        return np.minimum(cat * self.white_fraction, self.relerr_cap)

    def catalog_relerr(self, flux) -> np.ndarray:
        """The unscaled catalog fluxerr/flux relation (for plots/diagnostics)."""
        f = np.maximum(np.asarray(flux, dtype=float), 1e-30)
        lf = np.clip(np.log10(f), *self.log10_flux_domain)
        return np.minimum(10.0 ** np.polyval(self.coeffs, lf), self.relerr_cap)

    def to_dict(self) -> dict:
        return dict(coeffs=list(self.coeffs), log10_flux_domain=list(self.log10_flux_domain),
                    scatter_dex=self.scatter_dex, npoints=self.npoints,
                    nfiles=self.nfiles, relerr_cap=self.relerr_cap,
                    white_fraction=self.white_fraction,
                    form="sigma_F/F = white_fraction * 10**polyval(coeffs, log10(flux))")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NoiseModel":
        with open(path) as f:
            d = json.load(f)
        return cls(coeffs=tuple(d["coeffs"]),
                   log10_flux_domain=tuple(d["log10_flux_domain"]),
                   scatter_dex=float(d["scatter_dex"]), npoints=int(d["npoints"]),
                   nfiles=int(d["nfiles"]), relerr_cap=float(d.get("relerr_cap", 1.0)),
                   white_fraction=float(d.get("white_fraction", 1.0)))


def _collect_pairs(obs_files, relerr_max: float = 1.0):
    """(flux, relerr) measurements from every file's best filter.

    Uses the same best-filter rule as the synthetic geometry reader, so the
    noise model describes the same measurements the synthetic objects mimic.
    Points with non-positive flux/err or relerr >= ``relerr_max`` (SNR < 1;
    junk) are dropped.
    """
    flux_all, rel_all = [], []
    nfiles = 0
    for path in obs_files:
        try:
            obs = read_obs(path)
        except (OSError, ValueError):
            continue
        k = select_best_filter(obs)
        got = False
        for i in range(obs.n):
            fl, er = obs.flux[i][k], obs.fluxerr[i][k]
            if fl is None or er is None or fl <= 0 or er <= 0:
                continue
            rel = er / fl
            if rel >= relerr_max:
                continue
            flux_all.append(fl)
            rel_all.append(rel)
            got = True
        if got:
            nfiles += 1
    return np.asarray(flux_all), np.asarray(rel_all), nfiles


def measure_white_fraction(obs_files, *, percentile: float = 10.0,
                           min_points: int = 5, date_tol: float = 60.0) -> float:
    """Effective white-noise fraction of the catalog fluxerr.

    For every apparition with at least ``min_points`` measurements, compare the
    *observed* point-to-point flux scatter (successive-difference estimator,
    which removes slow phase-angle trends) with the catalog fluxerr. Real
    lightcurve variation only adds scatter, so the low-percentile envelope of
    the ratio — the quietest apparitions (round objects, pole-on viewing) — is
    an upper bound on the fraction of the catalog error budget that actually
    fluctuates per epoch. The remainder is systematic/calibration terms that
    the LEADER amplitude statistic never sees.

    Returns 1.0 (no calibration) when fewer than 20 apparitions qualify.
    """
    ratios = []
    for path in obs_files:
        try:
            obs = read_obs(path)
        except (OSError, ValueError):
            continue
        k = select_best_filter(obs)
        rows = [(obs.dates[i], obs.flux[i][k], obs.fluxerr[i][k]) for i in range(obs.n)
                if obs.flux[i][k] is not None and obs.fluxerr[i][k] is not None
                and obs.flux[i][k] > 0 and obs.fluxerr[i][k] > 0]
        if len(rows) < min_points:
            continue
        rows.sort()
        t = np.array([r[0] for r in rows])
        f = np.array([r[1] for r in rows])
        e = np.array([r[2] for r in rows])
        brk = np.where(np.diff(t) > date_tol)[0] + 1
        for seg in np.split(np.arange(len(t)), brk):
            if len(seg) < min_points:
                continue
            sig_obs = np.std(np.diff(f[seg])) / np.sqrt(2)
            sig_cat = np.median(e[seg])
            if sig_cat > 0:
                ratios.append(sig_obs / sig_cat)
    if len(ratios) < 20:
        return 1.0
    return float(min(np.percentile(ratios, percentile), 1.0))


def fit_noise_model(obs_files, *, deg: int = 2, relerr_max: float = 1.0,
                    calibrate: bool = True, white_percentile: float = 10.0,
                    return_data: bool = False):
    """Fit the population's flux -> relative-uncertainty relation.

    Returns a :class:`NoiseModel` (or ``(model, flux, relerr)`` when
    ``return_data`` — the pairs are useful for the diagnostic plot without
    re-scanning the files). The polynomial degree drops automatically if there
    are too few measurements. With ``calibrate`` (default) the model also
    measures the **white-noise fraction** of the catalog fluxerr (see
    :func:`measure_white_fraction`) and scales ``relerr()`` by it — the
    catalog budget includes calibration terms that do not fluctuate per epoch.
    """
    flux, rel, nfiles = _collect_pairs(obs_files, relerr_max)
    if len(flux) < 10:
        raise ValueError(
            f"Only {len(flux)} usable (flux, fluxerr) pairs in {len(list(obs_files))} "
            ".obs files — cannot fit an empirical noise model.")
    while len(flux) < 10 * (deg + 1) and deg > 1:
        deg -= 1

    lf, lr = np.log10(flux), np.log10(rel)
    # fit inside a robust domain so a handful of extreme fluxes cannot steer it
    lo, hi = np.percentile(lf, [0.5, 99.5])
    sel = (lf >= lo) & (lf <= hi)
    coeffs = np.polyfit(lf[sel], lr[sel], deg)
    scatter = float(np.sqrt(np.mean((lr[sel] - np.polyval(coeffs, lf[sel])) ** 2)))

    wf = (measure_white_fraction(obs_files, percentile=white_percentile)
          if calibrate else 1.0)
    model = NoiseModel(coeffs=tuple(float(c) for c in coeffs),
                       log10_flux_domain=(float(lo), float(hi)),
                       scatter_dex=scatter, npoints=int(sel.sum()), nfiles=nfiles,
                       relerr_cap=relerr_max, white_fraction=wf)
    return (model, flux, rel) if return_data else model


def plot_noise_model(model: NoiseModel, flux, relerr, out_png: str, *,
                     show: bool = False) -> str:
    """Document the fit: data density, binned medians, and the fitted curve."""
    import matplotlib.pyplot as plt

    flux = np.asarray(flux)
    relerr = np.asarray(relerr)
    fig, ax = plt.subplots(figsize=(7, 5))
    # subsample the scatter for a light-weight figure
    if len(flux) > 20000:
        idx = np.random.default_rng(0).choice(len(flux), 20000, replace=False)
        fs, rs = flux[idx], relerr[idx]
    else:
        fs, rs = flux, relerr
    ax.plot(fs, rs, ".", ms=1, color="0.7", alpha=0.4, label="measurements")

    lf = np.log10(flux)
    edges = np.linspace(*model.log10_flux_domain, 13)
    mids, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lf >= a) & (lf < b)
        if m.sum() >= 20:
            mids.append(10 ** ((a + b) / 2))
            meds.append(np.median(relerr[m]))
    ax.plot(mids, meds, "ks", ms=5, label="binned median")

    lx = np.linspace(*model.log10_flux_domain, 200)
    fx = 10 ** lx
    ax.plot(fx, model.catalog_relerr(fx), "r-", lw=2, label="catalog relation (fit)")
    band = 10 ** (np.polyval(model.coeffs, lx))
    ax.fill_between(fx, band * 10 ** -model.scatter_dex, band * 10 ** model.scatter_dex,
                    color="r", alpha=0.15, label=f"±{model.scatter_dex:.2f} dex scatter")
    if model.white_fraction < 1.0:
        ax.plot(fx, model.relerr(fx), "m--", lw=2,
                label=f"effective white noise (×{model.white_fraction:.2f}, applied)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("flux (as tabulated in the .obs files)")
    ax.set_ylabel("relative uncertainty  σ_F / F")
    parts = []
    for i, c in enumerate(model.coeffs):
        power = len(model.coeffs) - 1 - i
        mag = f"{abs(c):.3f}" + ("" if power == 0 else "·x" if power == 1 else f"·x^{power}")
        parts.append(mag if not parts and c >= 0 else ("+ " if c >= 0 else "− ") + mag)
    terms = " ".join(parts)
    wtxt = (f"; white-noise fraction {model.white_fraction:.2f}"
            if model.white_fraction < 1.0 else "")
    ax.set_title("Empirical photometric-noise model\n"
                 f"log10(σ_F/F) = {terms},  x = log10(flux)\n"
                 f"({model.npoints} points from {model.nfiles} objects{wtxt})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_png
