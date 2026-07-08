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
    ``log10(relerr)`` as a function of ``log10(flux)``. Evaluation clips
    ``log10(flux)`` to the fitted domain (no extrapolation) and caps the
    returned relative error at ``relerr_cap``.
    """

    coeffs: tuple                 # polynomial coefficients, highest power first
    log10_flux_domain: tuple      # (lo, hi) fitted domain of log10(flux)
    scatter_dex: float            # rms residual of the fit, dex
    npoints: int                  # measurements used in the fit
    nfiles: int                   # .obs files contributing
    relerr_cap: float = 1.0       # ceiling on the evaluated relative error

    def relerr(self, flux) -> np.ndarray:
        """Relative uncertainty sigma_F/F at the given flux(es)."""
        f = np.maximum(np.asarray(flux, dtype=float), 1e-30)
        lf = np.clip(np.log10(f), *self.log10_flux_domain)
        return np.minimum(10.0 ** np.polyval(self.coeffs, lf), self.relerr_cap)

    def to_dict(self) -> dict:
        return dict(coeffs=list(self.coeffs), log10_flux_domain=list(self.log10_flux_domain),
                    scatter_dex=self.scatter_dex, npoints=self.npoints,
                    nfiles=self.nfiles, relerr_cap=self.relerr_cap,
                    form="log10(fluxerr/flux) = polyval(coeffs, log10(flux))")

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
                   nfiles=int(d["nfiles"]), relerr_cap=float(d.get("relerr_cap", 1.0)))


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


def fit_noise_model(obs_files, *, deg: int = 2, relerr_max: float = 1.0,
                    return_data: bool = False):
    """Fit the population's flux -> relative-uncertainty relation.

    Returns a :class:`NoiseModel` (or ``(model, flux, relerr)`` when
    ``return_data`` — the pairs are useful for the diagnostic plot without
    re-scanning the files). The polynomial degree drops automatically if there
    are too few measurements.
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

    model = NoiseModel(coeffs=tuple(float(c) for c in coeffs),
                       log10_flux_domain=(float(lo), float(hi)),
                       scatter_dex=scatter, npoints=int(sel.sum()), nfiles=nfiles,
                       relerr_cap=relerr_max)
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
    ax.plot(fx, model.relerr(fx), "r-", lw=2, label="fitted polynomial")
    band = 10 ** (np.polyval(model.coeffs, lx))
    ax.fill_between(fx, band * 10 ** -model.scatter_dex, band * 10 ** model.scatter_dex,
                    color="r", alpha=0.15, label=f"±{model.scatter_dex:.2f} dex scatter")

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
    ax.set_title("Empirical photometric-noise model\n"
                 f"log10(σ_F/F) = {terms},  x = log10(flux)\n"
                 f"({model.npoints} points from {model.nfiles} objects)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return out_png
