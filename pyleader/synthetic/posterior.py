"""Posterior-inversion correction: recovered (p, β) → probability over true (p, β).

The quadratic correction (Step 5) maps a recovered peak to a single corrected
value — inadequate where the p–β degeneracy makes the recovered→true mapping
many-to-one. Here we instead treat the fixed-peak basis runs as a sampled **forward
model** — at each assigned grid point we know the mean and scatter of what
LEADER recovers — and invert it with Bayes' rule:

    P(true grid point | observed recovered peak)
        ∝ N(observed; mean_ij, cov_ij) × prior

The posterior over the true grid carries credible intervals, the p–β
covariance, and (crucially) **multimodality where the degeneracy makes several
true populations consistent with one recovery** — ambiguity is reported instead
of silently averaged away. Beta is in degrees throughout this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .basis import basis_units

# Covariance floor: recovered peaks are quantized to the inversion's bins
# (Δp = 0.05, Δβ ≈ 3.1°), so per-seed scatter can underestimate the true
# uncertainty; add half a bin width in quadrature.
_FLOOR_P = (0.05 / 2.0) ** 2
_FLOOR_B = (3.1 / 2.0) ** 2  # deg^2


def _recovered_stat(P, BETA, Pmargin, Bmargin, stat: str):
    """One summary of a recovered solution: the marginal peak or weighted median.

    Returns ``(p, beta_deg)``. Used identically for basis units and for the real
    analysis, so the observed statistic always matches the forward table's.
    """
    if stat == "peak":
        return (float(P[int(np.argmax(Pmargin))]),
                float(np.rad2deg(BETA[int(np.argmax(Bmargin))])))
    if stat == "median":
        from .stats import distribution_stats
        return (float(distribution_stats(P, Pmargin)["median"]),
                float(distribution_stats(np.rad2deg(BETA), Bmargin)["median"]))
    raise ValueError(f"stat must be 'peak' or 'median', got {stat!r}")


def recovered_stat_from_analysis(analysis_outdir: str, stat: str = "peak"):
    """The real analysis's recovered ``(p, beta_deg)`` statistic, averaged over trials.

    Computed from the per-trial joint solutions (``Trial*/W_trial*.npz``) with the
    same extractor the forward table uses, so both sides of the posterior are
    measured identically.
    """
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(analysis_outdir, "Trial*", "W_trial*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No Trial*/W_trial*.npz under {analysis_outdir} — re-run the analysis with "
            "this version of PyLEADER (it saves each trial's joint solution).")
    vals = []
    for p in paths:
        d = np.load(p)
        W = np.asarray(d["W"], float)
        vals.append(_recovered_stat(d["P"], d["BETA"], W.sum(axis=1), W.sum(axis=0), stat))
    v = np.asarray(vals)
    return float(v[:, 0].mean()), float(v[:, 1].mean())


@dataclass
class ForwardTable:
    """Mean/covariance of a recovered statistic at each assigned grid point."""

    p_grid: np.ndarray          # (Np,) assigned p peaks
    b_grid: np.ndarray          # (Nb,) assigned beta peaks, DEGREES
    mean: np.ndarray            # (Np, Nb, 2): recovered [p, beta_deg] mean over seeds
    cov: np.ndarray             # (Np, Nb, 2, 2): seed covariance + floor
    nseeds: np.ndarray          # (Np, Nb): seeds contributing per grid point
    stat: str = "peak"          # which recovered statistic this table describes

    def save(self, path: str) -> None:
        np.savez(path, p_grid=self.p_grid, b_grid=self.b_grid,
                 mean=self.mean, cov=self.cov, nseeds=self.nseeds, stat=self.stat)

    @classmethod
    def load(cls, path: str) -> "ForwardTable":
        d = np.load(path)
        stat = str(d["stat"]) if "stat" in d.files else "peak"
        return cls(d["p_grid"], d["b_grid"], d["mean"], d["cov"], d["nseeds"], stat)


def build_forward_table(basis_dir: str, *, stat: str = "peak",
                        cache: bool = True) -> ForwardTable:
    """Assemble (and cache) the forward table from a completed basis directory.

    ``stat`` selects the recovered summary used as the measurement channel:
    ``"peak"`` (marginal argmax; historic behaviour) or ``"median"`` (weighted
    median of the marginals — continuous, so less bin-quantized). Both channels
    can be built from the same basis; no new simulations are needed.
    """
    cache_path = os.path.join(basis_dir, f"forward_table_{stat}.npz")
    if cache and os.path.exists(cache_path):
        return ForwardTable.load(cache_path)

    # collect the recovered statistic per (assigned p, assigned b)
    samples: dict = {}
    for p_peak, b_peak, _s, path in basis_units(basis_dir):
        d = np.load(path)
        rec = _recovered_stat(d["P"], d["BETA"], d["Pmargin"], d["Bmargin"], stat)
        key = (round(p_peak, 6), round(np.rad2deg(b_peak), 6))
        samples.setdefault(key, []).append(rec)
    if not samples:
        raise FileNotFoundError(f"No completed basis units found in {basis_dir}")

    p_grid = np.array(sorted({k[0] for k in samples}))
    b_grid = np.array(sorted({k[1] for k in samples}))
    Np, Nb = len(p_grid), len(b_grid)
    mean = np.full((Np, Nb, 2), np.nan)
    cov = np.zeros((Np, Nb, 2, 2))
    nseeds = np.zeros((Np, Nb), dtype=int)

    floor = np.diag([_FLOOR_P, _FLOOR_B])
    for (pk, bk), vals in samples.items():
        ip = int(np.argmin(np.abs(p_grid - pk)))
        ib = int(np.argmin(np.abs(b_grid - bk)))
        v = np.asarray(vals)                       # (nseeds, 2)
        mean[ip, ib] = v.mean(axis=0)
        c = np.cov(v.T) if len(v) > 1 else np.zeros((2, 2))
        cov[ip, ib] = np.atleast_2d(c) + floor
        nseeds[ip, ib] = len(v)

    table = ForwardTable(p_grid, b_grid, mean, cov, nseeds, stat)
    if cache:
        table.save(cache_path)
    return table


@dataclass
class Posterior:
    """Posterior over the true (p, beta) grid for one observed recovery."""

    p_grid: np.ndarray          # (np_fine,)
    b_grid: np.ndarray          # (nb_fine,) degrees
    density: np.ndarray         # (np_fine, nb_fine), sums to 1
    observed: tuple             # (p_rec, b_rec_deg)
    # point estimates + central credible intervals from the marginals
    p_map: float; b_map: float
    p_median: float; p_lo68: float; p_hi68: float; p_lo95: float; p_hi95: float
    b_median: float; b_lo68: float; b_hi68: float; b_lo95: float; b_hi95: float
    multimodal: bool
    n_modes: int

    def save(self, path: str) -> None:
        np.savez(path, p_grid=self.p_grid, b_grid=self.b_grid, density=self.density,
                 observed=np.asarray(self.observed),
                 p_stats=np.array([self.p_map, self.p_median, self.p_lo68, self.p_hi68,
                                   self.p_lo95, self.p_hi95]),
                 b_stats=np.array([self.b_map, self.b_median, self.b_lo68, self.b_hi68,
                                   self.b_lo95, self.b_hi95]),
                 multimodal=self.multimodal, n_modes=self.n_modes)


def _interval(grid, marg, frac):
    """Central credible interval [lo, hi] of a 1-D marginal at probability `frac`."""
    c = np.cumsum(marg)
    c = c / c[-1]
    lo = float(np.interp((1 - frac) / 2, c, grid))
    hi = float(np.interp(1 - (1 - frac) / 2, c, grid))
    return lo, hi


def _count_modes(density, level_mass=0.68):
    """Number of disjoint regions in the highest-density `level_mass` credible set."""
    from scipy import ndimage
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat)
    thresh = flat[int(np.searchsorted(csum, level_mass * csum[-1]))]
    mask = density >= thresh
    _labels, n = ndimage.label(mask)
    return int(n)


def posterior_correct(p_rec: float, b_rec_deg: float, table: ForwardTable,
                      *, refine: int = 6) -> Posterior:
    """Posterior over true (p_peak, beta_peak) given one observed recovered peak.

    The forward mean/covariance are bilinearly interpolated onto a grid
    ``refine``× denser than the basis grid; the prior is uniform over that grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    pf = np.linspace(table.p_grid[0], table.p_grid[-1],
                     (len(table.p_grid) - 1) * refine + 1)
    bf = np.linspace(table.b_grid[0], table.b_grid[-1],
                     (len(table.b_grid) - 1) * refine + 1)
    PP, BB = np.meshgrid(pf, bf, indexing="ij")
    pts = np.stack([PP.ravel(), BB.ravel()], axis=1)

    def interp(field):
        f = RegularGridInterpolator((table.p_grid, table.b_grid), field)
        return f(pts)

    m_p = interp(table.mean[..., 0])
    m_b = interp(table.mean[..., 1])
    c00 = interp(table.cov[..., 0, 0])
    c01 = interp(table.cov[..., 0, 1])
    c11 = interp(table.cov[..., 1, 1])

    # Gaussian log-likelihood of the observation at every refined node
    dp = p_rec - m_p
    db = b_rec_deg - m_b
    det = c00 * c11 - c01 ** 2
    det = np.maximum(det, 1e-12)
    quad = (c11 * dp ** 2 - 2 * c01 * dp * db + c00 * db ** 2) / det
    loglike = -0.5 * (quad + np.log(det))
    loglike -= loglike.max()
    density = np.exp(loglike).reshape(PP.shape)
    density /= density.sum()

    p_marg = density.sum(axis=1)
    b_marg = density.sum(axis=0)
    imax = np.unravel_index(np.argmax(density), density.shape)

    p_lo68, p_hi68 = _interval(pf, p_marg, 0.68)
    p_lo95, p_hi95 = _interval(pf, p_marg, 0.95)
    b_lo68, b_hi68 = _interval(bf, b_marg, 0.68)
    b_lo95, b_hi95 = _interval(bf, b_marg, 0.95)
    n_modes = _count_modes(density)

    return Posterior(
        p_grid=pf, b_grid=bf, density=density, observed=(p_rec, b_rec_deg),
        p_map=float(pf[imax[0]]), b_map=float(bf[imax[1]]),
        p_median=float(np.interp(0.5, np.cumsum(p_marg) / p_marg.sum(), pf)),
        p_lo68=p_lo68, p_hi68=p_hi68, p_lo95=p_lo95, p_hi95=p_hi95,
        b_median=float(np.interp(0.5, np.cumsum(b_marg) / b_marg.sum(), bf)),
        b_lo68=b_lo68, b_hi68=b_hi68, b_lo95=b_lo95, b_hi95=b_hi95,
        multimodal=(n_modes > 1), n_modes=n_modes,
    )


def plot_posterior(post: Posterior, out_png: str, *, stat: str | None = None,
                   show: bool = False) -> None:
    """Three-panel figure: 2-D posterior map + the two marginals.

    ``stat`` names the measurement channel (``"peak"`` or ``"median"`` — which
    recovered statistic was inverted); when given it is stated in the figure
    title so the artifact is self-describing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.4),
                                        gridspec_kw={"width_ratios": [1.6, 1, 1]})

    # 2-D posterior with 68/95% highest-density contours
    flat = np.sort(post.density.ravel())[::-1]
    csum = np.cumsum(flat)
    lev68 = flat[int(np.searchsorted(csum, 0.68 * csum[-1]))]
    lev95 = flat[int(np.searchsorted(csum, 0.95 * csum[-1]))]
    pc = ax0.pcolormesh(post.p_grid, post.b_grid, post.density.T,
                        cmap="viridis", shading="auto")
    # levels can coincide when the posterior is extremely concentrated
    levels = sorted({lev95, lev68})
    if levels:
        ax0.contour(post.p_grid, post.b_grid, post.density.T,
                    levels=levels, colors=["w"] * len(levels),
                    linestyles=(["--", "-"] if len(levels) == 2 else ["-"]),
                    linewidths=1.2)
    ax0.plot(*post.observed, "rx", ms=10, mew=2, label="recovered (observed)")
    ax0.plot(post.p_map, post.b_map, "w*", ms=12, label="posterior mode")
    ax0.set_xlabel("true p")
    ax0.set_ylabel("true β (deg)")
    title = "Posterior over true (p, β)"
    if post.multimodal:
        title += f"  [MULTIMODAL: {post.n_modes} modes]"
    ax0.set_title(title)
    # legend: include the credible-region contours (proxy artists) and use a
    # gray box so the white contour lines / mode star stay visible
    handles, labels = ax0.get_legend_handles_labels()
    handles += [Line2D([], [], color="w", ls="-", lw=1.2),
                Line2D([], [], color="w", ls="--", lw=1.2)]
    labels += ["68% credible region", "95% credible region"]
    ax0.legend(handles, labels, fontsize=8, loc="upper left",
               facecolor="0.45", labelcolor="w", framealpha=0.85)
    fig.colorbar(pc, ax=ax0, label="probability")

    for ax, grid, axis, med, lo, hi, lbl in (
        (ax1, post.p_grid, 1, post.p_median, post.p_lo68, post.p_hi68, "p"),
        (ax2, post.b_grid, 0, post.b_median, post.b_lo68, post.b_hi68, "β (deg)"),
    ):
        marg = post.density.sum(axis=axis)
        ax.plot(grid, marg / marg.max(), "b-")
        ax.axvline(med, color="k", ls="-", lw=1, label=f"median {med:.3g}")
        ax.axvspan(lo, hi, color="b", alpha=0.15, label="68%")
        ax.set_xlabel(f"true {lbl}")
        ax.set_ylabel("marginal posterior")
        ax.legend(fontsize=8)

    if stat is not None:
        fig.suptitle(f"Posterior correction — {stat} channel "
                     f"(inverts the recovered {stat} statistic)")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
