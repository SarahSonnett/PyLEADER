"""Response-matrix unfolding: correct the full f(p, β) distribution.

The posterior correction (Step 5b) treats the population as a single peak. The
unfolding goes further: the fixed-peak basis runs define a **response matrix**
``R[i, j]`` — the recovered joint solution (bin *i* of the canonical 20×29
grid) produced by a population concentrated at true grid point *j*. Because the
amplitude CDF of a mixture is the mixture of the CDFs, the recovered solution
of an arbitrary population is approximately ``R @ f_true``, and ``f_true`` can
be recovered by regularized non-negative least squares — the same machinery as
``leader_invert`` itself, one level up.

The output is a coarse (basis-grid resolution) estimate of the **true joint
distribution** ``f(p, β)`` with per-bin uncertainty bands from a perturbation
ensemble. The p–β degeneracy appears as broad/correlated bands — that is the
honest information content, not a defect.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .basis import CANON_BETA, CANON_P, basis_units, rebin_to_canonical


@dataclass
class Response:
    """Response matrix built from a fixed-peak basis."""

    p_grid: np.ndarray        # (Np,) true-grid p values
    b_grid: np.ndarray        # (Nb,) true-grid beta values, DEGREES
    R: np.ndarray             # (580, Np*Nb): column = mean normalized recovered W
    col_std: np.ndarray       # (580,): pooled per-bin std across seeds (noise scale)

    def save(self, path: str) -> None:
        np.savez(path, p_grid=self.p_grid, b_grid=self.b_grid, R=self.R,
                 col_std=self.col_std)

    @classmethod
    def load(cls, path: str) -> "Response":
        d = np.load(path)
        return cls(d["p_grid"], d["b_grid"], d["R"], d["col_std"])


def build_response(basis_dir: str, *, cache: bool = True) -> Response:
    """Assemble (and cache) the response matrix from a completed basis."""
    cache_path = os.path.join(basis_dir, "response_matrix.npz")
    if cache and os.path.exists(cache_path):
        return Response.load(cache_path)

    groups: dict = {}
    for p_peak, b_peak, _s, path in basis_units(basis_dir):
        d = np.load(path)
        W = np.asarray(d["W"], float)
        s = W.sum()
        if s <= 0:
            continue
        key = (round(p_peak, 6), round(float(np.rad2deg(b_peak)), 6))
        groups.setdefault(key, []).append((W / s).ravel())
    if not groups:
        raise FileNotFoundError(f"No completed basis units found in {basis_dir}")

    p_grid = np.array(sorted({k[0] for k in groups}))
    b_grid = np.array(sorted({k[1] for k in groups}))
    ncols = len(p_grid) * len(b_grid)
    R = np.zeros((len(CANON_P) * len(CANON_BETA), ncols))
    stds = []
    for (pk, bk), vecs in groups.items():
        j = int(np.argmin(np.abs(p_grid - pk))) * len(b_grid) \
            + int(np.argmin(np.abs(b_grid - bk)))
        V = np.asarray(vecs)
        R[:, j] = V.mean(axis=0)
        if len(V) > 1:
            stds.append(V.std(axis=0))
    col_std = (np.mean(stds, axis=0) if stds
               else np.full(R.shape[0], R.max() * 0.02))
    col_std = np.maximum(col_std, R.max() * 1e-3)  # floor: avoid zero weights

    resp = Response(p_grid, b_grid, R, col_std)
    if cache:
        resp.save(cache_path)
    return resp


@dataclass
class UnfoldResult:
    p_grid: np.ndarray        # (Np,)
    b_grid: np.ndarray        # (Nb,) degrees
    f_mean: np.ndarray        # (Np, Nb) unfolded true distribution (sums to 1)
    f_lo: np.ndarray          # 16th percentile of the ensemble
    f_hi: np.ndarray          # 84th percentile
    alpha: float              # chosen regularization strength
    relerr: float             # ||R f - d|| / ||d|| at the mean solution
    # population medians of the unfolded distribution, with 16-84% bands from
    # the perturbation ensemble ("what is the median object's true p / beta?")
    pop_median_p: float = float("nan")
    pop_median_p_lo: float = float("nan")
    pop_median_p_hi: float = float("nan")
    pop_median_b: float = float("nan")
    pop_median_b_lo: float = float("nan")
    pop_median_b_hi: float = float("nan")

    def save(self, path: str) -> None:
        np.savez(path, p_grid=self.p_grid, b_grid=self.b_grid, f_mean=self.f_mean,
                 f_lo=self.f_lo, f_hi=self.f_hi, alpha=self.alpha, relerr=self.relerr,
                 pop_median_p=np.array([self.pop_median_p, self.pop_median_p_lo,
                                        self.pop_median_p_hi]),
                 pop_median_b=np.array([self.pop_median_b, self.pop_median_b_lo,
                                        self.pop_median_b_hi]))


def _smoothness_operator(np_, nb):
    """First-difference operator over the true (p, beta) grid (both directions)."""
    rows = []
    n = np_ * nb
    for i in range(np_):
        for j in range(nb):
            k = i * nb + j
            if i + 1 < np_:
                r = np.zeros(n); r[k] = -1; r[k + nb] = 1
                rows.append(r)
            if j + 1 < nb:
                r = np.zeros(n); r[k] = -1; r[k + 1] = 1
                rows.append(r)
    return np.asarray(rows)


def _solve(R, d, L, alpha):
    A = np.vstack([R, np.sqrt(alpha) * L])
    b = np.concatenate([d, np.zeros(L.shape[0])])
    res = lsq_linear(A, b, bounds=(0, np.inf), method="trf", lsmr_tol="auto")
    return res.x


def unfold(W_obs: np.ndarray, resp: Response, *, alphas=None,
           n_ensemble: int = 40, seed: int = 0) -> UnfoldResult:
    """Unfold an observed canonical-grid solution into f_true on the basis grid.

    ``W_obs`` is a (20, 29) joint solution on the canonical grid (rebinned if it
    came from a jittered-grid inversion). The regularization strength is chosen
    by the L-curve's maximum-curvature point over ``alphas``; uncertainties come
    from re-solving with the data vector perturbed at the basis noise scale.
    """
    d = np.asarray(W_obs, float).ravel()
    s = d.sum()
    if s <= 0:
        raise ValueError("Observed solution is empty (W sums to 0).")
    d = d / s

    np_, nb = len(resp.p_grid), len(resp.b_grid)
    L = _smoothness_operator(np_, nb)
    if alphas is None:
        alphas = np.logspace(-5, -1, 9)

    # Discrepancy principle: the smallest alpha sets the attainable misfit
    # floor; choose the LARGEST alpha whose misfit stays within 15% of that
    # floor — as much smoothing as the data allow, no more. (A max-curvature
    # L-curve pick proved unreliable here: the degeneracy makes response
    # columns highly correlated, and over-smoothing collapses the solution.)
    sols, errs = [], []
    for a in alphas:
        f = _solve(resp.R, d, L, a)
        sols.append(f)
        errs.append(np.linalg.norm(resp.R @ f - d))
    errs = np.asarray(errs)
    floor = errs.min()
    ok = np.where(errs <= 1.15 * floor)[0]
    ibest = int(ok.max()) if len(ok) else 0
    alpha = float(alphas[ibest])
    f_best = sols[ibest]

    # perturbation ensemble for per-bin uncertainty
    rng = np.random.default_rng(seed)
    ens = []
    for _ in range(n_ensemble):
        d_pert = np.maximum(d + rng.normal(0, resp.col_std), 0)
        ens.append(_solve(resp.R, d_pert, L, alpha))
    ens = np.asarray(ens)

    def norm_shape(v):
        v = v.reshape(np_, nb)
        t = v.sum()
        return v / t if t > 0 else v

    f_mean = norm_shape(f_best)
    f_lo = norm_shape(np.percentile(ens, 16, axis=0))
    f_hi = norm_shape(np.percentile(ens, 84, axis=0))
    relerr = float(np.linalg.norm(resp.R @ f_best - d) / np.linalg.norm(d))

    # population medians ("the median object's true p / beta") with ensemble bands
    from .stats import distribution_stats

    def _medians(vec):
        f = norm_shape(vec)
        return (distribution_stats(resp.p_grid, f.sum(axis=1))["median"],
                distribution_stats(resp.b_grid, f.sum(axis=0))["median"])

    med_p, med_b = _medians(f_best)
    ens_med = np.asarray([_medians(e) for e in ens])
    lo_p, hi_p = np.percentile(ens_med[:, 0], [16, 84])
    lo_b, hi_b = np.percentile(ens_med[:, 1], [16, 84])

    return UnfoldResult(resp.p_grid, resp.b_grid, f_mean, f_lo, f_hi, alpha, relerr,
                        pop_median_p=float(med_p), pop_median_p_lo=float(lo_p),
                        pop_median_p_hi=float(hi_p), pop_median_b=float(med_b),
                        pop_median_b_lo=float(lo_b), pop_median_b_hi=float(hi_b))


def observed_from_analysis(analysis_outdir: str) -> np.ndarray:
    """Average the per-trial joint solutions of a real analysis (canonical grid).

    Each ``Trial*/W_trial*.npz`` is rebinned from its (jittered) grid onto the
    canonical grid, normalized, and averaged.
    """
    paths = sorted(glob.glob(os.path.join(analysis_outdir, "Trial*", "W_trial*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No Trial*/W_trial*.npz in {analysis_outdir} — re-run the analysis with "
            "this version of PyLEADER (it now saves each trial's joint solution).")
    acc = np.zeros((len(CANON_P), len(CANON_BETA)))
    for p in paths:
        d = np.load(p)
        Wc = rebin_to_canonical(d["P"], d["BETA"], d["W"])
        s = Wc.sum()
        if s > 0:
            acc += Wc / s
    return acc / len(paths)


def plot_unfolded(res: UnfoldResult, out_png: str, *, truth=None, show: bool = False) -> None:
    """Joint map + marginal panels with 16–84% bands; optional truth overlay.

    ``truth`` may be a list of (p, beta_deg, weight) components to mark.
    """
    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.4),
                                        gridspec_kw={"width_ratios": [1.6, 1, 1]})
    pc = ax0.pcolormesh(res.p_grid, res.b_grid, res.f_mean.T, cmap="viridis", shading="auto")
    if truth is not None:
        for (pt, bt, wt) in truth:
            ax0.plot(pt, bt, "r*", ms=14 * max(wt, 0.4), mec="w")
    ax0.set_xlabel("true p"); ax0.set_ylabel("true β (deg)")
    ax0.set_title("Estimated true population distribution f(p, β)\n"
                  f"(unfolded; relerr={res.relerr:.3f}, α={res.alpha:.3g})")
    fig.colorbar(pc, ax=ax0, label="probability")

    for ax, axis, grid, lbl in ((ax1, 1, res.p_grid, "p"),
                                (ax2, 0, res.b_grid, "β (deg)")):
        m = res.f_mean.sum(axis=axis)
        lo = res.f_lo.sum(axis=axis)
        hi = res.f_hi.sum(axis=axis)
        ax.plot(grid, m, "b-", label="unfolded")
        ax.fill_between(grid, lo, hi, color="b", alpha=0.2, label="16–84%")
        if truth is not None:
            for (pt, bt, wt) in truth:
                ax.axvline(pt if lbl == "p" else bt, color="r", ls=":", alpha=0.8)
        ax.set_xlabel(f"true {lbl}"); ax.set_ylabel("marginal f")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
