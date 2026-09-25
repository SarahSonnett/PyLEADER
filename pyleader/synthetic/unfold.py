"""Response-matrix unfolding: correct the full f(p, β) distribution.

The posterior correction (Step 5b) treats the population as a single peak. The
unfolding goes further: the fixed-peak basis runs define a **response matrix**
``R[i, j]`` — the recovered joint solution (bin *i* of the canonical 20×29
grid) produced by a population concentrated at true grid point *j*. Because the
amplitude CDF of a mixture is the mixture of the CDFs, the recovered solution
of an arbitrary population is approximately ``R @ f_true``, and ``f_true`` can
be recovered by regularized non-negative least squares — the same machinery as
``leader_invert`` itself, one level up.

Two response spaces are available:

* **W-space** (:func:`build_response` / :func:`unfold`) — columns are the mean
  recovered joint solutions. Carries a small model error: the regularized NNLS
  inversion is not exactly linear in mixtures (measured with the mixture
  validation).
* **CDF-space** (:func:`build_response_cdf` / :func:`unfold_cdf`) — columns are
  each basis point's simulated **amplitude CDF**, and the observation is the
  real population's pooled amplitude CDF. Pooling amplitudes *is* mixing, so
  the forward model is **exactly linear in mixtures** — the W-space model error
  is removed by construction. This is the recommended space for evaluating
  systematics in the unfolded distribution; it requires a basis whose units
  saved their amplitude samples (bases built from 2026-07-08 on).

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

    @classmethod
    def load(cls, path: str) -> "UnfoldResult":
        """Reload a saved ``population_distribution.npz`` — e.g. to re-render the
        figure with modified titles/labels via :func:`plot_unfolded`, without
        re-running any simulation."""
        d = np.load(path)
        mp, mb = d["pop_median_p"], d["pop_median_b"]
        return cls(d["p_grid"], d["b_grid"], d["f_mean"], d["f_lo"], d["f_hi"],
                   float(d["alpha"]), float(d["relerr"]),
                   pop_median_p=float(mp[0]), pop_median_p_lo=float(mp[1]),
                   pop_median_p_hi=float(mp[2]), pop_median_b=float(mb[0]),
                   pop_median_b_lo=float(mb[1]), pop_median_b_hi=float(mb[2]))


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
    came from a jittered-grid inversion). Uncertainties come from re-solving
    with the data vector perturbed at the basis noise scale.
    """
    d = np.asarray(W_obs, float).ravel()
    s = d.sum()
    if s <= 0:
        raise ValueError("Observed solution is empty (W sums to 0).")
    d = d / s
    return _unfold_core(resp.R, d, resp.col_std, resp.p_grid, resp.b_grid,
                        alphas=alphas, n_ensemble=n_ensemble, seed=seed)


def _unfold_core(R, d, d_std, p_grid, b_grid, *, alphas=None,
                 n_ensemble: int = 40, seed: int = 0) -> UnfoldResult:
    """Shared regularized-NNLS unfolding engine (W-space and CDF-space).

    ``d_std`` is the per-row noise scale of the data vector, used for the
    perturbation ensemble.
    """
    np_, nb = len(p_grid), len(b_grid)
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
        f = _solve(R, d, L, a)
        sols.append(f)
        errs.append(np.linalg.norm(R @ f - d))
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
        d_pert = np.maximum(d + rng.normal(0, d_std), 0)
        ens.append(_solve(R, d_pert, L, alpha))
    ens = np.asarray(ens)

    def norm_shape(v):
        v = v.reshape(np_, nb)
        t = v.sum()
        return v / t if t > 0 else v

    f_mean = norm_shape(f_best)
    f_lo = norm_shape(np.percentile(ens, 16, axis=0))
    f_hi = norm_shape(np.percentile(ens, 84, axis=0))
    relerr = float(np.linalg.norm(R @ f_best - d) / np.linalg.norm(d))

    # population medians ("the median object's true p / beta") with ensemble bands
    from .stats import distribution_stats

    def _medians(vec):
        f = norm_shape(vec)
        return (distribution_stats(p_grid, f.sum(axis=1))["median"],
                distribution_stats(b_grid, f.sum(axis=0))["median"])

    med_p, med_b = _medians(f_best)
    ens_med = np.asarray([_medians(e) for e in ens])
    lo_p, hi_p = np.percentile(ens_med[:, 0], [16, 84])
    lo_b, hi_b = np.percentile(ens_med[:, 1], [16, 84])

    return UnfoldResult(np.asarray(p_grid), np.asarray(b_grid), f_mean, f_lo, f_hi,
                        alpha, relerr,
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


# --------------------------------------------------------------------------
# CDF-space response: exactly linear in mixtures (no NNLS model error)
# --------------------------------------------------------------------------

@dataclass
class ResponseCDF:
    """Response matrix whose columns are simulated amplitude CDFs.

    Each column is the mean (over seeds) empirical CDF of the amplitudes a
    fixed-peak population at that grid point produces, evaluated on the common
    ``a_grid``. Pooling amplitudes is mixing, so ``R @ f`` is *exactly* the
    amplitude CDF of a population with mixture weights ``f`` (up to per-object
    amplitude-count differences, which are second-order).
    """

    p_grid: np.ndarray        # (Np,) true-grid p values
    b_grid: np.ndarray        # (Nb,) true-grid beta values, DEGREES
    a_grid: np.ndarray        # (Na,) common amplitude grid
    R: np.ndarray             # (Na, Np*Nb): column = mean amplitude CDF
    col_std: np.ndarray       # (Na,): pooled per-row std across seeds

    def save(self, path: str) -> None:
        np.savez(path, p_grid=self.p_grid, b_grid=self.b_grid, a_grid=self.a_grid,
                 R=self.R, col_std=self.col_std)

    @classmethod
    def load(cls, path: str) -> "ResponseCDF":
        d = np.load(path)
        return cls(d["p_grid"], d["b_grid"], d["a_grid"], d["R"], d["col_std"])


def _ecdf_on_grid(a_sorted, a_grid):
    """Empirical CDF of a sorted amplitude sample, evaluated at ``a_grid``."""
    return np.searchsorted(a_sorted, a_grid, side="right") / len(a_sorted)


def build_response_cdf(basis_dir: str, *, n_a: int = 200,
                       cache: bool = True) -> ResponseCDF:
    """Assemble (and cache) the CDF-space response from a completed basis.

    Requires basis units that saved their amplitude samples (``A`` in
    ``synthetic_result.npz``; bases built from 2026-07-08 on) — older bases
    must be rebuilt.
    """
    cache_path = os.path.join(basis_dir, "response_cdf.npz")
    if cache and os.path.exists(cache_path):
        return ResponseCDF.load(cache_path)

    # amplitude grid: A is bounded in [0, 1) by construction
    a_grid = np.linspace(0.0, 1.0, n_a)

    groups: dict = {}
    n_missing = 0
    for p_peak, b_peak, _s, path in basis_units(basis_dir):
        d = np.load(path)
        if "A" not in d.files or len(d["A"]) == 0:
            n_missing += 1
            continue
        cdf = _ecdf_on_grid(np.sort(np.asarray(d["A"], float)), a_grid)
        key = (round(p_peak, 6), round(float(np.rad2deg(b_peak)), 6))
        groups.setdefault(key, []).append(cdf)
    if not groups:
        raise FileNotFoundError(
            f"No basis units with saved amplitude samples in {basis_dir}"
            + (f" ({n_missing} units predate amplitude saving)" if n_missing else "")
            + " — rebuild the basis with the current PyLEADER to use the "
              "CDF-space response.")
    if n_missing:
        print(f"NOTE: skipped {n_missing} basis unit(s) without saved amplitudes.")

    p_grid = np.array(sorted({k[0] for k in groups}))
    b_grid = np.array(sorted({k[1] for k in groups}))
    R = np.zeros((n_a, len(p_grid) * len(b_grid)))
    stds = []
    for (pk, bk), cdfs in groups.items():
        j = int(np.argmin(np.abs(p_grid - pk))) * len(b_grid) \
            + int(np.argmin(np.abs(b_grid - bk)))
        V = np.asarray(cdfs)
        R[:, j] = V.mean(axis=0)
        if len(V) > 1:
            stds.append(V.std(axis=0))
    col_std = (np.mean(stds, axis=0) if stds
               else np.full(n_a, 0.02))
    col_std = np.maximum(col_std, 1e-4)

    resp = ResponseCDF(p_grid, b_grid, a_grid, R, col_std)
    if cache:
        resp.save(cache_path)
    return resp


def observed_cdf_from_analysis(analysis_outdir: str, a_grid) -> tuple:
    """Observed amplitude CDF (mean, std) from a real analysis's saved trials.

    Each ``Trial*/W_trial*.npz`` (new analyses) carries the trial's pooled
    sorted amplitudes; each trial is an independent resampling of the
    population, so the across-trial std of the CDF *is* the sampling
    uncertainty of the observation.
    """
    paths = sorted(glob.glob(os.path.join(analysis_outdir, "Trial*", "W_trial*.npz")))
    cdfs = []
    for p in paths:
        d = np.load(p)
        if "Asort" in d.files and len(d["Asort"]):
            cdfs.append(_ecdf_on_grid(np.asarray(d["Asort"], float), a_grid))
    if not cdfs:
        raise FileNotFoundError(
            f"No saved amplitude samples in {analysis_outdir}/Trial*/ — either "
            "re-run the analysis with the current PyLEADER (it saves Asort per "
            "trial), or supply the .obs files (observed_cdf_from_obs / --obsdir).")
    C = np.asarray(cdfs)
    return C.mean(axis=0), np.maximum(C.std(axis=0), 1e-4)


def observed_cdf_from_obs(obs_files, acfg, a_grid, *, n_boot: int = 100,
                          seed: int = 0) -> tuple:
    """Observed amplitude CDF (mean, std) recomputed directly from ``.obs`` files.

    Pools every object's amplitudes once (no trial resampling) using the same
    reader/tolerances as the analysis (``acfg`` is an :class:`AnalysisConfig`);
    the std comes from bootstrap resampling over objects. Use this when the
    analysis predates per-trial amplitude saving.
    """
    from ..lightcurve import lcg_read_WISE

    per_object = []
    for fname in obs_files:
        try:
            _nppo, _nappar, A = lcg_read_WISE(fname, acfg)
        except (OSError, ValueError, IndexError):
            continue
        if len(A):
            per_object.append(np.asarray(A, float))
    if not per_object:
        raise ValueError("No usable amplitudes in the supplied .obs files.")

    pooled = np.sort(np.concatenate(per_object))
    cdf = _ecdf_on_grid(pooled, a_grid)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(per_object), len(per_object))
        sample = np.sort(np.concatenate([per_object[i] for i in idx]))
        boots.append(_ecdf_on_grid(sample, a_grid))
    std = np.maximum(np.asarray(boots).std(axis=0), 1e-4)
    return cdf, std


def unfold_cdf(cdf_obs, cdf_std, resp: ResponseCDF, *, alphas=None,
               n_ensemble: int = 40, seed: int = 0) -> UnfoldResult:
    """Unfold an observed amplitude CDF into f_true on the basis grid.

    Because the forward model is exactly linear in mixtures, the residual
    misfit here is measurement noise + basis sampling noise only — no
    inversion model error (unlike W-space).
    """
    d = np.asarray(cdf_obs, float)
    if d.shape[0] != resp.R.shape[0]:
        raise ValueError("cdf_obs must be evaluated on resp.a_grid "
                         f"({resp.R.shape[0]} points, got {d.shape[0]}).")
    return _unfold_core(resp.R, d, np.asarray(cdf_std, float),
                        resp.p_grid, resp.b_grid,
                        alphas=alphas, n_ensemble=n_ensemble, seed=seed)


def plot_unfolded(res: UnfoldResult, out_png: str, *, truth=None,
                  space: str | None = None, show: bool = False) -> None:
    """Joint map + marginal panels with 16–84% bands; optional truth overlay.

    ``truth`` may be a list of (p, beta_deg, weight) components to mark;
    ``space`` labels the response space ("W" or "CDF") in the title.
    """
    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.4),
                                        gridspec_kw={"width_ratios": [1.6, 1, 1]})
    pc = ax0.pcolormesh(res.p_grid, res.b_grid, res.f_mean.T, cmap="viridis", shading="auto")
    if truth is not None:
        for (pt, bt, wt) in truth:
            ax0.plot(pt, bt, "r*", ms=14 * max(wt, 0.4), mec="w")
    ax0.set_xlabel("true p"); ax0.set_ylabel("true β (deg)")
    tag = f", {space}-space response" if space else ""
    ax0.set_title("Estimated true population distribution f(p, β)\n"
                  f"(unfolded{tag}; relerr={res.relerr:.3f}, α={res.alpha:.3g})")
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
