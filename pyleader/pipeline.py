"""End-to-end per-population driver.

Given a dynamical-population ID (a Nesvorný collisional family or a background
MBA population), this chains the whole workflow:

    build .obs  ->  LEADER analysis  ->  bias map / fixed-peak basis on *this
    population's* observing geometry  ->  population-specific corrections  ->  apply.

Because the correction is derived from the population's own ``.obs`` cadence and
geometry, it is bespoke to the dataset — the scientifically appropriate choice.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analysis import diameter_matched_files, run_analysis
from .config import AnalysisConfig, ObsBuildConfig
from .populations import is_background
from .synthetic.config import SyntheticConfig
from .synthetic.correction import (
    apply_correction, fit_from_csv, plot_correction_fit, save_correction,
)
from .synthetic.stats import distribution_stats  # noqa: F401  (re-exported convenience)
from .synthetic.bias_map import run_bias_map


@dataclass
class PopulationConfig:
    """Everything needed to analyze one population end-to-end."""

    pop_id: str
    population_kind: Optional[str] = None      # inferred from pop_id if None
    cat: str = "allsky_4band_p1bs_psd"
    filterpriority: str = "w3"
    diam_low: float = 3.0
    diam_high: float = 5.0

    # analysis
    Ntrials: int = 100
    Ndraws: int = 1000
    phase_angle_limit: float = 40.0
    date_tol: float = 60.0
    wanted: int = 5
    convert2degrees: bool = True
    overwrite: bool = True
    neowise_fle: str = "neowise_mainbelt.csv"

    # bias map (quadratic correction). b_peaks are in RADIANS at the config/API
    # level (all internal math is radians); the CLIs accept degrees and convert.
    p_peaks: tuple = (0.35, 0.45, 0.55, 0.65, 0.75)
    b_peaks: tuple = tuple(np.deg2rad((10.0, 30.0, 50.0, 75.0)))
    bias_map_ndraws: int = 1000
    bias_map_nseeds: int = 1
    scattering: str = "ls_lambert"
    correction_stat: str = "peak"

    # which correction(s) to derive/apply:
    #   "quadratic"  — the deterministic recovered->true surface (bias map)
    #   "posterior"  — the probabilistic correction with credible intervals (fixed-peak basis)
    #   "both"       — run both (default; the report shows them side by side)
    correction_method: str = "both"
    # which recovered statistic the posterior inverts: "peak" (marginal argmax),
    # "median" (weighted median of the marginals; continuous, less bin-quantized),
    # or "both" (default) — running both doubles as a consistency check: for a
    # genuinely single-peaked population the two channels must agree.
    posterior_stat: str = "both"
    # fixed-peak basis campaign for the posterior correction (auto-built/resumed if absent)
    basis_dir: Optional[str] = None            # default: "<analysis outdir>_basis"
    basis_np: int = 8                          # p grid points
    basis_nb: int = 8                          # beta grid points
    basis_p_range: tuple = (0.30, 0.80)
    basis_b_range_deg: tuple = (6.0, 84.0)
    basis_nseeds: int = 4
    basis_nproc: Optional[int] = None          # default: 8, capped at cores - 2

    base_dir: str = None
    # Arbitrary .obs directory (bypasses the naming convention). Used for both
    # building (Step 2) and reading (Step 3); the bias-map/basis geometry follows.
    obsdir: Optional[str] = None

    def __post_init__(self):
        if self.population_kind is None:
            self.population_kind = "background" if is_background(self.pop_id) else "family"
        if self.base_dir is None:
            from .config import DEFAULT_BASE_DIR
            self.base_dir = DEFAULT_BASE_DIR
        if self.correction_method not in ("quadratic", "posterior", "both"):
            raise ValueError(
                f"correction_method must be 'quadratic', 'posterior' or 'both', "
                f"got {self.correction_method!r}")
        if self.posterior_stat not in ("peak", "median", "both"):
            raise ValueError(
                f"posterior_stat must be 'peak', 'median' or 'both', "
                f"got {self.posterior_stat!r}")

    def analysis_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            famid=self.pop_id, cat=self.cat, filterpriority=self.filterpriority,
            diam_low=self.diam_low, diam_high=self.diam_high,
            phase_angle_limit=self.phase_angle_limit, Ndraws=self.Ndraws, Ntrials=self.Ntrials,
            date_tol=self.date_tol, wanted=self.wanted, overwrite=self.overwrite,
            convert2degrees=self.convert2degrees, neowise_fle=self.neowise_fle,
            population_kind=self.population_kind, base_dir=self.base_dir,
            obsdir=self.obsdir,
        )

    def obs_config(self) -> ObsBuildConfig:
        return ObsBuildConfig(
            famid=self.pop_id, cat=self.cat, filterpriority=self.filterpriority,
            population_kind=self.population_kind, neowise_fle=self.neowise_fle,
            base_dir=self.base_dir, obsdir=self.obsdir,
        )

    def synthetic_base(self, geometry_files) -> SyntheticConfig:
        # tolerances matched to the analysis so the correction reflects the same cuts
        return SyntheticConfig(
            Ndraws=self.bias_map_ndraws, scattering=self.scattering,
            phase_angle_limit=self.phase_angle_limit, date_tol=self.date_tol, wanted=self.wanted,
            convert2degrees=self.convert2degrees, geometry_files=list(geometry_files),
            base_dir=self.base_dir,
        )


@dataclass
class PopulationResult:
    pop_id: str
    outdir: str
    recovered: tuple                       # (p, beta_deg) LEADER peak, averaged over trials
    corrected: Optional[tuple] = None      # quadratic correction (p, beta_deg), if run
    correction_path: Optional[str] = None
    bias_map_csv: Optional[str] = None
    r2: Optional[tuple] = None             # (r2_p, r2_beta) of the quadratic fit
    posterior: object = None               # headline Posterior (median channel if run)
    posteriors: Optional[dict] = None      # {stat: Posterior} for each channel run
    basis_dir: Optional[str] = None


def _require_damit_models() -> None:
    """Fail early (before the analysis) with actionable guidance if models are absent."""
    import glob
    from .synthetic.config import SyntheticConfig
    scfg = SyntheticConfig()
    if not glob.glob(os.path.join(scfg.damit_dir, "*.txt")):
        raise FileNotFoundError(
            f"No DAMIT shape models in {scfg.damit_dir} — the per-population correction "
            "needs them. Fetch them once with:\n"
            "    pyleader-download-models        (or: python scripts/download_models.py)\n"
            "then re-run. Or pass --refresh-models / run_population(..., refresh_models=True) "
            "to download the latest versions now."
        )


def _recovered_peak(analysis_outdir: str, acfg: AnalysisConfig):
    """Average LEADER peak (pmax, betamax_deg) across trials from the summary file."""
    summary = os.path.join(analysis_outdir, "summary", acfg.summary_name)
    pmax, betamax = np.genfromtxt(summary, unpack=True, usecols=(1, 2), dtype=float, skip_header=1)
    return float(np.mean(np.atleast_1d(pmax))), float(np.mean(np.atleast_1d(betamax)))


def run_population(cfg: PopulationConfig, *, do_build: bool = False,
                   refresh_models: bool = False, seed: int | None = None) -> PopulationResult:
    """Run the full per-population pipeline; returns a :class:`PopulationResult`.

    DAMIT shape models are assumed to already exist in the models directory;
    pass ``refresh_models=True`` to re-download the current DAMIT versions of the
    models listed in ``asteroideja.txt`` before the corrections run.
    """
    acfg = cfg.analysis_config()

    # 0. optionally refresh the DAMIT shape models, else ensure they exist
    if refresh_models:
        from .synthetic.config import SyntheticConfig
        from .synthetic.damit import download_damit_models, parse_model_list
        scfg = SyntheticConfig()
        download_damit_models(parse_model_list(scfg.damit_list), scfg.damit_dir, refresh=True)
    else:
        _require_damit_models()

    # 1. build .obs (optional; needs network + astropy/sunpy)
    if do_build:
        from .obsfiles.build import build_obs_files
        build_obs_files(cfg.obs_config())

    # 2. LEADER analysis on the real data
    print(f"[{cfg.pop_id}] LEADER analysis ...")
    outdir = run_analysis(acfg, seed=seed)
    rec_p, rec_b = _recovered_peak(outdir, acfg)

    summary_dir = os.path.join(outdir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    geom = diameter_matched_files(acfg)
    base_syn = cfg.synthetic_base(geom)
    result = PopulationResult(pop_id=cfg.pop_id, outdir=outdir, recovered=(rec_p, rec_b))
    coeffs = None

    # 3a–5a. quadratic correction: bias map -> fit -> apply
    if cfg.correction_method in ("quadratic", "both"):
        print(f"[{cfg.pop_id}] determining the bias map from {len(geom)} population geometries ...")
        # simulations live OUTSIDE the analysis dir (re-running the analysis wipes it)
        biasmap_dir = f"{outdir}_biasmap"
        biasmap_csv = run_bias_map(base_syn, cfg.p_peaks, cfg.b_peaks,
                                   nseeds=cfg.bias_map_nseeds, seed=(seed or 0),
                                   outdir=biasmap_dir)

        # surface the recovered-vs-assigned figure in the summary directory
        summary_src = os.path.join(biasmap_dir, "bias_map_summary.png")
        if os.path.exists(summary_src):
            shutil.copy(summary_src, os.path.join(summary_dir, "bias_map_summary.png"))

        coeffs = fit_from_csv(biasmap_csv, stat=cfg.correction_stat)
        corr_path = os.path.join(summary_dir, "quadratic_correction.json")
        save_correction(coeffs, corr_path)
        plot_correction_fit(biasmap_csv, coeffs,
                            os.path.join(summary_dir, "quadratic_correction_fit.png"))

        cor_p, cor_b = apply_correction([rec_p], [rec_b], coeffs)
        result.corrected = (float(cor_p[0]), float(cor_b[0]))
        result.correction_path = corr_path
        result.bias_map_csv = biasmap_csv
        result.r2 = (coeffs["diagnostics"]["r2_p"], coeffs["diagnostics"]["r2_b"])

    # 3b–5b. posterior correction: fixed-peak basis -> forward table(s) -> posterior(s)
    if cfg.correction_method in ("posterior", "both"):
        from .synthetic.basis import run_basis
        from .synthetic.posterior import (build_forward_table, posterior_correct,
                                          plot_posterior, recovered_stat_from_analysis)

        basis_dir = cfg.basis_dir or f"{outdir}_basis"
        print(f"[{cfg.pop_id}] posterior correction: fixed-peak basis at {basis_dir}")
        p_grid = np.linspace(cfg.basis_p_range[0], cfg.basis_p_range[1], cfg.basis_np)
        b_grid = np.deg2rad(np.linspace(cfg.basis_b_range_deg[0],
                                        cfg.basis_b_range_deg[1], cfg.basis_nb))
        run_basis(base_syn, p_grid, b_grid, nseeds=cfg.basis_nseeds,
                  seed=(seed or 0), outdir=basis_dir, nproc=cfg.basis_nproc)

        channels = (("peak", "median") if cfg.posterior_stat == "both"
                    else (cfg.posterior_stat,))
        result.posteriors = {}
        for st in channels:
            table = build_forward_table(basis_dir, stat=st)
            obs = recovered_stat_from_analysis(outdir, st)
            post = posterior_correct(obs[0], obs[1], table)
            post.save(os.path.join(summary_dir, f"posterior_{st}.npz"))
            plot_posterior(post, os.path.join(summary_dir, f"posterior_correction_{st}.png"))
            result.posteriors[st] = post
        # headline channel: median if available (continuous observable), else peak
        result.posterior = result.posteriors.get("median",
                                                 next(iter(result.posteriors.values())))
        result.basis_dir = basis_dir

    _write_report(cfg, outdir, (rec_p, rec_b), result.corrected, coeffs,
                  getattr(result, "posteriors", None))

    msg = f"[{cfg.pop_id}] recovered (p={rec_p:.3f}, β={rec_b:.1f}°)"
    if result.corrected is not None:
        msg += f"  ->  quadratic (p={result.corrected[0]:.3f}, β={result.corrected[1]:.1f}°)"
    if result.posterior is not None:
        po = result.posterior
        ch = "median" if "median" in (result.posteriors or {}) else "peak"
        msg += (f"  ->  posterior[{ch}] p={po.p_median:.3f} "
                f"[{po.p_lo68:.3f},{po.p_hi68:.3f}], β={po.b_median:.1f}° "
                f"[{po.b_lo68:.1f},{po.b_hi68:.1f}]"
                + ("  [MULTIMODAL]" if po.multimodal else ""))
    print(msg)
    return result


def _write_report(cfg, outdir, recovered, corrected=None, coeffs=None, posteriors=None):
    with open(os.path.join(outdir, "summary", "population_report.txt"), "w") as f:
        f.write(f"# Population pipeline report: {cfg.pop_id} ({cfg.population_kind})\n")
        f.write(f"# catalog={cfg.cat} filter={cfg.filterpriority} "
                f"diam=[{cfg.diam_low},{cfg.diam_high}] km  Ntrials={cfg.Ntrials} Ndraws={cfg.Ndraws}\n")

        if coeffs is not None and corrected is not None:
            d = coeffs["diagnostics"]
            pr, br = d["p_rec_range"], d["b_rec_range"]
            p_extrap = not (pr[0] <= recovered[0] <= pr[1])
            b_extrap = not (br[0] <= recovered[1] <= br[1])
            f.write(f"# quadratic correction: {cfg.correction_stat}-based, per-population "
                    f"geometry, n={d['n']}, terms={coeffs.get('n_terms')}, "
                    f"R2_p={d['r2_p']:.3f} R2_beta={d['r2_b']:.3f}\n")
            f.write(f"# synthetic recovered ranges: p in [{pr[0]:.2f},{pr[1]:.2f}], "
                    f"beta in [{br[0]:.0f},{br[1]:.0f}] deg\n")
            if p_extrap or b_extrap:
                f.write("# WARNING: recovered %s outside the synthetic range -> quadratic "
                        "correction EXTRAPOLATES; treat corrected value(s) with caution\n"
                        % (", ".join([x for x, e in (("p", p_extrap), ("beta", b_extrap)) if e])))
            f.write("\n== quadratic correction ==\n")
            f.write("quantity   recovered   corrected\n")
            f.write("p          %9.4f   %9.4f\n" % (recovered[0], corrected[0]))
            f.write("beta_deg   %9.2f   %9.2f\n" % (recovered[1], corrected[1]))

        if posteriors:
            for st, po in posteriors.items():
                f.write(f"\n== posterior correction, {st} channel "
                        "(probabilistic; per-population fixed-peak basis) ==\n")
                f.write("quantity   observed       median  68% interval        95% interval        mode\n")
                f.write("p          %9.4f   %9.4f  [%6.4f, %6.4f]  [%6.4f, %6.4f]  %6.4f\n"
                        % (po.observed[0], po.p_median, po.p_lo68, po.p_hi68,
                           po.p_lo95, po.p_hi95, po.p_map))
                f.write("beta_deg   %9.2f   %9.2f  [%6.2f, %6.2f]  [%6.2f, %6.2f]  %6.2f\n"
                        % (po.observed[1], po.b_median, po.b_lo68, po.b_hi68,
                           po.b_lo95, po.b_hi95, po.b_map))
                if po.multimodal:
                    f.write(f"# WARNING: the posterior is MULTIMODAL ({po.n_modes} modes) — the "
                            "p-beta degeneracy admits several true populations consistent with "
                            f"this recovery; see posterior_correction_{st}.png before quoting a "
                            "single value.\n")
                edge = (po.p_median <= po.p_grid[0] + 0.02 or po.p_median >= po.p_grid[-1] - 0.02
                        or po.b_median <= po.b_grid[0] + 2 or po.b_median >= po.b_grid[-1] - 2)
                if edge:
                    f.write("# NOTE: the posterior peaks near the edge of the basis grid — "
                            "consider widening basis_p_range / basis_b_range_deg.\n")

            # cross-channel consistency: for a genuinely single-peaked population,
            # the peak- and median-channel posteriors must agree within errors.
            if "peak" in posteriors and "median" in posteriors:
                a, b = posteriors["peak"], posteriors["median"]
                ok_p = max(a.p_lo68, b.p_lo68) <= min(a.p_hi68, b.p_hi68)
                ok_b = max(a.b_lo68, b.b_lo68) <= min(a.b_hi68, b.b_hi68)
                if ok_p and ok_b:
                    f.write("\n# CONSISTENCY: peak- and median-channel posteriors agree "
                            "(68% intervals overlap in both p and beta) — compatible with a "
                            "single-peaked population.\n")
                else:
                    which = ", ".join(x for x, ok in (("p", ok_p), ("beta", ok_b)) if not ok)
                    f.write(f"\n# WARNING: peak- and median-channel posteriors DISAGREE in "
                            f"{which} (68% intervals do not overlap) — the population may be "
                            "skewed or multimodal rather than single-peaked; inspect the "
                            "posterior maps and the population distribution "
                            "(pyleader-unfold).\n")
