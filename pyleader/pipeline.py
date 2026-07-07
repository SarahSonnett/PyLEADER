"""End-to-end per-population driver.

Given a dynamical-population ID (a Nesvorný collisional family or a background
MBA population), this chains the whole workflow:

    build .obs  ->  LEADER analysis  ->  synthetic sweep on *this population's*
    observing geometry  ->  fit a population-specific correction  ->  apply it.

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
from .synthetic.sweep import run_sweep


@dataclass
class PopulationConfig:
    """Everything needed to analyze one population end-to-end."""

    pop_id: str
    population_kind: Optional[str] = None      # inferred from pop_id if None
    cat: str = "allsky_4band_p1bs_psd"
    filterpriority: str = "w3"
    diam_low: float = 5.0
    diam_high: float = 10.0

    # analysis
    Ntrials: int = 100
    Ndraws: int = 1000
    phase_angle_limit: float = 40.0
    date_tol: float = 60.0
    wanted: int = 5
    convert2degrees: bool = True
    overwrite: bool = True
    neowise_fle: str = "neowise_mainbelt.csv"

    # synthetic sweep (correction)
    p_peaks: tuple = (0.35, 0.45, 0.55, 0.65, 0.75)
    b_peaks: tuple = (0.2, 0.5, 0.9, 1.3)
    sweep_ndraws: int = 1000
    nseeds: int = 1
    scattering: str = "ls_lambert"
    correction_stat: str = "peak"

    base_dir: str = None
    # Arbitrary .obs directory (bypasses the naming convention). Used for both
    # building (Stage 1) and reading (Stage 2), and the sweep geometry follows.
    obsdir: Optional[str] = None

    def __post_init__(self):
        if self.population_kind is None:
            self.population_kind = "background" if is_background(self.pop_id) else "family"
        if self.base_dir is None:
            from .config import DEFAULT_BASE_DIR
            self.base_dir = DEFAULT_BASE_DIR

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
            Ndraws=self.sweep_ndraws, scattering=self.scattering,
            phase_angle_limit=self.phase_angle_limit, date_tol=self.date_tol, wanted=self.wanted,
            convert2degrees=self.convert2degrees, geometry_files=list(geometry_files),
            base_dir=self.base_dir,
        )


@dataclass
class PopulationResult:
    pop_id: str
    outdir: str
    recovered: tuple                # (p, beta_deg) LEADER peak, averaged over trials
    corrected: tuple                # (p, beta_deg) after the population correction
    correction_path: str
    sweep_csv: str
    r2: tuple                       # (r2_p, r2_beta) of the correction fit


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


def _recovered_peak(analysis_outdir: str, pop_id: str):
    """Average LEADER peak (pmax, betamax_deg) across trials from the summary file."""
    summary = os.path.join(analysis_outdir, f"SummaryAnalysis_Famid{pop_id}.txt")
    pmax, betamax = np.genfromtxt(summary, unpack=True, usecols=(1, 2), dtype=float, skip_header=1)
    return float(np.mean(np.atleast_1d(pmax))), float(np.mean(np.atleast_1d(betamax)))


def run_population(cfg: PopulationConfig, *, do_build: bool = False,
                   refresh_models: bool = False, seed: int | None = None) -> PopulationResult:
    """Run the full per-population pipeline; returns a :class:`PopulationResult`.

    DAMIT shape models are assumed to already exist in the models directory;
    pass ``refresh_models=True`` to re-download the current DAMIT versions of the
    models listed in ``asteroideja.txt`` before the correction sweep.
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
    rec_p, rec_b = _recovered_peak(outdir, cfg.pop_id)

    # 3. synthetic sweep on THIS population's observing geometry
    geom = diameter_matched_files(acfg)
    print(f"[{cfg.pop_id}] correction sweep on {len(geom)} population geometries ...")
    sweep_dir = os.path.join(outdir, "correction_sweep")
    base_syn = cfg.synthetic_base(geom)
    sweep_csv = run_sweep(base_syn, cfg.p_peaks, cfg.b_peaks,
                          nseeds=cfg.nseeds, seed=(seed or 0), outdir=sweep_dir, verbose=False)

    # surface the recovered-vs-assigned summary figure at the top of the output dir
    summary_src = os.path.join(sweep_dir, "sweep_summary.png")
    if os.path.exists(summary_src):
        shutil.copy(summary_src, os.path.join(outdir, "sweep_summary.png"))

    # 4. fit the population-specific correction
    coeffs = fit_from_csv(sweep_csv, stat=cfg.correction_stat)
    corr_path = os.path.join(outdir, "correction_function.json")
    save_correction(coeffs, corr_path)
    plot_correction_fit(sweep_csv, coeffs, os.path.join(outdir, "correction_fit.png"))

    # 5. apply it to the population's recovered peak
    cor_p, cor_b = apply_correction([rec_p], [rec_b], coeffs)
    cor_p, cor_b = float(cor_p[0]), float(cor_b[0])

    _write_report(cfg, outdir, (rec_p, rec_b), (cor_p, cor_b), coeffs)

    print(f"[{cfg.pop_id}] recovered (p={rec_p:.3f}, β={rec_b:.1f}°)  ->  "
          f"corrected (p={cor_p:.3f}, β={cor_b:.1f}°)")
    return PopulationResult(
        pop_id=cfg.pop_id, outdir=outdir, recovered=(rec_p, rec_b), corrected=(cor_p, cor_b),
        correction_path=corr_path, sweep_csv=sweep_csv,
        r2=(coeffs["diagnostics"]["r2_p"], coeffs["diagnostics"]["r2_b"]),
    )


def _write_report(cfg, outdir, recovered, corrected, coeffs):
    d = coeffs["diagnostics"]
    pr, br = d["p_rec_range"], d["b_rec_range"]
    p_extrap = not (pr[0] <= recovered[0] <= pr[1])
    b_extrap = not (br[0] <= recovered[1] <= br[1])
    with open(os.path.join(outdir, "population_report.txt"), "w") as f:
        f.write(f"# Population pipeline report: {cfg.pop_id} ({cfg.population_kind})\n")
        f.write(f"# catalog={cfg.cat} filter={cfg.filterpriority} "
                f"diam=[{cfg.diam_low},{cfg.diam_high}] km  Ntrials={cfg.Ntrials} Ndraws={cfg.Ndraws}\n")
        f.write(f"# correction: {cfg.correction_stat}-based, per-population geometry, "
                f"n={d['n']}, terms={coeffs.get('n_terms')}, "
                f"R2_p={d['r2_p']:.3f} R2_beta={d['r2_b']:.3f}\n")
        f.write(f"# synthetic recovered ranges: p in [{pr[0]:.2f},{pr[1]:.2f}], "
                f"beta in [{br[0]:.0f},{br[1]:.0f}] deg\n")
        if p_extrap or b_extrap:
            f.write("# WARNING: recovered %s outside the synthetic range -> correction "
                    "EXTRAPOLATES; treat corrected value(s) with caution\n"
                    % (", ".join([x for x, e in (("p", p_extrap), ("beta", b_extrap)) if e])))
        f.write("\nquantity   recovered   corrected\n")
        f.write("p          %9.4f   %9.4f\n" % (recovered[0], corrected[0]))
        f.write("beta_deg   %9.2f   %9.2f\n" % (recovered[1], corrected[1]))
