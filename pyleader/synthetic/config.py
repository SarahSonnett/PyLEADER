"""Configuration for the synthetic-validation pipeline.

Replaces the scattered ``P_PEAK`` / ``B_PEAK`` / ``Nkierroksia`` script variables
of the MATLAB ``leader_synth_main_WISE.m`` and its helpers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from ..config import DEFAULT_BASE_DIR

# Location of the original MATLAB LEADER release, which ships the example DAMIT
# model listing (asteroideja.txt) and a subset of WISE geometry .obs files.
_LEADER_MASTER = f"{DEFAULT_BASE_DIR}/LEADER-master"

# Repository root (pyleader/synthetic/config.py -> PyLEADER/). The downloaded
# DAMIT shape models live in <repo>/damit_models so all inputs are centralized.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DAMIT_MODELS_DIR = os.path.join(_REPO_ROOT, "damit_models")

# Package data dir; ships the DAMIT model listing (asteroideja.txt) + default correction.
_SYN_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DAMIT_LIST = os.path.join(_SYN_DATA_DIR, "asteroideja.txt")


@dataclass
class SyntheticConfig:
    """Configuration for a synthetic LEADER validation run.

    A synthetic population is built with *assigned* shape-elongation and
    spin-latitude peaks (``p_peak``, ``b_peak``); the same inversion used for
    real data then attempts to recover them.
    """

    # --- assigned (true) distribution peaks ---
    # None => randomized per the MATLAB rules (0.6*rand+0.35 for p, 1.5*rand+0.05 for beta).
    # NOTE: b_peak is in RADIANS at the config/API level (all internal math is
    # radians); the command-line interfaces accept degrees and convert.
    p_peak: Optional[float] = None
    b_peak: Optional[float] = None

    # --- sample size / tolerances ---
    Ndraws: int = 1000                       # number of synthetic objects (Nkierroksia)
    date_tol: float = 60.0                   # max JD gap within an apparition
    wanted: int = 5                          # min points per apparition for an eta estimate
    phase_angle_limit: float = 30.0          # synth uses 30 deg (vs. 40 for the real analysis)

    # --- brightness model ---
    # Flat fractional Gaussian noise on L (the original LEADER release's 1%).
    # Superseded when `noise_model` holds a fitted empirical NoiseModel
    # (pyleader.synthetic.noise): then each epoch's noise follows the
    # population's own flux-fluxerr relation instead.
    noise_level: float = 0.01
    noise_model: Optional[object] = None     # NoiseModel or None (flat noise)
    scattering: str = "ls_lambert"           # "ls_lambert" (default) or "hapke"
    hapke_param: Tuple[float, float, float, float] = (0.63, 0.04, 1.4, -0.4)
    hapke_rough: float = 20.0
    trot_min_hr: float = 3.0                 # rotation period range (hours)
    trot_max_hr: float = 12.0

    # --- shape / spin sampling rules ---
    p_accept_tol: float = 0.075              # accept a stretched model if |p - p_peak| <= this
    p_escape_chance: float = 0.1             # chance to accept an off-target p ...
    p_escape_min: float = 0.45               # ... as long as p exceeds this
    beta_peak_chance: float = 0.75           # chance beta is drawn near b_peak (else uniform)
    beta_jitter: float = 0.05                # stddev of the Gaussian around b_peak

    # --- inversion / output ---
    convert2degrees: bool = True
    deltaP: float = 0.1
    deltaB: float = 1.0
    # Randomly perturb the inversion's recovered (P, BETA) bin grids (the
    # original LEADER behaviour). Basis runs disable this so every run shares
    # one canonical grid (required to stack them into a forward model).
    grid_jitter: bool = True

    # --- data paths ---
    damit_list: str = _DAMIT_LIST
    damit_dir: str = _DAMIT_MODELS_DIR
    geometry_dir: str = f"{_LEADER_MASTER}/WISE_3band_subset/WISE/WISE_3band/obs"
    # Explicit list of .obs geometry files; when set it overrides geometry_dir
    # (used by the per-population pipeline to match the analyzed objects).
    geometry_files: Optional[list] = None
    # Optional callable returning one (p_target, beta_rad) draw per object —
    # lets validation runs assign an arbitrary (mixture/broad) true population
    # instead of the single-peak rules. Overrides p_peak/b_peak sampling.
    truth_sampler: Optional[object] = None
    base_dir: str = DEFAULT_BASE_DIR
    outdir: Optional[str] = None             # defaults via `resolved_outdir`

    def __post_init__(self) -> None:
        if self.scattering not in ("ls_lambert", "hapke"):
            raise ValueError(
                f"scattering must be 'ls_lambert' or 'hapke', got {self.scattering!r}"
            )

    @classmethod
    def fixed_peak_preset(cls, p_peak: float, b_peak: float, **kwargs) -> "SyntheticConfig":
        """A fixed-peak synthetic population: every object at one assigned ``(p, beta)``.

        (A near-delta distribution, in statistical terms.) Used for the correction
        basis runs: essentially all objects sit at the assigned peak (no
        uniform-beta background, tight tolerances), and the
        inversion grid jitter is disabled so all runs share the canonical
        recovered grid. ``b_peak`` is in radians (as at the API level).
        """
        defaults = dict(
            p_peak=p_peak, b_peak=b_peak,
            p_accept_tol=0.02, p_escape_chance=0.0,
            beta_peak_chance=1.0, beta_jitter=0.01,
            grid_jitter=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @property
    def resolved_outdir(self) -> str:
        """Output directory for this run (labeled by the assigned peaks when fixed)."""
        if self.outdir is not None:
            return self.outdir
        tag = ""
        if self.p_peak is not None and self.b_peak is not None:
            tag = f"_p{self.p_peak:.2f}_b{math.degrees(self.b_peak):.0f}deg"
        return f"{self.base_dir}/synthetic_validation{tag}"
