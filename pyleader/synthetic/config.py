"""Configuration for the synthetic-validation pipeline.

Replaces the scattered ``P_PEAK`` / ``B_PEAK`` / ``Nkierroksia`` script variables
of the MATLAB ``leader_synth_main_WISE.m`` and its helpers.
"""

from __future__ import annotations

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
    p_peak: Optional[float] = None
    b_peak: Optional[float] = None

    # --- sample size / tolerances ---
    Ndraws: int = 1000                       # number of synthetic objects (Nkierroksia)
    date_tol: float = 60.0                   # max JD gap within an apparition
    wanted: int = 5                          # min points per apparition for an eta estimate
    phase_angle_limit: float = 30.0          # synth uses 30 deg (vs. 40 for the real analysis)

    # --- brightness model ---
    noise_level: float = 0.01                # fractional Gaussian noise added to L
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

    # --- data paths ---
    damit_list: str = _DAMIT_LIST
    damit_dir: str = _DAMIT_MODELS_DIR
    geometry_dir: str = f"{_LEADER_MASTER}/WISE_3band_subset/WISE/WISE_3band/obs"
    # Explicit list of .obs geometry files; when set it overrides geometry_dir
    # (used by the per-population pipeline to match the analyzed objects).
    geometry_files: Optional[list] = None
    base_dir: str = DEFAULT_BASE_DIR
    outdir: Optional[str] = None             # defaults via `resolved_outdir`

    def __post_init__(self) -> None:
        if self.scattering not in ("ls_lambert", "hapke"):
            raise ValueError(
                f"scattering must be 'ls_lambert' or 'hapke', got {self.scattering!r}"
            )

    @property
    def resolved_outdir(self) -> str:
        """Output directory for this run (labeled by the assigned peaks when fixed)."""
        if self.outdir is not None:
            return self.outdir
        tag = ""
        if self.p_peak is not None and self.b_peak is not None:
            tag = f"_p{self.p_peak:.2f}_b{self.b_peak:.2f}"
        return f"{self.base_dir}/synthetic_validation{tag}"
