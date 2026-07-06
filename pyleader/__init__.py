"""PyLEADER: a Python port of the MATLAB LEADER package (Nortunen et al. 2017).

Derives asteroid shape (``p``) and spin-axis (``beta``) distributions from
WISE/NEOWISE thermal photometry, with added diagnostics and error estimation.

This package is the modularized form of the project's original Jupyter
notebooks.  The two entry points are:

* :func:`pyleader.run_analysis` — the LEADER inversion experiment
  (replaces ``LEADER_python_final``/``_bg``/``_forcedN``).
* :func:`pyleader.build_obs_files` — build ``.obs`` input files from IRSA +
  JPL Horizons (replaces ``make_LEADER_obs_files``).

Importing :mod:`pyleader.analysis` only requires numpy/scipy/matplotlib;
:func:`build_obs_files` additionally needs ``astropy``, ``sunpy`` and
``requests`` and is imported lazily so the analysis path works without them.
"""

from __future__ import annotations

from .config import AnalysisConfig, ObsBuildConfig
from .analysis import run_analysis
from .inversion import InversionResult, leader_invert
from .lightcurve import lcg_read_WISE, leader_phasecorr
from .synthetic import SyntheticConfig, run_synthetic, compare_populations
from .pipeline import PopulationConfig, run_population

__all__ = [
    "AnalysisConfig",
    "ObsBuildConfig",
    "run_analysis",
    "build_obs_files",
    "InversionResult",
    "leader_invert",
    "lcg_read_WISE",
    "leader_phasecorr",
    "SyntheticConfig",
    "run_synthetic",
    "compare_populations",
    "PopulationConfig",
    "run_population",
]

__version__ = "0.1.0"


def build_obs_files(cfg: "ObsBuildConfig", **kwargs):
    """Lazy wrapper around :func:`pyleader.obsfiles.build.build_obs_files`.

    Imported on demand so that ``import pyleader`` does not require the
    obs-building dependencies (astropy/sunpy/requests).
    """
    from .obsfiles.build import build_obs_files as _build
    return _build(cfg, **kwargs)
