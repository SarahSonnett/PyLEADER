"""Synthetic validation of the LEADER method.

Build a synthetic asteroid population with *assigned* shape-elongation and
spin-latitude distributions (from DAMIT shape models + real WISE observing
geometry), recover them with the standard LEADER inversion, and compare
recovered vs. assigned — validating the method and/or deriving a correction
function for real-data results.

Modularized form of the MATLAB ``leader_synth_main_WISE.m`` and helpers.
"""

from __future__ import annotations

from .config import SyntheticConfig
from .population import run_synthetic, SyntheticResult
from .compare import ks_comparison, compare_populations

__all__ = [
    "SyntheticConfig",
    "run_synthetic",
    "SyntheticResult",
    "ks_comparison",
    "compare_populations",
]
