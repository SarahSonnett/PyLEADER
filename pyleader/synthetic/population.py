"""Synthetic-population validation driver.

Ports ``leader_synth_main_WISE.m``: build a synthetic asteroid population with
assigned shape/spin peaks, recover the (p, beta) distribution with the standard
LEADER inversion, and compare recovered vs. assigned.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import numpy as np

from ..inversion import InversionResult, leader_invert
from .brightness import synthetic_amplitudes
from .config import SyntheticConfig
from .damit import read_damit_model
from .ellipsoid import ellipsoid_properties
from .geometry import read_synth_geometry
from .plots import synthetic_plots
from .stats import compute_stats, write_stats_file


@dataclass
class SyntheticResult:
    """Outcome of one synthetic validation run (also what compare_populations needs)."""

    p_peak: float
    b_peak: float
    p_true: np.ndarray          # assigned shape elongations
    beta_true: np.ndarray       # assigned spin latitudes (radians)
    inversion: InversionResult
    P: np.ndarray               # recovered p grid
    BETA: np.ndarray            # recovered beta grid (radians)
    Pmargin: np.ndarray         # recovered marginal DF of p
    Bmargin: np.ndarray         # recovered marginal DF of beta
    stats: dict                 # min/max/mean/median, assigned vs recovered, p & beta(deg)
    outdir: str

    def save(self, path: str) -> None:
        np.savez(
            path,
            p_peak=self.p_peak, b_peak=self.b_peak,
            p_true=self.p_true, beta_true=self.beta_true,
            P=self.P, BETA=self.BETA, Pmargin=self.Pmargin, Bmargin=self.Bmargin,
            W=self.inversion.W,  # full joint occupation numbers (for posterior/unfolding)
        )

    @staticmethod
    def load_marginals(path: str):
        """Load just the fields needed for a population comparison."""
        d = np.load(path)
        return dict(P=d["P"], BETA=d["BETA"], Pmargin=d["Pmargin"], Bmargin=d["Bmargin"],
                    p_peak=float(d["p_peak"]), b_peak=float(d["b_peak"]))


def _model_files(cfg: SyntheticConfig):
    files = sorted(glob.glob(os.path.join(cfg.damit_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No DAMIT model files in {cfg.damit_dir}. Download them first "
            f"(pyleader.synthetic.damit.download_damit_models)."
        )
    return files


def _geometry_files(cfg: SyntheticConfig):
    if cfg.geometry_files is not None:
        files = list(cfg.geometry_files)
    else:
        files = sorted(glob.glob(os.path.join(cfg.geometry_dir, "*.obs")))
    files = [f for f in files if not os.path.basename(f).startswith("Nofilter")]
    if not files:
        raise FileNotFoundError(
            f"No .obs geometry files ({cfg.geometry_dir if cfg.geometry_files is None else 'geometry_files'})."
        )
    return files


def _draw_shape(model_files, cfg: SyntheticConfig, p_target=None):
    """Pick and stretch a DAMIT model until its elongation matches the target.

    ``p_target`` defaults to ``cfg.p_peak``; a ``truth_sampler`` passes its own
    per-object target instead.
    """
    if p_target is None:
        p_target = cfg.p_peak
    while True:
        fname = random.choice(model_files)
        x, y, z, F = read_damit_model(fname)
        R = np.column_stack([x, y, z])
        # Random stretch, factor >= 1 on each axis
        stretch = np.maximum(1.0, 2.0 * np.abs(np.random.randn(3)))
        R = R * stretch
        props = ellipsoid_properties(R, F)
        if abs(props.p - p_target) <= cfg.p_accept_tol:
            return props
        # small chance to accept an off-target (but still elongated) shape
        if np.random.rand() > (1 - cfg.p_escape_chance) and props.p > cfg.p_escape_min:
            return props


def _draw_beta(cfg: SyntheticConfig) -> float:
    """Assign a spin latitude: near b_peak most of the time, else uniform."""
    if np.random.rand() <= cfg.beta_peak_chance:
        beta = 0.0
        while beta <= 0 or beta >= np.pi / 2:
            beta = cfg.b_peak + cfg.beta_jitter * np.random.randn()
        return beta
    return np.pi / 2 * np.random.rand()


def run_synthetic(cfg: SyntheticConfig, *, seed: int | None = None, show: bool = False,
                  make_plots: bool = True, verbose: bool = True) -> SyntheticResult:
    """Run one synthetic validation experiment; returns a :class:`SyntheticResult`.

    ``make_plots=False`` skips the per-run figures (the stats file and ``.npz``
    are still written) and ``verbose=False`` silences the per-run terminal
    output — both are used by the bias map, which shows a single
    progress bar instead.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Resolve assigned peaks (randomized if not fixed), matching the MATLAB rules
    if cfg.p_peak is None:
        cfg.p_peak = 0.6 * np.random.rand() + 0.35
    if cfg.b_peak is None:
        cfg.b_peak = 1.5 * np.random.rand() + 0.05

    model_files = _model_files(cfg)
    geom_files = _geometry_files(cfg)

    A_tot, p_true, beta_true = [], [], []
    for k in range(cfg.Ndraws):
        if verbose and (k + 1) % 50 == 0:
            print(f"\r  synthetic objects: {k + 1}/{cfg.Ndraws}", end="", flush=True)

        if cfg.truth_sampler is not None:
            p_t, beta = cfg.truth_sampler()
            props = _draw_shape(model_files, cfg, p_target=p_t)
        else:
            props = _draw_shape(model_files, cfg)
            beta = _draw_beta(cfg)

        dates, e_sun, e_earth, ang = read_synth_geometry(
            random.choice(geom_files), cfg.phase_angle_limit
        )
        if len(dates) == 0:
            continue

        A, _ = synthetic_amplitudes(props.normals, props.areas, dates, e_sun, e_earth, ang, beta, cfg)

        A_tot.extend(list(A))
        p_true.append(props.p)
        beta_true.append(beta)

    A_tot = np.asarray(A_tot)
    Asort = np.sort(A_tot)
    CDFA = np.linspace(1 / len(Asort), 1, len(Asort))

    if verbose:
        print()  # end the object-count line
    result = leader_invert(Asort, CDFA, deltaP=cfg.deltaP, deltaB=cfg.deltaB,
                           grid_jitter=cfg.grid_jitter, verbose=verbose)

    Pmargin = np.sum(result.W, axis=1)
    Bmargin = np.sum(result.W, axis=0)

    p_true = np.asarray(p_true)
    beta_true = np.asarray(beta_true)
    stats = compute_stats(p_true, beta_true, result.P, Pmargin, result.BETA, Bmargin)

    outdir = cfg.resolved_outdir
    os.makedirs(outdir, exist_ok=True)

    if make_plots:
        synthetic_plots(result, p_true, beta_true, outdir, stats=stats,
                        convert2degrees=cfg.convert2degrees, show=show)
    write_stats_file(
        os.path.join(outdir, "distribution_stats.txt"), stats,
        label=f"synthetic run: p_peak={cfg.p_peak:.3f}, b_peak={cfg.b_peak:.3f} rad "
              f"({np.rad2deg(cfg.b_peak):.1f} deg), Ndraws={cfg.Ndraws}",
    )

    res = SyntheticResult(
        p_peak=cfg.p_peak, b_peak=cfg.b_peak,
        p_true=p_true, beta_true=beta_true,
        inversion=result, P=result.P, BETA=result.BETA,
        Pmargin=Pmargin, Bmargin=Bmargin, stats=stats, outdir=outdir,
    )
    res.save(os.path.join(outdir, "synthetic_result.npz"))

    # Console summary of recovered vs assigned peaks
    p_rec = result.P[np.argmax(Pmargin)]
    b_rec = np.rad2deg(result.BETA[np.argmax(Bmargin)])
    if verbose:
        print(f"Assigned  peak: p={cfg.p_peak:.3f}, beta={np.rad2deg(cfg.b_peak):.1f} deg")
        print(f"Recovered peak: p={p_rec:.3f}, beta={b_rec:.1f} deg")
        print(f"Output: {outdir}")
    return res
