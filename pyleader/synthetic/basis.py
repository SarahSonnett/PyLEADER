"""Delta-basis runs: the shared forward-model sampler for the probabilistic
(posterior) correction and the response-matrix unfolding.

One *basis* is a grid of near-delta synthetic populations — each concentrated
at one assigned ``(p_peak, b_peak)`` — observed at the target population's own
geometry and pushed through the full LEADER pipeline, with ``nseeds``
independent realizations per grid point. All runs share the canonical
(un-jittered) recovered bin grid so they can be stacked.

Layout of a basis directory::

    <basis_dir>/
      gp_p00_b00_seed0/synthetic_result.npz     # (grid indices, seed)
      gp_p00_b00_seed1/...
      ...
      basis_info.json                            # grid + config summary

Units already on disk are skipped (resumable); ``task="k/N"`` runs only the
k-th of N chunks (for job arrays / partial reruns). Runs execute in a
``multiprocessing`` pool (they are single-core and independent).
"""

from __future__ import annotations

import glob
import json
import math
import multiprocessing as mp
import os
from dataclasses import replace

import numpy as np

from .config import SyntheticConfig

# Canonical recovered-bin grids (leader_invert with grid_jitter=False).
CANON_P = np.linspace(0.025, 0.975, 20)
CANON_BETA = np.linspace(0.025, 1.55, 29)


def rebin_to_canonical(P, BETA, W):
    """Interpolate a (possibly jittered-grid) solution onto the canonical grid.

    Linear interpolation along each axis; used when correcting *real* analyses
    (whose inversions keep the historical grid jitter) against a basis built on
    the canonical grid. Returns ``W_canon`` with shape (20, 29).
    """
    P = np.asarray(P, float)
    BETA = np.asarray(BETA, float)
    W = np.asarray(W, float)
    # interp along beta (axis 1), then p (axis 0)
    Wb = np.empty((W.shape[0], len(CANON_BETA)))
    for i in range(W.shape[0]):
        Wb[i] = np.interp(CANON_BETA, BETA, W[i])
    Wc = np.empty((len(CANON_P), len(CANON_BETA)))
    for j in range(len(CANON_BETA)):
        Wc[:, j] = np.interp(CANON_P, P, Wb[:, j])
    return Wc


def unit_dir(basis_dir: str, ip: int, ib: int, seed_idx: int) -> str:
    return os.path.join(basis_dir, f"gp_p{ip:02d}_b{ib:02d}_seed{seed_idx}")


def _unit_seed(seed_base: int, ip: int, ib: int, s: int, nb: int, nseeds: int) -> int:
    """Deterministic per-unit RNG seed, stable under chunking/resume order."""
    return seed_base + (ip * nb + ib) * nseeds + s


def _run_unit(args):
    """Pool worker: one delta run. Top-level function so it pickles under spawn."""
    base_cfg, p_peak, b_peak, out, seed = args
    os.environ.setdefault("MPLBACKEND", "Agg")
    from .population import run_synthetic  # import inside worker (spawn-safe)

    cfg = replace(base_cfg,
                  p_peak=p_peak, b_peak=b_peak, outdir=out,
                  p_accept_tol=0.02, p_escape_chance=0.0,
                  beta_peak_chance=1.0, beta_jitter=0.01,
                  grid_jitter=False)
    run_synthetic(cfg, seed=seed, make_plots=False, verbose=False)
    return out


def run_basis(base_cfg: SyntheticConfig, p_grid, b_grid, *,
              nseeds: int = 4, seed: int = 0, outdir: str,
              nproc: int | None = None, task: str | None = None) -> str:
    """Run (or resume) the delta-basis campaign; returns ``outdir``.

    ``base_cfg`` supplies geometry (``geometry_files``/``geometry_dir``),
    ``Ndraws``, scattering, and tolerances; the delta-preset fields and
    ``grid_jitter=False`` are applied per unit. ``b_grid`` is in radians.
    ``nproc`` defaults to (cpu_count - 2); ``task="k/N"`` runs the k-th of N
    contiguous chunks of the remaining units (0-based k).
    """
    os.makedirs(outdir, exist_ok=True)
    p_grid = [float(p) for p in p_grid]
    b_grid = [float(b) for b in b_grid]
    nb = len(b_grid)

    # manifest (written every invocation; cheap and self-describing)
    info = dict(p_grid=p_grid, b_grid_rad=b_grid,
                b_grid_deg=[math.degrees(b) for b in b_grid],
                nseeds=nseeds, Ndraws=base_cfg.Ndraws, seed_base=seed,
                scattering=base_cfg.scattering, wanted=base_cfg.wanted,
                date_tol=base_cfg.date_tol,
                phase_angle_limit=base_cfg.phase_angle_limit)
    with open(os.path.join(outdir, "basis_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # all units, in a stable order
    units = []
    for ip, p_peak in enumerate(p_grid):
        for ib, b_peak in enumerate(b_grid):
            for s in range(nseeds):
                out = unit_dir(outdir, ip, ib, s)
                units.append((ip, ib, s, p_peak, b_peak, out))

    total = len(units)
    pending = [u for u in units
               if not os.path.exists(os.path.join(u[5], "synthetic_result.npz"))]
    done_already = total - len(pending)

    if task is not None:
        k, n = (int(x) for x in task.split("/"))
        chunks = np.array_split(np.arange(len(pending)), n)
        pending = [pending[i] for i in chunks[k]]
        print(f"Basis chunk {k}/{n}: {len(pending)} of the remaining units")

    print(f"Delta basis: {len(p_grid)}x{nb} grid x {nseeds} seed(s) = {total} units "
          f"({done_already} already done, {len(pending)} to run)")
    if not pending:
        return outdir

    jobs = [(base_cfg, p, b, out, _unit_seed(seed, ip, ib, s, nb, nseeds))
            for (ip, ib, s, p, b, out) in pending]

    nproc = nproc or max(1, (os.cpu_count() or 2) - 2)
    ndone = 0
    if nproc == 1:
        for job in jobs:
            _run_unit(job)
            ndone += 1
            print(f"\rBasis: {ndone}/{len(jobs)} ({100.0 * ndone / len(jobs):4.0f}%)",
                  end="", flush=True)
    else:
        ctx = mp.get_context("spawn")  # macOS-safe
        with ctx.Pool(processes=nproc) as pool:
            for _ in pool.imap_unordered(_run_unit, jobs):
                ndone += 1
                print(f"\rBasis: {ndone}/{len(jobs)} ({100.0 * ndone / len(jobs):4.0f}%)"
                      f"  [{nproc} workers]", end="", flush=True)
    print()
    return outdir


def basis_units(basis_dir: str):
    """Yield ``(p_peak, b_peak, seed_idx, npz_path)`` for every completed unit."""
    for path in sorted(glob.glob(os.path.join(basis_dir, "gp_p*_b*_seed*",
                                              "synthetic_result.npz"))):
        d = np.load(path)
        seed_idx = int(os.path.basename(os.path.dirname(path)).split("seed")[-1])
        yield float(d["p_peak"]), float(d["b_peak"]), seed_idx, path
