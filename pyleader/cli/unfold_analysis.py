#!/usr/bin/env python
"""Estimate the true population distribution f(p, β) by unfolding a LEADER analysis.

Uses a fixed-peak-basis response matrix (Step 4b's basis; build one with
``pyleader-basis``) to invert a real analysis into an estimate of the
population's true f(p, β), with per-bin 16–84% uncertainty bands from a
perturbation ensemble.

Two response spaces (``--space``):

* ``cdf`` (default) — columns are the basis units' simulated amplitude CDFs and
  the observation is the population's pooled amplitude CDF. Exactly linear in
  mixtures, so the W-space inversion model error is removed by construction.
  Needs saved amplitude samples: a basis built from 2026-07-08 on, and either
  an analysis that saved per-trial ``Asort`` or ``--obsdir`` to recompute the
  observed CDF from the ``.obs`` files directly.
* ``w`` — the original response over recovered joint solutions (works with any
  basis/analysis; carries a small measured mixture-linearity model error).

Example::

    python scripts/unfold_analysis.py \
        /path/to/Fam1128_analysis_..._1.0km_to_100.0km \
        --basis /path/to/Fam1128_analysis_..._basis
"""

from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.unfold import (  # noqa: E402
    build_response, build_response_cdf, observed_cdf_from_analysis,
    observed_cdf_from_obs, observed_from_analysis, plot_unfolded, unfold,
    unfold_cdf,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unfold a LEADER analysis into f_true(p, beta).")
    p.add_argument("analysis_outdir", help="analysis output directory (containing Trial*/W_trial*.npz)")
    p.add_argument("--basis", default=None,
                   help="fixed-peak basis directory (default: '<analysis_outdir>_basis')")
    p.add_argument("--space", choices=("cdf", "w"), default="cdf",
                   help="response space (default cdf: exactly linear in mixtures; "
                        "'w' reproduces the original behaviour)")
    p.add_argument("--obsdir", default=None,
                   help="(cdf) recompute the observed amplitude CDF from this .obs directory "
                        "instead of the analysis's saved per-trial amplitudes")
    p.add_argument("--wanted", type=int, default=5,
                   help="(cdf, with --obsdir) min points per apparition — match the analysis")
    p.add_argument("--date-tol", type=float, default=60.0,
                   help="(cdf, with --obsdir) apparition gap in days — match the analysis")
    p.add_argument("--phase-angle-limit", type=float, default=40.0,
                   help="(cdf, with --obsdir) max solar phase angle — match the analysis")
    p.add_argument("--n-ensemble", type=int, default=40, help="perturbation re-solves for error bands")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    outdir = args.analysis_outdir.rstrip("/")
    basis = args.basis or f"{outdir}_basis"
    if not os.path.isdir(basis):
        raise SystemExit(f"Basis directory not found: {basis}\n"
                         f"Build it first:  pyleader-basis <pop_id> ... --outdir {basis}")

    if args.space == "cdf":
        resp = build_response_cdf(basis)
        if args.obsdir is not None:
            from pyleader.config import AnalysisConfig
            acfg = AnalysisConfig(famid="_", wanted=args.wanted, date_tol=args.date_tol,
                                  phase_angle_limit=args.phase_angle_limit)
            files = sorted(glob.glob(os.path.join(args.obsdir, "*.obs")))
            files = [f for f in files if not os.path.basename(f).startswith("Nofilter")]
            cdf, std = observed_cdf_from_obs(files, acfg, resp.a_grid, seed=args.seed)
        else:
            cdf, std = observed_cdf_from_analysis(outdir, resp.a_grid)
        res = unfold_cdf(cdf, std, resp, n_ensemble=args.n_ensemble, seed=args.seed)
        space_label = "CDF"
    else:
        resp = build_response(basis)
        W_obs = observed_from_analysis(outdir)
        res = unfold(W_obs, resp, n_ensemble=args.n_ensemble, seed=args.seed)
        space_label = "W"

    summary_dir = os.path.join(outdir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    npz = os.path.join(summary_dir, "population_distribution.npz")
    png = os.path.join(summary_dir, "population_distribution.png")
    res.save(npz)
    plot_unfolded(res, png, space=space_label)
    print(f"Unfolded f(p, beta) [{space_label}-space]: "
          f"relerr={res.relerr:.4f}, alpha={res.alpha:.3g}")
    print(f"  population median p    = {res.pop_median_p:.3f} "
          f"[{res.pop_median_p_lo:.3f}, {res.pop_median_p_hi:.3f}] (16-84%, statistical only)")
    print(f"  population median beta = {res.pop_median_b:.1f} deg "
          f"[{res.pop_median_b_lo:.1f}, {res.pop_median_b_hi:.1f}] (16-84%, statistical only)")
    print(f"  {npz}\n  {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
