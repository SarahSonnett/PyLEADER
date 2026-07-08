#!/usr/bin/env python
"""Unfold a LEADER analysis into the true joint distribution f(p, β).

Uses a delta-basis response matrix (Step 4b's basis; build one with
``pyleader-basis``) to invert the recovered joint solution of a real analysis
into an estimate of the population's true f(p, β), with per-bin 16–84%
uncertainty bands from a perturbation ensemble.

Example::

    python scripts/unfold_analysis.py \
        /path/to/Fam1128_analysis_..._1.0km_to_100.0km \
        --basis /path/to/Fam1128_analysis_..._basis
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.unfold import (  # noqa: E402
    build_response, observed_from_analysis, plot_unfolded, unfold,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Unfold a LEADER analysis into f_true(p, beta).")
    p.add_argument("analysis_outdir", help="analysis output directory (containing Trial*/W_trial*.npz)")
    p.add_argument("--basis", default=None,
                   help="delta-basis directory (default: '<analysis_outdir>_basis')")
    p.add_argument("--n-ensemble", type=int, default=40, help="perturbation re-solves for error bands")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    outdir = args.analysis_outdir.rstrip("/")
    basis = args.basis or f"{outdir}_basis"
    if not os.path.isdir(basis):
        raise SystemExit(f"Basis directory not found: {basis}\n"
                         f"Build it first:  pyleader-basis <pop_id> ... --outdir {basis}")

    resp = build_response(basis)
    W_obs = observed_from_analysis(outdir)
    res = unfold(W_obs, resp, n_ensemble=args.n_ensemble, seed=args.seed)

    npz = os.path.join(outdir, "unfolded_fpb.npz")
    png = os.path.join(outdir, "unfolded_fpb.png")
    res.save(npz)
    plot_unfolded(res, png)
    print(f"Unfolded f(p, beta): relerr={res.relerr:.4f}, alpha={res.alpha:.3g}")
    print(f"  population median p    = {res.pop_median_p:.3f} "
          f"[{res.pop_median_p_lo:.3f}, {res.pop_median_p_hi:.3f}] (16-84%, statistical only)")
    print(f"  population median beta = {res.pop_median_b:.1f} deg "
          f"[{res.pop_median_b_lo:.1f}, {res.pop_median_b_hi:.1f}] (16-84%, statistical only)")
    print(f"  {npz}\n  {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
