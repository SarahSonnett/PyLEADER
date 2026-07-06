#!/usr/bin/env python
"""CLI to compare two recovered (p, beta) distributions.

Replaces ``ast_comparison_WISE.m`` + ``KS_comparison.m``. Takes two
``synthetic_result.npz`` files written by ``run_synthetic.py`` (or any analysis
that saved recovered marginals in that format) and reports the L1/L2/L-inf
distances between their marginal CDFs of ``p`` and ``beta``, with plots.

Example::

    python scripts/compare_populations.py \
        run_a/synthetic_result.npz run_b/synthetic_result.npz \
        --outdir comparison_a_vs_b --labels "b=0.3" "b=0.8"
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

from pyleader.synthetic.compare import compare_populations  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare two recovered p/beta distributions.")
    p.add_argument("npz1", help="first synthetic_result.npz")
    p.add_argument("npz2", help="second synthetic_result.npz")
    p.add_argument("--outdir", default="population_comparison", help="output directory for plots")
    p.add_argument("--labels", nargs=2, default=["pop 1", "pop 2"], help="labels for the two populations")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    dP, dB, _ = compare_populations(args.npz1, args.npz2, args.outdir,
                                    labels=tuple(args.labels), show=args.show)
    print("Differences [L1/4, L2, 2*Linf]:")
    print(f"  p    : {dP}")
    print(f"  beta : {dB}")
    print(f"Plots written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
