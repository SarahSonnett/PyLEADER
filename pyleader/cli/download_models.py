#!/usr/bin/env python
"""Download the representative DAMIT shape models listed in ``asteroideja.txt``.

By default this fetches only the models not already present in the target
directory (so it is a safe "get me the models" step on a fresh checkout). Pass
``--refresh`` to re-download the current DAMIT version of *every* listed model,
refreshing them to the latest published shapes.

Examples::

    python scripts/download_models.py            # fetch any missing models
    python scripts/download_models.py --refresh     # refresh all to the latest DAMIT versions
"""

from __future__ import annotations

import argparse

from pyleader.synthetic.config import SyntheticConfig
from pyleader.synthetic.damit import download_damit_models, parse_model_list


def build_parser() -> argparse.ArgumentParser:
    d = SyntheticConfig()
    p = argparse.ArgumentParser(description="Download the DAMIT models listed in asteroideja.txt.")
    p.add_argument("--damit-list", default=d.damit_list,
                   help="asteroideja.txt listing (defaults to the copy shipped with the package)")
    p.add_argument("--damit-dir", default=d.damit_dir, help="destination directory for the models")
    p.add_argument("--refresh", action="store_true",
                   help="re-download every listed model (refresh to the latest DAMIT versions)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    numbers = parse_model_list(args.damit_list)
    print(f"{len(numbers)} models listed in {args.damit_list}")
    ok = download_damit_models(numbers, args.damit_dir, force=args.refresh)
    print(f"{len(ok)}/{len(numbers)} models available in {args.damit_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
