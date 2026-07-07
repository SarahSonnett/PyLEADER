#!/usr/bin/env python
"""CLI entry point for building LEADER ``.obs`` input files.

Replaces ``make_LEADER_obs_files.ipynb``.  Queries IRSA for each member of a
collisional family and JPL Horizons for the observing geometry, then writes one
``.obs`` file per object.  Requires ``requests``, ``astropy`` and ``sunpy``
(``pip install -r requirements.txt``) and internet access.

Examples::

    python scripts/build_obs_files.py --famid 350
    python scripts/build_obs_files.py --famid 350 --curl-only   # just write the curl script
"""

from __future__ import annotations

import argparse
import os
import sys


from pyleader.config import ObsBuildConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    d = ObsBuildConfig()
    p = argparse.ArgumentParser(description="Build LEADER .obs files from IRSA + JPL Horizons.")
    p.add_argument("--famid", default=d.famid, help="collisional family identifier")
    p.add_argument("--cat", default=d.cat, help="IRSA catalog to query")
    p.add_argument("--min-obs", type=int, default=d.min_obs, help="min observations to write a .obs file")
    p.add_argument("--istart", type=int, default=d.istart, help="index to resume from")
    p.add_argument("--overwrite", action="store_true", default=d.overwrite, help="overwrite existing data dir")
    p.add_argument("--filterpriority", default=d.filterpriority, help="filter to analyze (lowercase)")
    p.add_argument("--family-file", default=d.family_file, help="MBA family membership file (name or absolute path)")
    p.add_argument("--neowise-fle", default=d.neowise_fle, help="NEOWISE catalog file (name or absolute path)")
    p.add_argument("--base-dir", default=d.base_dir, help="root working directory for inputs/outputs")
    p.add_argument("--population", dest="population_kind", choices=("family", "background"),
                   default=d.population_kind, help="directory-naming scheme / membership source")
    p.add_argument("--datadir", default=None,
                   help="write .obs files to this exact directory (bypasses the naming convention)")
    p.add_argument("--legacy-format", action="store_true", default=d.legacy_format,
                   help="write .obs in the legacy block format instead of the default tabular format")
    p.add_argument("--curl-only", action="store_true",
                   help="only write the bulk curl-download script, do not query per object")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ObsBuildConfig(
        famid=args.famid,
        cat=args.cat,
        min_obs=args.min_obs,
        istart=args.istart,
        overwrite=args.overwrite,
        filterpriority=args.filterpriority,
        family_file=args.family_file,
        neowise_fle=args.neowise_fle,
        base_dir=args.base_dir,
        population_kind=args.population_kind,
        data_dir_override=args.datadir,
        legacy_format=args.legacy_format,
    )

    if args.curl_only:
        from pyleader.obsfiles.curl import write_curl_script
        path = write_curl_script(cfg)
        print(f"Wrote curl script to: {path}")
        return 0

    from pyleader.obsfiles.build import build_obs_files
    outdir = build_obs_files(cfg)
    print(f"Done. .obs files written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
