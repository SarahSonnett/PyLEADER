"""Read and fetch DAMIT asteroid shape models.

Ports ``damit_model.m`` (the vertex/facet reader) and adds a downloader for the
models named in the MATLAB ``asteroideja.txt`` listing, since only the Juno
sample ships with the LEADER release.

DAMIT shape models (OBJ format) are served at::

    https://damit.cuni.cz/projects/damit/generated_files/open/AsteroidModel/<model_id>/shape.obj

and the asteroid-number -> asteroid-id -> model-id mapping comes from the CSV
table exports under ``.../projects/damit/exports/table/``.

If you use DAMIT models, cite Ďurech et al. (2010) and the original model
papers (see the repository README).
"""

from __future__ import annotations

import csv
import io
import os
import re
import urllib.request

import numpy as np

_DAMIT = "https://damit.cuni.cz/projects/damit"
_ASTEROIDS_CSV = f"{_DAMIT}/exports/table/asteroids"
_MODELS_CSV = f"{_DAMIT}/exports/table/asteroid_models"
_SHAPE_URL = _DAMIT + "/generated_files/open/AsteroidModel/{model_id}/shape.obj"
_UA = {"User-Agent": "PyLEADER/0.1 (synthetic validation; +https://github.com/SarahSonnett/PyLEADER)"}


def read_damit_model(path):
    """Read an asteroid shape model file; return ``(x, y, z, F)``.

    Handles the DAMIT OBJ format (lines ``v x y z`` / ``f i j k``) used by the
    MATLAB code, and also the DAMIT ``shape.txt`` format (a ``Nv Nf`` count
    header followed by vertex then facet rows). Vertices are scaled by their
    mean absolute value (as in ``damit_model.m``) and ``F`` is returned as
    0-based integer vertex indices.
    """
    with open(path, "r") as fid:
        raw = [ln.strip() for ln in fid if ln.strip()]

    first = raw[0].split()
    is_obj = first and first[0] in ("v", "f")

    V, F = [], []
    if is_obj:
        for ln in raw:
            parts = ln.split()
            tag = parts[0]
            if tag == "v":
                V.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f":
                # OBJ facets may be "i", "i/j", or "i//k" — take the vertex index
                F.append([int(p.split("/")[0]) for p in parts[1:4]])
    else:
        # count-header format: "Nv Nf", then Nv vertices, then Nf facets
        nv, nf = int(first[0]), int(first[1])
        for ln in raw[1:1 + nv]:
            p = ln.split()
            V.append([float(p[0]), float(p[1]), float(p[2])])
        for ln in raw[1 + nv:1 + nv + nf]:
            p = ln.split()
            F.append([int(float(p[0])), int(float(p[1])), int(float(p[2]))])

    V = np.asarray(V, dtype=float)
    V = V / np.mean(np.abs(V))
    F = np.asarray(F, dtype=int) - 1  # 1-based (file) -> 0-based (numpy)
    return V[:, 0], V[:, 1], V[:, 2], F


def parse_model_list(listing_path):
    """Parse a MATLAB ``asteroideja.txt`` listing into asteroid numbers.

    Entries look like ``Dimensiot\\10.txt``; returns ``[10, 1002, ...]``.
    """
    numbers = []
    with open(listing_path, "r") as f:
        text = f.read()
    for token in re.split(r"[\r\n]+", text):
        token = token.strip()
        if not token:
            continue
        base = token.replace("\\", "/").split("/")[-1]        # -> "10.txt"
        stem = os.path.splitext(base)[0]
        if stem.isdigit():
            numbers.append(int(stem))
    return numbers


def _fetch_text(url, timeout=120):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8-sig", "replace")


def _load_number_to_model(timeout=120):
    """Build ``{asteroid_number: first_model_id}`` from the DAMIT table exports."""
    # asteroids: number -> id
    num_to_astid = {}
    for row in csv.DictReader(io.StringIO(_fetch_text(_ASTEROIDS_CSV, timeout))):
        num = (row.get("number") or "").strip()
        if num.isdigit():
            num_to_astid[int(num)] = row["id"]

    # asteroid_models: asteroid_id -> first (lowest) model id
    astid_to_model = {}
    for row in csv.DictReader(io.StringIO(_fetch_text(_MODELS_CSV, timeout))):
        astid = row["asteroid_id"]
        mid = int(row["id"])
        if astid not in astid_to_model or mid < astid_to_model[astid]:
            astid_to_model[astid] = mid

    return {
        num: astid_to_model[astid]
        for num, astid in num_to_astid.items()
        if astid in astid_to_model
    }


def download_damit_models(numbers, dest, *, timeout=120, verbose=True):
    """Download one DAMIT shape model per asteroid number into ``dest``.

    Files are written as ``<number>.txt`` in OBJ format (matching the LEADER
    listing). Existing files are not re-downloaded. Numbers without an
    accessible model are skipped. Returns the list of numbers now available
    locally in ``dest``.
    """
    os.makedirs(dest, exist_ok=True)

    wanted = list(dict.fromkeys(numbers))  # de-dupe, keep order
    missing = [n for n in wanted if not os.path.exists(os.path.join(dest, f"{n}.txt"))]

    available = [n for n in wanted if os.path.exists(os.path.join(dest, f"{n}.txt"))]
    if not missing:
        if verbose:
            print(f"All {len(wanted)} models already present in {dest}")
        return available

    if verbose:
        print(f"Resolving DAMIT model ids for {len(missing)} asteroids...")
    num_to_model = _load_number_to_model(timeout)

    ok, fail = list(available), []
    for n in missing:
        mid = num_to_model.get(n)
        if mid is None:
            fail.append(n)
            continue
        url = _SHAPE_URL.format(model_id=mid)
        try:
            text = _fetch_text(url, timeout)
            with open(os.path.join(dest, f"{n}.txt"), "w") as f:
                f.write(text)
            ok.append(n)
        except Exception as exc:  # network / 404 / parse
            if verbose:
                print(f"  skip {n}: {exc}")
            fail.append(n)

    if verbose:
        print(f"Downloaded {len(ok) - len(available)} new models "
              f"({len(ok)} available, {len(fail)} unavailable) in {dest}")
    return ok
