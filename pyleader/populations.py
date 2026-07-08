"""Resolve a dynamical-population ID to the objects to query for photometry.

Two kinds of population are supported, distinguished by the ID:

* **collisional family** (a numeric / family ID, e.g. ``"3815"``) — members come
  from a family-membership listing (``AllMBAFamilyMembers.txt``, Nesvorný et al.)
  cross-matched against the NEOWISE catalog to recover query designations.
* **background population** (``BG_<REGION>_<TYPE>types``, e.g. ``"BG_IB_Ctypes"``)
  — members come directly from the already NEOWISE-matched
  ``BGobjs_<REGION>_<TYPE>type_neowise.txt`` file.

Both return the same pair used by the obs-builder:
``(matchids, matchids_curlformat)`` — the packed MPC names and the IDs to hand
to IRSA (numbered designation when available, else provisional designation).
"""

from __future__ import annotations

import os

import numpy as np

from .config import ObsBuildConfig, require_neowise, resolve_data_file


def is_background(pop_id: str) -> bool:
    """True for background-population IDs (``BG_...``)."""
    return str(pop_id).upper().startswith("BG_")


def background_neowise_path(cfg: ObsBuildConfig) -> str:
    """Map a background ID to its ``BGobjs_<REGION>_<TYPE>type_neowise.txt`` file.

    ``BG_IB_Ctypes`` -> ``BGobjs_IB_Ctype_neowise.txt``, resolved from
    ``base_dir`` if present there, else from the copy shipped with the package.
    """
    parts = cfg.famid.split("_")           # ["BG", "IB", "Ctypes"]
    if len(parts) < 3:
        raise ValueError(
            f"Background id {cfg.famid!r} must look like BG_<REGION>_<TYPE>types"
        )
    region = parts[1]
    typ = parts[2].rstrip("s")             # "Ctypes" -> "Ctype"
    return resolve_data_file(f"BGobjs_{region}_{typ}_neowise.txt", cfg.base_dir)


def _resolve_background(cfg: ObsBuildConfig):
    """Members of a background population, from its BGobjs neowise listing.

    Columns: ``objnum provdesig packed_name diam diamerr``.
    """
    path = background_neowise_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Background membership file not found: {path}")

    objnum, provdesig, packed = np.genfromtxt(
        path, unpack=True, usecols=(0, 1, 2), dtype=str
    )
    matchids = np.asarray([p.replace('"', "").strip() for p in np.atleast_1d(packed)])
    objnum = np.atleast_1d(objnum)
    provdesig = np.atleast_1d(provdesig)

    curl = np.where(objnum == "0", provdesig, objnum)
    curl = np.asarray([c.replace('"', "").replace(" ", "") for c in curl])
    print(f"{len(matchids)} objects in background population {cfg.famid}")
    return matchids, curl


def _resolve_family(cfg: ObsBuildConfig):
    """Members of a collisional family: membership listing x NEOWISE catalog.

    Ports the cross-match previously in ``obsfiles.build.prepare_matchids``.
    """
    famid_all, mpecobj_all, _objid_all = np.genfromtxt(
        cfg.family_path, unpack=True, dtype=str, usecols=(0, 1, 2)
    )
    objid_mpec = mpecobj_all.compress((famid_all == cfg.famid).flat)
    print("Number of asteroids in this family = " + str(len(objid_mpec)))

    require_neowise(cfg.neowise_path)
    objnum_n, provdesig_n, name_mpced_n, diam_n, diamerr_n = np.genfromtxt(
        cfg.neowise_path, unpack=True, usecols=(0, 1, 2, 11, 12), delimiter=",", dtype=str
    )

    # consolidate duplicate diameter determinations: keep smallest error
    u_objnum_n, u_provdesig_n, u_name_mpced_n = [], [], []
    for name in np.asarray(list(set(name_mpced_n))):
        imatch = np.where(name_mpced_n == name)[0]
        if len(imatch) > 1:
            ibest_group = np.where(diamerr_n[imatch] == min(diamerr_n[imatch]))[0]
            if len(ibest_group) > 1:
                ibest_group = ibest_group[0]
            ibest = imatch[ibest_group]
        elif len(imatch) == 1:
            ibest = imatch
        else:
            continue
        u_objnum_n.append(objnum_n[ibest][0])
        u_provdesig_n.append(provdesig_n[ibest][0].replace('"', "").replace(" ", ""))
        u_name_mpced_n.append(name_mpced_n[ibest][0].replace('"', "").rstrip())

    u_objnum_n = np.asarray(u_objnum_n)
    u_provdesig_n = np.asarray(u_provdesig_n)
    u_name_mpced_n = np.asarray(u_name_mpced_n)

    matchids, matchids_curlformat = [], []
    for target in objid_mpec:
        imatch = np.where(u_name_mpced_n == target)[0]
        if len(imatch) > 0:
            matchids.append(u_name_mpced_n[imatch][0])
            if u_objnum_n[imatch] == "0":
                matchids_curlformat.append(u_provdesig_n[imatch][0])
            else:
                matchids_curlformat.append(u_objnum_n[imatch][0])

    matchids = np.asarray(matchids)
    matchids_curlformat = np.asarray(matchids_curlformat)
    print(str(len(matchids)) + " asteroids in this family found in the WISE/NEOWISE catalog")
    return matchids, matchids_curlformat


def resolve_members(cfg: ObsBuildConfig):
    """Return ``(matchids, matchids_curlformat)`` for the configured population."""
    if is_background(cfg.famid):
        return _resolve_background(cfg)
    return _resolve_family(cfg)
