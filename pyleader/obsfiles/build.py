"""Build LEADER ``.obs`` files for a collisional family.

Modularized form of the main loop (and setup cells) of
``make_LEADER_obs_files.ipynb``.  Requires ``numpy``; the per-object loop also
needs ``requests``/``astropy``/``sunpy`` (imported lazily via the ephemeris and
irsa helpers).
"""

from __future__ import annotations

import os
import shutil

import numpy as np

from ..config import ObsBuildConfig
from .columns import determine_column_indices
from .naming import convert_to_mpecname
from .photometry import convert_mags_to_janskys, replace_null


def _ensure_data_dir(cfg: ObsBuildConfig) -> None:
    """Create the output data directory (notebook cell 3)."""
    try:
        os.mkdir(cfg.data_dir)
    except OSError:
        if cfg.overwrite:
            shutil.rmtree(cfg.data_dir)
            os.mkdir(cfg.data_dir)
        else:
            print(f"{cfg.data_dir} directory already exists")


def prepare_matchids(cfg: ObsBuildConfig):
    """Cross-match the family membership list against the NEOWISE catalog.

    Ports notebook cells 4, 6, 7 and 8.  Returns ``(matchids, matchids_curlformat)``:
    the MPC-ed names found in NEOWISE, and the identifier to use when querying IRSA
    (numbered designation when available, otherwise provisional designation).
    """
    # --- family membership (cell 4) ---
    famid_all, mpecobj_all, objid_all = np.genfromtxt(
        cfg.family_path, unpack=True, dtype=str, usecols=(0, 1, 2)
    )
    objid_mpec = mpecobj_all.compress((famid_all == cfg.famid).flat)
    print("Number of asteroids in this family = " + str(len(objid_mpec)))

    # --- NEOWISE catalog (cell 6) ---
    objnum_n, provdesig_n, name_mpced_n, diam_n, diamerr_n = np.genfromtxt(
        cfg.neowise_path, unpack=True, usecols=(0, 1, 2, 11, 12), delimiter=",", dtype=str
    )

    # --- consolidate duplicate diameter determinations (cell 7): keep smallest error ---
    u_objnum_n, u_provdesig_n, u_name_mpced_n, u_diam_n, u_diamerr_n = [], [], [], [], []
    unique_mpecs_n = np.asarray(list(set(name_mpced_n)))
    for i in range(len(unique_mpecs_n)):
        imatch = np.where(name_mpced_n == unique_mpecs_n[i])[0]
        if len(imatch) > 1:
            ibest_group = np.where(diamerr_n[imatch] == min(diamerr_n[imatch]))[0]
            if len(ibest_group) > 1:
                ibest_group = ibest_group[0]
            ibest = imatch[ibest_group]
        elif len(imatch) == 1:
            ibest = imatch
        else:
            print("problem finding index matches for " + str(unique_mpecs_n[i]))
            continue

        u_objnum_n.append(objnum_n[ibest][0])
        u_provdesig_n.append(provdesig_n[ibest][0].replace('"', "").replace(" ", ""))
        u_name_mpced_n.append(name_mpced_n[ibest][0].replace('"', "").rstrip())
        u_diam_n.append(diam_n[ibest][0])
        u_diamerr_n.append(diamerr_n[ibest][0])

    u_objnum_n = np.asarray(u_objnum_n)
    u_provdesig_n = np.asarray(u_provdesig_n)
    u_name_mpced_n = np.asarray(u_name_mpced_n)

    # --- match family members to NEOWISE objects (cell 8) ---
    matchids, matchids_curlformat, nomatch = [], [], []
    for i in range(len(objid_mpec)):
        imatch = np.where(u_name_mpced_n == objid_mpec[i])[0]
        if len(imatch) > 0:
            matchids.append(u_name_mpced_n[imatch][0])
            if u_objnum_n[imatch] == "0":
                matchids_curlformat.append(u_provdesig_n[imatch][0])
            else:
                matchids_curlformat.append(u_objnum_n[imatch][0])
        else:
            nomatch.append(objid_mpec[i])

    matchids = np.asarray(matchids)
    matchids_curlformat = np.asarray(matchids_curlformat)
    print(str(len(matchids)) + " asteroids in this family found in the WISE/NEOWISE catalog")

    return matchids, matchids_curlformat


def _write_obs_file(cfg: ObsBuildConfig, curlformat, jd_f, wbflux, wbfluxerr, wrflux, wrfluxerr,
                    astx, asty, astz, ast_to_wisex, ast_to_wisey, ast_to_wisez) -> None:
    """Write a single ``.obs`` file (notebook cell 18 inner block)."""
    path = f"{cfg.data_dir}/{curlformat}.obs"
    with open(path, "w+") as wfile:
        wfile.write(str(int(len(wbflux))) + "\n")
        for ii in range(len(astx)):
            wfile.write(str(jd_f[ii]) + " 1\n")
            wfile.write("%1.8f %1.8f %1.8f\n" % (-astx[ii], -asty[ii], -astz[ii]))
            wfile.write("%1.8f %1.8f %1.8f\n" % (-ast_to_wisex[ii], -ast_to_wisey[ii], -ast_to_wisez[ii]))
            if cfg.cat in ("allsky_4band_p1bs_psd", "allsky_3band_p1bs_psd"):
                if cfg.filterpriority == "w2":
                    wfile.write("4.6028 " + str(round(wbflux[ii], 10)) + " " + str(round(wbfluxerr[ii], 10)) + " 1\n")
                if cfg.filterpriority == "w3":
                    wfile.write("11.0984 " + str(round(wrflux[ii], 10)) + " " + str(round(wrfluxerr[ii], 10)) + " 2\n")
            elif cfg.cat == "neowiser_p1bs_psd":
                wfile.write("4.6028 " + str(round(wrflux[ii], 10)) + " " + str(round(wrfluxerr[ii], 10)) + " 1\n")
            else:
                continue
            wfile.write("\n")
            wfile.write("\n")


def build_obs_files(cfg: ObsBuildConfig) -> str:
    """Query IRSA + Horizons for every family member and write ``.obs`` files.

    Returns the directory the files were written to.
    """
    from .ephemeris import get_positions
    from .irsa import query_irsa

    _ensure_data_dir(cfg)
    ifilt = cfg.ifilt
    matchids, matchids_curlformat = prepare_matchids(cfg)

    for jj in range(int(cfg.istart), len(matchids)):
        print("Working on object ID " + str(matchids[jj]) + ", index = " + str(jj))

        irsaoutput = query_irsa(cfg.cat, str(matchids_curlformat[jj]), str(matchids[jj]))

        # locate the data rows beneath the column header
        idata = []
        try:
            icolhead = [i for i in range(len(irsaoutput)) if irsaoutput[i].startswith("|       cntr_u|")][0]
            colheads = irsaoutput[icolhead]
            for i in range(len(irsaoutput[icolhead:])):
                if not irsaoutput[icolhead + i].startswith("|"):
                    idata.append(i)
        except IndexError:
            print("Couldn't retrieve IRSA output for " + matchids[jj] + ".  Query returned: " + str(irsaoutput))
            continue

        if len(idata) == 0:
            print("No data matches for " + matchids[jj] + " in this catalog: " + cfg.cat)
            continue

        idata = np.asarray(idata, dtype=int)[0] + icolhead
        datalines = irsaoutput[idata:]

        (imjd, icc_flags, iph_qual, iwbflg, iwbmpro, iwbsigmpro, iwbsnr,
         iwrflg, iwrmpro, iwrsigmpro, iwrsnr) = determine_column_indices(colheads, cfg.cat)

        mjd = np.asarray([float(line.split()[imjd]) for line in datalines])
        jd = mjd + 2400000.5
        cc_flags = np.asarray([str(line.split()[icc_flags]) for line in datalines])
        ph_qual = np.asarray([str(line.split()[iph_qual]) for line in datalines])
        wbflg_1 = np.asarray([line.split()[iwbflg] for line in datalines])
        wrflg_1 = np.asarray([line.split()[iwrflg] for line in datalines])
        wbmpro = np.asarray([line.split()[iwbmpro] for line in datalines])
        wbsigmpro = np.asarray([line.split()[iwbsigmpro] for line in datalines])
        wrmpro = np.asarray([line.split()[iwrmpro] for line in datalines])
        wrsigmpro = np.asarray([line.split()[iwrsigmpro] for line in datalines])

        flgs = np.asarray([wbflg_1, wrflg_1])
        foo2 = np.empty((2, len(flgs[0, :])))
        foo2[0, :] = [int(flgs[0, iii].replace("null", "9")) for iii in range(len(flgs[0, :]))]
        foo2[1, :] = [int(flgs[1, iii].replace("null", "9")) for iii in range(len(flgs[0, :]))]
        flgs = np.asarray(foo2, dtype=int)

        # quality filtering (cc_flags clean, ph_qual A/B/C, redder-band flag 0)
        jd_f, wbmpro_f, wrmpro_f, wbsigmpro_f, wrsigmpro_f, bflgs_f = [], [], [], [], [], []
        for i in range(len(mjd)):
            if cc_flags[i][ifilt] in ("0", "p", "P"):
                if ph_qual[i][ifilt] in ("A", "B", "C"):
                    if flgs[1, i] == 0:
                        jd_f.append(jd[i])
                        wbmpro_f.append(wbmpro[i])
                        wbsigmpro_f.append(wbsigmpro[i])
                        wrmpro_f.append(wrmpro[i])
                        wrsigmpro_f.append(wrsigmpro[i])
                        bflgs_f.append(flgs[0, i])

        jd_f = np.asarray(jd_f, dtype=float)
        wbmpro_f = replace_null(wbmpro_f)
        wbsigmpro_f = replace_null(wbsigmpro_f)
        wrmpro_f = replace_null(wrmpro_f)
        wrsigmpro_f = replace_null(wrsigmpro_f)
        bflgs_f = np.asarray(bflgs_f, dtype=int)

        if len(wbmpro_f) < cfg.min_obs:
            print(f"Not enough filtered measurements for object ID {matchids[jj]} in this catalog: {cfg.cat}")
            # preserve the notebook's raw-output dump for inspection
            with open(f"{cfg.data_dir}/Nofilter_{matchids[jj]}.obs", "w+") as wfile:
                for line in irsaoutput:
                    wfile.write(line + "\n")
            continue

        wbflux, wbfluxerr, wrflux, wrfluxerr = convert_mags_to_janskys(
            wbmpro_f, wbsigmpro_f, wrmpro_f, wrsigmpro_f, bflgs_f, cfg.cat
        )

        try:
            astx, asty, astz, ast_to_wisex, ast_to_wisey, ast_to_wisez = get_positions(
                matchids_curlformat[jj], jd_f
            )
        except ValueError:
            print("No horizons match for obj id " + matchids[jj])
            continue

        _write_obs_file(
            cfg, matchids_curlformat[jj], jd_f, wbflux, wbfluxerr, wrflux, wrfluxerr,
            astx, asty, astz, ast_to_wisex, ast_to_wisey, ast_to_wisez,
        )

    return cfg.data_dir
