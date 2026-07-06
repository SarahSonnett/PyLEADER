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
from ..obsio import ObsData, write_obs_table
from ..populations import resolve_members
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
    """Return ``(matchids, matchids_curlformat)`` for the population.

    Back-compat wrapper around :func:`pyleader.populations.resolve_members`
    (handles both collisional families and background populations).
    """
    return resolve_members(cfg)


# Filter -> (WISE filter index, wavelength, which flux column) per catalog.
def _filter_slot(cfg: ObsBuildConfig):
    if cfg.cat in ("allsky_4band_p1bs_psd", "allsky_3band_p1bs_psd"):
        return (1, 4.6028, "wb") if cfg.filterpriority == "w2" else (2, 11.0984, "wr")
    if cfg.cat == "neowiser_p1bs_psd":
        return (1, 4.6028, "wr")
    raise ValueError(f"catalog not recognized: {cfg.cat!r}")


def _build_obsdata(cfg, jd_f, wbflux, wbfluxerr, wrflux, wrfluxerr,
                   astx, asty, astz, ast_to_wisex, ast_to_wisey, ast_to_wisez) -> ObsData:
    """Assemble an :class:`ObsData` from the per-epoch geometry and fluxes."""
    n = len(jd_f)
    e_sun = np.column_stack([-np.asarray(astx), -np.asarray(asty), -np.asarray(astz)])
    e_earth = np.column_stack([-np.asarray(ast_to_wisex), -np.asarray(ast_to_wisey), -np.asarray(ast_to_wisez)])
    fi, wl, which = _filter_slot(cfg)
    fx, fe = (wbflux, wbfluxerr) if which == "wb" else (wrflux, wrfluxerr)
    flux = [[None] * 4 for _ in range(n)]
    fluxerr = [[None] * 4 for _ in range(n)]
    wave = [[None] * 4 for _ in range(n)]
    for k in range(n):
        flux[k][fi] = round(float(fx[k]), 10)
        fluxerr[k][fi] = round(float(fe[k]), 10)
        wave[k][fi] = wl
    return ObsData(np.asarray(jd_f, dtype=float), e_sun, e_earth, flux, fluxerr, wave)


def _write_block(path: str, data: ObsData) -> None:
    """Write ObsData in the legacy block format."""
    with open(path, "w+") as w:
        w.write(str(int(data.n)) + "\n")
        for k in range(data.n):
            nf = sum(1 for filt in range(4) if data.flux[k][filt] is not None)
            w.write("%s %d\n" % (data.dates[k], nf))
            w.write("%1.8f %1.8f %1.8f\n" % tuple(data.e_sun[k]))
            w.write("%1.8f %1.8f %1.8f\n" % tuple(data.e_earth[k]))
            for filt in range(4):
                if data.flux[k][filt] is None:
                    continue
                w.write("%s %s %s %d\n" % (data.wavelength[k][filt],
                                           data.flux[k][filt], data.fluxerr[k][filt], filt))
            w.write("\n")
            w.write("\n")


def _write_obs_file(cfg: ObsBuildConfig, curlformat, jd_f, wbflux, wbfluxerr, wrflux, wrfluxerr,
                    astx, asty, astz, ast_to_wisex, ast_to_wisey, ast_to_wisez) -> None:
    """Write a single ``.obs`` file (tabular by default; block if ``cfg.legacy_format``)."""
    data = _build_obsdata(cfg, jd_f, wbflux, wbfluxerr, wrflux, wrfluxerr,
                          astx, asty, astz, ast_to_wisex, ast_to_wisey, ast_to_wisez)
    path = f"{cfg.data_dir}/{curlformat}.obs"
    if cfg.legacy_format:
        _write_block(path, data)
    else:
        write_obs_table(path, data)


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
