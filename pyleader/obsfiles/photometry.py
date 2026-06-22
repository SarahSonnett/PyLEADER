"""Photometric conversions: WISE magnitudes -> fluxes (Janskys)."""

from __future__ import annotations

import numpy as np


def replace_null(arr, fill: float = -10.0) -> np.ndarray:
    """Coerce an array to float, substituting ``fill`` for ``'null'`` entries."""
    try:
        return np.asarray(arr, dtype=float)
    except ValueError:
        arr_fix = [fill if val == "null" else val for val in arr]
        return np.asarray(arr_fix, dtype=float)


def convert_mags_to_janskys(wbmag, wbmagerr, wrmag, wrmagerr, bflg, cat: str):
    """Convert bluer/redder-band magnitudes to fluxes with color corrections.

    Color corrections are from Wright et al. (2010), Table 1.  ``cat`` selects
    which band pair (and zero points) apply.
    """
    igood = np.where(bflg == 0)

    if len(igood) > 2:
        color = np.median(wbmag[igood] - wrmag[igood])
    else:
        color = np.median(wbmag - wrmag)

    if cat == "neowiser_p1bs_psd":
        colorcode = np.array([-0.4040, -0.0538, 0.2939, 0.6393, 0.9828, 1.3246, 1.6649, 2.0041])  # W2
        f_wb = [1.0283, 1.0084, 0.9961, 0.9907, 0.9921, 1.0, 1.0142, 1.0347]
        f_wr = [1.0206, 1.0066, 0.9976, 0.9935, 0.9943, 1.0, 1.0107, 1.0265]
    elif cat in ("allsky_4band_p1bs_psd", "allsky_3band_p1bs_psd"):
        colorcode = np.array([-0.9624, -0.0748, 0.8575, 1.8357, 2.8586, 3.9225, 5.0223, 6.1524])  # W3
        f_wb = [1.0206, 1.0066, 0.9976, 0.9935, 0.9943, 1.0, 1.0107, 1.0265]
        f_wr = [1.1344, 1.0088, 0.9393, 0.9169, 0.9373, 1.0, 1.1081, 1.2687]
    else:
        raise ValueError(f"catalog name not recognized: {cat!r}")

    diff = list(np.abs(colorcode - color))
    i_nu = diff.index(min(diff))
    print("i_nu = " + str(i_nu))

    if cat == "neowiser_p1bs_psd":
        wbflux = (306.682 / f_wb[i_nu]) * 10 ** (-wbmag / 2.5)
        wrflux = (170.663 / f_wr[i_nu]) * 10 ** (-wrmag / 2.5)
    else:  # allsky 4band / 3band
        wbflux = (170.663 / f_wb[i_nu]) * 10 ** (-wbmag / 2.5)
        wrflux = (29.045 / f_wr[i_nu]) * 10 ** (-wrmag / 2.5)

    wbfluxerr = wbflux * np.log(10) * wbmagerr
    wrfluxerr = wrflux * np.log(10) * wrmagerr

    return wbflux, wbfluxerr, wrflux, wrfluxerr
