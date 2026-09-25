"""Locate data-column indices in an IRSA Gator table header."""

from __future__ import annotations


def determine_column_indices(colheads: str, cat: str):
    """Return the column indices needed to parse an IRSA table for ``cat``.

    Returns ``(imjd, icc_flags, iph_qual, iwbflg, iwbmpro, iwbsigmpro, iwbsnr,
    iwrflg, iwrmpro, iwrsigmpro, iwrsnr)`` where the "wb"/"wr" (bluer/redder)
    bands depend on the catalog.
    """
    # The [1:] drops the leading empty field before the first '|'
    colnames = colheads.split("|")[1:]
    colnames = [colname.strip() for colname in colnames]

    imjd = colnames.index("mjd")
    icc_flags = colnames.index("cc_flags")
    iph_qual = colnames.index("ph_qual")

    if cat == "neowiser_p1bs_psd":
        iwbflg = colnames.index("w1flg_1")
        iwbmpro = colnames.index("w1mpro")
        iwbsigmpro = colnames.index("w1sigmpro")
        iwbsnr = colnames.index("w1snr")

        iwrflg = colnames.index("w2flg_1")
        iwrmpro = colnames.index("w2mpro")
        iwrsigmpro = colnames.index("w2sigmpro")
        iwrsnr = colnames.index("w2snr")

    elif cat in ("allsky_4band_p1bs_psd", "allsky_3band_p1bs_psd"):
        iwbflg = colnames.index("w2flg_1")
        iwbmpro = colnames.index("w2mpro")
        iwbsigmpro = colnames.index("w2sigmpro")
        iwbsnr = colnames.index("w2snr")

        iwrflg = colnames.index("w3flg_1")
        iwrmpro = colnames.index("w3mpro")
        iwrsigmpro = colnames.index("w3sigmpro")
        iwrsnr = colnames.index("w3snr")

    else:
        raise ValueError(f"Catalog name not recognized: {cat!r}")

    return (imjd, icc_flags, iph_qual, iwbflg, iwbmpro, iwbsigmpro, iwbsnr,
            iwrflg, iwrmpro, iwrsigmpro, iwrsnr)
