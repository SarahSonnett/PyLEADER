"""IRSA Gator moving-object query helpers.

Requires ``requests`` (imported lazily).
"""

from __future__ import annotations

_BASE = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

_SELCOLS = {
    "allsky_4band_p1bs_psd": (
        "mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,"
        "w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,"
        "w3mpro,w3sigmpro,w3snr,w3flg"
    ),
    "neowiser_p1bs_psd": (
        "mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,"
        "w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg"
    ),
}
_SELCOLS["allsky_3band_p1bs_psd"] = _SELCOLS["allsky_4band_p1bs_psd"]


def _selcols(cat: str) -> str:
    try:
        return _SELCOLS[cat]
    except KeyError:
        raise ValueError(f"catalog not recognized: {cat!r}")


def query_url(cat: str, mobjstr: str) -> str:
    """Build a Gator moving-object cone-search URL for ``mobjstr``."""
    return (
        f"{_BASE}?outfmt=1&searchForm=MO&spatial=cone&catalog={cat}"
        f"&mobj=smo&mobjstr={mobjstr}&selcols={_selcols(cat)}"
    )


def query_irsa(cat: str, curlformat: str, matchid: str):
    """Run the IRSA query with the notebook's fallback cascade.

    Tries, in order, until the response has more than one line:
      1. ``<curlformat>:AST``
      2. ``<matchid>:AST``
      3. ``<curlformat>:AST`` (retry, as in the original notebook)
      4. ``<curlformat>``     (without the ``:AST`` suffix)

    Returns the response split into lines.
    """
    import requests

    candidates = [
        f"{curlformat}:AST",
        f"{matchid}:AST",
        f"{curlformat}:AST",
        f"{curlformat}",
    ]

    irsaoutput = []
    for mobjstr in candidates:
        res = requests.get(query_url(cat, mobjstr))
        irsaoutput = res.text.splitlines()
        if len(irsaoutput) > 1:
            break
    return irsaoutput
