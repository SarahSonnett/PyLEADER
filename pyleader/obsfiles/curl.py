"""Optional: write a curl script to bulk-download IRSA tables.

Auxiliary helper carried over from ``make_LEADER_obs_files.ipynb`` (cell 27).
Useful when you would rather fetch all of a family's ``.tbl`` files in one
batch instead of querying object-by-object via :func:`build_obs_files`.
"""

from __future__ import annotations

from ..config import ObsBuildConfig
from .build import prepare_matchids

_BASE = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

_CURL_SELCOLS = {
    "allsky_4band_p1bs_psd": (
        "mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w4flg_1,"
        "w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg,"
        "w3mpro,w3sigmpro,w3snr,w3flg,w4mpro,w4sigmpro,w4snr,w4flg"
    ),
    "neowiser_p1bs_psd": (
        "mjd,ra,dec,cc_flags,ph_qual,w1flg_1,w2flg_1,w3flg_1,w4flg_1,"
        "w1mpro,w1sigmpro,w1snr,w1flg,w2mpro,w2sigmpro,w2snr,w2flg"
    ),
}


def write_curl_script(cfg: ObsBuildConfig, matchids=None) -> str:
    """Write a curl script that downloads one ``.tbl`` per family member.

    Returns the path of the script written (``cfg.curl_script``).
    """
    try:
        selcols = _CURL_SELCOLS[cfg.cat]
    except KeyError:
        raise ValueError(f"Catalog defined is not recognized: {cfg.cat!r}")

    if matchids is None:
        matchids, _ = prepare_matchids(cfg)

    with open(cfg.curl_script, "w+") as wfile:
        for mid in matchids:
            url = (
                f"{_BASE}?outfmt=1&searchForm=MO&spatial=cone&catalog={cfg.cat}"
                f"&moradius=0.3&mobj=smo&mobjstr={mid}&selcols={selcols}"
            )
            wfile.write(f'curl -o {mid}.tbl "{url}"\n')
            wfile.write("\n")

    return cfg.curl_script
