"""Designation conversion for the obs-building pipeline.

This is the narrower ``convert_to_mpecname`` from ``make_LEADER_obs_files.ipynb``,
which only handles provisional designations (4 digits + letters).  It is kept
separate from the analysis-side :func:`pyleader.naming.convert_to_mpecname`,
which additionally handles numbered designations of various lengths.
"""

from __future__ import annotations

from ..naming import convert_to_letter

__all__ = ["convert_to_letter", "convert_to_mpecname"]


def convert_to_mpecname(objid: str) -> str:
    """Convert a provisional designation to packed MPC ("mpec") format."""
    newobjid = list(convert_to_letter(objid[0:2]))
    newobjid += objid[2:4]
    if len(objid) == 6:
        newobjid += objid[4] + "00" + objid[5]
    elif len(objid) == 7:
        newobjid += objid[4] + "0" + objid[6] + objid[5]
    elif len(objid) == 8:
        newobjid += objid[4] + objid[6:] + objid[5]
    elif len(objid) == 9:
        newobjid += objid[4] + convert_to_letter(objid[6:8]) + objid[8] + objid[5]
    return "".join(newobjid)
