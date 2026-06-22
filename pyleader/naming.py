"""Asteroid designation conversion helpers (analysis side).

Ported from the analysis notebooks' ``convert_to_letter`` / ``convert_to_mpecname``
cells.  The obs-building pipeline has its own, narrower ``convert_to_mpecname``
(see :mod:`pyleader.obsfiles.naming`) that only handles provisional designations;
the two are intentionally kept separate.
"""

from __future__ import annotations


def convert_to_letter(text: str) -> str:
    """Convert a two-digit numeric prefix to its packed-designation letter."""
    if int(text) <= 35:
        return str(chr(int(text) + 55))
    elif int(text) <= 75:
        return str(chr(int(text) + 61))
    else:
        raise ValueError(f"conversion not found for {text!r}")


def convert_to_mpecname(objid: str) -> str:
    """Convert an asteroid id/designation to packed MPC ("mpec") format.

    Handles provisional designations, numbered designations of varying length,
    and zero-pads short numbered designations.
    """
    objid = objid.replace("+", "")

    if len(objid) > 5:
        # provisional designation -> packed format
        if objid[0:4].isdigit() and objid[4:6].isalpha():
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

        # 6+ digit numbered designation -> packed format
        elif objid[:].isdigit():
            newobjid = list(convert_to_letter(objid[0:2]))
            newobjid += objid[2:]
            return "".join(newobjid)

        else:
            raise ValueError(
                f"Object name {objid!r} not recognized for conversion to mpec format"
            )

    # numbered designations of 5 or fewer digits: left-pad to 5 chars
    elif len(objid) == 5:
        return objid
    elif len(objid) == 4:
        return "0" + objid
    elif len(objid) == 3:
        return "00" + objid
    elif len(objid) == 2:
        return "000" + objid
    elif len(objid) == 1:
        return "0000" + objid

    raise ValueError(f"Empty object name {objid!r}")


def convertMPCed(text: str) -> str:
    """Convert a single packed-designation letter back to its numeric code.

    Auxiliary helper carried over from ``make_LEADER_obs_files.ipynb``.
    """
    if text.isupper():
        return str(ord(text) - 55)
    elif text.islower():
        return str(ord(text) - 61)
    raise ValueError(f"cannot convert {text!r}")
