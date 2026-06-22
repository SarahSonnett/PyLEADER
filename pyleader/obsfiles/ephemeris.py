"""Asteroid / observer geometry via JPL Horizons (through sunpy).

Requires ``astropy`` and ``sunpy``; both are imported lazily so this module can
be imported without them present.
"""

from __future__ import annotations

from .naming import convert_to_mpecname


def get_positions(objid: str, jd_def):
    """Return Sun- and observer-relative vectors of the asteroid at each epoch.

    Computes, in au, the asteroid->Sun and asteroid->WISE (Horizons body -163)
    vectors for every Julian date in ``jd_def``.

    Returns ``(a2s_x, a2s_y, a2s_z, a2o_x, a2o_y, a2o_z)``.
    """
    from astropy.time import Time
    from astropy.coordinates import CartesianRepresentation
    from sunpy.coordinates import get_horizons_coord

    # Provisional designations need packing for Horizons
    if len(objid) > 5 and objid[0:4].isdigit() and objid[4:6].isalpha():
        objid = convert_to_mpecname(objid)

    t = Time(jd_def, format="jd")
    utc = t.to_datetime()

    ast_positions = get_horizons_coord(objid, time=utc)
    astxyz = ast_positions.represent_as(CartesianRepresentation)

    wise_positions = get_horizons_coord("-163", time=utc)
    wisexyz = wise_positions.represent_as(CartesianRepresentation)

    sun_positions = get_horizons_coord("sun", time=utc)
    sunxyz = sun_positions.represent_as(CartesianRepresentation)

    a2s = sunxyz - astxyz
    a2o = wisexyz - astxyz

    return (a2s.x.value, a2s.y.value, a2s.z.value,
            a2o.x.value, a2o.y.value, a2o.z.value)
