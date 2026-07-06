"""Shape-elongation and facet geometry from a polyhedral model.

Ports ``leader_ellipsoid.m``: from vertices ``R`` and triangular faces ``F`` it
computes each facet's outward normal and area (needed by the brightness model)
and the shape elongation ``p = b/a`` from the model's projected extents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EllipsoidProps:
    p: float                 # shape elongation b/a
    normals: np.ndarray      # (Nfaces, 3) unit facet normals  (MATLAB `normaali`)
    areas: np.ndarray        # (Nfaces,) facet areas           (MATLAB `ala`)
    semiaxes: np.ndarray     # (a, b, c) / c


def ellipsoid_properties(R: np.ndarray, F: np.ndarray) -> EllipsoidProps:
    """Compute facet normals/areas and the elongation ``p = b/a`` for a model.

    ``R`` is an (Nvert, 3) array of (possibly stretched) vertices; ``F`` is an
    (Nface, 3) array of 0-based vertex indices.
    """
    R = np.asarray(R, dtype=float)
    F = np.asarray(F, dtype=int)

    # --- facet normals and areas ---
    p1 = R[F[:, 0]]
    p2 = R[F[:, 1]]
    p3 = R[F[:, 2]]
    a1 = p2 - p1
    a2 = p3 - p2
    cross = np.cross(a1, a2)
    normtemp = np.linalg.norm(cross, axis=1)
    normals = cross / normtemp[:, None]
    areas = 0.5 * normtemp

    # --- semiaxes via maximum projected extent (after Kaasalainen) ---
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
    phi = np.arange(1, 181) / 180.0 * np.pi

    # x-direction: longest projected width over rotation angle phi
    xphi = np.array([np.ptp(X * np.cos(ph) + Y * np.sin(ph)) for ph in phi])
    a = xphi.max()
    phimax = phi[np.argmax(xphi)]

    # y-direction: width perpendicular to the a-axis
    yy = Y * np.cos(phimax) - X * np.sin(phimax)
    b = np.ptp(yy)

    # z-direction
    c = np.ptp(Z)

    semiaxes = np.array([a, b, c]) / c
    p = b / a

    return EllipsoidProps(p=p, normals=normals, areas=areas, semiaxes=semiaxes)
