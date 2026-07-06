"""Read/write LEADER ``.obs`` files, in either the legacy block layout or a
friendlier tabular layout.

Both formats are accepted by the whole package — :func:`read_obs` auto-detects
which one a file uses, so datasets produced by older analyses (block format)
and by this package (tabular by default) interoperate freely.

Legacy **block** format::

    <n_epochs>
    <jd> <n_filters>
    <sun_x> <sun_y> <sun_z>
    <obs_x> <obs_y> <obs_z>
    <wavelength> <flux> <fluxerr> <filter_index>      # one line per filter
    <blank>
    <blank>
    ... (next epoch)

Tabular format — a ``#`` comment header then one whitespace-delimited row per
measurement (epochs with multiple filters become multiple rows sharing the same
jd and geometry)::

    # jd  sun_x sun_y sun_z  obs_x obs_y obs_z  wavelength flux fluxerr filter
    2455369.4824 1.235 -2.641 -0.174 0.219 -2.641 -0.206 11.0984 0.00366 0.00116 2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TABLE_HEADER = ("# jd  sun_x sun_y sun_z  obs_x obs_y obs_z  "
                "wavelength flux fluxerr filter")


@dataclass
class ObsData:
    """Parsed contents of an ``.obs`` file (format-agnostic).

    ``flux``/``fluxerr`` are ``n_epochs x 4`` lists (one slot per WISE filter
    index 0-3); missing measurements are ``None``. ``wavelength`` holds the
    per-filter wavelength seen for each epoch (or ``None``).
    """

    dates: np.ndarray                 # (n,)
    e_sun: np.ndarray                 # (n, 3)
    e_earth: np.ndarray               # (n, 3)
    flux: list                        # n x 4, entries float or None
    fluxerr: list                     # n x 4, entries float or None
    wavelength: list                  # n x 4, entries float or None

    @property
    def n(self) -> int:
        return len(self.dates)


def _blank(f4):
    return [[None, None, None, None] for _ in range(f4)]


def read_obs(path: str) -> ObsData:
    """Read an ``.obs`` file in either supported format (auto-detected)."""
    with open(path, "r") as fid:
        lines = [ln.rstrip() for ln in fid]

    first = next((ln for ln in lines if ln.strip()), "")
    tok = first.split()
    is_block = len(tok) == 1 and tok[0].lstrip("+-").isdigit()

    if is_block:
        return _read_block(lines)
    return _read_table(lines)


def _read_block(lines) -> ObsData:
    nblocks = int(lines[0])
    dates = np.zeros(nblocks)
    e_sun = np.zeros((nblocks, 3))
    e_earth = np.zeros((nblocks, 3))
    flux = _blank(nblocks)
    fluxerr = _blank(nblocks)
    wavelength = _blank(nblocks)

    i = 1
    for b in range(nblocks):
        if i >= len(lines):
            break  # tolerate files that omit the final trailing blank lines
        if lines[i]:
            dates[b] = float(lines[i].split()[0])
            nfilters = int(lines[i].split()[1])
            e_sun[b, :] = lines[i + 1].split()
            e_earth[b, :] = lines[i + 2].split()
            for j in range(nfilters):
                parts = lines[i + 3 + j].split()
                fidx = int(parts[3])
                wavelength[b][fidx] = float(parts[0])
                flux[b][fidx] = float(parts[1])
                fluxerr[b][fidx] = float(parts[2])
            i += 5 + nfilters
        else:
            i += 1

    return ObsData(dates, e_sun, e_earth, flux, fluxerr, wavelength)


def _read_table(lines) -> ObsData:
    # group rows by epoch (unique jd + geometry, in first-seen order)
    epochs = {}
    order = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        jd = float(p[0])
        sun = (float(p[1]), float(p[2]), float(p[3]))
        obs = (float(p[4]), float(p[5]), float(p[6]))
        wl, fx, fe, filt = float(p[7]), float(p[8]), float(p[9]), int(p[10])
        key = (jd, sun, obs)
        if key not in epochs:
            epochs[key] = {"wl": [None] * 4, "fx": [None] * 4, "fe": [None] * 4}
            order.append(key)
        epochs[key]["wl"][filt] = wl
        epochs[key]["fx"][filt] = fx
        epochs[key]["fe"][filt] = fe

    n = len(order)
    dates = np.zeros(n)
    e_sun = np.zeros((n, 3))
    e_earth = np.zeros((n, 3))
    flux, fluxerr, wavelength = [], [], []
    for k, key in enumerate(order):
        jd, sun, obs = key
        dates[k] = jd
        e_sun[k, :] = sun
        e_earth[k, :] = obs
        wavelength.append(epochs[key]["wl"])
        flux.append(epochs[key]["fx"])
        fluxerr.append(epochs[key]["fe"])

    return ObsData(dates, e_sun, e_earth, flux, fluxerr, wavelength)


def write_obs_table(path: str, data: ObsData) -> None:
    """Write an :class:`ObsData` to the tabular ``.obs`` format."""
    with open(path, "w") as f:
        f.write(TABLE_HEADER + "\n")
        for k in range(data.n):
            sx, sy, sz = data.e_sun[k]
            ox, oy, oz = data.e_earth[k]
            for filt in range(4):
                if data.flux[k][filt] is None:
                    continue
                wl = data.wavelength[k][filt]
                f.write("%.6f %.8f %.8f %.8f %.8f %.8f %.8f %s %.10f %.10f %d\n" % (
                    data.dates[k], sx, sy, sz, ox, oy, oz,
                    ("%.4f" % wl) if wl is not None else "0",
                    data.flux[k][filt], data.fluxerr[k][filt], filt))
