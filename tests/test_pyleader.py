"""Regression tests for PyLEADER data-boundary behavior (2026-10 audit).

Each test pins a historical data-boundary bug or a guard added in response. Runnable with pytest or directly:

    python tests/test_pyleader.py

Network-dependent tests are skipped unless PYLEADER_NET_TESTS=1.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyleader.obsio import read_obs, write_obs_table, ObsData  # noqa: E402
from pyleader import populations                               # noqa: E402


def _tmpfile(text, suffix=".txt"):
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    f.write(text)
    f.close()
    return f.name


def test_obs_tabular_roundtrip_preserves_sign():
    """Sign convention: a comparison tool once assumed .obs stores +a2s; it stores the
    NEGATED vectors. The read->write round trip must be exact and sign-
    preserving, so any comparison tool can trust read_obs verbatim."""
    dates = np.array([2455227.25, 2455228.5])
    e_sun = np.array([[1.59770649, -4.69508511, 1.77967660],
                      [1.58760171, -4.69850506, 1.77963096]])
    e_earth = np.array([[0.61784849, -4.69504225, 1.88132314],
                        [0.60774352, -4.69846215, 1.88144693]])
    flux = [[None, None, 0.2025407627, None], [None, None, 0.2580550967, None]]
    fluxerr = [[None, None, 0.0069955101, None], [None, None, 0.0083187135, None]]
    wave = [[None, None, 11.0984, None], [None, None, 11.0984, None]]
    path = _tmpfile("", suffix=".obs")
    write_obs_table(path, ObsData(dates, e_sun, e_earth, flux, fluxerr, wave))
    back = read_obs(path)
    os.unlink(path)
    assert np.allclose(back.e_sun, e_sun, atol=1e-8)
    assert np.allclose(back.e_earth, e_earth, atol=1e-8)
    assert back.flux[0][2] is not None and abs(back.flux[0][2] - 0.2025407627) < 1e-9
    assert back.n == 2


def test_obs_block_format_read():
    """Legacy block files must parse to the same structure as tabular."""
    block = ("1\n"
             "2455227.25 1\n"
             "1.59770649 -4.69508511 1.77967660\n"
             "0.61784849 -4.69504225 1.88132314\n"
             "11.0984 0.2025407627 0.0069955101 2\n"
             "\n\n")
    path = _tmpfile(block, suffix=".obs")
    o = read_obs(path)
    os.unlink(path)
    assert o.n == 1
    assert abs(o.e_sun[0][0] - 1.59770649) < 1e-8
    assert o.flux[0][2] is not None


class _Cfg:
    def __init__(self, famid, base_dir):
        self.famid = famid
        self.base_dir = base_dir


def test_bgobjs_internal_space_rejected():
    """Concatenation bug: designations with internal spaces shifted the
    whitespace-parsed columns and queried the WRONG OBJECTS for every
    unnumbered member. Malformed rows must now raise, not shift."""
    bad = ("       588             -       00588  130.099    0.554\n"
           "         0     2008 RK58     K08R58K    8.549    1.043\n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "BGobjs_XX_Dtype_neowise.txt")
        open(path, "w").write(bad)
        cfg = _Cfg("BG_XX_Dtypes", d)
        try:
            populations._resolve_background(cfg)
        except ValueError as e:
            assert "internal" in str(e) or "5 columns" in str(e)
        else:
            raise AssertionError("spaced designation row was not rejected")


def test_bgobjs_valid_rows_resolve():
    good = ("       588             -       00588  130.099    0.554\n"
            "         0      2008RK58     K08R58K    8.549    1.043\n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "BGobjs_XX_Dtype_neowise.txt")
        open(path, "w").write(good)
        cfg = _Cfg("BG_XX_Dtypes", d)
        matchids, curl = populations._resolve_background(cfg)
        assert list(curl) == ["588", "2008RK58"]
        assert list(matchids) == ["00588", "K08R58K"]


def test_diameter_catalog_mismatch_raises():
    """A non-mainbelt population silently matched against the mainbelt
    catalog inverts near-empty draw pools.
    A <50% catalog match must now raise with the --neowise-fle hint."""
    from pyleader.analysis import diameter_matched_files

    class ACfg:
        diam_low, diam_high = 5.0, 50.0

    with tempfile.TemporaryDirectory() as d:
        for stem in ("11111", "22222", "33333", "44444"):
            open(os.path.join(d, stem + ".obs"), "w").write("# jd\n")
        open(os.path.join(d, "Nofilter_55555.obs"), "w").write("raw dump\n")
        cfg = ACfg()
        cfg.datadir = d
        cfg.neowise_path = "<test catalog>"
        # catalog knows only one of the four objects
        names = np.array(["11111", "99999"])  # post-strip forms, as _load_neowise_diameters returns
        diams = np.array([10.0, 20.0])
        try:
            diameter_matched_files(cfg, names, diams)
        except RuntimeError as e:
            assert "neowise-fle" in str(e)
        else:
            raise AssertionError("catalog mismatch was not detected")
        # and with full coverage it matches normally (Nofilter excluded)
        names = np.array(["11111", "22222", "33333", "44444"])
        diams = np.array([10.0, 20.0, 3.0, 60.0])
        got = sorted(os.path.basename(p) for p in
                     diameter_matched_files(cfg, names, diams))
        assert got == ["11111.obs", "22222.obs"]


def test_horizons_small_body_resolution_network():
    """Bare '624' once resolved to Kiviuq (Saturn satellite code)
    instead of 624 Hektor. Requires network; set PYLEADER_NET_TESTS=1."""
    if os.environ.get("PYLEADER_NET_TESTS") != "1":
        print("  (skipped: set PYLEADER_NET_TESTS=1 to run)")
        return
    from pyleader.obsfiles.ephemeris import get_positions
    sx, sy, sz, ox, oy, oz = get_positions("624", np.array([2455227.25]))
    r = float(np.sqrt(sx[0] ** 2 + sy[0] ** 2 + sz[0] ** 2))
    assert 4.5 < r < 6.0, f"624 resolved to r={r:.2f} au (Hektor is ~5.3)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
