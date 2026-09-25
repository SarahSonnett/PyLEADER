"""Build LEADER ``.obs`` input files from IRSA + JPL Horizons.

Modularized form of ``make_LEADER_obs_files.ipynb``.  The main entry point is
:func:`pyleader.obsfiles.build.build_obs_files` (also exposed as
``pyleader.build_obs_files``).

Note: :mod:`pyleader.obsfiles.ephemeris` and the build loop require
``astropy``, ``sunpy`` and ``requests``.  Those imports are deferred to call
time so this subpackage can be imported for its pure helpers without them.
"""
