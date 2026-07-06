# TODO

## Packaging

A `pyproject.toml` (setuptools) is now in place. Done:

- [x] `pyproject.toml` with project metadata + dependencies (core = numpy/scipy/matplotlib;
      `[obs]` extra = astropy/sunpy/requests).
- [x] **Correction data shipped as package data** — `[tool.setuptools.package-data]`
      `"pyleader.synthetic" = ["data/*.json"]`, so `default_correction()` works from an installed
      copy. Verified: the built wheel contains `correction_function.json` and a clean-venv install
      loads it. The CLI logic lives in `pyleader/cli/*` (importable/shippable); `scripts/*.py` are
      thin source-tree shims.
- [x] Console entry points (`pyleader-analysis`, `pyleader-build-obs`, `pyleader-synthetic`,
      `pyleader-compare`, `pyleader-sweep`, `pyleader-plot-sweep`, `pyleader-fit-correction`).

Remaining / nice-to-have:

- [ ] Switch `default_correction()` from a filesystem path to `importlib.resources` so it also
      works from zipped (non-extracted) installs.
- [ ] Set `version` dynamically from `pyleader.__version__` instead of duplicating it in
      `pyproject.toml`.
- [ ] Publish (tag + build sdist/wheel) if distributing beyond source checkouts.

## Data (not committed; regenerable)

- `damit_models/` — 347 DAMIT shape models, gitignored (~29 MB). Regenerate with
  `scripts/run_synthetic.py --download` or `scripts/sweep_synthetic.py` after downloading.
