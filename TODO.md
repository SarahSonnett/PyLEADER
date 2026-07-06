# TODO

## Packaging

The package currently runs from a source checkout (the CLIs insert the repo root on
`sys.path`, and imports resolve via `PYTHONPATH`). There is no build system yet. When adding one:

- [ ] Add a `pyproject.toml` (e.g. setuptools or hatchling) with project metadata and dependencies
      (mirror `requirements.txt`: core = numpy/scipy/matplotlib; extras for obs-building =
      astropy/sunpy/requests).
- [ ] **Include the shipped correction data as package data.** The file
      `pyleader/synthetic/data/correction_function.json` is loaded at runtime by
      `pyleader.synthetic.correction.default_correction()`. It is **not** Python, so it will be
      omitted from wheels unless explicitly declared. Without it, `default_correction()` raises
      `FileNotFoundError` on an installed copy.
    - setuptools: set `[tool.setuptools.package-data]` `pyleader.synthetic = ["data/*.json"]`
      (or `include-package-data = true` + a `MANIFEST.in` with
      `recursive-include pyleader/synthetic/data *.json`).
    - hatchling: ensure `pyleader/synthetic/data/*.json` is under `[tool.hatch.build] include`.
    - Consider loading it via `importlib.resources` instead of a filesystem path so it also works
      from zipped installs.
- [ ] Expose the CLIs as console entry points (`run_analysis`, `build_obs_files`, `run_synthetic`,
      `compare_populations`, `sweep_synthetic`, `plot_sweep`, `fit_correction`) so users don't need
      `python scripts/...`.

## Data (not committed; regenerable)

- `damit_models/` — 347 DAMIT shape models, gitignored (~29 MB). Regenerate with
  `scripts/run_synthetic.py --download` or `scripts/sweep_synthetic.py` after downloading.
