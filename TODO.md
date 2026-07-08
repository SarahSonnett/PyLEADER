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
- [x] Console entry points (`pyleader-analysis`, `pyleader-build-obs`, `pyleader-spot-check`,
      `pyleader-compare`, `pyleader-bias-map`, `pyleader-plot-bias-map`, `pyleader-fit-correction`).

Remaining / nice-to-have:

- [ ] Switch `default_correction()` from a filesystem path to `importlib.resources` so it also
      works from zipped (non-extracted) installs.
- [ ] Set `version` dynamically from `pyleader.__version__` instead of duplicating it in
      `pyproject.toml`.
- [ ] Publish (tag + build sdist/wheel) if distributing beyond source checkouts.

## Correction v2 (planned — next session)

- [ ] Implement the **posterior-inversion correction + response-matrix unfolding** per the design
      doc: [docs/plans/correction_v2_posterior_and_unfolding.md](docs/plans/correction_v2_posterior_and_unfolding.md).
      Phase 0 (shared delta-basis runs, multiprocessing pool + `--task k/N` chunking) → Phase 1
      (posterior correction with credible intervals; degeneracy shows up as widened/multimodal
      posteriors) → Phase 2 (response-matrix unfolding of full f(p, β)). Local compute is fine
      (M3 Max, ~10–30 min/population pooled); cluster notes in the doc.

## Optional follow-on tooling (deferred)

- [ ] Port the `Compare_LEADER_results_*.ipynb` notebooks (currently in
      `~/Desktop/work/MBA_SFDs/`) as **optional** follow-on code — comparison plots across
      populations/regions/sizes/taxonomies built from the `DF_p_all.txt` / `DF_b_all.txt`
      outputs. Lower priority / narrower audience: mostly reproduces the author's own analyses,
      though a few datasets are usable by others. Keep it out of the core pipeline.
    - Heads-up when porting: these read `MarginalDF_p_beta_trial*.txt`, which now has 29 rows with
      `nan`-padded `p` columns (full β marginal); ensure the `p`-column handling is NaN-aware.

## Data (not committed; regenerable)

- `damit_models/` — 347 DAMIT shape models, gitignored (~29 MB). Fetch with
  `pyleader-download-models` (or `scripts/download_models.py`); the listing (`asteroideja.txt`)
  ships in `pyleader/synthetic/data/`. Use `--force` to refresh to the latest DAMIT versions.
