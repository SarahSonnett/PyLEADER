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

## Correction v2 (IMPLEMENTED on branch `posterior-correction`)

- [x] Posterior-inversion correction + response-matrix unfolding per
      [docs/plans/correction_v2_posterior_and_unfolding.md](docs/plans/correction_v2_posterior_and_unfolding.md):
      `pyleader-basis` (parallel/resumable/chunkable fixed-peak basis), posterior correction with
      68/95% credible intervals + multimodality flag (`run_population --correction-method`),
      and `pyleader-unfold` (full f(p, β) with 16–84% bands).
- [ ] **CDF-space response refinement** — remove the measured mixture-linearity model error of the
      W-space unfolding by building response columns from simulated amplitude CDFs (exactly
      linear in mixtures). See the plan doc's "Follow-on refinement".
- [ ] **Covariance pooling across neighboring basis grid points** (optional ultra-precise mode) —
      each forward-table covariance is estimated from only `nseeds` realizations, so it carries
      ~`1/sqrt(2(nseeds-1))` relative noise that propagates into the credible intervals. Pooling
      (or smoothing) the covariance estimates over neighboring `(p, β)` grid points would
      stabilize the forward table **without new simulations** — worthwhile when someone needs
      ultra-precise intervals but cannot afford 16+ seeds. Expose as an opt-in flag
      (e.g. `--pool-covariance`) so the default stays strictly local.
- [ ] **Regenerate production bases/bias maps under the empirical noise model** — bases built
      before the per-population flux-fluxerr noise model (added 2026-07-08) used flat 1% noise,
      which badly understates NEOWISE uncertainties (Fam1128 median relerr ≈ 30%), so their
      corrections understate the bias. `run_basis` warns when resuming such a directory; rebuild
      in a fresh `--outdir` (or pass `--noise-model flat` to reproduce the old behaviour).

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
