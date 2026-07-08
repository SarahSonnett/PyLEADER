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
- [x] **CDF-space response refinement** (2026-07-08) — `pyleader-unfold --space cdf` (now the
      default) builds response columns from each basis unit's simulated amplitude CDF, which is
      exactly linear in mixtures — the W-space model error is removed by construction. Basis
      units and analysis trials now persist their amplitude samples (`A` in
      `synthetic_result.npz`, `Asort` in `W_trial*.npz`); older analyses can use `--obsdir` to
      recompute the observed CDF. `--space w` keeps the original behaviour.
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
- [ ] **Calibrate the effective per-epoch noise (BLOCKER for production empirical-noise use)** —
      the 2026-07-08 Fam3556 rerun showed that applying the catalog fluxerr as *independent
      per-epoch Gaussian* scatter yields a forward model inconsistent with the real data: every
      synthetic grid point recovers p ≈ 0.16–0.31 while the real analysis recovers p ≈ 0.45–0.51,
      pinning the posterior at the p = 0.80 grid edge (multimodal, huge β intervals); meanwhile
      the real amplitude CDF (median A ≈ 0.62) sits *above* every synthetic column (max ≈ 0.53).
      Likely cause: NEOWISE fluxerr contains correlated/systematic components that do not appear
      in the intra-apparition point-to-point scatter that η measures, so treating it as white
      noise overstates the η-relevant noise (while the ellipsoid model may simultaneously
      understate the real amplitude variance — non-ellipsoidal shapes, albedo variegation,
      p < 0.30 objects). Candidate fixes: (a) estimate the effective white-noise fraction from
      intra-apparition residual scatter of low-amplitude objects; (b) calibrate a global noise
      scale factor so synthetic η distributions match the real population's; (c) extend
      basis_p_range beyond 0.80 regardless. Until then, treat empirical-noise corrections with
      caution and compare against `--noise-model flat`.

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
