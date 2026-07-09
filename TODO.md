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
- [x] **Calibrate the effective per-epoch noise** (2026-07-08) — applying the catalog fluxerr as
      *independent per-epoch Gaussian* scatter made the forward model inconsistent with the real
      Fam3556 data (all synthetic grid points recovered p ≈ 0.16–0.31 vs the real 0.45–0.51;
      posterior pinned at the p = 0.80 grid edge). Cause: the catalog fluxerr is a *total* error
      budget — most of it is calibration/systematic terms that shift a whole apparition
      coherently and cancel out of the differential amplitude statistic η. Fix implemented:
      `measure_white_fraction` (noise.py) takes the 10th-percentile envelope of
      (observed intra-apparition scatter ÷ catalog fluxerr) over all apparitions — the quietest
      apparitions bound the genuinely random epoch-to-epoch component — and scales the noise
      model by it (Fam3556: 0.32). Recorded in `noise_model.json` / the diagnostic figure /
      `basis_info.json`. **Verified** on a full Fam3556 rerun (256-unit basis, `_basis_cal`):
      the posterior came off the grid edge and is unimodal in both channels — median channel
      p = 0.586 [0.546, 0.624], β = 44.6° [36.0, 53.4] — with the peak/median consistency check
      passing. Remaining follow-ups: consider extending `basis_p_range` beyond 0.80, and
      re-examine the `white_percentile` choice once a few populations have been calibrated.
      Note from the same rerun: the CDF-space unfolding fits the observed amplitude CDF to
      relerr ≈ 0.004, but the mixture validation shows β localization in the *unfolded
      distribution* stays degeneracy-limited at realistic noise — quote the posterior for the
      peak and treat the unfolded β marginal as indicative.

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
