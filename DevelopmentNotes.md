# Development Notes

Companion to the [README](README.md): the deeper explanations, derivations, measured runtimes,
and format details that would clutter the quick-start documentation. Section numbers follow the
README's pipeline steps. (This file also serves as source material for the PyLEADER
publication.)

## Measured runtimes

All measurements: M3 Max (10P+4E cores), family 3556, 5–10 km members (362 objects), 8 basis
workers, package defaults unless stated.

| stage | wall time |
| --- | --- |
| Step 3 analysis (100 trials × 1000 draws) | ~5 min |
| Step 4a bias map (20 runs) + Step 5a fit | ~3 min |
| noise-model fit + repeatability calibration | ~0.5 min |
| Step 4b basis, 12×12 × 8 seeds = 1152 units | ~78 min |
| Steps 5b/6b posteriors + unfolding | seconds |
| **end-to-end** | **~87 min** |

Per-step detail:

**Step 2 (build .obs).** Roughly ~10 s per catalogued object at typical home-internet speeds,
limited by connection strength on both ends; a population can span ~100–5000 objects, so heavily
populated groups can take many hours.

**Step 3 (analysis).** ~3 s per trial at the default `ndraws=1000` (Apple-silicon laptop; no network) —
  about **5–6 minutes** for the default 100 trials, scaling roughly linearly with
  `ntrials × ndraws`. If a run takes far longer than this, something is likely wrong (e.g. an
  unexpectedly empty or malformed `.obs` directory).

**Step 4b (basis).** Wall time ≈ units × (time per unit) ÷ workers, exactly linear in grid
points × `--nseeds`. A unit costs ≈ 33 s single-core, dominated by the fixed-peak shape
sampling (the tight ±0.02 tolerance accepts only ~1 in 14 stretched DAMIT shapes; the
parsed models are cached in memory per worker). The default 12×12 grid × 4 seeds (576
units) is therefore **~40 min** on 8 workers; × 8 seeds ≈ 80 min (measured: 1152 units in
78 min). Seed-count guidance is under "Step 4b" below.

**Whole pipeline.** The
  LEADER analysis takes ~5 min (100 trials × 1000 draws), the bias map ~3 min (20 runs), and the
  fixed-peak basis dominates — ~40 min at the default 12×12 × 4 seeds, ~78 min at 8 seeds. The
  corrections, posteriors, and report add seconds. **End-to-end: ~50–90 min** depending on
  `--basis-nseeds`; everything scales linearly with trials, draws, and basis units.

## Step 3 — analysis details

### `ndraws` vs `ntrials`

One *trial* = draw `ndraws` objects at random (with replacement) from
  the population's `.obs` files, pool their lightcurve amplitudes into a single CDF, and run one
  LEADER inversion on it. `ndraws` therefore sets the *statistical size of each sample*, while
  `ntrials` sets *how many times that whole experiment is repeated* (with independent random
  draws) — the scatter of the recovered peaks across trials is what provides the spread on the
  result. Total inversions = `ntrials`; total object draws = `ntrials × ndraws`.

### Progress & logging

The terminal shows a single self-updating progress bar
  (`trial k/N (xx%)`); the full run record — configuration, per-trial results, and timestamps —
  is written to `analysis.log` inside the output directory.

## Step 4a — how the synthetic populations are built

### Assigned-distribution scatter

The synthetic objects are not all placed *exactly* at the
  assigned peaks; five `SyntheticConfig` fields (library API; not CLI flags) control the spread,
  with defaults taken directly from the original LEADER release (Nortunen & Kaasalainen 2017):
  `p_accept_tol = 0.075` (a stretched DAMIT shape is accepted when its `p` is within this of the
  peak), `p_escape_chance = 0.1` and `p_escape_min = 0.45` (a 10% chance to accept an off-peak
  shape anyway, provided `p` exceeds 0.45 — a broad tail), `beta_peak_chance = 0.75` (75% of
  objects draw `β` near the peak, the rest uniformly over 0–90°) and `beta_jitter = 0.05` rad
  (the Gaussian width around `β_peak`). The Step-4b fixed-peak preset overrides these to
  near-delta values (`0.02 / 0.0 / – / 1.0 / 0.01`).

### Photometric noise (empirical model + repeatability calibration)

By default (`--noise-model empirical`) the synthetic fluxes receive
  noise matched to the population's **own photometry**, built in two measured steps:

  1. **The error-bar relation.** A quadratic in log–log space,
     `log₁₀(σ_F/F) = c₀ + c₁·log₁₀(F) + c₂·log₁₀(F)²`, is fit once to every (flux, fluxerr)
     pair in the population's `.obs` files. It reproduces the familiar two-regime shape of
     survey photometry: at the faint end the absolute error is flat (set by the sky background
     and detector, independent of source brightness), while for bright sources the error grows
     in proportion to the flux (set by calibration). Each synthetic brightness is scaled to its
     borrowed object's mean measured flux and the relation is evaluated point by point, so
     fainter objects — and the fainter rotational phases of a single lightcurve — get
     proportionally larger errors.
  2. **The repeatability calibration.** A catalog error bar answers *"how far might this
     measurement be from the true flux?"* — it includes systematic terms (zero-point and other
     calibration uncertainties) that shift **all** of an object's measurements up or down
     *together*. But LEADER's amplitude statistic is differential: it only feels the
     **epoch-to-epoch scatter** — how much the measurements jitter relative to *each other* —
     and a shared calibration offset cancels out of it exactly. So the pipeline also measures,
     from the data themselves, what fraction of the quoted error bar shows up as genuine
     point-to-point scatter: in the *quietest* apparitions (near-spherical objects, pole-on
     viewing — negligible real lightcurve variation) the observed scatter divided by the quoted
     fluxerr directly reveals that fraction. This **repeatability fraction** (stored as
     `white_fraction`; the statistical term is "white noise" for random, uncorrelated
     scatter) multiplies the relation of step 1 before it is applied. For family 3556 it is
     0.32 — i.e. only about a third of the NEOWISE error bar behaves as epoch-to-epoch jitter.

  The fit, the fraction, and a diagnostic figure are recorded as `noise_model.json` +
  `noise_model_fit.png` so every simulation's noise is traceable to a measurement. Getting this
  right matters in both directions: the original release's flat 1% Gaussian (`--noise-model
  flat`) understates the real jitter for faint NEOWISE sources (which inflates apparent
  brightness variation and biases the recovery), while injecting the **full** catalog error bar
  as jitter overstates it several-fold — enough to push the recovered `p` of every synthetic
  population far below anything the real data produce.

## Step 5a — the quadratic correction, equation form

The fitted correction is a pair of degree-2 polynomial surfaces in the
  recovered values (`p` ≡ recovered `p`, `β` ≡ recovered `β` in degrees):

$$p_{\rm corr} = c_0 + c_1 p + c_2 \beta + c_3 p^2 + c_4\, p\beta + c_5 \beta^2$$

$$\beta_{\rm corr} = d_0 + d_1 p + d_2 \beta + d_3 p^2 + d_4\, p\beta + d_5 \beta^2$$

  The twelve coefficients are fit by least squares to the bias-map points (each run's recovered
  statistic vs. its assigned peak) and stored in `quadratic_correction.json`; results are clipped
  to the physical ranges `p ∈ [0, 1]`, `β ∈ [0°, 90°]`. With too few points for 6 terms the fit
  automatically drops to linear (3) or constant (1).

## Step 4b — the fixed-peak basis

### One noise model per basis

A basis must be built under **one** noise model throughout — mixing units is
  invalid, so `pyleader-basis` warns when resuming a directory whose existing units used a
  different one (bases predating the empirical model count as flat). Rebuild in a fresh
  `--outdir` after changing the noise model.

### How many seeds (precision vs cost)

Wall time ≈ units × (time per unit) ÷ workers, and is exactly
  linear in grid points × `--nseeds`. A unit costs ≈ 33 s single-core (measured on family 3556,
  5–10 km, M3 Max), dominated by the fixed-peak shape sampling (the tight ±0.02 tolerance
  accepts only ~1 in 14 stretched DAMIT shapes; the parsed models are cached in memory per
  worker). The default 12×12 grid × 4 seeds (576 units) is therefore **~40 min** on 8 workers,
  and × 8 seeds ≈ 80 min (measured: 1152 units in 78 min). The seeds measure
  the *scatter* of the recovery,
  and the scatter estimate improves as $1/\sqrt{2(n_{\rm seeds}-1)}$: **4 seeds** → known to ~±40%
  (fine for exploration), **8–10** → ~±25% (recommended for publication-grade credible intervals),
  **16+** → diminishing returns. Because the basis is **resumable**, you can start at 4 seeds and
  later re-run with `--nseeds 8` (or `--basis-nseeds 8` in the pipeline): only the *new* units are
  computed, and re-running the posterior afterwards re-determines the credible intervals with the
  improved scatter estimate.

## Step 5b — the posterior correction

**How Bayes' rule enters.** The basis answers the *forward* question by brute force: "if the truth
were `(p, β)`, what would LEADER recover?" — at each assigned grid point the `nseeds` realizations
give the mean and scatter of the recovered statistic, i.e. an empirical likelihood
`P(recovered | truth)` (modeled as a Gaussian with that mean and covariance). Bayes' rule turns
this around into the question actually being asked:

$$P(\mathrm{truth} \mid \mathrm{recovered}) \;\propto\; P(\mathrm{recovered} \mid \mathrm{truth}) \times P(\mathrm{truth})$$

With a flat (uninformative) prior `P(truth)` over the basis grid, the posterior at each candidate
truth is simply the Gaussian likelihood of the *one observed* recovery evaluated against that grid
point's forward mean/covariance, normalized over the grid. No fitting is involved — the basis
**is** the model.

**Reading the 2-D posterior map.** A **diagonal ridge** (or a tilted, elongated credible region)
in the `(p, β)` probability heatmap is *expected* and is not a numerical problem: it is the
p–β degeneracy made visible. Different `(p, β)` combinations produce nearly identical amplitude
statistics (a rounder shape at low `β` mimics an elongated one seen pole-on), so a whole diagonal
family of truths is consistent with one recovery, and the posterior honestly assigns them all
probability. When the ridge breaks into disjoint islands, the multimodality flag fires.

## The `.obs` file format

PyLEADER reads **either** layout — `read_obs()` auto-detects them, so datasets from prior analyses
work unchanged. **Both are fully supported.**

- **Tabular** (written by default): a `#` comment header, then one whitespace-delimited row per
measurement — `jd  sun_x sun_y sun_z  obs_x obs_y obs_z  wavelength flux fluxerr filter`.
- **Legacy block:** the original format (count header; per-epoch Sun/observer vectors and
`λ flux σ filter` lines separated by blank lines). Pass `--legacy-format` to write it.

## Assumptions, limitations, and caveats

An honest inventory, compiled for the public release and the journal paper. Items marked
**(inherited)** come from the LEADER method as published (Nortunen & Kaasalainen 2017); the rest
arise from this package's per-population correction machinery and data handling.

### Physical model

1. **Triaxial ellipsoids with `a ≥ b = c` (inherited).** Every object is modeled as an ellipsoid
   characterized by a single elongation `p = b/a`. Real asteroids have concavities, albedo
   variegation, and independent flattening (`c < b`), all of which modulate brightness; the
   method attributes *all* brightness variation to elongation + viewing geometry. Consequence:
   `p` is an *effective photometric elongation*, not a literal axis ratio, and any non-shape
   variability biases it low (more apparent variation → more elongated).
2. **Reflected-light scattering laws applied to thermal fluxes (inherited).** The synthetic
   brightness uses Lommel–Seeliger+Lambert (or Hapke) scattering, while the W3/W4 measurements
   are dominated by thermal emission. The published method makes the same approximation; thermal
   effects (thermal inertia, night-side emission, phase-integral differences) are not modeled.
   The per-population correction absorbs this *partially* — synthetic and real objects share
   geometry and noise, but not thermal physics.
3. **Rotation periods drawn uniformly over 3–12 h.** Real period distributions differ (and vary
   with size); slow rotators, tumblers, and binaries violate the assumption that each apparition
   samples many rotations at effectively random phase. Objects far outside 3–12 h are
   represented incorrectly in the synthetics.
4. **Spin-latitude sign is folded.** `β ∈ [0°, 90°]`: prograde and retrograde spins are
   indistinguishable to the amplitude statistic; results say nothing about spin sense.
   *Documentation item:* the code labels `β` inconsistently ("spin-axis latitude" in the README,
   "spin pole polar angle" in one plot label) — resolve to one convention before release.
5. **Pole longitudes and rotation phases assumed uniform (inherited).** True for most collisional
   families; violated by any aligned sub-population (e.g. YORP-clustered spin vectors).

### Shape library

6. **Stretched DAMIT models stand in for the population's shapes.** Synthetic objects are drawn
   from ~350 DAMIT shape models, each randomly stretched (factor `max(1, 2|N(0,1)|)` per axis)
   until the target `p` is met. DAMIT is biased toward large, bright, high-amplitude objects
   with well-sampled lightcurves; stretching produces a particular family of shapes that may not
   represent small family members. There is no direct test of shape-library representativeness;
   it is a structural systematic shared by all synthetics (bias map, basis, and hence both
   corrections).

### Photometric noise model

7. **The white-noise fraction comes from the quietest apparitions.** The empirical noise model
   applies `white_fraction` × the catalog flux–fluxerr relation, with the fraction estimated
   from the 10th-percentile envelope of (observed intra-apparition scatter ÷ catalog fluxerr).
   Assumptions: (a) the quietest decile of apparitions is nearly signal-free — residual
   lightcurve signal there means the noise is *over*estimated; (b) the non-white remainder of
   the error budget is perfectly common-mode within an apparition and cancels in η — partially
   correlated terms violate this in either direction; (c) one global fraction applies at all
   flux levels. The successive-difference scatter estimator removes slow trends, but with
   rotation periods comparable to the WISE sampling cadence, some signal leaks into the
   estimator regardless.
8. **Noise realism is bounded by the catalog.** Fluxerr values are taken at face value as the
   total error budget; unflagged cosmetics, blends, and background sources are not separately
   modeled.

### Correction machinery

9. **The quadratic correction is a single-valued map through a many-to-one problem.** The p–β
   degeneracy means several truths produce the same recovery; the quadratic silently picks one.
   It is fit to few points (default 20 grid points × few seeds), can clip at physical bounds
   (a recovered `β_corr = 0°` is an extrapolation artifact, not a measurement), and the report's
   extrapolation warning triggers only outside the *fitted* range. Treat it as the fast
   cross-check; the posterior is the primary statement.
10. **The posterior assumes an approximately single-peaked population.** It is calibrated on
    fixed-peak (near-delta) synthetic populations; for strongly skewed or multimodal true
    distributions "the peak" is ill-defined. The multimodality flag and the peak-vs-median
    channel consistency check are alarms, not guarantees.
11. **The posterior is grid-bounded and Gaussian-approximated.** Truths outside the basis grid
    (default `p ∈ [0.30, 0.95]`, `β ∈ [6°, 84°]`) carry zero prior probability — an
    edge-hugging posterior means the grid, not the data, is binding (the report flags this).
    The forward scatter at each grid point is modeled as a Gaussian estimated from only
    `nseeds` realizations (interval precision `1/√(2(n−1))`: ±40% at 4 seeds, ±25% at 8), with
    a covariance floor of half a recovery bin — a choice, not a measurement.
12. **Credible intervals are conditional on the forward model.** They propagate sampling noise
    and basis scatter, but *not* the structural systematics above (shape library, thermal
    physics, noise-model assumptions). They are best interpreted as lower bounds on the total
    uncertainty.
13. **Unfolded β structure is degeneracy-limited.** The mixture validation shows the CDF-space
    unfolding fits the amplitude CDF nearly perfectly (relerr ~0.01) while still failing to
    localize β structure at NEOWISE noise levels — many f(p, β) fit the same data. Quote the
    posterior for β; treat the unfolded β marginal as indicative. The unfolding's 16–84% bands
    are statistical-only (perturbation ensemble) and the solution depends on the smoothness
    regularization (α chosen by discrepancy principle). W-space unfolding additionally carries a
    measured mixture-nonlinearity model error; CDF-space (default) removes it by construction.
14. **CDF-space mixing assumes comparable amplitude counts per object.** Pooling amplitudes is
    exactly linear in mixtures only when basis units contribute similar numbers of amplitudes
    per object; differences are a second-order effect.

### Data and selection

15. **Survey selection enters only through the geometry sample.** Synthetics inherit the
    *detected* population's cadences and (via the noise anchor) fluxes; objects NEOWISE never
    detected, apparitions with <5 points, and any flux-dependent detection bias within the
    sample are not otherwise modeled. Results describe the *observed* (diameter-cut, detected)
    population, not the intrinsic one.
16. **Family membership is taken as given.** Nesvorný HCM family lists (and the
    region/taxonomy-based background definitions) contain interlopers; contamination
    biases the population statistics and is not modeled.
17. **Diameter cuts rely on NEOWISE diameters**, which carry their own ~10%+ uncertainties and
    NEATM assumptions; objects without catalog diameters are excluded.
18. **Single-band analysis.** The analysis uses one filter (typically W3); the synthetic
    geometry reader independently selects each file's best filter, so band choice can differ
    per object. Phase behavior is handled by the LEADER release's empirical phase correction
    with a 40° phase-angle limit; apparitions are defined by a 60-day gap. All three are
    inherited conventions, not fitted.

### Statistical interpretation

19. **Trial scatter is sampling uncertainty only.** The `ntrials` resampling captures
    object-draw variance within the fixed dataset, not systematics.
20. **Regularization defaults are inherited** (`δ_p`, `δ_β` from the MATLAB release) and the
    recovered grid is fixed (20 × 29 bins); results at the bin scale are quantized (the
    posterior's median channel and the covariance floor mitigate, not remove, this).
21. **Reproducibility:** seeded runs are deterministic; results are not bit-for-bit identical to
    the original MATLAB/notebook implementations (documented bug fixes — see the conversion
    notes below).

### Validation status (honest summary)

- Posterior coverage was spot-checked (3/3 truths inside 95% intervals with correct
  multimodality flags) — consistent, but a small sample; a systematic coverage campaign over
  the grid would strengthen the paper.
- The noise calibration and grid convergence (8×8 vs 12×12 agreeing to Δp = 0.002, Δβ = 1°)
  were verified on family 3556; other populations inherit the method but have not been
  individually validated.
- The mixture validation quantifies unfolding fidelity on one family's geometry; its β caveat
  (item 13) is measured, not hypothetical.
- Regimes where the method degrades: very small populations (few pooled amplitudes), very faint
  populations (noise-dominated amplitudes), strongly multimodal or skewed true distributions,
  and any population whose recovered values pin against the basis-grid edge.

## Notes on the notebook → package conversion

The package supersedes the original Jupyter notebooks (`LEADER_python_final`, `_bg`, `_forcedN`),
which are unified into one configurable driver (`_bg` = `--population background`, `_forcedN` =
`--forced-n`). A few clear bugs were fixed during conversion (each marked `# FIX:` in the source:
phase-correction return value, an apparition off-by-one, the forced-N subsampling, and removal of
dead `interp2d`/`mlab` imports), so results are not bit-for-bit identical to the notebooks.
