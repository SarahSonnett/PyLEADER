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

## Notes on the notebook → package conversion

The package supersedes the original Jupyter notebooks (`LEADER_python_final`, `_bg`, `_forcedN`),
which are unified into one configurable driver (`_bg` = `--population background`, `_forcedN` =
`--forced-n`). A few clear bugs were fixed during conversion (each marked `# FIX:` in the source:
phase-correction return value, an apparition off-by-one, the forced-N subsampling, and removal of
dead `interp2d`/`mlab` imports), so results are not bit-for-bit identical to the notebooks.
