# PyLEADER

A Python version of the **LEADER** method (originally MATLAB; Nortunen & Kaasalainen 2017), with
enhancements for diagnostics, error determination, and per-population bias correction. PyLEADER
recovers the distributions of asteroid **shape elongation** (`p`) and **spin-axis latitude** (`β`)
for a whole population from WISE/NEOWISE thermal photometry.

Give it a **dynamical population ID** — a Nesvorný collisional family or a background main-belt
population — and the end-to-end driver queries NEOWISE, writes one `.obs` file per object, runs
LEADER, derives a correction function from *that population's own observing geometry*, and applies
it:

```sh
pyleader-population 1128            # a collisional family
pyleader-population BG_IB_Ctypes    # a background population (add --build to fetch photometry)
```

Every step is also available as a standalone command (see [Usage](#usage)).

## How it works

PyLEADER implements the **LEADER** method (*Latitudes and Elongations of Asteroid Distributions
Estimated Rapidly*) of Nortunen & Kaasalainen (2017). For large, sparsely sampled populations,
inverting individual lightcurves is infeasible — so instead of solving for one object at a time,
LEADER recovers the **joint distribution of shape elongation** `p` **and spin-axis latitude** `β` **for the
whole population** from the statistics of brightness variations. Each object is modeled as a triaxial
ellipsoid with axes `a ≥ b = c`; the shape elongation is `p = b/a ∈ (0, 1]` (`p = 1` is a sphere),
and `β` is the spin-axis latitude relative to the ecliptic.

**1. Per object — brightness amplitude.** For each apparition, from the phase-corrected intensities
`L` we compute the brightness-dispersion statistic and convert it to an amplitude `A`
(Eq. 7 of Nortunen & Kaasalainen 2017):

$$\eta = \frac{\Delta(L^2)}{\langle L^2\rangle}, \quad \Delta(L^2)=\sqrt{\big\langle (L^2-\langle L^2\rangle)^2\big\rangle}, \qquad A = \sqrt{1 - \dfrac{1}{\dfrac{1}{\sqrt{8}\eta} + \tfrac{1}{2}}}$$

In the code this is `eta = std(L**2)/mean(L**2)` and the `A` formula in
`[lightcurve.py](pyleader/lightcurve.py)`.

**2. Population — forward model.** Pooling `A` over all sampled objects gives the cumulative
distribution `C(A)`. LEADER writes it as a weighted sum of analytic basis functions `F_ij` over a
grid of `(p_i, β_j)` bins, a linear system in the **occupation numbers** `w_ij` (the unnormalized
joint distribution of `p` and `β`):

$$C(A) = \sum_{i,j} w_{ij} F_{ij}(A; p_i, \beta_j) \equiv M\mathbf{w}$$

The matrix `M` is assembled in `[inversion.py](pyleader/inversion.py)`.

**3. Population — regularized inversion.** The weights are recovered by non-negative least squares
with smoothness operators `R_p`, `R_β` that penalize sharp gradients in the `p` and `β` directions
(strengths `δ_p`, `δ_β`):

$$\min_{\mathbf{w}\ge0} \left\lVert \tilde{M}\mathbf{w} - \tilde{C} \right\rVert, \qquad \tilde{M} = \begin{bmatrix} M  \sqrt{\delta_p}R_p  \sqrt{\delta_\beta}R_\beta \end{bmatrix}$$

solved with SciPy's `lsq_linear` under the positivity bound `w ≥ 0`. The peak of `w` gives the
population's most likely `(p, β)`; repeating the experiment over many random draws of the sample
(the *trials*) yields the spread used for error determination.

**4. Bias correction (this package's main addition).** LEADER's recovered `(p, β)` is biased, and the
bias depends on the observing geometry — so it differs from dataset to dataset. PyLEADER therefore
derives a **per-population correction**: it builds synthetic populations of known `(p, β)` observed
at the *same population's* cadence/geometry, measures how LEADER recovers them, and fits a
recovered→true mapping to apply to the real result. This per-trial error determination and
correction machinery are the enhancements used in Sonnett, Lilly & Grav (2025).

## Install

```sh
pip install -e .            # editable install from a checkout
pip install -e ".[obs]"     # also install the .obs-building dependencies
```

This codebase requires Python version >= 3.9 and `numpy`/`scipy`/`matplotlib`; the `[obs]` extra adds
`astropy`/`sunpy`/`requests`, needed only for **building** `.obs` files (which also requires internet
access). Installing puts one console command per script on your `PATH`:


| command                    | script equivalent                       |
| -------------------------- | --------------------------------------- |
| `pyleader-population`      | `python scripts/run_population.py`      |
| `pyleader-download-models` | `python scripts/download_models.py`     |
| `pyleader-build-obs`       | `python scripts/build_obs_files.py`     |
| `pyleader-analysis`        | `python scripts/run_analysis.py`        |
| `pyleader-synthetic`       | `python scripts/run_synthetic.py`       |
| `pyleader-sweep`           | `python scripts/sweep_synthetic.py`     |
| `pyleader-fit-correction`  | `python scripts/fit_correction.py`      |
| `pyleader-plot-sweep`      | `python scripts/plot_sweep.py`          |
| `pyleader-compare`         | `python scripts/compare_populations.py` |


The commands and `python scripts/<name>.py` forms are interchangeable; the examples below use the
installed commands.

## Usage

The pipeline flows in steps. `pyleader-population` runs steps **2–5** in one call (and step 1 with
`--build`); each step is also a standalone command.

```
   population ID  (family "1128"  or  background "BG_IB_Ctypes")
         │
   [0] pyleader-download-models      fetch DAMIT shape models        ── once, prerequisite
         │
   [1] pyleader-build-obs            query NEOWISE → one .obs/object  ── needs [obs] extras + internet
         │
   [2] pyleader-analysis             LEADER inversion → recovered (p, β) + spread
         │
   [3] pyleader-sweep                synthetic (p,β) grid on THIS population's geometry
         │
   [4] pyleader-fit-correction       fit recovered→true correction_function.json
         │
   [5] apply                         corrected (p, β) → population_report.txt

   └── pyleader-population wraps [2]–[5] (and [1] with --build) ──┘
```



### Step 0 — Fetch DAMIT shape models  (`pyleader-download-models` if in the virtual environment or `python scripts/download_models.py` if not)

- **What it does:** 
downloads the representative DAMIT shape models the synthetic step needs. Run
once after cloning. The model *listing* (`asteroideja.txt`) ships with the package; the models
themselves (~29 MB) do not. Since this code connects to the DAMIT database (see References), runtime
is dependent on the internet connection strength, but on my local laptop with typical home internet
speeds, it took about 5 minutes to fully refresh all 347 models specified in this repo.  
- **Input:** 
none (reads the shipped `asteroideja.txt`; queries the DAMIT database).
- **Arguments:** 
`--refresh` re-download every listed model to its latest DAMIT version *(optional;
default fetches only missing ones)*; 
`--damit-dir PATH` destination *(default* `damit_models/`*)*.
- **Output:** 
`<number>.txt` shape files in `damit_models/`.



### Step 1 — Build `.obs` files  (`pyleader-build-obs` or `pyleader-population --build` in virtual environment)

- **What it does:** 
resolves the population to its member objects, queries NEOWISE @ IPAC for clean
photometry, and writes one `.obs` file per object (photometry + Sun/observer geometry per point).
- **Input:** 
  - membership files under `--base-dir` 
  - `AllMBAFamilyMembers.txt` (families) or `BGobjs_<REGION>_<TYPE>type_neowise.txt` (backgrounds)
  - plus `neowise_mainbelt.csv`.
- **Arguments:** 
`--famid ID` *(required)*; 
`--population {family,background}` *(default inferred)*;
`--cat CATALOG` IRSA catalog *(default* `allsky_4band_p1bs_psd`*)*; 
`--filterpriority {w2,w3}`*(default* `w3`*)*; 
`--min-obs N` minimum points to keep an object *(int ≥ 1, default 5)*;
`--legacy-format` write the old block format *(optional; default tabular)*;
`--obsdir DIR` write the `.obs` files to an exact directory instead of the derived path.
- **Output:** 
`<base-dir>/<Fam|>{id}_data_<cat>_<filter>/*.obs` (or `--obsdir` if given).



### Step 2 — Recover the distributions  (`pyleader-analysis`)

- **What it does:** 
runs the LEADER inversion over `Ntrials` random draws of the population and writes
the recovered `(p, β)` distributions with their trial-to-trial spread.
- **Input:** 
  - the population's `.obs` directory (from Step 1) + `neowise_mainbelt.csv` for diameters.
  The directory is derived as `<base-dir>/<Fam|>{id}_data_<cat>_<filter>/`; 
  use `--obsdir DIR` to read from an exact directory that doesn't follow this naming.
- **Arguments:** 
`--famid ID` the integer designation for the collisional family represented by the .obs files *(required)* ; 
`--diam-low` / `--diam-high` diameter window in km *(≥ 0, low < high; default 5–10)*;** 
`--ndraws N` **number of real objects to randomly draw from the .obs files per trial *(int ≥ 1, default 1000)*;** 
`--ntrials N` **number of times to randomly draw** `ndraws` **objects and repeat the analysis *(int ≥ 1, default 100)*;** 
`--phase-angle-limit DEG` **max solar phase angle *(0–90, default 40)*;** 
`--wanted N` **min points per apparition *(int ≥ 3, default 5)*;** 
`--date-tol DAYS` **apparition gap *(> 0, default 60)*; 
`--population {family,background}`; 
`--obsdir DIR` read `.obs` from an exact directory; 
`--forced-n` subsample each object to `wanted` amplitudes; 
`--overwrite`; 
`--seed N`.
- **Output:** 
`<...>_analysis_<...>_<lo>km_to_<hi>km/` with `SummaryAnalysis_*.txt`, per-`Trial*/` diagnostics, and `Summary_pmax/betamax_*.png`.



### Step 3 — Synthetic sweep  (`pyleader-sweep`; single point: `pyleader-synthetic`)

- **What it does:** 
builds synthetic populations of *known* `(p, β)` from DAMIT shapes observed at the
target geometry, runs them through LEADER, and tabulates recovered-vs-assigned statistics across a
grid of assigned peaks.
- **Input:** 
DAMIT models (`damit_models/`) + a geometry source (a directory of `.obs`, or — inside
`pyleader-population` — the analyzed population's own files).
- **Arguments:** 
`--p-peaks P …` assigned elongation peaks *(each* `0 < p ≤ 1`*; default 0.35 0.45 0.55 0.65 0.75)*; 
`--b-peaks B …` assigned latitude peaks in **radians** *(each* `0 < β < π/2`*; default 0.2 0.5 0.9 1.3)*; 
`--ndraws N` synthetic objects per grid point *(int ≥ 1, default 1000)*; 
`--nseeds N` realizations per grid point for error bars *(int ≥ 1, default 1)*;
`--scattering {ls_lambert,hapke}` *(default* `ls_lambert`*, matching the MATLAB code)*;
`--geometry-dir PATH`; 
`--outdir PATH` *(required)*; 
`--seed N`.
- **Output:** 
  - `sweep_stats.csv` (one row per grid point × seed: min/max/mean/median of assigned vs. recovered `p`, `β`)  
  - `sweep_summary.png`. 
  `pyleader-plot-sweep <csv>` re-renders the summary.



### Step 4 — Fit the correction  (`pyleader-fit-correction`)

- **What it does:** 
fits the recovered→true mapping (a 2-D quadratic in recovered `p`, `β`) from a
sweep CSV — the correction to apply to real LEADER output.
- **Input:** 
  - a `sweep_stats.csv` from Step 3.
- **Arguments:** 
`csv` path *(required)*; 
`--stat {peak,mean,median}` which statistic to correct *(default* `mean`*; the pipeline uses* `peak`*, matching LEADER's reported pmax/betamax)*; 
`-o PATH` output JSON.
- **Output:** 
  - `correction_function.json` (coefficients + fit diagnostics) and a predicted-vs-true `correction_fit.png`.



### Step 5 — Apply the correction

Apply a fitted (or the shipped default) correction to real LEADER output:

```python
from pyleader.synthetic import default_correction, load_correction, apply_correction

corr = default_correction()                              # shipped with the package
# corr = load_correction("correction_function.json")     # or a population-specific fit
p_true, beta_true = apply_correction(p_recovered, beta_recovered_deg, corr)
```

`pyleader-compare A.npz B.npz --outdir cmp` reports the L1/L2/L∞ distances between two recovered
distributions.

### The whole pipeline  (`pyleader-population`)

Runs steps 2–5 for one population, deriving the correction from **that population's own** `.obs`
**observing geometry** (the scientifically appropriate choice, since the geometry — hence the bias —
differs per dataset):

```sh
# a collisional family, analyzing an existing .obs dataset end-to-end
pyleader-population 1128 --diam-low 1 --diam-high 100

# a background population, fetching .obs from NEOWISE first
pyleader-population BG_IB_Ctypes --build
```

- **Input:** 
the population's `.obs` directory (or `--build` to create it) + DAMIT models (`pyleader-download-models`; the run stops early with instructions if they are missing).
- **Arguments:** 
the positional population `ID` *(required)*; 
the Step-2 analysis options: 
(`--diam-low/-high`, `--ntrials`, `--ndraws`, `--phase-angle-limit`, `--date-tol`, `--wanted`); 
the Step-3 sweep options: (`--p-peaks`, `--b-peaks`, `--sweep-ndraws`, `--nseeds`, `--scattering`);
`--correction-stat {peak,mean,median}` *(default* `peak`*)*; 
`--build`; 
`--refresh-models` re-download the latest DAMIT models first; 
`--base-dir PATH`;
`--obsdir DIR` read/write `.obs` from an exact directory (the correction sweep's geometry follows it); 
`--seed N`.
- **Output:** 
the analysis directory plus `correction_sweep/`, the population-specific
`correction_function.json` + `correction_fit.png`, and `population_report.txt` (recovered → corrected
`p`, `β`, with an extrapolation warning when the recovered value falls outside the synthetic range).



### `.obs` file format

PyLEADER reads **either** layout — `read_obs()` auto-detects them, so datasets from prior analyses
work unchanged. **Both are fully supported.**

- **Tabular** (written by default): a `#` comment header, then one whitespace-delimited row per
measurement — `jd  sun_x sun_y sun_z  obs_x obs_y obs_z  wavelength flux fluxerr filter`.
- **Legacy block:** the original format (count header; per-epoch Sun/observer vectors and
`λ flux σ filter` lines separated by blank lines). Pass `--legacy-format` to write it.



### As a library

```python
from pyleader import PopulationConfig, run_population

result = run_population(PopulationConfig(pop_id="1128", diam_low=1, diam_high=100), seed=0)
print(result.recovered, "->", result.corrected)   # (p, β_deg) before and after correction
```



## Example: the Hygiea family

The figures below are for the **Hygiea family** (family 10; 3–5 km diameter range). The per-trial
and summary diagnostics come from a 100-trial LEADER analysis.

**Per-trial diagnostics.** The inversion fits the cumulative distribution of light-curve amplitudes
`A`; the relative error measures how well the reconstructed CDF (∑ wᵢⱼFᵢⱼ) matches the observed one.
The solved occupation numbers `w` over the `(p, β)` grid, and the smoothed joint distribution:

Fit of the amplitude CDF
Occupation numbers over (p, beta)
Smoothed joint distribution f(p, beta)

**Population summaries (across all trials).** The peak of the shape (`p`) and spin-axis (`β`)
distributions over all trials, each with a Gaussian fit giving the population value and its spread:

Distribution of p peaks
Distribution of beta peaks

**Per-population bias correction.** Running the full pipeline on this dataset

```sh
pyleader-population 10 --diam-low 3 --diam-high 5 --ntrials 100 --nseeds 3
```

derives a correction from Hygiea's *own* observing geometry: a synthetic `(p_peak, β_peak)` sweep
(20 grid points × 3 seeds) observed at the family's cadence. The **sweep summary** shows how LEADER's
recovered means (colored, ±1σ over seeds) depart from the assigned truth (dashed) as a function of
each input parameter — making the direction of the bias and the `p`–`β` interdependence explicit:

Hygiea sweep summary: recovered vs assigned p and beta

`p` is recovered biased low everywhere, and by *more* at low spin latitude (the blue `β_peak=11°`
curve sits farthest below the diagonal); `β` is compressed toward mid-range (over-estimated below
~50°, under-estimated above), nearly independent of `p_peak`. Fitting a recovered→true mapping to
these points recovers the assigned peaks well (R² = 0.93 for both `p` and `β`):

Hygiea correction fit: corrected vs true p and beta

Applying it de-biases the LEADER result for the population (`population_report.txt`):


| quantity  | recovered | corrected |
| --------- | --------- | --------- |
| `p`       | 0.497     | **0.642** |
| `β` (deg) | 30.4      | **3.1**   |


As the sweep predicts, `p` is corrected upward, and `β` — only weakly constrained by amplitudes and
here near the low edge of the synthetic recovered range (`β_rec ∈ [28°, 90°]`) — shifts toward the
pole; the report flags such near/out-of-range cases as uncertain.

## Package layout

```
pyleader/
  pipeline.py      run_population(): the end-to-end per-population driver
  populations.py   resolve a family / background ID to its member objects
  config.py        AnalysisConfig / ObsBuildConfig / (SyntheticConfig, PopulationConfig)
  obsio.py         read/write .obs files (auto-detects tabular or legacy block format)
  lightcurve.py    read & phase-correct .obs -> amplitudes  (lcg_read_WISE)
  inversion.py     linear inversion for (p, β) occupation numbers  (leader_invert)
  postprocess.py   solution smoothing;  plotting.py  per-trial & summary plots
  analysis.py      run_analysis(): the LEADER experiment driver
  obsfiles/        build .obs files from IRSA + JPL Horizons
  synthetic/       synthetic validation, sweep, and bias correction (from DAMIT models)
  cli/             console-command implementations (scripts/*.py are thin shims)
```



## Notes on the notebook → package conversion

The package supersedes the original Jupyter notebooks (`LEADER_python_final`, `_bg`, `_forcedN`),
which are unified into one configurable driver (`_bg` = `--population background`, `_forcedN` =
`--forced-n`). A few clear bugs were fixed during conversion (each marked `# FIX:` in the source:
phase-correction return value, an apparition off-by-one, the forced-N subsampling, and removal of
dead `interp2d`/`mlab` imports), so results are not bit-for-bit identical to the notebooks.

## References

- Nortunen, H., & Kaasalainen, M. 2017, *LEADER: fast estimates of asteroid shape elongation
and spin latitude distributions from scarce photometry*, Astronomy & Astrophysics, 608, A91.
[doi:10.1051/0004-6361/201731360](https://doi.org/10.1051/0004-6361/201731360)
- Sonnett, S., Lilly, E., & Grav, T. 2025, *Exploring Dynamical and Evolutionary Processes via
Debiased Main Belt Asteroid Size-Frequency Distributions*, EPSC-DPS Joint Meeting 2025,
EPSC-DPS2025-2069. [doi:10.5194/epsc-dps2025-2069](https://doi.org/10.5194/epsc-dps2025-2069)
- Ďurech, J., Sidorin, V., & Kaasalainen, M. 2010, *DAMIT: a database of asteroid models*,
Astronomy & Astrophysics, 513, A46. [doi:10.1051/0004-6361/200912693](https://doi.org/10.1051/0004-6361/200912693)
— source of the shape models used by the synthetic-validation pipeline
([DAMIT database](https://damit.cuni.cz/)).

