# PyLEADER

A Python version of the LEADER package (originally written in MATLAB; Nortunen & Kaasalainen 2017),
with a few enhancements for diagnostics and error determination. PyLEADER derives the
distributions of asteroid shape elongation (`p`) and spin-axis latitude (`beta`) for a
population from WISE/NEOWISE thermal photometry.

## How it works

PyLEADER implements the **LEADER** method (*Latitudes and Elongations of Asteroid
Distributions Estimated Rapidly*) of Nortunen & Kaasalainen (2017). For large, sparsely
sampled populations, inverting individual lightcurves is infeasible — so instead of solving
for one object at a time, LEADER recovers the **joint distribution of shape elongation `p`
and spin-axis latitude `β` for the whole population** from the statistics of brightness
variations. Each object is modeled as a triaxial ellipsoid with axes `a ≥ b = c`; the shape
elongation is `p = b/a ∈ (0, 1]` (`p = 1` is a sphere), and `β` is the spin-axis latitude
relative to the ecliptic.

**1. Per object — brightness amplitude.** For each apparition, from the phase-corrected
intensities `L` we compute the brightness-dispersion statistic and convert it to an
amplitude `A` (Eq. 7 of Nortunen & Kaasalainen 2017):

$$\eta = \frac{\Delta(L^2)}{\langle L^2\rangle}, \quad \Delta(L^2)=\sqrt{\big\langle (L^2-\langle L^2\rangle)^2\big\rangle}, \qquad A = \sqrt{1 - \dfrac{1}{\dfrac{1}{\sqrt{8}\,\eta} + \tfrac{1}{2}}}$$

In the code this is `eta = std(L**2)/mean(L**2)` and the `A` formula in
[`lightcurve.py`](pyleader/lightcurve.py).

**2. Population — forward model.** Pooling `A` over all sampled objects gives the cumulative
distribution `C(A)`. LEADER writes it as a weighted sum of analytic basis functions
`F_ij` over a grid of `(p_i, β_j)` bins, which is a linear system in the **occupation numbers**
`w_ij` (the unnormalized joint distribution of `p` and `β`):

$$C(A) = \sum_{i,j} w_{ij}\, F_{ij}(A; p_i, \beta_j) \;\equiv\; M\mathbf{w}$$

The matrix `M` is assembled in [`inversion.py`](pyleader/inversion.py).

**3. Population — regularized inversion.** The weights are recovered by non-negative least
squares with smoothness operators `R_p`, `R_β` that penalize sharp gradients in the `p` and
`β` directions (strengths `δ_p`, `δ_β`):

$$\min_{\mathbf{w}\,\ge\,0} \left\lVert \tilde{M}\mathbf{w} - \tilde{C} \right\rVert, \qquad \tilde{M} = \begin{bmatrix} M \\\\ \sqrt{\delta_p}\,R_p \\\\ \sqrt{\delta_\beta}\,R_\beta \end{bmatrix}$$

solved with SciPy's `lsq_linear` under the positivity bound `w ≥ 0`. The peak of `w` gives the
population's most likely `(p, β)`; repeating the whole experiment over many random draws of the
sample (the *trials*) yields the spread used for error determination — the Gaussian-fit summary
histograms shown below. This per-trial error determination and the accompanying diagnostics are
the enhancements added here, used in Sonnett, Lilly & Grav (2025).

## Repo contents

The science is now a modular `pyleader` package (the original notebooks are kept for reference):

```
pyleader/
  config.py        AnalysisConfig / ObsBuildConfig (replaces the notebook "top cell")
  naming.py        designation conversions (analysis side)
  lightcurve.py    read & phase-correct .obs files -> amplitudes  (lcg_read_WISE)
  inversion.py     linear inversion for (p, beta) occupation numbers  (leader_invert)
  postprocess.py   solution smoothing  (leader_postprocess_WISE)
  plotting.py      per-trial and summary plots
  analysis.py      run_analysis(): the main experiment driver
  obsfiles/        build .obs input files from IRSA + JPL Horizons
scripts/
  run_analysis.py      CLI for the analysis
  build_obs_files.py   CLI for building .obs input files
```

The three original analysis notebooks (`LEADER_python_final`, `_bg`, `_forcedN`) are unified
into one configurable driver: `_bg` is `--population background`, and `_forcedN` is `--forced-n`.

## Install

```sh
pip install -r requirements.txt
```

The analysis path needs only `numpy`/`scipy`/`matplotlib`. Building `.obs` files additionally
requires `astropy`/`sunpy`/`requests` and internet access.

## Usage

### Run the analysis

Defaults reproduce the original `LEADER_python_final` configuration:

```sh
python scripts/run_analysis.py
```

Common overrides:

```sh
# Background population
python scripts/run_analysis.py --famid BG_PB_Ctypes --population background

# Forced-N (subsample each object to `wanted` amplitudes)
python scripts/run_analysis.py --famid 4 --forced-n --wanted 11 --diam-low 3 --diam-high 5

# Quick test run
python scripts/run_analysis.py --ntrials 2 --ndraws 50 --overwrite --seed 0
```

Run `python scripts/run_analysis.py --help` for the full list of options. Inputs are read from
`<base-dir>/<Fam><famid>_data_<cat>_<filter>/` and results are written to a sibling
`*_analysis_*` directory.

### Build .obs input files

```sh
python scripts/build_obs_files.py --famid 350
python scripts/build_obs_files.py --famid 350 --curl-only   # just write the bulk curl script
```

### As a library

```python
from pyleader import AnalysisConfig, run_analysis

cfg = AnalysisConfig(famid="3815", diam_low=5.0, diam_high=10.0, Ntrials=2, Ndraws=50, overwrite=True)
outdir = run_analysis(cfg, seed=0)
```

## Example output

The figures below come from a run on the Hygiea family (family 10; 3–5 km diameter range,
100 trials), produced by the original notebook workflow. A run writes per-trial diagnostics
into each `Trial*/` subdirectory and population-level summaries at the top of the output
directory.

**Per-trial diagnostics**

The inversion fits the cumulative distribution of light-curve amplitudes `A`. The relative
error quantifies how well the reconstructed CDF (∑ wᵢⱼFᵢⱼ) matches the observed one:

![Fit of the amplitude CDF](docs/images/RelativeError.png)

The solved occupation numbers `w` over the (shape `p`, spin-axis `β`) grid, and the same
solution after smoothing into a joint distribution f(p, β):

![Occupation numbers over (p, beta)](docs/images/OccupationNumbers_w.png)
![Smoothed joint distribution f(p, beta)](docs/images/Solutions_smoothed.png)

**Population summaries (across all trials)**

The peak of the shape (`p`) and spin-axis (`β`) distributions over all 100 trials, each with a
Gaussian fit giving the population value and its spread:

![Distribution of p peaks](docs/images/Summary_pmax.png)
![Distribution of beta peaks](docs/images/Summary_betamax.png)

## Notes on the notebook → package conversion

A few clear bugs in the notebooks were fixed during conversion; each fix is marked `# FIX:` in
the source (phase-correction return value, an apparition off-by-one, the forced-N subsampling,
and removal of dead `interp2d`/`mlab` imports). Because of these fixes, results will not be
bit-for-bit identical to the original notebooks.

## References

- Nortunen, H., & Kaasalainen, M. 2017, *LEADER: fast estimates of asteroid shape elongation
  and spin latitude distributions from scarce photometry*, Astronomy & Astrophysics, 608, A91.
  [doi:10.1051/0004-6361/201731360](https://doi.org/10.1051/0004-6361/201731360)
- Sonnett, S., Lilly, E., & Grav, T. 2025, *Exploring Dynamical and Evolutionary Processes via
  Debiased Main Belt Asteroid Size-Frequency Distributions*, EPSC-DPS Joint Meeting 2025,
  EPSC-DPS2025-2069. [doi:10.5194/epsc-dps2025-2069](https://doi.org/10.5194/epsc-dps2025-2069)
