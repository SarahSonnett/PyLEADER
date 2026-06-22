# PyLEADER

A Python version of the LEADER package (originally written in MATLAB; Nortunen et al. 2017),
with a few enhancements for diagnostics and error determination. PyLEADER derives the
distributions of asteroid shape elongation (`p`) and spin-axis orientation (`beta`) for a
population from WISE/NEOWISE thermal photometry.

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

## Notes on the notebook → package conversion

A few clear bugs in the notebooks were fixed during conversion; each fix is marked `# FIX:` in
the source (phase-correction return value, an apparition off-by-one, the forced-N subsampling,
and removal of dead `interp2d`/`mlab` imports). Because of these fixes, results will not be
bit-for-bit identical to the original notebooks.
