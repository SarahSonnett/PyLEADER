# Correction v2: posterior inversion + response-matrix unfolding

**Status:** planned, not implemented (written 2026-07-08; target: next work session).
**Prerequisite reading:** README "How it works" + "Per-population pipeline"; `pyleader/synthetic/`.

## Why (scientific motivation)

The current per-population correction (Steps 4–6) fits a single-valued 2-D quadratic
`(p_rec, β_rec) → (p_true, β_true)` from the sweep. That captures the *mean* systematic bias but is
scientifically incomplete:

1. **Degeneracy** (e.g., sphere at β=0 vs. elongated object pole-on) makes the recovered→true mapping
   **many-to-one**; a deterministic correction silently picks one answer where several are consistent.
2. **No uncertainties** are propagated to the corrected values (bare point estimates in the report).
3. The correction is **conditional on the assumed synthetic family** (narrow p peak ±0.075; β 75%
   Gaussian σ≈3° + 25% uniform) — real populations may be broad/multimodal.
4. Only **scalar summaries** (peak/mean/median) are corrected, never the distribution shape, and p/β
   are fit as independent surfaces (no covariance).

Fix: treat the sweep as a sampled **forward model** and invert it probabilistically (Phase 1), then
extend the same basis runs into a **response matrix** to unfold full distributions (Phase 2).

---

## Phase 0 — shared delta-basis infrastructure (both phases consume this)

One basis-run campaign per population: near-delta synthetic populations on a fine true-(p, β) grid,
observed at the population's own geometry, each pushed through the full pipeline.

Changes:

1. **Delta preset** — the config knobs already exist on `SyntheticConfig`
   ([pyleader/synthetic/config.py](../../pyleader/synthetic/config.py)):
   `beta_peak_chance=1.0`, `beta_jitter≈0.01`, `p_accept_tol≈0.02`, `p_escape_chance=0.0`.
   Add a `SyntheticConfig.delta_preset(p_peak, b_peak, ...)` classmethod (or `--delta` CLI flag).

2. **Save the joint solution** — `SyntheticResult.save()`
   ([pyleader/synthetic/population.py](../../pyleader/synthetic/population.py)) currently stores only
   marginals; add the full `W` (20×29 occupation numbers) + the run's `P`/`BETA` grids to the `.npz`.

3. **Canonical recovered-bin grid** — `leader_invert`
   ([pyleader/inversion.py](../../pyleader/inversion.py)) jitters the recovered `P`/`BETA` grids with
   truncated-Gaussian noise on every call (inherited from MATLAB), so bins differ run-to-run.
   Two-part fix: (a) add `grid_jitter: bool = True` param to `leader_invert` so basis runs can
   disable it; (b) add a `rebin_to_canonical(P, BETA, W)` helper (linear interp onto the unjittered
   `linspace` grids) for correcting *real* analyses, which keep jitter. Decide at implementation
   whether real-data analysis should also default to no-jitter (cleaner) — check effect on results.

4. **Parallel + chunked + resumable runner** — new `pyleader/synthetic/basis.py`:
   - `run_basis(cfg, grid, nseeds, outdir, nproc=None, task=None)`:
     `multiprocessing.Pool` over (grid-point × seed); each unit writes
     `outdir/gp{ij}_seed{s}/synthetic_result.npz`; **skip units whose npz already exists**
     (resumability); `task="k/N"` runs only the k-th of N chunks (cluster arrays / restarts).
   - matplotlib must stay `Agg` and figures off (`make_plots=False`) in workers.
   - CLI `pyleader-basis` (+ `scripts/basis_runs.py` shim, pyproject entry point):
     `pyleader-basis <pop_id> --grid-np 8 --grid-nb 8 --nseeds 4 --nproc 10 [--task k/N]
     [--obsdir DIR]` — reuses `PopulationConfig` plumbing for geometry/tolerances.

Grid defaults to start with: `p ∈ linspace(0.30, 0.80, 8)`, `β ∈ linspace(0.10, 1.45, 8)` rad,
`nseeds=4`, `Ndraws=1000` → 256 runs/population.

---

## Phase 1 — posterior-inversion correction

New `pyleader/synthetic/posterior.py`:

1. `build_forward_table(basis_dir) -> ForwardTable`: per true grid point, the mean vector and
   covariance (across seeds) of the recovered summary `(p_peak_rec, β_peak_rec)` (optionally also
   mean/median variants). Persist as `forward_table.npz`.
2. `posterior_correct(p_rec, b_rec, table) -> Posterior`: Gaussian likelihood of the observed
   recovered pair at every true grid point (bilinear-interpolated mean/cov between grid points);
   normalized posterior over the true grid. `Posterior` carries: MAP, marginal medians,
   68/95% credible intervals for p and β, the full 2-D posterior array, and a **multimodality flag**
   (>1 disjoint 68% region).
3. Plots: 2-D posterior map with credible contours + the recovered point; 1-D marginal posteriors.
4. Integration:
   - `run_population` gains `correction_method: {"quadratic","posterior","both"}` (default `"both"`
     during validation; decide final default later). Posterior path requires a basis dir
     (`--basis DIR`, default `<outdir>/basis/`); if absent, instruct to run `pyleader-basis`.
   - `population_report.txt` gains `p = X (+u/−l, 68%)`, `β = Y (+u/−l, 68%)`, multimodality warning.
   - Also (cheap interim, independent of basis): propagate quadratic-fit uncertainty
     `σ = RMSE ⊕ local seed scatter` into the existing report.
5. **Validation:** (a) coverage test — synthetic populations at *off-grid* true peaks; the 68%/95%
   intervals must cover truth at the right rates; (b) leave-one-out on grid points; (c) degeneracy
   demo — a near-spherical + a pole-on-elongated population should yield visibly widened/bimodal
   posteriors (this is the acceptance test that the method expresses the ambiguity).

## Phase 2 — response-matrix unfolding (full distributions)

New `pyleader/synthetic/unfold.py`:

1. `build_response(basis_dir) -> R`: columns = true grid points (e.g. 64), rows = canonical
   recovered bins — **joint `W`** (580 = 20×29) preferred to preserve p–β covariance; stacked
   marginals (49) as a cheaper fallback. Average over seeds; keep per-bin seed variance for weights.
   Persist `response_matrix.npz`.
2. `unfold(recovered, R, ...)`: solve `R f = d`, `f ≥ 0`, with 2-D smoothness regularization on the
   true grid — same `lsq_linear` + Tikhonov-block pattern as `leader_invert`. Regularization
   strength via L-curve or CV on synthetic mixtures. Uncertainties: ensemble of solutions from
   data-vector perturbations (using seed covariance) → per-bin bands on `f_true`.
3. **Mixture generator for validation** — extend the synthetic population sampler so true (p, β) can
   be drawn from a user-specified mixture/broad distribution (new `SyntheticConfig` option, e.g.
   `truth_sampler=callable` or parametric `(weights, components)`), *not* just single peaks.
   Unfold these known-truth mixtures; the residual quantifies the mixture-linearity (NNLS) model
   error — fold it into the error budget. This validation is mandatory before trusting results.
4. Outputs: corrected joint `f_true(p, β)` map + corrected marginals with uncertainty bands;
   CLI `pyleader-unfold <analysis_outdir> --basis DIR`.
5. Note honestly in docs: output resolution ~8×8–10×10 true bins is the information content;
   degeneracy appears as broad/correlated bands (feature, not bug).

## Docs / bookkeeping

- README: new "Step 4b/5b — probabilistic correction" subsection once Phase 1 lands.
- Gitignore basis outputs (`basis/` dirs; large, regenerable).
- Keep quadratic path working throughout (back-compat; comparisons in validation).

---

## Compute strategy

Measured: `run_synthetic` at `Ndraws=1000` ≈ **21 s single-core** (this machine, M3 Max).

| Campaign | runs | serial | pooled (10 workers, local) |
|---|---|---|---|
| Basis, 1 population (8×8×4) | 256 | ~1.5 h | **~10–15 min** |
| Response-grade, 1 pop (10×10×5) | 500 | ~3 h | **~20–30 min** |
| 50-population campaign (8×8×4) | 12,800 | ~75 h | **~8–10 h (overnight)** |

**Local-first is the plan**: the M3 Max (14 cores / 36 GB) handles single populations in minutes and
a full campaign overnight once the Phase-0 pool exists. Memory per worker is small (<1 GB).

**Cluster escape hatch** (only needed for much larger campaigns / finer grids / fast turnaround):
the workload is embarrassingly parallel and the `--task k/N` chunk flag makes submission trivial —
one array element per chunk, no inter-task communication, pip-installable package (wheel verified).
- **OSPool (OSG)** — free for US-based researchers; ideal high-throughput fit (many independent
  single-core tasks); HTCondor submit file ≈ 10 lines.
- **NSF ACCESS "Explore" allocation** — free, lightweight application (abstract-level); gives SLURM
  clusters (Anvil/Bridges-2/Expanse). SLURM array: `#SBATCH --array=0-63`, each task
  `pyleader-basis ... --task ${SLURM_ARRAY_TASK_ID}/64`.
- **NASA HEC (Pleiades/Aitken)** — free if the work is NASA-funded (likely applies); allocation
  request through the HEC portal.
- **Commercial cloud** — ~150 CPU-h ≈ **$5–15** at spot prices; cheap but setup overhead likely
  exceeds the benefit vs. running locally.

## Open questions (decide at implementation)

1. Freeze `leader_invert` grid jitter for real analyses too, or rebin? (Check effect on results.)
2. Match basis `Ndraws` to the *real* population's usable-amplitude count (noise realism) vs. fixed 1000?
3. Grid ranges/resolution per population (p 0.30–0.80? β full 0–π/2?) and `nseeds`.
4. Default `correction_method` after validation (`posterior` vs `both`).
5. Joint-`W` vs stacked-marginal response rows (start joint; fall back if too noisy).

## Effort estimate

Phase 0 ≈ half a day; Phase 1 ≈ one day (+ validation runs); Phase 2 ≈ 1–1.5 days (+ the real
science time: regularization tuning and mixture-validation interpretation).
