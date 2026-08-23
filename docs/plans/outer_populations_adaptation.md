# Adapting the pipeline for Hildas and Jupiter Trojans (and Cybeles)

Plan drafted 2026-08-23. Governing principle: every MBA-tuned assumption gets a measured
sensitivity test at outer-population geometry before production runs; the paper's caveats are
then measured numbers, not guesses. Cybeles are close enough to the outer belt that the MBA
machinery likely transfers with only the standard per-population validation; Hildas and
Trojans share the deeper concerns and are treated together below.

## Phase 0 — zero-compute preparation

1. **NEOWISE tables**: fetch the Hilda and Trojan tables from the PDS Diameters & Albedos
   V2.0 bundle (same DOI as the main-belt table); verify whether the Cybele region is inside
   the main-belt table's semimajor-axis bound or needs its own extraction. (Also a fresh-
   download stress test of the documented ingest path.)
2. **Rotation-period priors from literature**: assemble measured period distributions —
   K2 campaigns are the best uniform source (Szabó et al. 2017 for Trojans; Kalup et al. 2021
   for Hildas + Trojans), supplemented by ground-based compilations. Deliverable: an empirical
   period sample (or fitted lognormal) per population, with provenance.
3. **DAMIT audit**: count Trojan/Hilda entries in the shape library (there are a few — e.g.
   (624) Hektor — but not many). Expectation: too few to build a dedicated library; the test
   in Phase 2 quantifies the consequence instead.
4. **Simmer range checks**: confirm the NEATM flux lookup and cadence/FOV tables cover
   r = 4-5.5 au temperatures and the L4/L5 sky tracks; note the known L4/L5 sample asymmetry
   in the cryo mission.

## Phase 1 — one small code change

**Configurable rotation-period prior.** Replace the hard-coded uniform 3-12 h draw with an
optional `trot_sampler` callable on `SyntheticConfig` (mirroring the existing
`truth_sampler` pattern), defaulting to the current behavior. Record the prior in
`basis_info.json` (same consistency rule as the noise model: one prior per basis; warn on
resume mismatch). This is the only code the adaptation strictly requires — everything else
in the per-population design (geometry borrowing, empirical noise calibration, per-population
basis) transfers automatically, which is the design's whole point.

Why it matters mechanically: a WISE apparition spans ~1-3 days sampled every ~1.6 h. For
P ~ 3-12 h rotators the points sample rotation phase quasi-randomly and η measures the full
lightcurve variance. A P ≳ 30-50 h rotator's points cover only part of a cycle per apparition,
suppressing the measured η — a bias the synthetics must reproduce with the *right period
distribution*, or the correction miscalibrates in exactly the population where slow rotators
are common (Trojans; Hildas to a lesser degree).

## Phase 2 — calibration experiments (order matters; ~2-3 nights total)

- **2a. Period-prior sensitivity, run on a known MBA family first.** Rebuild one existing
  family's basis with the literature MBA period distribution instead of uniform 3-12 h;
  compare posteriors. This measures the knob's leverage where we have ground truth and tells
  us (i) how much period priors matter at all, and (ii) whether the existing MBA results
  carry a retroactive caveat. Cheap: one basis rebuild (~1.5 h).
- **2b. Trojan-geometry coverage + mixture tests.** Build a basis at real Trojan geometry
  (after Phase 0's data build), then run the standard validation battery *before* looking at
  real results: inject known fixed peaks and count 68/95% posterior coverage; run the mixture
  test to measure β localization there. Expect degeneracy to be *worse* than the main belt:
  Trojans are observed over a tiny phase-angle range (α ≲ 12°) with less geometric leverage,
  and the honest outcome may be "β is unconstrained for Trojans; quote p only." Better to
  learn that from synthetics than from a referee. Same battery for Hildas.
- **2c. Simmer truth-recovery at outer orbits.** SynthPop population on Hilda/Trojan orbits
  with a known SFD -> Simmer -> verify the debiased SFD recovers truth. The η(D) window
  geometry is a-dependent by design but validated only at main-belt synodic periods; this is
  the direct test. Run L4 and L5 separately (different cryo epoch coverage).

## Phase 3 — production, with structure decisions locked in advance

- L4 and L5 as **separate populations** end-to-end (geometry, noise, basis, corrections);
  combine only if their posteriors agree within uncertainties.
- Diameter bins set by each population's detection floor (likely 10-25 and 25-100 km for
  Trojans; 5-10 possible for Hildas/Cybeles).
- Pre-registered reporting rules from 2b: which quantities (p peak; β only if coverage tests
  support it) each population is allowed to claim, written into §4.2 of the summary document
  *before* results are inspected.
- Membership by dynamical-region cuts (BGobjs-style), each with an Eulalia-grade provenance
  README.

## Standing scientific caveats (to be quantified, not assumed)

- DAMIT shape library remains MBA-trained; bound via half-library jackknife bases (2b add-on)
  rather than by restricting to the handful of Trojan shapes.
- The reflected-light scattering approximation on thermal fluxes is inherited everywhere; at
  Trojan temperatures the approximation error may differ from the main belt — absorbed to
  first order by the per-population correction, stated in the caveats inventory regardless.
- Cross-population comparisons (Trojans vs main belt) inherit differential systematics
  beyond those in §4.2 (period priors, shape libraries, thermal regime); treat such
  comparisons as qualitative unless the Phase-2 tests bound them.
