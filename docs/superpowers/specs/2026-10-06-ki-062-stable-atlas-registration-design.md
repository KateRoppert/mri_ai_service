# KI-062: Atlas Registration from Several Starts, Best by Metric — Design

> **Status 2026-10-07: fallback option, not the chosen path.** A follow-up
> experiment found skull-stripping before registration (brain → brain
> template) as accurate at roughly today's cost, while multi-start costs
> +70–90 s per session (the +15 s below was measured without the search
> itself). See `2026-10-07-skull-strip-before-registration-design.md`.

**Date:** 2026-10-06
**Branch:** `fix/ki-062-stable-atlas-registration` (from `main`)
**Type:** Bugfix (wrong and unrepeatable atlas registration)

## Problem

Stage 05 registers the reference modality (`t1` for MS) to the 1 mm MNI152
atlas with one rigid `ants.registration` call: identity start, MI on a 20 %
random point sample, no seed (`registration.register_to_atlas`). Everything in
atlas space depends on it: AI masks, lobar / McDonald localisation,
longitudinal diffs, the files uploaded to Kappa.

On SibBMS `P000067` (6 sessions: four 2D with 6 mm slices, two 3D 0.35 mm):

- **Not repeatable.** Same input, same settings: `t1 → atlas` differs between
  runs by up to 29 mm on stored runs, up to 63 mm in a controlled repeat
  (ses-001, 6 repeats). AI masks of one session from two runs: Dice 0.31.
- **Mostly wrong, not just noisy.** A fixed seed makes it repeatable but
  equally bad. The current result has ANTs MI ≈ −0.45 against the atlas; a
  rotation search start reaches ≈ −0.69. On the 3D ses-005 the overlay shows
  ventricles and corpus callosum displaced/rotated today and on the atlas
  with the search start.

## Evidence for the design (2026-10-06, experiments in a 8g container)

Independent truth check: all sessions are the same patient, so a correct
atlas registration puts them on top of each other. Reference: 3D ses-005
(search start, visually verified). Score = correlation with it in its head
mask.

| Session | current | search 1 mm | search 3 mm | picked by lowest MI |
|---|---|---|---|---|
| ses-001 (2D) | 0.30–0.32 | **0.388** | 0.382 | search 3 mm (0.382) |
| ses-002 (2D) | 0.27–0.29 | **0.385** | 0.376 | search 1 mm ✓ |
| ses-003 (2D) | 0.29–0.44 | **0.448** | 0.410 | search 1 mm ✓ |
| ses-004 (2D) | 0.23 | 0.132 ✗ | **0.281** | search 3 mm ✓ |
| ses-006 (3D) | 0.26 | 0.496 | **0.512** | search 3 mm ✓ |

No single start is always right (search 1 mm fails ses-004), but **the
lowest atlas MI among the starts picks the truth-best one in 4/5 and a
near-tie in the fifth.** Principal-axes start never won — dropped.

## Design

In `register_to_atlas` (reference modality → atlas):

1. **Register a ≤1 mm copy** of the moving image (same rule as
   `_downsample_for_registration`, KI-058 fix): the atlas grid is 1 mm, finer
   input adds nothing. Also removes the full-resolution 231 Mvox registration
   from stage 05's memory peak.
2. **Three starts**, each a full rigid registration with `random_seed=42`:
   - `plain` — identity start (today's behaviour, kept as a fallback);
   - `search-1mm` — `ants.affine_initializer` on the ≤1 mm copy vs the atlas;
   - `search-3mm` — the same on 3 mm copies of both.
   (`search_factor=15°`, `radian_fraction=0.2`, `local_search_iterations=10`.)
3. **Pick the lowest `ants.image_mutual_information(atlas, warped)`**; save its
   transform and warped output exactly as today. Downstream (modality
   registration, original-image resampling, stage 07 inverse) is unchanged.
4. **Log the decision**: each start's MI and which won, so a bad case is
   visible in the stage log.
5. Config: `registration.params.atlas_starts: ["plain", "search-1mm",
   "search-3mm"]` (order irrelevant) and `random_seed: 42`. Changes
   `preprocessing_id` — expected: results change.

Cost: ~20 s per session instead of ~5 s (stage 05 is minutes per session).

### Consequences to state, not fix here

- Every atlas-space result produced so far (all runs, both lesion types, and
  what is already in Kappa) used the old registration. They should be
  regenerated; old and new must not be mixed in one longitudinal comparison.
- KI-050 (mask path frozen at the first upload) makes mixing likely — it is
  the next fix.
- Glioblastoma: UPENN inputs arrive already registered; clinical GBM (BO,
  KA*) go through the same code — check one case after the fix.

## Out of scope

- Registering a patient's sessions to each other first (KI-046) — the right
  long-term structure for longitudinal work, bigger change.
- Affine (12 DOF) instead of rigid.

## Testing

- Unit (`scripts/preprocessing_steps/test_atlas_registration_starts.py`):
  chooses the start with the lowest MI (stubbed registrations); logs every
  start; registers on a ≤1 mm copy (spy); seed passed; `atlas_starts` config
  respected; a failing start (exception) is skipped, the others still chosen.
- Repeatability on real data: ses-001 three times → transforms within 1.5 mm.
- Truth check on real data: P000067 full rerun → inter-session consistency
  vs ses-005 ≥ the table above; the lobar/McDonald outputs and the
  longitudinal table regenerated.
