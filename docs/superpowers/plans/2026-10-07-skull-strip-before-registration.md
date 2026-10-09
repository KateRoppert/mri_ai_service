# Experiment: Skull Stripping Before Registration — Plan

Spec: `docs/superpowers/specs/2026-10-07-skull-strip-before-registration-design.md`
Branch: `exp/skull-strip-before-registration`, worktree
`/media/storage/home/demob/mri_ai_service-exp`. The main checkout and the
running `web` stay on `main`. Commit only when Kate asks.

Every container run here uses `--memory 9g` (an unlimited one took the laptop
into a global OOM on 2026-10-06).

## Step 1 — Switch + helpers (TDD)
Tests next to the code: `scripts/preprocessing_steps/test_skull_strip_position.py`.
- `skull_strip_position(params)` → `after_registration` by default, validates.
- `resolve_brain_template(atlas_path)` → sibling `<name>_brain.nii.gz`, else
  `$FSLDIR/data/standard/<name>_brain.nii.gz`, else a clear error.
- `create_native_brain_mask(ref_file, work_dir, params, gpu_pool)` — ≤1 mm
  copy, same stripper / fallback / GPU pinning as today (shared helper
  extracted from `process_subject_skull_stripping`, behaviour unchanged).
- `warp_mask_to_atlas(native_mask, atlas, transform)` — nearest neighbour,
  binary, atlas grid.
- `register_to_atlas(..., moving_override=None, atlas_override=None,
  random_seed=None)` — defaults reproduce today's call exactly.

## Step 2 — Wire into Stage 05
`process_single_subject`: if `position == before_registration` →
native mask after N4; registration on brain ↔ brain template (seeded); step 4
warps + applies the mask instead of running the stripper. `after` path
untouched. Check: all existing suites green (default path unchanged).

## Step 3 — Smoke
One MS 2D session + one MS 3D session + one GBM session, `before`, in a
throwaway container: stage 05 completes, outputs have today's names/spaces,
mask in `transformations/` in atlas space, peak memory logged.

## Step 4 — Runs
`demo_workspace/exp_ss_order/{after,before}/{run1,run2}/<set>/` via
`orchestrator.py` in containers on the compose network (stage 06 needs the
seg services), worktree code mounted, laptop GPU config (`exclude_gpus: []`)
in the experiment's preprocessing configs. Sets: SibBMS P000067 + P000010,
MS_5 P000915, dropbox_33 ×6, BO ×6, UPENN ×3 (check skull first).

## Step 5 — Analysis + report
Scripts under `research/ss_order_experiment/`; report
`docs/experiments/2026-10-07-ss-order.md`: registration (MI, repeatability,
inter-session consistency, overlays), masks, downstream lesions / zones,
cost. Recommendation for production.
