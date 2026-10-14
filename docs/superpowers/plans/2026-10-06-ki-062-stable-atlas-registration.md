# KI-062: Atlas Registration from Several Starts — Plan

> **Status 2026-10-07: fallback option, not the chosen path.** A follow-up
> experiment found skull-stripping before registration (brain → brain
> template) as accurate at roughly today's cost, while multi-start costs
> +70–90 s per session (the +15 s below was measured without the search
> itself). See `2026-10-07-skull-strip-before-registration-design.md`.

Spec: `docs/superpowers/specs/2026-10-06-ki-062-stable-atlas-registration-design.md`
Branch: `fix/ki-062-stable-atlas-registration`

Experiments run in containers with `--memory 8g` (an unlimited run took the
laptop down to a global OOM on 2026-10-06). Commit per step only when Kate asks.

## Step 1 — Tests first
`scripts/preprocessing_steps/test_atlas_registration_starts.py`: cases from
the spec, with `ants.registration` / `affine_initializer` /
`image_mutual_information` stubbed so tests are fast and deterministic; one
real tiny-volume smoke test that a start produces a transform file.

## Step 2 — Implement
- `registration.py`: `_atlas_start_kwargs(name, atlas, moving_1mm)`,
  `register_to_atlas(..., atlas_starts=…, random_seed=42)` — copy ≤1 mm,
  run starts, pick min MI, log table, save as today.
- `process_subject_registration`: pass the new params.
- `configs/preprocessing_config.yaml`: `atlas_starts`, `random_seed` with a
  comment pointing to this spec (index-only staging: the laptop's
  `exclude_gpus: []` must stay out of the commit).
- Check: unit tests + all suites.

## Step 3 — Real data
- Repeatability: ses-001 ×3 via the real function → ≤1.5 mm.
- Kate reruns P000067 into a fresh folder (no rebuild: scripts/configs are
  mounted). Then: consistency vs ses-005, overlay PNGs, stage 05 memory and
  time per session from the peak lines.
- Update KI-062 (fixed part, numbers) and note the regeneration need.
