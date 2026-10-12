# Stage 05 Memory — Plan

Spec: `docs/superpowers/specs/2026-10-12-stage05-memory-design.md`
Branch: `fix/stage05-memory` (worktree `../mri_ai_service-mem`)

Tests run in a throwaway `brain-lesion-web:demo` container with the
worktree mounted read-only (`--user 1000:1000`,
`PYTHONDONTWRITEBYTECODE=1`), so the worktree gets no root-owned files.

## Step 1 — N4 on a ≤1 mm copy

- `bias_correction.py`:
  - `apply_n4_bias_correction(..., max_spacing_mm=None)`: read with
    `sitk.ReadImage(..., sitkFloat32)`; if `max_spacing_mm` is set and an
    axis is finer, resample (linear, same origin/direction, size from the
    extent) before Otsu/N4; write with `sitk.WriteImage` (header from the
    resampled image).
  - `process_subject_bias_correction(..., max_spacing_mm=None)`: passes it
    through only when writing to the temp dir for registration; for
    `use_for_registration_only: false` it is ignored (warning if set).
- `05_preprocessing.py`: pass
  `registration.params.max_registration_spacing_mm` (default 1.0).
- Tests `scripts/preprocessing_steps/test_bias_correction_spacing.py`:
  0.35 mm phantom → 1 mm output, 1 × 1 × 5 mm untouched, physical bounds
  and direction preserved, full-resolution output when not
  registration-only; bias of a synthetic field removed (sanity).

## Step 2 — single-process HD-BET

- New `skull_stripping/hdbet_runner.py` (`-i -o -device [--disable_tta]`,
  `if __name__ == "__main__"`, 1 + 1 workers, removes nnU-Net's json
  side files, `apply_bet`).
- `hdbet.py`: `_hdbet_python()` from the `hd-bet` shebang; command =
  `[python, runner, ...]`; `is_available` also requires the shebang to be
  readable; docstring updated.
- Tests in `skull_stripping/tests/test_hdbet_runner.py`: command
  construction, `--disable_tta` rules unchanged, missing CLI →
  unavailable; a container-only test that imports the HD-BET internals and
  checks `predict_from_files` accepts the two `num_processes_*` arguments
  (skipped when HD_BET is not importable).

## Step 3 — verification

1. Full test suites: `scripts/preprocessing_steps`, `skull_stripping/tests`,
   `backend/` (no regressions).
2. Memory profile on SibBMS ses-005, both positions (worktree scripts
   mounted), compare with the 2026-10-12 table.
3. HD-BET mask identity CLI vs runner: one GBM and one MS session.
4. Registration shift old vs new vs old-vs-old (2 runs) on SibBMS 3D,
   MS 2D and GBM; report in the spec's results section.
5. Update `k_bytes_per_voxel` for stage 05 in `configs/resource_config.yaml`
   from the new peaks; KNOWN_ISSUES: KI-058 follow-up entry.
6. Kate: multi-session SibBMS run with both services up (after merge or
   with the branch checked out).

No commit / merge / push without Kate's request.
