# KI-049: BET Fails on a Path with a Space, Says Nothing, and the Step Passes — Design

**Date:** 2026-10-21
**Branch:** `fix/ki-049-bet-silent-failure` (from `main` 3e8d69f)
**Type:** Fix

## Why

KI-049 (2026-08, patient KA117 on the second machine): the output root was
under `…/Рабочий стол/…`, BET failed, the log said
`BET failed with return code 1: ` (empty), the step logged
`✓ Skull stripping completed` and the session counted as a success, with the
skull still on the images that went to segmentation.

Re-checked 2026-10-21 in the current image (`brain-lesion-web`, FSL `bet`):

| path | result |
|---|---|
| `/w/plain/t1.nii.gz` | rc 0, brain + mask |
| `/w/with space/t1.nii.gz` | rc 1, `Error: input image /w/with space/t1 not valid` on **stdout**, stderr empty |
| `/w/кириллица/t1.nii.gz` | rc 0 — Cyrillic is fine; the space breaks it |

Still true in the code:

1. `run_bet` builds its error from `result.stderr` only → empty message.
2. `run_bet` reports success without checking that the stripped image and
   the mask were written.
3. Stage 05 step 4 (`after_registration`): a failed mask only sets
   `steps.skull_stripping.success = False`; the step still logs ✓ and the
   session ends `success = True`. (`before_registration`, the default since
   2026-10-14, already raises on a failed native mask — but with the same
   empty message.) A failure of `get_stripper` returns
   `{"success": False, "error": …}`, whose values are not per-modality
   dicts, so `all(r.get(...))` dies with `AttributeError` — the session fails
   by accident, with a misleading error.

## Change

1. **Space-proof BET.** `run_bet` runs BET in a private `tempfile.mkdtemp()`
   directory (system temp, no spaces): the input is symlinked there as
   `in.nii.gz`, BET writes `out.nii.gz` / `out_mask.nii.gz`, and the results
   are moved to the requested paths. The caller's paths can be anything.
   Temp dir removed in `finally`.
2. **Real error text.** Error = return code + stdout + stderr (trimmed),
   as `check_fsl_installed` already does.
3. **Success means files.** BET rc 0 but no stripped image, or no mask when
   one was asked for → `success: False` with the reason.
4. **A failed skull strip fails the session.** Stage 05 step 4 raises when
   the reference mask failed or any modality could not be masked, exactly as
   the native path already does; the per-session error then lands in the
   stage report and the "lost patients" view (KI-054). The `get_stripper`
   failure shape is handled explicitly.

No change to HD-BET (subprocess with list args; checked with a spaced path
in verification) or to the fallback policy (fallback stays "tool
unavailable", not "tool failed").

## Verification

- Unit tests (`skull_stripping/tests/test_bet_paths.py`, real FSL `bet` in the
  image, skipped without it): spaced output root → success, brain + mask at
  the requested paths; forced failure (invalid input) → `success: False`
  with BET's stdout text in `error`; rc 0 without mask → failure.
- Stage 05 test: a stripper returning failure → session `success: False`,
  error names skull stripping, step not logged as ✓.
- Live: stage 05 on one SibBMS session with `root_output_dir` under
  `…/Рабочий стол/…`, both skull-strip positions, BET forced (`method: bet`)
  and HD-BET; output skull-stripped, session success.

## Results (2026-10-21)

SibBMS sub-003_ses-001 (`14_10_1651`), output root `/w/Рабочий стол/…`:

| code | tool | position | session | mask | non-zero voxels in t1 |
|---|---|---|---|---|---|
| old (main) | BET | before | failed: "BET failed with return code 1: " | no | — |
| old (main) | BET | after | **success** | **no** | 0.526 (skull on) |
| new | BET | before | success | yes | 0.243 |
| new | BET | after | success | yes | 0.253 |
| new | HD-BET | before | success | yes | 0.203 |
| new | HD-BET | after | success | yes | 0.205 |

Tests: `test_bet_paths.py` (5, real `bet` for the spaced path),
`test_stage05_skull_strip_failure.py` (5); `scripts/` + `backend/`
597 passed, the 3 known environment failures unchanged.
