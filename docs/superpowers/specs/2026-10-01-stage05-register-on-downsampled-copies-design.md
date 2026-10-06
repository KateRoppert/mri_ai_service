# Stage 05: Compute Intra-Session Registration on ≤1 mm Copies — Design

**Date:** 2026-10-01
**Branch:** `fix/stage05-register-on-downsampled-copies` (from `main`)
**Type:** Bugfix (OOM on high-resolution 3D sessions)

## Problem

SibBMS `P000067` (MS, 6 sessions) loses ses-005 and ses-006 in stage 05 on
every run — on CPU and on GPU skull stripping alike (runs `30_09_1752`,
`1_10_1916`). The worker is killed while registering `t2fl` to `t1`:

- 2026-10-01: `CONSTRAINT_MEMCG` in the 9g `web` container, worker at 9.2 GB
  anon RSS. The container cap did its job (the host survived); the work itself
  needs more than the whole container.

The two sessions are a different acquisition from the other four:

| Sessions | `t1` | `t2fl` |
|---|---|---|
| 001–004 | 1.4–2.6 Mvox (2D, 6 mm slices) | 1.8–4.5 Mvox |
| 005 | **231 Mvox** (310×864×864, 0.35 mm) | **171 Mvox** |
| 006 | **120 Mvox** | **186 Mvox** |

## Cause

`registration.register_modalities()` (`scripts/preprocessing_steps/registration.py:294`)
runs `ants.registration(fixed=native t1, moving=native t2fl)` at full native
resolution with ANTs defaults, and writes `warpedmovout` on the native `t1`
grid (231 Mvox) to a temp file that is deleted a few lines later.

Nothing downstream needs native resolution here: every stage-05 output is
resampled onto the 1 mm atlas grid (`apply_transforms(fixed=atlas, …)`). A
rigid transform lives in physical (mm) space, so a transform computed on
1 mm copies applies unchanged to the full-resolution originals.

## Design

In `register_modalities()`:

- Before registering, resample **fixed and moving** to `spacing_i =
  max(native_spacing_i, registration_max_spacing_mm)` per axis (default
  1.0 mm; never upsample; linear interpolation). Images already ≥1 mm on
  every axis pass through untouched.

  **Correction after the verification run (2026-10-01):** this originally
  said sessions 001–004 pass untouched. They do not — they are 2D with
  thick slices, but their in-plane spacing is 0.47–0.98 mm, so their copies
  are coarsened in-plane too. Measured on `1_10_1939` vs the full-resolution
  run `1_10_1916`: transform differences ≤1 mm at the corners of a
  200×240×200 mm box (one case, ses-001 `t2`, 3 mm), against a run-to-run
  spread of ≤1.2 mm between two full-resolution runs; t2/t2fl–t1 mutual
  information equal within ±0.002 (5 of 8 slightly better). Accepted.
- Register the copies; save the transform exactly as today.
- Keep writing the temp `warpedmovout` (now small) — nothing else changes in
  the call chain (`apply_transforms` to atlas, inverse transforms in stage 07).
- New optional param `registration.params.max_registration_spacing_mm`
  (default 1.0) in `configs/preprocessing_config.yaml`, documented.
  Note: adding the key changes the preprocessing_id hash (expected — results
  for high-res sessions differ).

`register_to_atlas()` is not changed: its fixed image is the 1 mm atlas
(7 Mvox) and it already survives the 231 Mvox moving `t1`.

### Evidence (probe, 2026-10-01, 9g container, ses-005 `t2fl`→`t1`)

- Copies: t1 (171,300,300), t2fl (174,290,290). Registration 22.7 s.
- Whole-process peak RSS 6.66 GB (mostly the two full-size images loaded for
  the check) vs >9.2 GB killed before.
- Transform: <1° rotation, ~0.3 mm shift (same visit, as expected).
  Correlation with `t1` on the 1 mm grid: 0.754 unregistered → 0.764 registered.

### Risks / follow-ups (not in this branch)

- Stage 07 warps masks back onto native grids (231 Mvox for ses-005).
  Nearest-neighbour on uint8 is cheaper than registration but unmeasured on
  these sizes — check on the verification run; if it OOMs, record in KI-058.
- KI-058: correct the earlier hypothesis — the stage-05 9 GB worker was the
  full-resolution registration of 3D 0.35 mm sessions, not HD-BET on CPU.

## Testing

- Unit (pytest, next to the code — `tests/` dirs are gitignored except the
  skull-stripping one): `scripts/preprocessing_steps/test_registration_downsample.py`:
  - spacing helper: (0.35,0.35,0.55) → (1,1,1); (0.47,0.47,5.0) → (1,1,5);
    (1.2,1.2,6) unchanged;
  - `register_modalities` on two small synthetic volumes (fine spacing) with a
    known shift recovers the shift within 0.5 mm, and `ants.registration` is
    called with downsampled images (spy).
- Real: rerun `P000067` into a fresh folder → 6/6 sessions through stage 05–08,
  `Kappa 6/6`; sessions 001–004 transforms identical to the previous run
  (they are not resampled).
