# Stage 05 Memory: N4 on a ≤1 mm Copy, Single-Process HD-BET — Design

**Date:** 2026-10-12
**Branch:** `fix/stage05-memory` (from `origin/main`)
**Type:** Fix (follow-up to KI-058)

## Why

SibBMS 3D sessions (P000067 ses-005/006: t1 231 Mvox at 0.35 mm, t2fl up
to 186 Mvox) are OOM-killed in stage 05 when both segmentation services and
the desktop run on the laptop (~6.5 GB available to the run). A
multi-session MS run then finishes only part of the sessions. Stopping
`service-gbm-seg` is the current workaround.

## Measurements (2026-10-12)

ses-005, one session, container `--memory 9g`, anonymous memory of the whole
container (pipeline process + children, no file cache), sampled every 0.2 s:

| Step | after_registration | before_registration (exp branch) |
|---|---|---|
| reorient | 2.4 GB | 2.5 GB |
| **N4 bias correction** | **7.0 GB** | **7.0 GB** |
| native skull strip (HD-BET) | — | 5.7 GB |
| t1 → atlas | 3.3 GB | 0.9 GB |
| modalities → t1 | 3.1 GB | 3.4 GB |
| apply transforms to originals | 4.2–5.8 GB | 4.3 GB |
| skull strip in atlas space (HD-BET) | 5.6 GB | — |

Two causes, both unrelated to what the outputs need:

1. **N4 runs at full resolution.** `bias_correction.py` estimates the field
   on a 4× shrunk copy, but keeps several full-size copies of the image
   (float64 `get_fdata`, its SimpleITK copy, float32 cast, Otsu mask, the
   field resampled back, `exp(field)`, the quotient, the numpy result):
   ~7 GB at 231 Mvox. With `use_for_registration_only: true` (the config
   value) the N4 output is read **only** to compute transforms — modalities
   are registered on ≤1 mm copies (`max_registration_spacing_mm`), the atlas
   metric is evaluated on the 1 mm atlas grid, the native skull strip makes
   its own ≤1 mm copy. All final images are re-warped from the originals.
   The full-resolution N4 image is never used at full resolution.

2. **HD-BET CLI starts ~13 processes for one file.** HD-BET 2.0.1
   `hdbet_predict` hardcodes `num_processes_preprocessing=4`,
   `num_processes_segmentation_export=8` and a `Pool(4)` for applying the
   mask; every worker imports torch. Measured on the same image:

   | | anon peak | processes | time | output |
   |---|---|---|---|---|
   | `hd-bet` CLI (today) | 5.1–5.2 GB | 14–15 | 17 s / 29 s | — |
   | same predictor, 1 + 1 workers | 1.9–2.0 GB | 7 | 16 s / 28 s | mask and brain **bit-identical** |

   TTA on/off does not change memory.

## Changes

### 1. N4 on a ≤1 mm copy

In `process_subject_bias_correction`, when the output is for registration
only (`use_for_registration_only: true` and a temp dir is given), the image
is first resampled so that no axis is finer than the registration working
spacing (`registration.params.max_registration_spacing_mm`, default 1.0),
and N4 runs on and saves that copy. The rule is the existing
`registration_spacing`: only axes finer than the limit are coarsened (2D
0.47 × 0.47 × 5 mm → 1 × 1 × 5 mm), resampling keeps physical space
(origin, direction), so a rigid transform found on the copy is the same
physical transform.

When `use_for_registration_only: false` the N4 image is a final output and
stays at full resolution — unchanged.

The full-size float64 load is also replaced by reading straight into
SimpleITK as float32 (no nibabel round trip), which lowers the 2D / non-
downsampled path too.

`05_preprocessing.py` passes the registration spacing into the bias
correction step. No new config key.

### 2. HD-BET without the worker fan-out

`HdBetStripper` keeps running HD-BET in a subprocess (CUDA state stays out
of the pipeline worker), but instead of the `hd-bet` CLI it runs a small
runner, `skull_stripping/hdbet_runner.py`, with the Python interpreter of
the installed `hd-bet` (taken from the CLI's shebang; HD-BET lives in the
FSL Python in the image). The runner does what the CLI does —
`maybe_download_parameters`, `get_hdbet_predictor`,
`predictor.predict_from_files`, `apply_bet` — with 1 preprocessing and
1 export worker, and writes the same files (`-o` image and the `_bet` mask).
Arguments, timeout, TTA, device and error reporting stay as today.

If the runner cannot import HD-BET's internals (e.g. a future HD-BET
version renames them), it exits non-zero with the import error, and the
existing `fallback_method` path applies; a unit test pins the signatures
used, so an image upgrade fails the test rather than silently degrading.

### 3. Transforms applied in single precision (added during implementation)

The first after-fix profile sampled memory every 0.2 s and missed short
spikes. Exact per-step peaks (VmHWM reset per step) showed the real peak:
`ants.apply_transforms` defaults to double — it clones fixed and moving to
double and allocates the output as another clone of moving, at the moving
image's size: +5.6 GB on the 231 Mvox t1, +4.2 GB on the 171 Mvox t2fl.
`singleprecision=True` halves it (+2.1 GB); output differs by at most 1.9 on
an intensity range of 141 000 (~1e-5). Applied to all three calls in
`registration.py`: `apply_transform`, the combined modality warp, and
`inverse_transform_mask` (stage 07, where the native image is the fixed one;
labels stay exact labels). SimpleITK resampling was tried too (+1.7 GB) but
its output did not match without reworking transform conventions — dropped.

## Effect on results

- **HD-BET:** none — masks and stripped images are bit-identical (same
  model, same TTA; worker count only changes parallelism).
- **N4 / registration:** final images are still produced from the
  originals with one interpolation. Only the image the transform is
  computed on changes: the bias field is estimated on a ~4 mm grid instead
  of ~1.4 mm (the field is a smooth B-spline), and the t1 → atlas moving
  image is the 1 mm copy. The transform can shift slightly. Verified
  against run-to-run spread before merging (see Verification); if the
  shift is larger than that spread, this part is not merged.

`preprocessing_id`: unchanged (no config key changes). Open question for
Kate: if the measured shift is within run-to-run spread, keep the id (no
new Kappa dataset key, existing results stay comparable); otherwise bump
it by adding an explicit config key.

## Expected peak

ses-005: from 7.0 GB to roughly 4–4.5 GB (estimate from sampled memory; the
measured result is in Results below), set by applying transforms to the
full-resolution originals (the only step that genuinely needs them). To be
confirmed by measurement; the resource planner's `k` for stage 05 is then
re-measured from the "peak RSS" log lines and updated in
`configs/resource_config.yaml`.

## Out of scope

- Downsampling the originals early: would add a second interpolation to the
  final images, and gains at most ~1 GB beyond this fix.
- The exp branch (`before_registration`) benefits from both changes without
  edits: its native strip reads the N4 reference and runs `HdBetStripper`.

## Verification

1. Unit tests (`scripts/preprocessing_steps/`): N4 copy spacing (0.35 mm
   axes → 1 mm, ≥1 mm axes untouched, `use_for_registration_only: false`
   keeps full size, physical space preserved); HD-BET runner command and
   output names; HD-BET internals' signatures.
2. Memory: the profiling script on ses-005, both skull-strip positions,
   before/after.
3. HD-BET: mask identity on a GBM and an MS session vs the CLI.
4. Registration: t1 atlas-space displacement old vs new on SibBMS 3D, an
   MS 2D session and a GBM session; compared with the old code's own
   run-to-run spread (2 runs each).
5. End to end (Kate): a multi-session SibBMS run with both segmentation
   services up; all sessions reach stage 08.

## Results (2026-10-12)

Exact peak of the pipeline process (VmHWM per step), SibBMS ses-005, 3D
0.35 mm; container peak includes HD-BET and file cache:

| | old | new |
|---|---|---|
| stage 05, after_registration — process / container | 7.96 / 8.47 GB | **5.01 / 5.30 GB** |
| stage 05, before_registration — process / container | 8.57 / 8.99 GB | **5.63 / 6.28 GB** |
| stage 05 time | 188–191 s | 69–78 s |
| stage 07 (3 native masks) | 8.81 GB | **5.02 GB** |

The remaining peak is applying transforms to the 0.35 mm originals in float.

Planner: `k_bytes_per_voxel` stage 05 35.0 → 25.0, stage 07 37.7 → 23.0.

**HD-BET:** runner vs CLI — mask and brain bit-identical on atlas-space and
native 1 mm t1. (Runner failures seen in throwaway containers were weight
downloads from zenodo; production mounts `backend/data/hdbet_weights`. The
error message now carries the tail of stderr, where the cause is, instead of
the first 500 characters.)

**Stage 07:** native masks differ in 1–2 voxels out of 58 k–468 k lesion
voxels (nearest-neighbour ties).

**Registration**, t1 → atlas transform, max shift over a 180×220×160 mm box,
old = 2 runs from the 2026-10-07 experiment, new = 2 runs of this branch:

| session | mode | old–old | new–new | old–new | template corr old → new |
|---|---|---|---|---|---|
| MS SibBMS 3D 0.35 mm | after | 3.15 | 0.73 | 1.9–4.6 | 0.360/0.364 → 0.364/0.361 |
| MS SibBMS 2D | after | 7.76 | 1.64 | 3.0–8.5 | 0.344/0.337 → 0.385/0.376 |
| MS ms5 | after | 18.8 | 35.9 | 18.3–33.2 | 0.285/0.300 → 0.283/0.289 |
| GBM bo | after | 10.2 | 5.5 | 1.7–10.1 | 0.256/0.264 → 0.258/0.258 |
| GBM upenn | after | 1.58 | 3.53 | 2.6–6.2 | 0.483/0.482 → 0.481/0.481 |
| MS SibBMS 3D 0.35 mm | before | 0.27 | 0.23 | 0.75–1.04 | 0.520 → 0.519 |
| MS SibBMS 2D | before | 0.47 | 0.49 | 0.42–0.58 | 0.501 → 0.502 |
| MS ms5 | before | 8.47 | 1.94 | 4.0–13.4 | 0.400/0.410 → 0.414/0.414 |
| GBM bo | before | 0.57 | 0.58 | 0.43–0.83 | 0.485 → 0.485 |
| GBM upenn | before | 1.47 | 1.58 | 0.42–1.80 | 0.486 → 0.486/0.487 |

- after_registration (today's default): old-vs-new shifts sit inside the
  old code's own run-to-run spread (KI-062); quality unchanged.
- before_registration (repeatable): shifts 0.4–1.8 mm, at the size of the
  repeat spread, except SibBMS 3D: ~1 mm against a 0.27 mm repeat spread —
  a real but sub-voxel change (the box corners are the extreme points), with
  the same template correlation. ms5 is unstable in both versions.
- GBM upenn sub-305 "t2" is a 4D diffusion series (78 volumes): N4 failed
  on it in the old code too ("axes don't match array"); stage 01 no longer
  picks diffusion (`fix/stage01-exclude-diffusion`).
