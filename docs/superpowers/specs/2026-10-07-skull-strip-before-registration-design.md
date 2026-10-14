# Experiment: Skull Stripping Before Atlas Registration — Design

**Date:** 2026-10-07
**Branch:** `exp/skull-strip-before-registration` (from `origin/main`, in a separate
git worktree — `main` and the running app stay untouched)
**Type:** Experiment (decides the fix for KI-062 and the input space of the
skull-stripping research)

## Why

KI-062: the stage 05 rigid `t1 → MNI152` registration lands in a wrong place
and differs between runs (up to 63 mm on 2D sessions, a stable rotation on 3D).
A side experiment on SibBMS `P000067` (2026-10-07) compared one-shot variants:
an atlas brain mask on the metric failed; **registering a skull-stripped brain
to the brain template** (`MNI152_T1_1mm_brain`) was the most even fix
(inter-session consistency 0.38–0.45 vs 0.23–0.45 today, repeat spread
≤1.9 mm) at roughly today's cost, but needs skull stripping to run first.
Before changing production, measure what that order does on glioblastoma and
MS data end to end, including segmentation.

## The config question

`configs/preprocessing_config.yaml` lists `steps` as an ordered list, but
`05_preprocessing.py` ignores the order: steps run in a fixed sequence
(reorient → N4 → registration → [resampling] → skull stripping), only
`enabled` is read, and each step reads a hard-coded predecessor's directory.
Free reordering was never implemented and mostly cannot be meaningful (N4
must precede registration; resampling only makes sense without registration).
The one real choice is **where skull stripping sits**. The experiment adds
exactly that switch:

```yaml
- name: skull_stripping
  params:
    position: after_registration   # default — today's behaviour, unchanged
    # position: before_registration
```

Default = today, so the code is safe to merge later even if the experiment
says "keep the old order".

## `before_registration` flow (reference modality only changes path)

1. Reorient, N4 — as today.
2. **Skull strip the reference modality in native space**, on its N4-corrected
   ≤1 mm copy (same stripper and fallback as production: HD-BET → BET).
3. **Register brain → brain template** (`MNI152_T1_1mm_brain`, same grid as
   `MNI152_T1_1mm`), rigid, `random_seed` fixed. Save the transform under the
   same name as today.
4. Other modalities → reference: unchanged (skull-on, ≤1 mm copies).
5. Apply transforms to the originals: unchanged.
6. **Warp the native brain mask into atlas space** (nearest neighbour) and
   apply it to every modality — instead of running the stripper a second time.
   Save it as `transformations/…_brain_mask.nii.gz` (atlas space), as today, so
   stage 07/08 and `research/…/atlas_space.py` find it where they expect.

Outputs keep today's names and spaces; only how they are produced changes.

## Data (both lesion types, both acquisition kinds)

| Set | Lesion | Why |
|---|---|---|
| SibBMS P000067 (6 sessions), P000010 (2) | MS | 2D thick-slice + 3D 0.35 mm, longitudinal → consistency check |
| MS_5 P000915 | MS | another site |
| dropbox_33: 6 KA patients | GBM | clinical GBM, the research calibration set |
| clinical_dicom BO: 6 patients | GBM | the large production set |
| UPENN-GBM: 3 patients | GBM | public reference (check first: skull present in the raw DICOM?) |

Each set runs **twice per order** (after / before) into separate folders under
`demo_workspace/exp_ss_order/`. No Kappa upload (CLI runs, no session).

## Measures

1. **Registration**: MI to the brain template (whole-head MI for "after");
   repeat spread between the two runs of the same order; for longitudinal
   patients, inter-session consistency in atlas space; overlay PNGs for a
   sample (2D and 3D, GBM and MS).
2. **Skull stripping**: mask volume (ml), LCC, holes, leakage — the v2
   research metrics where available on `main`, otherwise volume + visual
   contact sheet; native-space edge-touch on 2D slabs noted, not gated.
3. **Downstream**: lesion volume and count per session (after vs before);
   Dice of native-space lesion masks between orders; lobar / McDonald zone
   assignments that change.
4. **Cost**: stage 05 time and peak memory per session (peak lines, KI-058).

## How it runs without touching `main`

- `git worktree add ../mri_ai_service-exp exp/skull-strip-before-registration`.
- Stages run by `orchestrator.py` in throwaway `brain-lesion-web:demo`
  containers that mount the **worktree's** `scripts/ utils/ configs/`, join the
  compose network (stage 06 reaches `service-ms-seg` / `service-gbm-seg`), and
  run with `--memory 9g`. The running `web` and its mounts stay on `main`.
- Results and a short report land in `docs/experiments/2026-10-07-ss-order.md`
  on the experiment branch.

## Out of scope

- The research v2 cascade / validation (not on `main`): the experiment uses
  production's HD-BET → BET. If the order is adopted, the cascade moves with
  it when v2 is merged.
- Changing the order in production: a separate decision after the report.
