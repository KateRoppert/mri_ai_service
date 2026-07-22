# Fix: MPRAGE contrast loss (Stage 01) + multi-echo conversion loss (Stage 03)

**Date:** 2026-07-22
**Branch:** `fix/stage01-mprage-t1c-and-stage03-multiecho`
**Status:** design — awaiting approval

## Goal

Two independent, well-diagnosed defects surface on the clinical dropbox_33 dataset
(German-labelled MRI). Both cause a patient to silently lose a required modality and
therefore be dropped from glioblastoma segmentation. Fix both so the target batch
segments completely.

---

## Bug A — Contrast-enhanced MPRAGE never classified as `t1c` (Stage 01)

### Symptom
Patients whose only post-contrast series is an MPRAGE (or `t1_mpr_*`) sequence get no
`t1c`. Both their T1 series collapse to plain `t1`; de-duplication keeps one and
**discards the contrast series entirely**. Downstream, the glioblastoma model requires
`t1c`, so the patient is skipped. Observed on KA01, KA07 (calibration run) and — before
this fix — 14 of 51 patients across both batches.

### Root cause
Contrast **is** detected correctly. `_detect_contrast()` reads `ContrastBolusAgent`
(0018,0010) — anonymisation left the literal value `'anonymized'`, which the code treats
as "contrast present" — and the German `KM` marker (KI-048) is honoured. The failure is
in **classification**, not detection.

`MODALITY_PATTERNS['t1c']['exclude']` contains the text token `'mpr'`, intended to reject
derived **MPR reconstructions** (multi-planar reformats). But `'mpr'` is a substring of:

- `mprage` — the primary Siemens T1 sequence (`t1_mprage_fs_sag_1mm_iso`), and
- `t1_mpr_*` — also a primary T1 acquisition (`t1_mpr_sag_KM`, `t1_mpr_tra_p2_iso`).

`_match_modality()` line ~463:
```python
if has_t1 and (has_contrast_kw or has_contrast) and not has_excl:
    return 't1c'
```
For `t1_mprage_fs_sag_1mm_iso`: `has_t1=True`, `has_contrast=True`, but
`has_excl=True` (because `'mpr' in 'mprage'`) → the branch is skipped, the series falls
through to plain `t1`. The explicit `KM` marker (KI-048) is overridden the same way
(KA12: `t1_mprage_sag_1mm_KM` currently gets no `t1c`).

The same `'mpr'` token sits in the `t2` and `t2fl` exclude lists with the same latent
flaw (harmless there today only because MPRAGE never matches T2/FLAIR keywords).

### The correct discriminator
Primary acquisitions vs. derived reconstructions are distinguished cleanly by the DICOM
**`ImageType`** tag (0008,0008), verified on this data:

| Series | `ImageType[0]` |
|---|---|
| `t1_mprage_fs_sag_1mm_iso` (primary) | `ORIGINAL` |
| `t1_mpr_sag_KM` (primary) | `ORIGINAL` |
| `t1_mpr_tra_p2_iso` (primary) | `ORIGINAL` |
| `..._MPR_MPR cor` (reconstruction) | `DERIVED` |

`'mpr'` in the text was a crude, buggy proxy for "this is a reconstruction". `ImageType`
is the real signal.

### Design
1. Add helper `ModalityDetector._is_derived_reconstruction(dcm) -> bool`:
   returns `True` when `ImageType` (0008,0008) is present and its first element is
   `DERIVED` (case-insensitive), or any element is `SECONDARY`. Missing/empty tag →
   `False` (fail open: treat as primary; see fallback below).
2. In `detect_modality()`, after loading the DICOM, if `_is_derived_reconstruction(dcm)`
   is `True`, exclude the series from primary-modality classification (return `None`
   with method `"excluded: derived reconstruction (ImageType)"`), so reconstructions
   never compete with primary acquisitions.
3. Remove the `'mpr'` token from the `exclude` lists of `t1c`, `t2`, and `t2fl`
   patterns. All three now rely on `ImageType` for reconstruction rejection.

### Fallback / robustness
- **`ImageType` missing or anonymised:** helper returns `False`, series is treated as a
  primary acquisition. This is the safe direction for THIS dataset (it prefers keeping a
  real acquisition over dropping it). A dataset that both anonymises `ImageType` **and**
  ships standalone MPR-reconstruction series could let a reconstruction through; that is
  not present in dropbox_33 and is out of scope (note in KNOWN_ISSUES if desired).
- The existing `dyn`, `pit`, `spir` tokens in the `t1c` exclude stay — they target
  distinct sequences (dynamic, pituitary, SPIR), not reconstructions.

### Expected impact
`t1c` coverage 35/51 → **49/51** patients. The remaining two (KA03, KA14) genuinely have
no contrast series in the source data (no `KM`, no `ContrastBolusAgent`) — a data
reality, not a bug.

### Tests (TDD)
- `t1_mprage_fs_sag_1mm_iso` + `ContrastBolusAgent='anonymized'` → `t1c` (regression for
  KA01/KA07).
- `t1_mprage_sag_1mm_KM` (KM marker, no agent) → `t1c` (regression for KA12; proves KM no
  longer suppressed).
- `t1_mpr_sag_KM` primary (`ImageType ORIGINAL`) → `t1c` (KA06).
- Series with `ImageType[0]='DERIVED'` and a T1+contrast name → excluded (`None`).
- Plain `t1_mprage_sag_p2`, no contrast → `t1` (unchanged).
- `t2_tse_tra_4mm` → `t2`, `t2_space_flair_fs` → `t2fl` (unchanged; no `mpr` regression).

---

## Bug B — Multi-echo series reported as failed conversion (Stage 03)

### Symptom
`sub-028` (KA130) `t2` series: *"dcm2niix finished successfully but output file was not
created"*. The patient loses `t2` and is the only one of 33 in the target batch not
segmented. The converted data actually exists, orphaned as `..._t2_e2.nii.gz`.

### Root cause
The series is a dual-echo `pd+t2_tse_tra` (PD = echo 1, T2 = echo 2). dcm2niix tags the
output with the echo number from `EchoNumbers` (=2), producing
`sub-028_ses-001_t2_e2.nii.gz` and **no** base-named file. The converter
([`scripts/03_convert_to_nifti.py:251`](../../../scripts/03_convert_to_nifti.py))
checks only the exact name `sub-028_ses-001_t2.nii.gz`:

```python
expected_file = anat_dir / f"{filename_pattern}.nii.gz"
if expected_file.exists():
    for extra in anat_dir.glob(f"{filename_pattern}_*.nii.gz"):
        extra.unlink()          # treats every suffixed file as a throwaway artifact
    ... success
else:
    reason = "dcm2niix finished successfully but output file was not created"  # FAIL
```

When only a suffixed file exists, the exact check fails → the series is reported failed
**and** the real data is left orphaned (the cleanup loop is gated behind
`expected_file.exists()`, so it never runs here). The cleanup's assumption — "suffixed
files are always redundant artifacts" — is wrong when the suffixed file is the only
output.

### Design
Modify the `returncode == 0` branch:

1. If the exact `{filename_pattern}.nii.gz` exists → current behaviour unchanged (keep it,
   delete suffixed extras as artifacts).
2. Else, gather `sorted(anat_dir.glob(f"{filename_pattern}_*.nii.gz"))`:
   - **Exactly one** suffixed file → rename it (and its sidecar `.json`, if present) to
     the canonical `{filename_pattern}.nii.gz` / `.json`, log an INFO explaining the
     multi-echo/reconstruction rename, count success.
   - **Zero** suffixed files → current failure (`"output file was not created"`).
   - **More than one** suffixed file → fail with a descriptive reason listing the
     candidates (e.g. real dual-echo producing `_e1` **and** `_e2`), flag for manual
     review. Echo-aware auto-selection is deliberately out of scope (YAGNI) — not present
     in dropbox_33; revisit only if such data appears.

### Expected impact
`sub-028` (KA130) gains `t2`; all 33 target-batch patients (that have contrast) segment.
No change for any single-output series.

### Tests (TDD)
- Directory with only `pat_ses_t2_e2.nii.gz` (+ `.json`) → renamed to `pat_ses_t2.nii.gz`
  (+ `.json`), reported success. (Unit test at the rename-resolution level; may stub the
  dcm2niix call and pre-create the suffixed file.)
- Directory with base `pat_ses_t2.nii.gz` + extra `pat_ses_t2_Eq_1.nii.gz` → base kept,
  extra deleted (unchanged behaviour).
- Directory with `_e1` and `_e2` (two suffixed, no base) → failure with descriptive
  reason, nothing renamed.
- Directory with no output at all → failure `"output file was not created"` (unchanged).

---

## Global constraints
- Both datasets that already work (MS_5, UPENN-GBM, KA117–152 that currently segment)
  must keep working — no regression. Verified via the unchanged-behaviour tests above and
  a re-run of the target batch after implementation.
- Code comments in English; conventional-commit messages; commit per sub-step; run tests
  before committing.
- No new dependencies. `ImageType`/`EchoNumbers`/`ContrastBolusAgent` are already-present
  standard DICOM tags read via pydicom.

## Out of scope
- KI-047 (MosMed fully-anonymised tags) — unrelated, separately tracked.
- Echo-aware selection among multiple echoes in one folder (Bug B multi-suffix case).
- The `pd+t2` `'+'` contrast false-positive — harmless (no `t1` token, never becomes t1c).
- Worker/parallelism tuning — separate task.
