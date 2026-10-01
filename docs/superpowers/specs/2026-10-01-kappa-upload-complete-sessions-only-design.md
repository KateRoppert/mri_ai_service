# Kappa Upload: Complete Sessions Only, Counted Against the Input — Design

**Date:** 2026-10-01
**Branch:** `fix/kappa-upload-complete-sessions-only` (from `main`)
**Type:** Bugfix (wrong data sent to Kappa + misleading delivery counter)

## Problem

Run `30_09_1752` (SibBMS `P000067`, MS, 6 sessions, dataset 353):

| Session | What happened | In Kappa |
|---|---|---|
| ses-001…004 | Processed fully | Uploaded ✓ |
| ses-005 | Stage 05 worker OOM-killed after `t1`, `t2` were written, before `t2fl`; stage 06 skipped it (missing `t2fl`) → **no mask** | **Uploaded** with 2 images and no mask ✗ |
| ses-006 | Stage 05 pool died before it started → nothing in `preprocessed/` | Not uploaded, **not counted** |

History showed **"Kappa 5/5"**. Expected: 4 uploaded, out of 6.

Both stage 05 and 06 recorded the losses correctly in their
`incomplete_data.json` reports; only the uploader ignores them.

## Cause

`KappaUploader._discover_sessions()` (`backend/kappa_uploader.py:234`) builds
the session list from whatever `*.nii.gz` exists in `preprocessed/`, then
attaches masks if any. So:

1. a session is uploaded as soon as **any** preprocessed file exists — no check
   for the required modalities (`configs/lesion_types.yaml`) or for the mask;
2. `total` = sessions found in `preprocessed/`, so a session lost before or
   during stage 05 never enters the denominator.

## Design

### Uploader: the universe is stage 01's output, upload only ready sessions

- **Universe (denominator):** every complete session stage 01 produced —
  `bids_organized/sub-*/ses-*` (the `_incomplete/` tree is excluded: those are
  sessions the doctor still has to resolve in the incomplete-patients queue,
  they never entered processing). Fallback to today's `preprocessed/` scan
  when `bids_organized/` is absent (old/CLI layouts).
- **Ready** = preprocessed has every `required_modalities` entry for the run's
  `lesion_type` **and** the main segmentation mask exists.
- Ready sessions upload exactly as today (dedup, name-clash, reconcile).
- Not-ready sessions are **not uploaded**; each is reported as
  `{"session", "success": False, "error": "not_processed", "message"}` with a
  readable reason, e.g. `нет t2fl и маски сегментации` /
  `нет данных после предобработки`.
- `total` = size of the universe.

### Delivery state: "not processed" is terminal, not a retry

`kappa_delivery.classify()` today retries whenever `delivered < total`
(`kappa_delivery.py:161-166`, `_transient`). With the new denominator that
would retry 4/6 forever. So:

- `not_processed` sessions are neither delivered nor blocking; they are
  listed in `detail.not_processed` (session + message).
- If every **ready** session is delivered → status `done`, counters
  `delivered=4, total=6`. Nothing to retry: the data does not exist.
- Zero ready sessions → existing `needs_attention / missing_files` path.
- `name_clash` keeps its `needs_attention` priority.

### UI (history column)

- Label for `done` with `delivered < total`:
  **`Kappa 4/6 · 2 не обработаны`**.
- Hint (tooltip / details): one line per session — `sub-003_ses-005: нет t2fl
  и маски сегментации` — plus a pointer to the lost-patients report, which
  already explains the stage-level reason.
- `done` with `delivered == total` stays `Kappa N/N`.

### Out of scope

- The incomplete `ses-005` entity already in dataset 353
  (`7a695977-0dd8-4ecc-8ccc-b6ace894e54c`) — delete by hand in Kappa. If left,
  a re-run will treat the same study as a duplicate and not replace it.
- Why stage 05 was OOM-killed (HD-BET forced to CPU by `exclude_gpus: [0]` on a
  single-GPU laptop; worker reached ~9 GB) — laptop config fixed locally;
  note to add to KI-058 once `fix/ms-report-dynamics-and-timeouts` (which
  introduces KI-058) is merged.

## Testing (pytest, next to the code)

- `_discover_sessions` / readiness: full session ready; missing one required
  modality → not ready; modalities present but no mask → not ready; session in
  `bids_organized/` with nothing downstream → counted, not ready; `_incomplete/`
  ignored; no `bids_organized/` → falls back to `preprocessed/`.
- GBM vs MS required sets (`t1c` required only for GBM).
- `upload_results` does not call `upload_entity` for not-ready sessions;
  `total` = universe.
- `kappa_delivery.classify`: 4 delivered + 2 not_processed of 6 → `done`, no
  next attempt, `detail.not_processed` has 2 entries; 0 ready → `missing_files`;
  network failure on a ready session still retries.
- Existing `backend/` suite stays green; frontend `npm run lint` (in the
  `web` image) adds no new findings.
