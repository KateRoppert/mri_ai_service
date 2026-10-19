# KI-035: Completing an Incomplete Kappa Entity — Design

**Date:** 2026-10-19
**Branch:** `fix/ki-035-kappa-completes-incomplete-study` (from `main` 3d0ccb9)
**Type:** Fix

## Why — what is still true of KI-035

KI-035 (written July): a study first run with missing data lands in Kappa
incomplete; a later complete run of the same study has the same
`study_hash` (`PatientID:StudyInstanceUID`) and is skipped as a duplicate,
so the incomplete version stays forever.

Since then:

- `KappaUploader._readiness` uploads only sessions with every required
  modality and the atlas-space mask. `lesion_types.yaml` has no optional
  modalities, so **new** runs can no longer put an incomplete session into
  Kappa.
- The replace flow (`kappa-replace-on-reprocess`) handles a session rebuilt
  from a corrected modality set (`reprocessed_sessions` → `supersedes` →
  per-session button).

What is left are entities uploaded before `_readiness`. Read-only audit
of user 26's datasets, 2026-10-19:

| dataset | entity | missing |
|---|---|---|
| 337 | sub-214_ses-001 | t1c + mask |
| 249 | sub-003, sub-055, sub-084, sub-214 (ses-001) | t1c + mask |
| 249 | sub-007, sub-079 (ses-001) | t2fl + mask |

353, 338, 158: none. Re-running any of these patients with complete data
is skipped today as "уже в датасете", and the doctor never sees a mask.

## Change

In `upload_results`, where a ready local session's `study_hash` is already
in the dataset (today: duplicate → skip), look at that entity's file list —
the listing already returns `files` per entity, so no extra request:

1. **Kappa entity complete** (every required modality + a non-native
   `segmask`) → duplicate, as today.
2. **Kappa entity incomplete, no expert masks** (`EXPERT_MASK_RE`) →
   **complete it automatically**: `replace_entity_contents(entity, this
   session's files)`. It replaces same-name files in place, adds the
   missing ones and drops stale pipeline files; expert masks are never
   touched. Then status → Labeled (3) if lower, update `dsEntityInfo`
   (`data_files`, `prediction_files`, `modalities`, `preprocessing_id`,
   `pipeline_run_id`) — deferred, see Open points — and local registration (`patient_registry` +
   AI mask v1). Result: success, `completed: true`; run log
   "дополнено (entity …)".
3. **Kappa entity incomplete, with expert masks** → not automatic:
   reported as `supersedes`, so it goes through the existing
   "требует решения" + replace button. (An expert mask drawn on an
   entity without an AI mask is unlikely, but it would be drawn on the old
   images.) The replace endpoint used to accept only sessions in the run's
   `reprocessed_sessions`; it now also accepts a session the uploader itself
   blocked as `supersedes_kappa` in the run's delivery state (other block
   reasons are still refused).

All files are replaced, not only the missing ones: the session's existing
images come from an earlier run, and the new mask belongs to this run's
images (run-to-run registration differences, KI-062). Appending only the
missing modality and mask would mix two registrations in one entity.

Failure (any file not patched/added, delete job failed) → the session
result is `upload failed` with what failed; the next delivery attempt sees
the entity still incomplete and tries again. No POST is ever made for a
hash already in the dataset, so this cannot create duplicates.

## Open points

- **Updating `dsEntityInfo`.** Kappa v2 `PUT /datasets/datasetEntities/
  {dataset_id}/{entity_id}` (multipart: `update_dataset_entity` JSON with
  required `remark`, plus `files`) "updates metadata or replaces files".
  Whether an empty `files` list leaves files alone is not documented.
  Verify on a throwaway entity in a test dataset before using it; if it
  is unsafe, leave `dsEntityInfo` as uploaded and record that its file lists
  can be stale (Kappa's own `files` is the truth the uploader reads).
- **Decisions for Kate:** (a) automatic completion without a button when
  there are no expert masks — proposed yes: nothing on an entity without
  a mask could have been validated; (b) completing an entity whose
  `preprocessing_id` differs from this run's — proposed yes: every file is
  replaced, so the entity is internally consistent, and its info records
  the new id.

## Out of scope

- KI-026 (silent dedup confuses the validation UX) — separate.
- Re-running the 7 audited patients — an operational step after merge.
- KI-073 (manual retry racing the background attempt) — separate.

## Verification

- Unit tests (`backend/test_kappa_uploader_completion.py`), fakes for the
  listing and `replace_entity_contents`:
  complete entity → duplicate, no replace; incomplete → replace with the
  session's files, success `completed`, local registration; incomplete
  with an expert mask → `supersedes`, no replace; replace failure →
  `upload failed`, no POST; listing with no `files` key → treated as
  unknown, duplicate (never replace on missing information).
- Existing uploader / delivery / replace tests stay green.
- Live (after Kappa answers, with Kate): one of the 7 audited patients
  re-run; the entity gains t1c/t2fl + mask, one entity per name remains.

## Results (2026-10-19, live, throwaway dataset 357)

| case | result |
|---|---|
| complete entity | duplicate, unchanged |
| incomplete (t1/t2/t2fl, no mask), ses-002 and ses-004 | completed to 5 files, one entity per name, `completed` |
| incomplete + expert `_segmask_v2` | `supersedes`, run `needs_attention`; replace keeps the expert mask |
| rerun after all of the above | done 4/4, nothing changed |
| `PUT` with metadata only | 200, files unchanged, info replaced — now used after completion |

Found on the way (separate fixes): an empty dataset's listing is 404
"No record found." (`fix/kappa-empty-dataset-listing`); Kappa refuses status
3 without labels / `output_image` (KI-075).
Added during implementation: an incomplete entity under a different name
than the local session is not completed automatically (its files would keep
the local name and the entity would look incomplete forever).
