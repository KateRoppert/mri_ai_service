# Manual Session Merge ("Piece A") — Design

## Context

A patient can have multiple sessions (visits), each with its own subset of modalities — sometimes no single session is complete on its own, but two together cover everything required (e.g. `ses-001` = t1+t2, `ses-002` = t1c+t2fl, split across a logistically-delayed second registration). This was analyzed early in the incomplete-patient work (`docs/superpowers/specs/2026-08-06-incomplete-patient-workflow-design.md`, "Piece A") as a session-merge *heuristic* — automatic detection of merge candidates via scanner vendor/model, patient weight, and date-gap signals — but that auto-detection was explicitly deferred and never built. The review queue built since (`feat/incomplete-patients-frontend`, the audit-trail follow-up) works strictly at single-session granularity: no way to see a patient's sessions together, no way to combine them.

The doctor has now explicitly asked for the manual half of this: (1) visual grouping of a patient's sessions in the review list, (2) a way to manually merge two sessions through the UI. The heuristic (auto-suggesting candidates) remains out of scope — this is a doctor-initiated, doctor-controlled action only.

## Design

### Merge mechanics (backend)

The design reuses existing machinery rather than building a new one. Every series ever assigned to a modality — not just entries in `excluded_series`, but also the currently-assigned `series[modality]` — carries its `original_path`, pointing at the original raw DICOM source directory, not the anonymized BIDS copy. This means "merging" two sessions doesn't require any new file-copying or anonymization code: it's enough to pull the donor session's series (both currently-assigned and its own leftover `excluded_series`) into the primary session's `excluded_series` pool as new candidates, tagged with a new reason `"from_other_session"`. The doctor then assigns them using the exact same `relabel_series` flow already built and tested for the ordinary fill/replace case — including the conflict case (both sessions already have a `t1`, say): both versions are offered as alternatives, and the doctor picks via the same replace-with-swap-back mechanism that already exists.

New method `merge_sessions(output_path, patient_id, primary_session_id, donor_session_id)` in `backend/pipeline_manager.py`:
1. Validate `patient_id`/`primary_session_id`/`donor_session_id` against the existing BIDS-ID regex patterns (same validation already used by `relabel_series`/`discard_session`).
2. Load `dataset_mapping.json`, locate both sessions under the same patient.
3. Build one `excluded_series`-shaped entry per donor series — from `donor_session_data['series']` (currently-assigned modalities) and from `donor_session_data['excluded_series']` (donor's own leftover alternatives) — each with `reason: "from_other_session"`, `detected_modality` set to the modality it came from (or the donor's own `detected_modality` for its already-excluded entries). Skip any `original_path` already present in the primary's `excluded_series` (idempotency — merging twice, or merging after a prior partial merge, doesn't duplicate candidates).
4. Append these to `primary_session_data['excluded_series']`.
5. Mark `donor_session_data['status'] = 'merged'` and `donor_session_data['merged_into_session_id'] = primary_session_id`.
6. Write the mapping file back.

No physical files are touched — the donor's already-anonymized BIDS copy is simply no longer referenced going forward; its `original_path` pointers (now living inside the primary's `excluded_series`) still work when `relabel_series` re-copies+re-anonymizes them into the primary session. The primary's own `status`/`series` are untouched by the merge itself — merging only makes more candidates available; completeness is still recomputed by `relabel_series` exactly as it already is today, as the doctor assigns them.

New endpoint: `POST /api/incomplete-patients/{run_id}/{patient_id}/merge`, body `{primary_session_id, donor_session_id}`.

### Donor session's fate in the review queue

`merged` is a new terminal status, distinct from `discarded` — it means something different (consolidated into another session, not skipped) and deserves its own audit-trail entry, matching the project's existing bias toward accurate, permanent history over silent disappearance. `get_incomplete_patients()`'s inclusion rule gains one more clause: `status == 'merged'` is always included (permanent, like `discarded` — not one-shot like the `manually_reviewed` "became complete" case), rendered as a 6th status tag: "Объединена → ses-XXX" (grey, like discarded, but naming the target). No "undo merge" action in this pass — consistent with "no undo discard" already established.

### Frontend: grouping (item 1)

`IncompletePatients.jsx`'s table sorts rows by `(original_id, session_id)` and merges the "Пациент" column's cell across consecutive rows of the same patient via antd `Table`'s standard `rowSpan` technique — no collapsing/tree view, just keeping a patient's sessions adjacent and not repeating the name on every row.

### Frontend: merge action (item 2)

No new modal. A new section is added directly inside the already-open `IncompletePatientDetail.jsx` — the session currently being viewed is always the merge *target* (primary). The new section, placed below the existing "Неотобранные серии" list: "Объединить с другой сессией пациента" + a `Select` listing that patient's other sessions (filtered client-side from the already-loaded `sessions` array — no new fetch — excluding the current session itself and any already `merged`/`discarded` ones) + a "Объединить" button (`Popconfirm`-wrapped, matching the discard/replace confirm pattern already used throughout this component). On success, `onActionComplete()` (already wired) refetches the list — the primary's `excluded_series` now includes the donor's candidates, ready to assign through the existing flow, and the donor's row (visible elsewhere in the list, once refetched) shows its new "Объединена" tag.

### Testing

Backend (TDD, pytest, matching `TestRelabelSeries*`/`TestDiscardSession*` conventions in `backend/test_incomplete_patients_api.py`):
- Merging pulls both sessions' assigned series and leftover `excluded_series` into the primary's `excluded_series`, each tagged `reason: "from_other_session"`.
- A modality present in both sessions produces two separate candidate entries (no silent preference) — both reach the primary's `excluded_series`.
- Donor session's status becomes `merged` with `merged_into_session_id` set correctly.
- Merging twice (or merging after a prior partial merge already pulled some of the same series) does not duplicate candidates already present in the primary's `excluded_series`.
- `get_incomplete_patients()` includes a `merged` session permanently (same pattern as the existing discarded-permanence test).
- Input validation: invalid `patient_id`/`primary_session_id`/`donor_session_id` rejected before any mapping mutation (mirrors the existing `TestRelabelSeriesInputValidation`/`TestDiscardSessionInputValidation` pattern).

Frontend: no automated test framework exists in this project — manual browser verification (dev server) plus `npm run lint`, same as every other frontend change this session.

## Out of Scope (explicitly deferred)

- The session-merge *heuristic* (automatic candidate detection via vendor/model/weight/date-gap signals) — this spec covers only the manual, doctor-initiated merge action it was meant to feed into.
- Undoing a merge.
- Any physical cleanup of the donor session's now-unreferenced BIDS files.
- Merging sessions across different patients (not a real scenario — merge only ever operates within one patient's own sessions).
