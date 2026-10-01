# Manual Modality Assignment for Incomplete Patients — Design

**Date:** 2026-09-30
**Branch:** `feat/incomplete-patients-assignment` (from `main`)
**Type:** Feature + bugfix (lesion-type-aware modality list)

## Problem

The incomplete-patient queue only shows what is *missing*. The doctor sees
"нет t1c" and a pile of leftover series, but never sees **what the algorithm
did pick** — so there is no way to disagree with a choice that is wrong rather
than absent. A series the detector mislabelled is invisible; only an empty slot
is actionable.

Three further gaps make the queue harder to use than it needs to be:

- The modality list in the UI is hardcoded to glioblastoma's four
  (`IncompletePatientDetail.jsx:13`). A multiple-sclerosis run is offered
  `t1c`, which that lesion type does not use.
- Assignments apply the instant they are made — DICOM copied, folders moved.
  There is no draft, so there is nothing to reconsider and nothing to cancel.
- A patient whose set the doctor fixes is never reprocessed: `skip_existing`
  sees the old outputs and skips them, so the correction changes nothing
  downstream.

## What already works

Worth stating, because it narrows the work considerably:

- **Unrecognized series are already assignable.** They are not greyed out
  anywhere. Their modality dropdown simply starts empty (the detector offered
  no guess), and the button stays disabled until one is chosen — which reads
  as "cannot be assigned" but is not.
- **Replacement is already a genuine swap.** `relabel_series` pushes the
  previous occupant back into `excluded_series` with reason
  `replaced_by_manual_relabel`, so nothing is lost.
- **The protocol name of every selected modality is already stored**
  (`session_data['series'][modality]['series_description']`). The API just
  never returns it.

So the work is mostly about exposing state that exists, restructuring the
screen around it, and making the save/reprocess cycle real.

## Goals

- The doctor sees the selected set and the rejected set side by side, and can
  move a series between them.
- Any series can be assigned to any required modality, including ones the
  detector could not classify.
- Changes are a draft until saved; saving is all-or-nothing as far as the
  pipeline's source of truth is concerned.
- A session whose set changed is reprocessed on the next run, and only it.
- The logic is identical for every lesion type; only the modality set differs.

## Non-goals

- **Updating a patient already delivered to Kappa.** See "The Kappa gap".
- Changing how the automatic detector picks modalities (stage 01).
- Touching `relabel_series`; it stays, covered by its tests, unused by the new
  path.

## Data model

`IncompletePatientSession` gains two fields:

```python
selected: List[SelectedModality]   # modality, series_description,
                                   # original_path, slice_count
required: List[str]                # required modalities for THIS lesion type
```

Both come from data that already exists: `selected` from
`session_data['series']`, `required` from `configs/lesion_types.yaml` via
`load_lesion_type_config(lesion_type)['required_modalities']` — the same
source stage 01 and `relabel_series` already use.

`required` is what removes the hardcoded list from the frontend. After this the
UI knows nothing about modalities; it renders what it is told, which is what
makes the behaviour identical across lesion types by construction rather than
by remembering to update two places.

`excluded_series` is unchanged. `detected_modality: null` already marks the
unrecognized ones; grouping them is the UI's job.

## The assignment endpoint

```
PUT /api/incomplete-patients/{run_id}/{patient_id}/{session_id}/assignment
    { "assignments": { "t1": "<original_path>", "t1c": "<original_path>" } }
  → { status, selected, excluded_series, needs_reprocess }
```

The client sends **the desired final set**, not a list of operations. That
choice is the heart of this design:

- Order stops mattering. Two edits touching the same slot cannot produce
  different results depending on which arrived first.
- Re-sending the same set is a no-op, so a retry after a dropped connection is
  safe.
- The whole set can be validated before anything is written, which is what
  makes "save" honest — see below.

The response carries the session's complete new state so the screen can
re-render without a second request.

## Applying a set

Three steps; the first two touch nothing.

**1. Validate the whole set.**

- the session is in a state that can be edited at all — `incomplete` or
  `complete`. A `discarded` or `merged` session is refused: both are decisions
  already taken, and quietly re-opening them would undo a choice nobody asked
  to undo;
- every `original_path` belongs to this session (present in `series` or
  `excluded_series`);
- every modality is in `required` for this lesion type;
- no series is assigned to two modalities.

Any violation returns `400` and **nothing is written** — not the mapping, not
a single DICOM file. This is the property that makes a save button meaningful:
the failures that a doctor can actually cause (a stale path from an old screen,
a modality that does not belong to this lesion type) cannot half-apply.

**2. Diff against the current set.**

| Case | Action |
|---|---|
| Modality now points at a different series | copy and anonymize the new one; the previous occupant returns to `excluded_series` |
| Modality was set, absent from the new set | delete its directory; the series returns to `excluded_series` |
| Modality unchanged | nothing at all |

The third row matters: without it, every save would re-copy every DICOM in the
session.

**3. Apply, then write once.**

Copying reuses `copy_and_anonymize_series` with its existing guard — if fewer
files are copied than found, it raises and `dataset_mapping.json` is *not*
rewritten. The mapping is written exactly once, at the end.

Two behaviours that do not exist today:

- **Clearing a modality without replacing it is allowed.** "This is not the
  t1c" is a legitimate statement. The session becomes incomplete and moves
  back into `_incomplete/`. Today the move only runs the other way
  (incomplete → main tree); the reverse has to be written.
- **`needs_reprocess: true`** is set whenever the resulting set differs from
  the previous one — with no check for whether the session was ever processed.
  If there are no artifacts, the later purge is simply a no-op. Deciding this
  from the presence of files would make the flag depend on filesystem state
  that other things also change.

There is no filesystem-level transaction here; copying DICOM is not atomic.
The boundary is drawn where it matters: `dataset_mapping.json` is the source of
truth every stage reads, and it either describes the new set completely or is
untouched.

## Reprocessing

**`backend/session_artifacts.py`** — one responsibility, testable alone:

```python
STAGE_DIRS = ("nifti", "preprocessed",
              "quality_reports", "segmentation", "transformations")

delete_session_artifacts(output_path, patient_id, session_id) -> list[str]
purge_sessions_marked_for_reprocess(output_path) -> dict
```

`metadata/` is **not** in that list, though it has the same layout. It is
written by stage 01 as it copies DICOM, and stage 02 — which could rebuild
it — is disabled in this pipeline. Since `bids_organized/` is kept on
purpose, stage 01 skips the patient on a requeue and never rewrites it, so
deleting metadata loses it permanently. The loss is not cosmetic either:
`_compute_study_hash` reads `PatientID` and `StudyInstanceUID` from there,
and without it every session already in the dataset comes back as
`name_clash` and nothing uploads at all.

Every stage writes per-patient output as `{stage_dir}/{sub-XXX}/{ses-YYY}/`,
verified against a real run, so deletion is a predictable walk over five
directories plus removing the patient directory if it is left empty.

`bids_organized/` is deliberately **not** deleted: it holds the corrected
assignment and is the pipeline's input, not its output.

`patient_id` and `session_id` are validated against the existing
`_BIDS_PATIENT_ID_PATTERN` / `_BIDS_SESSION_ID_PATTERN` before any path is
built. This is a delete driven by values read out of a JSON file; escaping the
run's directory must be impossible by construction, not by luck.

**When:** the requeue endpoint calls `purge_sessions_marked_for_reprocess`
before launching, clears the flags, and starts the run. `skip_existing` then
does the rest — what is missing gets rebuilt, every other patient is untouched.

## The Kappa gap

`_compute_study_hash` hashes `PatientID:StudyInstanceUID` — the identity of the
DICOM study, not the images. Reprocessing the same study with a corrected
modality set therefore produces **the same hash**, so the uploader treats the
patient as already delivered and Kappa keeps the mask computed from the wrong
set. Locally corrected, remotely stale, and silent about it.

Replacing the entity is possible in principle — `kappa_client.replace_entity_file`
already exists for expert-edited masks — but it takes one file at a time and it
is not established whether it replaces a multi-file entity or adds to it. That
is its own investigation.

**Observed in practice (run `fbc37b13`, 2026-10-01):** the first version of
this work deleted `metadata/` along with the stage outputs, which made the
hash `None` and turned both sessions into `name_clash` — nothing uploaded
at all. So the realistic failure is not only that Kappa quietly keeps the
old version; it can also refuse the new one. Either way the correction does
not reach Kappa, which is what the warning has to convey.

**Decision: out of scope here; warn instead.** When a session marked for
reprocessing already has a `kappa_entity_id`, the save response and the UI say
plainly that the version in Kappa is the old one and will not update by itself,
and the same line goes into the run's `logs/kappa.log`. Replacing it is a
separate task.

The registry lookup is scoped to the run's own `kappa_dataset_id`
(`find_by_bids_id(session_key, {dataset_id})`), never unscoped. `sub-NNN` is
unique only within a dataset, and an unscoped lookup would report another
account's patient as this one — the exact failure that
`2026-09-21-bids-numbering-per-dataset-design.md` exists to prevent, and that
bit us once already in `count_local_progress`.

## The screen

The session dialog becomes two lists.

**Отобранные модальности** — one row per required modality of this lesion type:

```
☑ t1     t1_mprage_sag_p2_iso            176 срезов
☑ t1c    t1_mprage_sag_p2_iso_KM         176 срезов
☐ t2     —  не назначена
☑ t2fl   t2_flair_sag_p2_iso             176 срезов
```

The checkbox is the mechanism for replacing, not a record of review: nothing
about its state is persisted. Clearing it frees the slot, which offers
"Назначить сюда" on eligible series below. Assigning moves that series up and
the previous occupant down. All of it stays in the browser until saved.

**Неотобранные серии** — as today, but in two groups: first the ones the
detector recognized (with the reason they lost), then, under their own heading,
**the unrecognized ones**, assignable the same way. Their separation is
presentational; it exists so the doctor can tell "the algorithm considered this
and rejected it" from "the algorithm had no opinion".

**Сохранить** is enabled only when something changed, with «Отменить» beside it
to return to the loaded state. After saving, the dialog states that the patient
will be reprocessed on the next run — and, when applicable, that the Kappa copy
stays as it is.

## Testing

| File | Covers |
|---|---|
| `backend/test_session_assignment.py` | Validation before any write: a path from another session, a modality outside this lesion type's `required`, one series assigned twice — each rejected with `dataset_mapping.json` byte-identical afterwards. An unchanged modality is not re-copied. Replacement returns the previous occupant to `excluded_series`. Clearing a modality makes the session incomplete and moves it back to `_incomplete/`. `needs_reprocess` is set only when the set actually changed |
| `backend/test_session_artifacts.py` | The six stage directories lose the session; `bids_organized/` keeps it. Neighbouring sessions and patients are untouched. A malformed `patient_id` (`../..`) is refused rather than resolved. Purge clears the flags it acted on |
| `backend/test_incomplete_patients_api.py` (extend) | The response carries `selected` with protocol names, and `required` from the lesion type config: `t1/t2/t2fl` for multiple sclerosis, **no `t1c`** |
| `backend/test_app_requeue_endpoint.py` (extend) | Requeue purges flagged sessions before the run starts |

Frontend is covered by `npm run lint`; the repo has no JS test harness.

**Manual check:** change the set of an already-processed patient, requeue, and
confirm that exactly that session was rebuilt and no other patient was touched.
