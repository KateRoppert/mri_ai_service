# Replacing a Reprocessed Patient in Kappa — Design

**Date:** 2026-10-02
**Branch:** `feat/kappa-replace-on-reprocess` (from `main`)
**Type:** Feature + a corrected assumption about the Kappa API

## Problem

When a doctor fixes a patient's modality set and the pipeline reruns, Kappa
keeps the result computed from the wrong set. The uploader dedups by
`study_hash`, which covers `PatientID:StudyInstanceUID` — the identity of the
DICOM study, not the images — so the reprocessed session looks like a
duplicate and is skipped. The correction lands on disk and never reaches the
dataset anyone actually reviews.

The previous spec
(`2026-09-30-incomplete-patient-assignment-design.md`) deferred this,
warning the operator instead, because the API we use appeared unable to
replace anything.

## What the API actually does

That conclusion was wrong, and worth recording precisely because it was
wrong for a specific reason: it was drawn from the one API version our client
happens to use.

Probed live against a throwaway dataset (355 `ZZ_TEST_replace_probe`), never
against clinical data:

| | v1 (`kappa_client.py` uses this) | v2 |
|---|---|---|
| Replace a file | **adds a duplicate** — same filename, new id | `PATCH /datasets/{ds}/{entity}/{file_id}` replaces in place, id preserved |
| Delete files | — | `DELETE /datasets/datasetEntities/files` (async) |
| Delete entities | `DELETE` → `405` | `DELETE /datasets/datasetEntities` (async) |
| Recover deleted | — | `POST /datasets/datasetEntities/recover` |
| Our existing token | works | **works, unchanged** |
| Response shape | — | **identical keys** |
| URL shape | carries `{user_id}/{user_type_id}` | identity comes from the token |

Two findings have consequences beyond this feature:

- `replace_entity_file` in v1, despite its name, **appends**. Uploading a file
  under a name that already exists produces a second file with that name and a
  different id. The expert-mask flow (`app.py:2152`) is built on it, so expert
  masks accumulate in Kappa. That turns out to be what we want (see below), but
  `MaskVersion`'s docstring states the opposite — "в Каппе хранится только
  актуальная (последняя)" — and is simply untrue. It gets corrected here.
- Deletion is asynchronous: `202` plus a `jobId` to poll. Even after the job
  reports `succeeded`, the entity listing still returned the deleted record for
  a second or so. Nothing may treat the listing as immediately consistent after
  a delete.

## Two kinds of new version, deliberately different

| | Behaviour | Why |
|---|---|---|
| Reprocessed patient | **Replaced** | The old result was computed from a modality set now known to be wrong. It is not an alternative, it is a mistake. |
| Expert mask edit | **Kept, all versions** | Not a correction of an error but a specialist's opinion. Each version has standing. |

Until the probe, these two behaved the same way only because the API allowed
nothing else. They now differ because they are different things.

## Goals

- A reprocessed session can replace its Kappa entity's contents, on the
  operator's explicit confirmation.
- Expert masks survive a replacement, and the operator is told they predate
  the new data.
- Nothing is overwritten in Kappa without a human saying so.
- No new delivery state: the existing queue, banner and per-run modal carry it.

## Non-goals

- Migrating `kappa_client.py` to v2. See "Where v2 enters".
- Automatic replacement. Overwriting data in someone else's system is not a
  thing to do silently.
- Deleting entities or datasets. Only an entity's files are touched.

## Where v2 enters

A new module, `backend/kappa_entity_files.py`: operations on an entity's files,
v2 only. `kappa_client.py` stays on v1 untouched.

Migrating the whole client is tempting now that the token and the response
shapes are known to match — but v1 works, is covered by tests, and the
uploader's dedup and name-clash logic depends on its exact behaviour. Putting
the upload path we spent a week stabilising at risk to gain one function is a
bad trade. The evidence for a later migration is recorded above; it is its own
task.

A separate module also makes the version boundary visible. Two base URLs mixed
through one file would be something a reader discovers by accident.

## Replacing an entity's contents

```python
replace_entity_contents(
    token, user_id, user_type_id, dataset_id, entity_id, files: list[Path],
) -> ReplaceResult   # patched / added / deleted counts, plus any job ids
```

It reads the entity's current files, matches by filename, and acts per file:

| Case | Action | Synchronous |
|---|---|---|
| Name present in both | `PATCH` by `file_id` — contents change, id preserved | yes |
| Name only in the new set | `POST /datasetEntities/files/{ds}/{entity}` | yes |
| Name only in the entity | `DELETE /datasetEntities/files` | **no** — `202` + `jobId` |

**Expert masks are exempt from the delete rule.** Files matching
`*_segmask_v{N}.nii.gz` are not part of a recomputed set and must survive, per
the decision above. Without this exemption the first replacement would wipe
every expert edit on that patient.

Deletion polls `GET /datasets/datasetEntities/bulk-mutation/jobs/{jobId}?datasetId=`
every 2 seconds for at most 60, then gives up and reports the job as still
running rather than claiming either outcome. It does **not** re-read the entity
listing to confirm, because that listing was observed to lag behind a job that
had already reported `succeeded`.

**A replacement can fail halfway.** Each file is its own request, so a failure
after three of five leaves the entity holding a mix. There is no transaction to
reach for and inventing one would mean staging a copy of every file in Kappa.
Instead the result says exactly which files were replaced and which were not,
the session stays `needs_attention`, and the action can simply be repeated —
`PATCH` by `file_id` is idempotent, so replaying it costs nothing and converges.
Claiming success on a partial write would be the real failure here.

## Knowing which sessions supersede Kappa

`needs_reprocess` cannot answer this: it is cleared during the purge, long
before upload.

So the requeue endpoint, which already purges flagged sessions, records what it
purged onto the run it creates — `pipeline_runs.reprocessed_sessions`, a JSON
list of `"sub-NNN_ses-NNN"`. The knowledge then survives to the moment it is
needed.

At upload, a session that dedup would skip as a duplicate **and** which appears
in that list is reported as `supersedes` rather than as a success.
`kappa_delivery.classify` maps it to the existing `needs_attention` with reason
`supersedes_kappa`.

No fourth status. The column tag, the summary banner and the per-run modal
already handle `needs_attention`; a new reason needs a label and an action, not
a new state for the history, the worker and the summary to each learn about.

## The confirmation

In the per-run modal, beside «Повторить сейчас», blocked sessions with reason
`supersedes_kappa` get **«Заменить в Kappa»**:

> Заменить версию в Kappa? Файлы пациента будут перезаписаны результатами новой
> обработки. Прежняя версия не сохранится.
>
> У этого пациента есть экспертные маски (2). Они останутся, но нарисованы по
> прежним данным.

The second paragraph appears only when such masks exist; the count comes from
`mask_versions`.

`POST /api/kappa/replace-entity/{run_id}/{patient_id}/{session_id}` performs it,
and refuses with `400` for a session not marked as superseding — the endpoint
must not be a general-purpose overwrite tool reachable by guessing a URL.

## Testing

| File | Covers |
|---|---|
| `backend/test_kappa_entity_files.py` | The diff: matching name → `PATCH`, new → add, absent → delete. **Expert masks are never deleted.** Job polling: success, failure, timeout. An empty set issues no calls at all |
| `backend/test_kappa_delivery.py` (extend) | `supersedes` → `needs_attention`, reason `supersedes_kappa`, no retry scheduled |
| `backend/test_app_replace_endpoint.py` | Replacement only for a session marked as superseding; any other is `400`. The response reports how many files were patched, added and deleted |
| `backend/test_app_requeue_endpoint.py` (extend) | Requeue records the purged sessions onto the new run |

Every Kappa call in the suite is mocked. The probing was done live against a
throwaway dataset, but a test suite that reaches a third-party service stops
being reproducible and starts failing for reasons that have nothing to do with
the code.

**Manual check:** reprocess a patient, replace, and confirm in Kappa that the
file count did **not** grow, that the file ids are unchanged, and that the
expert masks are still there.
