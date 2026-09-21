# BIDS Subject Numbering Scoped to the Kappa Dataset — Design

**Date:** 2026-09-21
**Branch:** `feat/bids-per-dataset` (from `main`)
**Type:** Behaviour change (patient identifiers) + bugfix

## Problem

`utils/bids_allocator.py` numbers subjects per `lesion_type`, and says so in its
own docstring: "lesion_type maps 1:1 to a Kappa dataset in the current
configuration". The per-user dataset change (`2026-09-15-kappa-per-user-dataset`)
ended that: every Kappa user now has their own dataset per lesion type, and Kate
also creates datasets by hand and points `kappa_datasets.yaml` at them. Numbering
did not follow. A brand-new dataset therefore starts at whatever the global
counter has reached — `sub-151`, not `sub-001`.

Wanted: **a new dataset starts at `sub-001`; a dataset that already holds data
continues after its highest number** — without anything getting mixed up.

### What the live data shows (2026-09-21)

| fact | value |
|---|---|
| GBM numbers allocated globally | 317 (`sub-001`…`sub-317`) |
| datasets those patients are spread over | 133, 249, 266, 349 |
| people whose sessions span two datasets | none |
| one number held by two different people **within** GBM | 1 — `sub-003`, in datasets 133 and 249 |

That collision predates the allocator (May 2026, per-run numbering). Scoping
lookups by dataset resolves it rather than perpetuating it.

## Goals

- The numbering space is the Kappa dataset the run uploads into.
- An empty dataset starts at `sub-001`; a non-empty one continues after the
  highest `sub-NNN` **that the dataset itself contains**.
- A patient is never ambiguous: a lookup resolves within a dataset first.
- A run never fails to start because Kappa is unreachable.
- No existing number changes.

## Non-goals

- Renaming entities already in Kappa. Their names are the numbers.
- Contiguous numbering. An incomplete patient keeps its number so it can be
  completed later; gaps (e.g. `sub-003` missing in 349) are intentional.
- Deferred upload with automatic retry — that is Spec B, separate.
- Safe *simultaneous* allocation from two machines into one dataset. See
  Limitations.

## Decisions

**D1 — The scope is the dataset id.** A string: `ds:337` for runs bound to a
Kappa dataset, `local:glioblastoma` for CLI runs with no Kappa at all (their
numbering stays as it is today). One string keeps the allocator dependency-free.

**D2 — The dataset is resolved once, at run start, and recorded on the run.**
Today it is resolved at upload, after numbering has already happened, so the two
ends can disagree — if `current` is repointed mid-run they will. Resolving at the
start also surfaces a permissions error (the 403 that `test_med1` hit) before GPU
time is spent, which is recommendation 1 of KI-057.

**D3 — The next number comes from the dataset, not only from our table.**
`next = max(highest local number for the scope, highest sub-NNN among the
dataset's entity names) + 1`. Entity names are `sub-001_ses-001` and are readable
via `get_dataset_entities()` (`backend/kappa_client.py`), already used by the
validation tab. This is what makes a hand-made dataset behave correctly, and it
is also what keeps two machines — this laptop and barguzin, each with its own
SQLite — from independently issuing `sub-001` to two different people in the same
dataset. The local table becomes a cache; the dataset is the authority.

**D4 — A provisional scope exists only for one corner.** No dataset for this
(user, lesion) anywhere *and* Kappa unreachable at start: number into
`pending:<run_id>` and bind it to the dataset once one is created, renaming the
scope on those allocation rows. A provisional scope always binds to a **newly
created** dataset, never merges into an existing one — merging could collide with
numbers already in that dataset. In Kate's workflow (dataset made by hand, id in
the config) this corner never occurs.

**D5 — Lookups resolve within a dataset, then join by the real patient.**
`sub-001` alone stops being unique, so a lookup needs the dataset. From it the
registry gives `original_patient_id`, which is unambiguous, and the patient's
sessions are then collected across the datasets **of the same Kappa account**.
This is what keeps an MS timeline whole when `current` moves from 158 to 338.
Datasets of other accounts are excluded, matching the isolation the per-user
change introduced.

**D6 — Migration preserves every number.** Existing allocations move into the
scope of the dataset they were uploaded to; allocations never uploaded move to
`local:<lesion_type>`. Nothing is renumbered.

## Data model

```
pipeline_runs
  + kappa_dataset_id INTEGER NULL      -- the dataset this run numbers in and uploads to

bids_patient_allocation  (rebuilt)
    scope               TEXT NOT NULL  -- 'ds:337' | 'local:glioblastoma' | 'pending:<run_id>'
    original_patient_id TEXT NOT NULL
    bids_id             TEXT NOT NULL
    created_at          TEXT NOT NULL
    PRIMARY KEY (scope, original_patient_id)
    UNIQUE      (scope, bids_id)
```

The column migration follows the existing pattern in `backend/database.py`
(`_migrate_add_lesion_type`, `_migrate_add_parent_run_id`). The old allocation
table is renamed to `bids_patient_allocation_legacy`, not dropped, so the change
is reversible.

## Flow

```
run start ──► resolve dataset ──► seed the scope ──► stage 01 numbers ──► … ──► upload
   │               │                    │                                        │
   │               │                    └─ max(local, Kappa entity names)         │
   │               └─ mapping hit → use it; miss → create in Kappa                │
   └─ kappa_dataset_id stored on the run ──────────────────────────────────────────┘
```

1. **Start** (`POST /api/pipeline/start`): `user_id` from the Kappa session,
   `preprocessing_id` from the preprocessing config, then
   `get_dataset_id(user_id, lesion_type, preprocessing_id)`. On a miss, create the
   dataset in Kappa. The resolve-or-create logic moves out of
   `KappaUploader._resolve_dataset_id()` into a shared helper used by both ends.
   Kappa unreachable and no mapping entry → provisional scope (D4) and a warning
   on the run.
2. **Seed**: when Kappa is reachable, read the dataset's entity names and record
   the highest `sub-NNN` seen, so allocation starts above it.
3. **Stage 01**: `pipeline_manager` writes `general.numbering_scope`;
   `orchestrator.py` passes `--numbering-scope` to `stage_01_reorganize` only,
   beside the existing `--lesion-type`; `IDMapper` calls
   `get_or_allocate(scope, original_id)`.
4. **Upload**: `KappaUploader` uses the run's `kappa_dataset_id` instead of
   resolving again. For a provisional scope it creates the dataset, rebinds the
   scope, then uploads.
5. **Resume / requeue**: the new run inherits the parent's `kappa_dataset_id`.

## Lookups

- `find_by_bids_id()` and `find_by_bids_subject()` (`backend/patient_registry.py`)
  take an optional dataset filter. Today they scan the whole registry, which is
  why GBM `sub-003` returns two different people.
- `GET /api/longitudinal/{patient_id}` and `/diff` take `run_id`. The run gives
  the dataset, the dataset plus subject give `original_patient_id`, and the
  sessions are collected across that account's datasets. Records with no dataset
  (run never uploaded) are included — they exist only on the machine that
  produced them.
- The account's datasets are the values of `kappa_datasets.yaml` keys that start
  with that `user_id`.
- Frontend: `ClinicalReportContent` already knows `runId` and passes it to
  `LongitudinalTimeline`, which passes it to the two API calls.

## Migration

1. Refuse to run if, inside any one dataset, a number maps to two people or a
   person maps to two numbers. Print the conflicts and stop — do not guess. On
   the current data this check passes (the only collision spans two datasets).
2. For every registry row with a `kappa_dataset_id`, write
   `(ds:<id>, original_patient_id) -> sub-NNN`.
3. Allocations with no registry row (numbered locally, never uploaded — 317 GBM
   numbers allocated, 113 of them uploaded) move to `local:<lesion_type>` with
   their numbers unchanged.
4. Keep `bids_patient_allocation_legacy` for rollback.

## Testing

- **Allocator**: two datasets, same person → a number in each, both starting at
  `sub-001`; stable on a second run in the same scope; existing
  `BEGIN IMMEDIATE` concurrency test still passes with the new key.
- **Seeding**: a scope whose dataset already holds `sub-001…sub-007` allocates
  `sub-008`, even when the local table is empty.
- **Migration**: run against a copy of the production DB — no uploaded patient
  changes number, row counts reconcile, and an injected conflict aborts.
- **Lookups**: two different people both named `sub-001` in different datasets do
  not merge; one person's sessions in 158 and 338 do merge; another account's
  sessions never appear.
- **Manual**: a run into a fresh dataset yields `sub-001`; a run into a dataset
  holding data continues after its last number.

## Limitations

- **Two machines starting a run into the same dataset at the same moment** both
  read the same maximum and pick the same next number. Sequential use is safe.
- **Kappa unreachable at start** means the maximum comes from the local table
  only, which is stale if another machine has added entities since.

Both are caught at upload rather than silently: duplicates are already detected
by `study_hash` **against the dataset itself**, so a session uploaded from
barguzin is recognised from the laptop. A name clash (same `sub-NNN`, different
study) will be reported as a warning on the run.
