# Per-User Kappa Dataset Mapping — Design

**Date:** 2026-09-15
**Branch:** `feat/kappa-per-user-dataset` (from `main`)
**Type:** Bugfix + behavior change (Kappa integration)

## Problem

When a pipeline run completes, the backend uploads the results to a Kappa
dataset. The dataset is chosen by `_resolve_dataset_id()`
(`backend/kappa_uploader.py`) via `get_dataset_id(lesion_type, preprocessing_id)`
against `configs/kappa_datasets.yaml`. The mapping key is
`{lesion_type}:{preprocessing_id}` (with a `:current` fallback) — **it does not
include the Kappa user**.

Observed: a new operator (`user_id=52`, `test_med1`) ran the glioblastoma
pipeline. The uploader resolved `glioblastoma:current → 249` — a dataset owned
by a different user (`user_id=26`, `e.roppert`/Kate) — and every write returned
**403 Forbidden** ("You don't have permission to perform this action"). Result:
`Upload complete: 0/1`, nothing was written anywhere.

Expected: a run by a given Kappa user writes to **that user's** dataset; a user
with no dataset yet for the lesion type gets a **new one created under their
token**.

## Root Cause

The dataset mapping and its accessors are keyed only by
`(lesion_type, preprocessing_id)`, shared across all users. So:
- `get_dataset_id` never "misses" once any user has created a dataset for a
  lesion type — it returns that (other user's) dataset id regardless of who is
  authenticated.
- The new user therefore never triggers dataset creation, and their upload to
  the other user's dataset is rejected with 403.

## Goals

- A run by Kappa `user_id` U writes to a dataset owned by U. A new user with no
  dataset for `(lesion_type, preprocessing_id)` gets one auto-created under
  their token, and subsequent runs reuse it.
- Existing datasets keep working for their original owner (user 26): no
  re-creation, no orphaning.
- The lesion-type dropdown in the UI reflects the *current user's* dataset ids.

## Non-Goals

- No change to how Kappa auth / sessions are established.
- No sharing/permissions model between users (each user's datasets are their
  own). Cross-user dataset sharing is out of scope.
- No change to the entity/session upload format or dedup logic.

## Design (Variant B — migrate existing mappings to their owner)

### Mapping key scheme
`configs/kappa_datasets.yaml` `datasets:` keys become user-scoped:

```
{user_id}:{lesion_type}:{preprocessing_id}
{user_id}:{lesion_type}:current
```

`get_dataset_id(user_id, lesion_type, prep)` looks up the exact user-scoped key,
then falls back to the user-scoped `:current`. `set_dataset_id(user_id, …)`
writes both the exact and `:current` user-scoped keys. A miss (user has no
dataset for that lesion type yet) returns `None`, so `_resolve_dataset_id`
creates a new dataset under the current user's token — the desired behavior.

### One-time migration (owner = user 26, confirmed from the DB)
`backend/data/brain_lesion.db` shows datasets **249** (glioblastoma, prep
`3a183dc7`) and **158** (multiple_sclerosis, prep `1099b9cd`) were created while
only `user_id=26` (Kate) existed, and `validations` for 158 are by user 26. So
both belong to user 26. Rewrite the existing userless keys:

```
glioblastoma:current: 249         →  26:glioblastoma:current: 249
                                     (+ 26:glioblastoma:3a183dc7: 249)
multiple_sclerosis:current: 158   →  26:multiple_sclerosis:current: 158
multiple_sclerosis:1e2b93ad: 158  →  26:multiple_sclerosis:1e2b93ad: 158
```

User 26 keeps writing to 249/158; user 52 (and any new user) gets fresh
datasets. The migration is a direct edit of the committed
`configs/kappa_datasets.yaml` (small, reviewed, no runtime migration code
needed). On load, any *remaining* userless `datasets:` key is ignored by the
new user-scoped lookups (kept only as an inert record; a follow-up may prune).

### The lesion-types endpoint needs the current user
`GET /api/kappa/lesion-types` (`backend/app.py`) currently calls
`get_lesion_types()` with no user and attaches `dataset_id` from the userless
`:current` mapping. It must resolve the current user:
- Accept `kappa_session_id` (the frontend already holds it — it is sent to
  `/api/pipeline/start`).
- Look up `user_id` from the `kappa_sessions` table for that session.
- Call `get_lesion_types(user_id)`, which attaches each lesion type's
  `dataset_id` from that user's `:current` key (or `None` if the user has none
  yet).

### Frontend
The lesion-type dropdown request must pass the current `kappa_session_id`.
Update `frontend/src/services/api.js` (the `getLesionTypes`/equivalent call)
and its caller to include the session id, mirroring how `startPipeline` already
passes it. When a lesion type has `dataset_id: null` for the current user, the
UI simply shows no existing dataset (a new one is created on first upload).

## Touch points
1. `backend/kappa_dataset_mapping.py` — `get_dataset_id(user_id, …)`,
   `set_dataset_id(user_id, …)`, `get_lesion_types(user_id)`; user-scoped keys.
2. `backend/kappa_uploader.py` — pass `self.user_id` at the two call sites
   (lines ~134, ~170).
3. `backend/app.py` — `/api/kappa/lesion-types` takes `kappa_session_id`,
   resolves `user_id` via `kappa_sessions`, passes it through.
4. `frontend/src/services/api.js` (+ caller) — send `kappa_session_id` with the
   lesion-types request.
5. `configs/kappa_datasets.yaml` — migrate 249/158 keys to owner 26.
6. `backend/test_preprocessing_version.py` — update to the new signatures; add
   a test for user-scoping (different users → different / newly-created keys).

## Behavior after the change
| Scenario | Result |
|----------|--------|
| User 26 runs glioblastoma | resolves `26:glioblastoma:current → 249`, writes to 249 (as before) |
| User 52 runs glioblastoma (no dataset yet) | miss → creates a new dataset under 52's token, stores `52:glioblastoma:…`, writes there |
| User 52 runs again | reuses their `52:glioblastoma:current` |
| lesion-types dropdown for user 52 | shows `dataset_id=null` until their first upload |

## Testing
- Unit: `get_dataset_id`/`set_dataset_id` user isolation — user A's write is not
  visible to user B; `:current` fallback per user; miss returns None.
- Unit: `get_lesion_types(user_id)` attaches the right per-user dataset id.
- Migration check: after editing the yaml, `get_dataset_id(26,'glioblastoma','3a183dc7')`
  and `get_dataset_id(26,'glioblastoma','anything')` both resolve 249; a
  different user id returns None.
- Endpoint: `/api/kappa/lesion-types?kappa_session_id=…` resolves the session's
  user and returns per-user dataset ids; unknown/absent session handled
  (400/empty, decided in the plan).
- Manual: re-run the failing case as user 52 → a new glioblastoma dataset is
  created under 52 and the upload succeeds (no 403).

## Risks / notes
- If `kappa_session_id` is missing/expired at the lesion-types call, the
  endpoint must degrade gracefully (return lesion types with `dataset_id=null`
  rather than 500) — pinned in the plan.
- Userless keys left in the yaml are inert but slightly confusing; a later
  cleanup can remove them once all datasets are user-scoped.
- Ownership of 249/158 is inferred (user 26 was the only pre-Sep-15 user);
  recorded here so it is auditable.
