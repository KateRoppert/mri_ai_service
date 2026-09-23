# Deferred Kappa Upload with Automatic Retry — Design

**Date:** 2026-09-23
**Branch:** `feat/deferred-kappa-upload` (from `main`)
**Type:** Reliability feature + bugfix (silent upload failures)

## Problem

A pipeline run that finishes successfully can still fail to reach Kappa, and
today nobody finds out. The results are on disk and nothing is lost in
principle, but there is no record that delivery is owed, no retry, and no
warning. Kate hit this twice during testing on 2026-09-22: one run uploaded
nothing at all and looked completed, another delivered part of its patients.

Wanted: **a run whose upload fails must not lose the results — the data is
delivered later, automatically, with a visible warning meanwhile.**

### The four ways delivery fails today

| # | Path | What the operator sees |
|---|---|---|
| 1 | No Kappa session at run start → `kappa_uploader = None` (`backend/pipeline_monitor.py:77`) | Nothing. No attempt is ever made |
| 2 | Kappa unreachable → `_resolve_dataset_id()` returns `None` → `upload_results()` **returns** `{"error": "Failed to resolve dataset_id"}` | `logger.info("Kappa upload results: ...")` — logged at INFO, indistinguishable from success |
| 3 | Exception during upload | `logger.error("Kappa upload error: %s")` and nothing else |
| 4 | Some sessions fail (`name_clash`, a 5xx on one patient) | In the log only; the run's status stays `completed` |

Path 2 is the important one: `_kappa_upload_safe` wraps the call in
`try/except`, but `upload_results` reports this failure by **return value**,
not by raising. Error handling written only around exceptions is blind to code
that returns its errors.

A second latent defect: the uploader is constructed **once, at the top of the
monitor loop**, holding the token as it was at that moment
(`backend/pipeline_monitor.py:78`). A three-hour run finishes with a
three-hour-old token. A long-lived task that caches a credential at scheduling
time implicitly treats that credential's lifetime as the task's boundary.

## Goals

- A completed run that owes data to Kappa is recorded as owing it.
- Transient failures (network, 5xx, Kappa down, no token yet) are retried
  automatically in the background, with backoff, without the browser open.
- Failures that retrying cannot fix are marked for a human instead of being
  retried forever.
- The operator sees delivery state per run, with a counter and a per-patient
  breakdown, plus an immediate warning when the upload right after a run fails.
- No patient is ever uploaded twice.

## Non-goals

- Changing `kappa_uploader.py` itself. Its `study_hash` dedup, `name_clash`
  check and `pending:<run_id>` scope rebinding were stabilised on 2026-09-21
  and stay as they are. Only its *callers* change.
- Resurrecting the entire run history. The migration backfills a bounded
  window (see Migration).
- Retrying anything for CLI / `orchestrator.py` runs, which never intended to
  upload.

## Key insight

`upload_results()` is idempotent: it discovers sessions from `output_path`,
compares them against the dataset by `study_hash`, and skips what is already
there (reconciling the local registry row instead). Retrying it is therefore
safe and resumable, and a retry is not "send again" but "reconcile with
reality".

This is what lets the stored delivery state be a **cache that self-heals**
rather than an independent ledger: every attempt recomputes the truth from disk
and from Kappa. If someone uploads a session by hand, the next attempt notices
and settles on `done`.

## State model

Four new columns on `pipeline_runs` — two queryable, one JSON blob, one key:

```
kappa_upload_status        TEXT      pending | done | needs_attention | NULL
kappa_upload_next_attempt  DATETIME  when the worker may pick this run up again
kappa_upload_detail        TEXT      JSON: counters, blocked sessions, attempts, last error
kappa_user_id              INTEGER   which Kappa account started this run
```

### Why `kappa_user_id` and not `kappa_session_id`

The worker needs a live token at attempt time. Sessions expire after ~7 days
and a re-login issues a **new** `session_id`, so a remembered session id is
dead exactly when the deferred upload needs it. The Kappa `user_id` is stable:
the worker looks up the newest unexpired session for that user.

For runs that started while Kappa was reachable the owner could also be derived
via the existing `_dataset_owner()` (`backend/app.py:1372`), but a run that
started while Kappa was **down** has no `kappa_dataset_id` yet (it numbers into
`pending:<run_id>`). The column covers both cases uniformly.

### Intent is recorded at birth

The start endpoint already knows whether the caller supplied a
`kappa_session_id`. If it did, the run is created with
`kappa_upload_status = 'pending'`; if not, the column stays `NULL` forever and
the worker never sees the run. This removes any guessing about whether a run
was meant to be uploaded, and keeps CLI runs out of the queue by construction.

### Transitions

```
   run created with a Kappa session
            |
         pending ---------- everything delivered ------> done
            |  ^                                          ^
            |  +-- network / 5xx / no token --+           |
            |      (backoff 1-2-5-15-30-60 min)           |
            |                                             |
            +-- name_clash / 403 / files missing          |
                        |                                 |
                 needs_attention --- human fixes it ------+
                                     and retries
```

`done` is set only when **every** session discovered on disk is present in the
dataset — uploaded now, or already there via the dedup path. Partial success
stays `pending` and carries its counter.

### `kappa_upload_detail` shape

```json
{
  "total": 5,
  "delivered": 3,
  "blocked": [
    {"session": "sub-003_ses-001", "reason": "name_clash",
     "message": "В датасете уже есть sub-003_ses-001 с другими данными"}
  ],
  "reason": "network",
  "last_error": "ConnectTimeout: kappa.nsu.ru:8061",
  "attempts": 7,
  "first_failure_at": "2026-09-23T09:02:11Z",
  "last_attempt_at": "2026-09-23T10:14:02Z"
}
```

`reason` is what the UI turns into words: `network`, `no_session`,
`name_clash`, `missing_files`, `stuck`. `first_failure_at` is the anchor for
the `stuck` rule below; it is cleared whenever a session is delivered.

## Components

### `backend/kappa_delivery.py` (new)

The retry policy as a pure function. No network, no database, no clock reads
beyond an injected `now`.

```python
def classify(result: dict | None, exc: BaseException | None, state: dict,
             now: datetime) -> dict
```

In: what `upload_results()` returned (or the exception it raised) plus the
current delivery state. Out: the new state — `status`, `next_attempt`, and the
`detail` dict above. Keeping the whole policy behind one signature is what
makes it testable without a Kappa instance.

The worker has one input of its own — "no live session for this user" — which
it expresses as `result={"error": "no_session"}` so that everything still
enters through the single `classify()` door.

### What is actually observable

`backend/kappa_client.py` logs HTTP status codes and returns `None`
(`kappa_client.py:302`, `:136`). **Status codes never reach the uploader**, so
the classifier cannot see a 403, a 401 or a 500. The vocabulary that actually
crosses the boundary is:

| From | Value |
|---|---|
| `upload_results()` top level | `{"error": "Failed to resolve dataset_id"}` |
| per-session `error` | `"name_clash"`, `"duplicate"`, `"no files"`, `"upload failed"` |
| raised | any exception |
| worker-supplied | `{"error": "no_session"}` |

Widening that vocabulary would mean changing `kappa_client` and
`kappa_uploader`, which this design deliberately leaves alone. The policy is
built on what exists.

Classification table:

| Outcome | New status | Retry |
|---|---|---|
| exception (network, timeout, anything raised) | `pending` | backoff, attempt counted |
| `{"error": "Failed to resolve dataset_id"}` — Kappa unreachable, dataset gone, or no rights; indistinguishable here | `pending` | backoff, attempt counted |
| per-session `"upload failed"` | `pending` | backoff, attempt counted |
| `{"error": "no_session"}` | `pending` | fixed 5 min, **attempt not counted** — waiting for a human to log in is not a failure |
| per-session `"name_clash"` | `needs_attention` | none |
| per-session `"no files"` | `needs_attention` (`missing_files`) | none |
| every session `success` or `duplicate` | `done` | — |

Backoff schedule: 1, 2, 5, 15, 30, 60 minutes, then 60 forever.

### Escalation: the `stuck` rule

Because a permission failure is indistinguishable from a network failure at
this layer, a run that only ever retries would hide a permanent problem behind
a reassuring "досылка выполняется автоматически" forever. So: **a run that has
been `pending` for more than 24 hours since its first failure without
delivering a single session becomes `needs_attention` with reason `stuck`.**

`no_session` waits do not count toward that window — a Kappa account nobody has
logged into over the weekend is not a stuck run. Partial progress resets it:
delivering any session means the path works and the rest is worth retrying.

This rule is load-bearing, not a nicety. It is the only route by which a 403,
a deleted dataset or a revoked token ever reaches a human.

Mixed outcomes take the worst: any blocked session makes the run
`needs_attention` even if others went through, because that run needs a human
regardless. Delivered sessions are never re-attempted (the dedup path skips
them), so escalating costs nothing.

### `backend/kappa_delivery_worker.py` (new)

An asyncio task started in the existing `lifespan`
(`backend/app.py:138`), beside the orphaned-run reconciliation. No new process,
container or scheduler — `pipeline_monitor` already works this way.

Every 60 seconds:

1. Select at most 5 runs matching
   `status = 'completed' AND kappa_upload_status = 'pending'
    AND (kappa_upload_next_attempt IS NULL OR kappa_upload_next_attempt <= now)`.
2. Resolve the owning Kappa account — `kappa_user_id` when set, otherwise
   `_dataset_owner(kappa_dataset_id)` for rows the migration backfilled — then
   find a live session for it. None → apply the `no_session` branch and move
   on.
3. Build a **fresh** `KappaUploader` (current token, `dataset_id` from the run)
   and `await upload_results()`.
4. Feed the outcome to `classify()` and persist the returned state.
5. Broadcast the new state over WebSocket for that run.

The 5-run cap bounds a tick's work: `upload_results` sleeps 2 s between
sessions, so a backlog cannot monopolise the loop.

### `backend/kappa_dataset_mapping.py`

`_dataset_owner()` currently lives in `backend/app.py:1372`, and the worker
cannot import it from there — `app` imports `pipeline_monitor`, and the worker
is started from `app`'s lifespan, so the dependency would close a cycle. It
moves to `kappa_dataset_mapping.py` as `owner_of_dataset(dataset_id)`, beside
`_load_mapping()` and `datasets_of_user()`, which is where it always belonged;
`app.py` calls the moved function. Pure relocation, no behaviour change.

### `backend/kappa_auth.py`

One addition: `find_live_session_for_user(user_id) -> dict | None` — newest
`kappa_sessions` row for that user whose `token_expiry` is in the future.

### `backend/pipeline_monitor.py`

- `_kappa_upload_safe` routes both its success and its failure through
  `classify()` and persists the result. This is where defect #2 dies: the
  `{"error": ...}` return is now an outcome, not a log line.
- Path #1 (no uploader at all) also records state: a run created with
  `pending` whose uploader could not be built simply stays `pending` — the
  worker will pick it up once a session exists.
- The uploader is rebuilt at completion time instead of being reused from the
  top of the loop, so the token is fresh.

### `backend/app.py`

- Start endpoint: set `kappa_upload_status='pending'` and `kappa_user_id` when
  the request carried a Kappa session.
- `/api/kappa/retry-upload/{run_id}`: persist the outcome through `classify()`
  like every other caller. Three call sites, one policy.
- `GET /api/kappa/delivery/summary` → `{"pending": 2, "needs_attention": 1}`.
- History endpoint returns the nested `kappa_upload` object per run.

### `backend/models.py`

`PipelineRunHistoryItem` gains:

```python
kappa_upload: Optional[KappaDeliveryStatus]
```

with `status`, `delivered`, `total`, `blocked`, `next_attempt_at`, `reason`.
Carried inside the existing history response — no extra round trip.

### Frontend

Three surfaces, because "did not reach Kappa" is noticed at three different
moments.

1. **Right after the run** — the monitor already broadcasts
   `kappa_upload_complete`; a `kappa_upload_deferred` message joins it and
   `ProgressMonitor.jsx` renders a warning: "Результаты не ушли в Kappa
   (сервис недоступен). Досылка выполняется автоматически, данные не
   потеряны."
2. **A "Kappa" column in the run history** (`PipelineHistory.jsx:164`), one tag
   per run: `✓ Kappa 5/5`, `⏳ 3 из 5 · повтор через 4 мин`,
   `⚠ 4 из 5 · 1 требует внимания`, `⚠ не удаётся выгрузить — нужна проверка`
   (reason `stuck`), or `—` when `status` is NULL. Clicking
   opens a modal with the per-patient breakdown and a **"Повторить сейчас"**
   button wired to `/api/kappa/retry-upload/{run_id}` — an endpoint that has
   existed since the 504 incident and that the frontend has never called.
3. **A summary `Alert` above the table**, driven by
   `/api/kappa/delivery/summary`, for the run that failed three days ago and
   has scrolled out of sight.

## Migration

`_migrate_add_kappa_delivery()` in `backend/database.py`, following the
existing `_migrate_add_*` + `PRAGMA table_info` pattern: add the four columns
if absent, idempotent.

Backfill, once, in the same migration:

```sql
UPDATE pipeline_runs SET kappa_upload_status = 'pending'
 WHERE status = 'completed'
   AND kappa_upload_status IS NULL
   AND kappa_dataset_id IS NOT NULL
   AND completed_at >= datetime('now', '-30 days')
```

Bounded on purpose. Without the window the migration would revive every run
ever completed, including ones nobody intended to upload. `kappa_user_id` stays
NULL for backfilled rows; the worker falls back to `_dataset_owner()` for those
— which works precisely because they have a `kappa_dataset_id`.

## Failure handling and edge cases

- **Two workers.** Only one backend process runs; if that ever changes, the
  `next_attempt` bump happens before the upload, so a second worker skips the
  run. Not a distributed lock, and does not need to be.
- **Backend restart mid-upload.** The run stays `pending` with an old
  `next_attempt`; the worker picks it up on the next tick and reconciles. A
  partially uploaded run is exactly the case the dedup path handles.
- **Run deleted / output_path gone.** `_discover_sessions()` finds nothing →
  `needs_attention` with `missing_files`, not an endless retry.
- **Token expires between selection and upload.** Kappa's 401 is swallowed by
  `kappa_client`, so this surfaces as `"upload failed"` → transient retry. The
  next tick finds no live session and switches to the `no_session` branch, so
  it settles correctly within a minute.
- **Dataset deleted in Kappa, or rights revoked.** Resolve fails →
  `{"error": "Failed to resolve dataset_id"}`, which looks exactly like Kappa
  being down. Retried for 24 hours, then escalated to `stuck`. The uploader
  never re-creates a dataset on this path (`dataset_id` is pinned on the run),
  because silently creating a replacement would split a patient cohort in two.

## Testing

| File | Covers |
|---|---|
| `backend/test_kappa_delivery.py` | `classify()`: every row of the classification table; the `{"error": "Failed to resolve dataset_id"}` return (today's defect #2); partial success counters; mixed outcome escalating to `needs_attention`; the 1-2-5-15-30-60 backoff schedule and its cap; `no_session` not incrementing `attempts`; the `stuck` rule — fires after 24 h with nothing delivered, does **not** fire when `no_session` waits fill the window, and is reset by any delivered session |
| `backend/test_kappa_delivery_worker.py` | Candidate selection honours status, `next_attempt` and the 5-run cap; no live session defers without counting an attempt; the uploader is built with the *current* token, not a stale one |
| `backend/test_pipeline_monitor_delivery.py` | `_kappa_upload_safe` persists state on all four failure paths, including the returned-error path and the no-uploader path |
| `backend/test_migrate_kappa_delivery.py` | Columns added idempotently; backfill respects the 30-day window and skips rows without `kappa_dataset_id` |

Tests follow the repo convention: `test_`-prefixed identifiers, explicit
cleanup, routed to the shared temp DB by `backend/conftest.py`. Frontend is
covered by `npm run lint` only — there is no JS test harness in the repo.

**Manual verification on the live stack:** point `kappa_datasets.yaml` at an
unreachable host, run the pipeline, confirm `pending` plus both warnings;
restore the host, confirm the status reaches `done` within a minute and that
the dataset contains each patient exactly once.
