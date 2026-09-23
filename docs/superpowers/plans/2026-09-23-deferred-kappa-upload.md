# Deferred Kappa Upload with Automatic Retry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A completed pipeline run whose Kappa upload fails records that it still owes data, retries automatically in the background, and shows the operator a warning until the data lands.

**Architecture:** Delivery state lives on `pipeline_runs` as a self-healing cache — every retry recomputes the truth by re-running the existing idempotent `KappaUploader.upload_results()`, which dedups against the dataset by `study_hash`. The retry policy is extracted into one pure function (`kappa_delivery.classify`) that three callers share: the post-run upload, a background asyncio worker, and the manual retry endpoint.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy + SQLite, pytest, React 19 + Ant Design.

**Spec:** `docs/superpowers/specs/2026-09-23-deferred-kappa-upload-design.md`

## Global Constraints

- **Do not modify `backend/kappa_uploader.py` or `backend/kappa_client.py`.** Their dedup, `name_clash` and `pending:<run_id>` rebinding logic was stabilised on 2026-09-21. Only callers change.
- **Status vocabulary is exactly** `pending`, `done`, `needs_attention`, or SQL NULL. No other value is ever written.
- **Reason vocabulary is exactly** `network`, `no_session`, `name_clash`, `missing_files`, `stuck`.
- **Backoff schedule is exactly** `[1, 2, 5, 15, 30, 60]` minutes, last value repeating forever.
- **`no_session` retry delay is exactly 5 minutes** and must NOT increment `attempts`.
- **Stuck threshold is exactly 24 hours** since `first_failure_at` with `delivered == 0`.
- **Worker tick is 60 seconds; at most 5 runs per tick.**
- **Migration backfill window is exactly 30 days.**
- Tests live beside the code as `backend/test_*.py` (the `tests/` directory is in `.gitignore`). Use `test_`-prefixed identifiers for any data written to the shared temp DB, and clean up explicitly.
- Code comments in English. Operator-facing UI strings in Russian.
- Commit style: conventional commits. End every commit message with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `backend/kappa_delivery.py` (new) | The retry policy as a pure function. No I/O, no clock, no DB. |
| `backend/kappa_delivery_worker.py` (new) | The asyncio loop: pick due runs, get a token, call the uploader, persist the verdict. |
| `backend/database.py` | Four new columns, their migration, and the accessors that read/write delivery state. |
| `backend/kappa_dataset_mapping.py` | Gains `owner_of_dataset()`, moved out of `app.py` to break an import cycle. |
| `backend/kappa_auth.py` | Gains `find_live_session_for_user()`. |
| `backend/pipeline_monitor.py` | Post-run upload routes its outcome through `classify()` and persists it. |
| `backend/app.py` | Records upload intent at run start; manual retry persists; summary endpoint; history carries delivery state. |
| `backend/models.py` | `KappaDeliveryStatus` response model, nested into `PipelineRunHistoryItem`. |
| `frontend/src/components/PipelineHistory.jsx` | The "Kappa" column, the per-patient modal, the summary alert. |
| `frontend/src/components/ProgressMonitor.jsx` | Handles the deferred-upload WebSocket message. |
| `frontend/src/services/api.js` | `retryKappaUpload()`, `getKappaDeliverySummary()`. |

Dependency order is strict: Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.

## Before You Start

```bash
cd /home/ubuntu/mri_ai_service
git checkout feat/deferred-kappa-upload   # already exists, branched from main
source venv/bin/activate
python -m pytest backend/ -q              # must be green before you touch anything
```

**Do not `git add -A` or `git commit -a` in this repo.** `configs/kappa_datasets.yaml` and `pipeline_config.yaml` are intentionally dirty in the working tree and must stay uncommitted. Stage files by explicit path, every time.

---

### Task 1: Delivery state columns, migration and accessors

**Files:**
- Modify: `backend/database.py` (model at `:30-75`, `init_db` at `:302`, `create_pipeline_run` at `:106`)
- Test: `backend/test_kappa_delivery_db.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PipelineRun.kappa_upload_status: Optional[str]`, `.kappa_upload_next_attempt: Optional[datetime]`, `.kappa_upload_detail: Optional[str]`, `.kappa_user_id: Optional[int]`
  - `create_pipeline_run(..., kappa_upload_status: Optional[str] = None, kappa_user_id: Optional[int] = None)`
  - `set_kappa_delivery(db, run_id: str, status: Optional[str], next_attempt: Optional[datetime], detail: dict) -> None`
  - `get_kappa_delivery(run: PipelineRun) -> dict` — always returns a dict, `{}` when unset
  - `runs_due_for_delivery(db, now: datetime, limit: int = 5) -> List[PipelineRun]`

**Background:** SQLite has no native datetime type and this codebase mixes naive and timezone-aware values. New code standardises on **timezone-aware UTC in Python** and **naive UTC in the database**, converted at exactly these two accessors so the conversion lives in one place.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_delivery_db.py`:

```python
"""Delivery-state columns on pipeline_runs and their accessors."""
from datetime import datetime, timedelta, timezone

import database as db_mod
from database import (
    SessionLocal,
    create_pipeline_run,
    get_kappa_delivery,
    runs_due_for_delivery,
    set_kappa_delivery,
)


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


def test_delivery_state_roundtrips_through_the_db():
    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out",
            run_id="test_delivery_roundtrip",
            kappa_upload_status="pending", kappa_user_id=26,
        )
        assert run.kappa_upload_status == "pending"
        assert run.kappa_user_id == 26
        assert get_kappa_delivery(run) == {}

        nxt = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
        set_kappa_delivery(
            db, "test_delivery_roundtrip", "pending", nxt,
            {"total": 5, "delivered": 3, "attempts": 2},
        )
        db.expire_all()
        run = db_mod.get_pipeline_run(db, "test_delivery_roundtrip")
        assert get_kappa_delivery(run)["delivered"] == 3
        # stored naive, returned aware
        assert run.kappa_upload_next_attempt.replace(tzinfo=timezone.utc) == nxt
    finally:
        _cleanup(db, "test_delivery_roundtrip")
        db.close()


def test_runs_due_for_delivery_selects_only_completed_pending_and_due():
    db = SessionLocal()
    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    ids = [
        "test_due_ready", "test_due_future", "test_due_running",
        "test_due_done", "test_due_null",
    ]
    try:
        for rid, status, upload, nxt in [
            ("test_due_ready", "completed", "pending", now - timedelta(minutes=1)),
            ("test_due_future", "completed", "pending", now + timedelta(minutes=30)),
            ("test_due_running", "running", "pending", None),
            ("test_due_done", "completed", "done", None),
            ("test_due_null", "completed", None, None),
        ]:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out",
                run_id=rid, kappa_upload_status=upload,
            )
            run.status = status
            db.commit()
            if nxt is not None:
                set_kappa_delivery(db, rid, upload, nxt, {})

        due = [r.run_id for r in runs_due_for_delivery(db, now)]
        assert "test_due_ready" in due
        assert "test_due_future" not in due
        assert "test_due_running" not in due
        assert "test_due_done" not in due
        assert "test_due_null" not in due
    finally:
        _cleanup(db, *ids)
        db.close()


def test_runs_due_for_delivery_honours_the_limit():
    db = SessionLocal()
    ids = [f"test_due_many_{i}" for i in range(7)]
    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    try:
        for rid in ids:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out",
                run_id=rid, kappa_upload_status="pending",
            )
            run.status = "completed"
            db.commit()
        assert len(runs_due_for_delivery(db, now, limit=5)) == 5
    finally:
        _cleanup(db, *ids)
        db.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_delivery_db.py -q`
Expected: FAIL — `ImportError: cannot import name 'get_kappa_delivery' from 'database'`.

- [ ] **Step 3: Add the columns to the model**

In `backend/database.py`, after the `parent_run_id` column (around `:73`):

```python
    # Deferred Kappa delivery. `kappa_upload_status` is NULL for runs that
    # never intended to upload (CLI, no Kappa session) — the worker ignores
    # those entirely. `kappa_upload_detail` is a JSON blob (see
    # backend/kappa_delivery.py). `kappa_user_id` is the Kappa account that
    # started the run: session ids rotate on re-login, user ids do not, so
    # this is what finds a live token days later.
    kappa_upload_status = Column(String, nullable=True)
    kappa_upload_next_attempt = Column(DateTime, nullable=True)
    kappa_upload_detail = Column(Text, nullable=True)
    kappa_user_id = Column(Integer, nullable=True)
```

- [ ] **Step 4: Add the migration**

In `backend/database.py`, append a new migration function beside the others (after `_migrate_add_kappa_dataset_id`):

```python
def _migrate_add_kappa_delivery():
    """Add the deferred-delivery columns to pipeline_runs, and mark recent
    completed runs as owing delivery so they are picked up once.

    The backfill window is deliberately bounded: without it the migration
    would revive every run ever completed, including ones nobody intended to
    upload. Rows it marks have no kappa_user_id — the worker falls back to
    owner_of_dataset(kappa_dataset_id), which works precisely because the
    WHERE clause requires that column to be set.
    """
    import sqlalchemy
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]
        added = False
        for name, ddl in [
            ("kappa_upload_status", "kappa_upload_status VARCHAR"),
            ("kappa_upload_next_attempt", "kappa_upload_next_attempt DATETIME"),
            ("kappa_upload_detail", "kappa_upload_detail TEXT"),
            ("kappa_user_id", "kappa_user_id INTEGER"),
        ]:
            if name not in cols:
                conn.execute(sqlalchemy.text(
                    f"ALTER TABLE pipeline_runs ADD COLUMN {ddl}"
                ))
                added = True
        if added:
            conn.execute(sqlalchemy.text(
                "UPDATE pipeline_runs SET kappa_upload_status = 'pending' "
                " WHERE status = 'completed' "
                "   AND kappa_upload_status IS NULL "
                "   AND kappa_dataset_id IS NOT NULL "
                "   AND completed_at >= datetime('now', '-30 days')"
            ))
        conn.commit()
```

Register it in `init_db()` at `backend/database.py:302`:

```python
    _migrate_add_kappa_dataset_id()
    _migrate_add_kappa_delivery()
```

- [ ] **Step 5: Extend `create_pipeline_run` and add the accessors**

In `backend/database.py`, add the two parameters to `create_pipeline_run` (`:106`) and pass them to the `PipelineRun(...)` constructor:

```python
def create_pipeline_run(
    db: Session,
    input_path: str,
    output_path: str,
    lesion_type: str = 'glioblastoma',
    parent_run_id: Optional[str] = None,
    kappa_dataset_id: Optional[int] = None,
    run_id: Optional[str] = None,
    kappa_upload_status: Optional[str] = None,
    kappa_user_id: Optional[int] = None,
) -> PipelineRun:
```

```python
        kappa_dataset_id=kappa_dataset_id,
        kappa_upload_status=kappa_upload_status,
        kappa_user_id=kappa_user_id,
```

Then append the accessors at the end of the module:

```python
def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite stores no timezone. Strip it on the way in so that comparisons
    inside the database are between like and like."""
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def set_kappa_delivery(
    db: Session,
    run_id: str,
    status: Optional[str],
    next_attempt: Optional[datetime],
    detail: dict,
) -> None:
    """Persist the verdict of one delivery attempt."""
    run = get_pipeline_run(db, run_id)
    if run is None:
        return
    run.kappa_upload_status = status
    run.kappa_upload_next_attempt = _naive_utc(next_attempt)
    run.kappa_upload_detail = json.dumps(detail, ensure_ascii=False)
    db.commit()


def get_kappa_delivery(run: PipelineRun) -> dict:
    """The detail blob, or {} when unset or unparseable. Never raises: a
    corrupt blob must not take down the history page."""
    raw = getattr(run, "kappa_upload_detail", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def runs_due_for_delivery(
    db: Session, now: datetime, limit: int = 5
) -> List[PipelineRun]:
    """Completed runs that still owe data and whose next attempt is due."""
    return (
        db.query(PipelineRun)
        .filter(
            PipelineRun.status == "completed",
            PipelineRun.kappa_upload_status == "pending",
            or_(
                PipelineRun.kappa_upload_next_attempt.is_(None),
                PipelineRun.kappa_upload_next_attempt <= _naive_utc(now),
            ),
        )
        .order_by(PipelineRun.completed_at.asc())
        .limit(limit)
        .all()
    )
```

Make sure the module imports what these need — add to the existing imports at the top of `backend/database.py`:

```python
import json
from datetime import timezone
from typing import List
from sqlalchemy import or_
```

(`datetime`, `Optional` and `Session` are already imported; do not duplicate them.)

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest backend/test_kappa_delivery_db.py -q`
Expected: 3 passed.

- [ ] **Step 7: Test the migration itself**

The accessors above never exercise `_migrate_add_kappa_delivery` — by the time
any test runs, `conftest.py` has already called `init_db()` on the temp
database, so the columns exist and the backfill has already happened on an
empty table. The backfill window needs its own database.

Create `backend/test_migrate_kappa_delivery.py`:

```python
"""The delivery migration: idempotent, and bounded in what it revives."""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlalchemy

import database as db_mod


def _legacy_db(tmp_path: Path) -> Path:
    """A pipeline_runs table as it looked BEFORE this feature — no delivery
    columns — with four runs that exercise each arm of the backfill's WHERE."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE pipeline_runs ("
        " run_id VARCHAR PRIMARY KEY, input_path VARCHAR, output_path VARCHAR,"
        " status VARCHAR, completed_at DATETIME, kappa_dataset_id INTEGER)"
    )
    recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(" ")
    ancient = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(" ")
    conn.executemany(
        "INSERT INTO pipeline_runs VALUES (?, '/in', '/out', ?, ?, ?)",
        [
            ("recent_with_dataset", "completed", recent, 350),
            ("recent_no_dataset", "completed", recent, None),
            ("ancient_with_dataset", "completed", ancient, 350),
            ("failed_with_dataset", "failed", recent, 350),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _statuses(path: Path) -> dict:
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT run_id, kappa_upload_status FROM pipeline_runs"
    ).fetchall()
    conn.close()
    return dict(rows)


def test_migration_adds_columns_and_backfills_only_recent_dataset_runs(
    tmp_path, monkeypatch
):
    path = _legacy_db(tmp_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(db_mod, "engine", engine)

    db_mod._migrate_add_kappa_delivery()

    with engine.connect() as conn:
        cols = [r[1] for r in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]
    for name in ("kappa_upload_status", "kappa_upload_next_attempt",
                 "kappa_upload_detail", "kappa_user_id"):
        assert name in cols

    statuses = _statuses(path)
    assert statuses["recent_with_dataset"] == "pending"
    assert statuses["recent_no_dataset"] is None      # no dataset to upload to
    assert statuses["ancient_with_dataset"] is None   # outside the 30-day window
    assert statuses["failed_with_dataset"] is None    # never completed


def test_migration_is_idempotent_and_does_not_re_backfill(tmp_path, monkeypatch):
    path = _legacy_db(tmp_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(db_mod, "engine", engine)

    db_mod._migrate_add_kappa_delivery()

    # Operator resolves the run; a second run of the migration must not
    # resurrect it, or every restart would re-queue finished work.
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE pipeline_runs SET kappa_upload_status = 'done' "
        " WHERE run_id = 'recent_with_dataset'"
    )
    conn.commit()
    conn.close()

    db_mod._migrate_add_kappa_delivery()

    assert _statuses(path)["recent_with_dataset"] == "done"
```

Run: `python -m pytest backend/test_migrate_kappa_delivery.py -q`
Expected: 2 passed.

Note why the second test holds even though the backfill `UPDATE` only runs
when a column was added: on the second call every column already exists, so
`added` is False and the `UPDATE` is skipped entirely. The test pins that
behaviour rather than assuming it.

- [ ] **Step 8: Run the whole backend suite**

Run: `python -m pytest backend/ -q`
Expected: no new failures versus the baseline you recorded in "Before You Start".

- [ ] **Step 9: Commit**

```bash
git add backend/database.py backend/test_kappa_delivery_db.py \
        backend/test_migrate_kappa_delivery.py
git commit -m "feat(db): delivery-state columns on pipeline_runs

A run that owes data to Kappa needs somewhere to say so. Four columns:
status, next attempt, a JSON detail blob and the owning Kappa user id
(session ids rotate on re-login, user ids do not).

The migration backfills 'pending' for completed runs with a dataset id
from the last 30 days — bounded, because an unbounded backfill would
revive every run ever completed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `classify()` — the retry policy as a pure function

**Files:**
- Create: `backend/kappa_delivery.py`
- Test: `backend/test_kappa_delivery.py` (create)

**Interfaces:**
- Consumes: nothing (pure; no imports from other project modules).
- Produces:
  - `classify(result: Optional[dict], exc: Optional[BaseException], state: dict, now: datetime) -> dict`
    returning `{"status": str, "next_attempt": Optional[datetime], "detail": dict}`
  - `BACKOFF_MINUTES: list[int]`, `NO_SESSION_RETRY_MINUTES: int`, `STUCK_AFTER_HOURS: int`
  - `NO_SESSION: dict` — the sentinel result the worker passes when no token exists

**Background — what actually crosses the boundary.** `backend/kappa_client.py` logs HTTP status codes and returns `None` (`:302`, `:136`), so **no status code ever reaches this function**. The complete input vocabulary is:

| Source | Value |
|---|---|
| `upload_results()` top level | `{"error": "Failed to resolve dataset_id"}` |
| `upload_results()` success | `{"dataset_id": int, "uploaded": int, "total": int, "sessions": [...]}` |
| `upload_results()` no sessions on disk | `{"uploaded": 0, "sessions": []}` (no `total` key) |
| per-session `error` | `"name_clash"`, `"duplicate"`, `"no files"`, `"upload failed"` |
| raised | any exception |
| worker sentinel | `{"error": "no_session"}` |

A session counts as **delivered** when `success` is true **or** its error is `"duplicate"` — a duplicate means the data is already in the dataset, which is what the operator cares about.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_delivery.py`:

```python
"""The Kappa delivery retry policy. Pure function, no network, no DB."""
from datetime import datetime, timedelta, timezone

import pytest

from kappa_delivery import NO_SESSION, classify

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _ok(session):
    return {"session": session, "success": True, "entity_id": "e1"}


def _fail(session, error):
    return {"session": session, "success": False, "error": error,
            "message": f"{session}: {error}"}


def test_all_sessions_delivered_is_done():
    result = {"dataset_id": 350, "uploaded": 2, "total": 2,
              "sessions": [_ok("sub-001_ses-001"), _ok("sub-002_ses-001")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "done"
    assert out["next_attempt"] is None
    assert out["detail"]["delivered"] == 2


def test_duplicate_counts_as_delivered():
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "duplicate")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "done"
    assert out["detail"]["delivered"] == 2


def test_unreachable_kappa_is_transient():
    out = classify({"error": "Failed to resolve dataset_id"}, None, {}, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "network"
    assert out["detail"]["attempts"] == 1
    assert out["next_attempt"] == NOW + timedelta(minutes=1)


def test_exception_is_transient():
    out = classify(None, RuntimeError("boom"), {}, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "network"
    assert "boom" in out["detail"]["last_error"]


def test_upload_failed_session_is_transient_with_counters():
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "upload failed")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["delivered"] == 1
    assert out["detail"]["total"] == 2


def test_name_clash_needs_attention_and_never_retries():
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "name_clash")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "needs_attention"
    assert out["next_attempt"] is None
    assert out["detail"]["reason"] == "name_clash"
    assert out["detail"]["blocked"][0]["session"] == "sub-002_ses-001"


def test_no_files_needs_attention():
    result = {"dataset_id": 350, "uploaded": 0, "total": 1,
              "sessions": [_fail("sub-001_ses-001", "no files")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "missing_files"


def test_nothing_on_disk_needs_attention():
    out = classify({"uploaded": 0, "sessions": []}, None, {}, NOW)
    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "missing_files"


def test_no_session_waits_without_counting_an_attempt():
    state = {"attempts": 3, "first_failure_at": "2026-09-23T11:00:00+00:00"}
    out = classify(NO_SESSION, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "no_session"
    assert out["detail"]["attempts"] == 3          # unchanged
    assert out["next_attempt"] == NOW + timedelta(minutes=5)


@pytest.mark.parametrize("prior_attempts,minutes", [
    (0, 1), (1, 2), (2, 5), (3, 15), (4, 30), (5, 60), (9, 60), (99, 60),
])
def test_backoff_schedule_and_cap(prior_attempts, minutes):
    state = {"attempts": prior_attempts,
             "first_failure_at": NOW.isoformat()}
    out = classify({"error": "Failed to resolve dataset_id"}, None, state, NOW)
    assert out["next_attempt"] == NOW + timedelta(minutes=minutes)


def test_stuck_after_24h_with_nothing_delivered():
    state = {"attempts": 40,
             "first_failure_at": (NOW - timedelta(hours=25)).isoformat()}
    out = classify({"error": "Failed to resolve dataset_id"}, None, state, NOW)
    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "stuck"
    assert out["next_attempt"] is None


def test_no_session_waits_do_not_make_a_run_stuck():
    state = {"attempts": 0,
             "first_failure_at": (NOW - timedelta(hours=48)).isoformat()}
    out = classify(NO_SESSION, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "no_session"


def test_progress_resets_the_stuck_window():
    state = {"attempts": 40,
             "first_failure_at": (NOW - timedelta(hours=25)).isoformat()}
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "upload failed")]}
    out = classify(result, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["first_failure_at"] is None
    assert out["detail"]["delivered"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_delivery.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kappa_delivery'`.

- [ ] **Step 3: Write the implementation**

Create `backend/kappa_delivery.py`:

```python
"""Retry policy for deferred Kappa uploads.

Pure: no network, no database, no clock. Everything it needs arrives as an
argument and everything it decides comes back as a return value, so the
whole policy is testable without a Kappa instance.

Design note — the vocabulary is narrow on purpose. backend/kappa_client.py
logs HTTP status codes and returns None, so a 403 and a dropped connection
are indistinguishable by the time they reach here. The `stuck` rule below is
what keeps a permanent failure from hiding behind an endless retry.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

BACKOFF_MINUTES = [1, 2, 5, 15, 30, 60]
NO_SESSION_RETRY_MINUTES = 5
STUCK_AFTER_HOURS = 24

# Sentinel the worker passes when no live Kappa session exists for the run's
# user. Routed through classify() like everything else so there is exactly
# one place where delivery state is decided.
NO_SESSION: Dict[str, Any] = {"error": "no_session"}

# Per-session errors that retrying cannot fix.
_BLOCKING = {
    "name_clash": "name_clash",
    "no files": "missing_files",
}


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _detail(state: Dict[str, Any], now: datetime, **over: Any) -> Dict[str, Any]:
    detail = {
        "total": state.get("total", 0),
        "delivered": state.get("delivered", 0),
        "blocked": [],
        "reason": None,
        "last_error": None,
        "attempts": state.get("attempts", 0),
        "first_failure_at": state.get("first_failure_at"),
        "last_attempt_at": now.isoformat(),
    }
    detail.update(over)
    return detail


def _transient(state, now, detail_over) -> Dict[str, Any]:
    """Schedule another attempt, or give up and ask for a human.

    A run only becomes `stuck` when it has delivered nothing at all: partial
    progress means the path works and the rest is worth retrying.
    """
    attempts = state.get("attempts", 0) + 1
    delivered = detail_over.get("delivered", 0)

    first_failure = None if delivered else (
        state.get("first_failure_at") or now.isoformat()
    )

    detail = _detail(state, now, attempts=attempts,
                     first_failure_at=first_failure, **detail_over)

    started = _parse(first_failure)
    if (not delivered and started
            and now - started >= timedelta(hours=STUCK_AFTER_HOURS)):
        detail["reason"] = "stuck"
        return {"status": "needs_attention", "next_attempt": None,
                "detail": detail}

    index = min(attempts - 1, len(BACKOFF_MINUTES) - 1)
    return {
        "status": "pending",
        "next_attempt": now + timedelta(minutes=BACKOFF_MINUTES[index]),
        "detail": detail,
    }


def classify(
    result: Optional[Dict[str, Any]],
    exc: Optional[BaseException],
    state: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    """Turn the outcome of one upload attempt into the run's new state.

    `result` is what KappaUploader.upload_results() returned (or NO_SESSION),
    `exc` is what it raised, `state` is the previous detail blob.
    """
    state = state or {}

    if exc is not None:
        return _transient(state, now,
                          {"reason": "network", "last_error": repr(exc)})

    if result is None:
        return _transient(state, now,
                          {"reason": "network",
                           "last_error": "uploader returned nothing"})

    error = result.get("error")

    if error == "no_session":
        # Waiting for a human to log in is not a failure: it must not burn
        # backoff attempts and must not count toward the stuck window.
        return {
            "status": "pending",
            "next_attempt": now + timedelta(minutes=NO_SESSION_RETRY_MINUTES),
            "detail": _detail(state, now, reason="no_session"),
        }

    if error:
        # Kappa unreachable, dataset gone, or no rights — indistinguishable.
        return _transient(state, now,
                          {"reason": "network", "last_error": str(error)})

    sessions = result.get("sessions") or []
    total = result.get("total", len(sessions))

    delivered = sum(
        1 for s in sessions
        if s.get("success") or s.get("error") == "duplicate"
    )
    blocked = [
        {"session": s.get("session"),
         "reason": _BLOCKING[s.get("error")],
         "message": s.get("message") or ""}
        for s in sessions
        if not s.get("success") and s.get("error") in _BLOCKING
    ]

    counters = {"total": total, "delivered": delivered}

    if not sessions:
        # A completed run that produced nothing to upload. Not transient —
        # retrying an empty directory forever tells the operator nothing.
        return {
            "status": "needs_attention",
            "next_attempt": None,
            "detail": _detail(state, now, reason="missing_files",
                              total=0, delivered=0),
        }

    if blocked:
        return {
            "status": "needs_attention",
            "next_attempt": None,
            "detail": _detail(state, now, reason=blocked[0]["reason"],
                              blocked=blocked, **counters),
        }

    if delivered >= total:
        return {
            "status": "done",
            "next_attempt": None,
            "detail": _detail(state, now, first_failure_at=None, **counters),
        }

    return _transient(state, now, {"reason": "network", **counters})
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest backend/test_kappa_delivery.py -q`
Expected: 20 passed (13 tests, 8 of them from the parametrised backoff case).

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_delivery.py backend/test_kappa_delivery.py
git commit -m "feat(kappa): retry policy as a pure classify() function

One function decides every delivery verdict, with no network, DB or clock
of its own, so the whole policy is testable without a Kappa instance.

The input vocabulary is narrow because kappa_client swallows HTTP status
codes: a 403 and a dropped connection arrive identical. The 24h 'stuck'
rule is what stops a permanent failure hiding behind endless retries.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Find the owner and a live token

**Files:**
- Modify: `backend/kappa_dataset_mapping.py` (add after `datasets_of_user` at `:46-57`)
- Modify: `backend/app.py:1372-1379` (delete `_dataset_owner`, import the moved one)
- Modify: `backend/kappa_auth.py` (add after `get_session` at `:72-97`)
- Test: `backend/test_kappa_session_lookup.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `kappa_dataset_mapping.owner_of_dataset(dataset_id: int) -> Optional[int]`
  - `kappa_auth.find_live_session_for_user(user_id: int, now: Optional[datetime] = None) -> Optional[dict]` — same dict shape `get_session()` returns

**Background:** `_dataset_owner()` sits in `app.py` today, but `app` imports `pipeline_monitor` and starts the worker from its lifespan, so a worker importing `app` would close an import cycle. The function belongs next to `_load_mapping()` anyway.

`kappa_sessions.token_expiry` is a **string** in Kappa's format, e.g. `2026-09-29T09:36:18.391277Z`. Python 3.12's `datetime.fromisoformat` parses the trailing `Z` directly. A row whose expiry is NULL or unparseable is treated as usable — the upload attempt is the real test, and refusing to try would strand runs over a formatting change.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_session_lookup.py`:

```python
"""Finding a live Kappa session for a user, days after the run started."""
from datetime import datetime, timedelta, timezone

from database import SessionLocal
from kappa_auth import find_live_session_for_user
from registry_models import KappaSession

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _add(db, session_id, user_id, expiry, created):
    db.add(KappaSession(
        session_id=session_id, kappa_token=f"token-{session_id}",
        user_id=user_id, user_type_id=1, token_expiry=expiry,
        created_at=created,
    ))
    db.commit()


def _cleanup(db, *session_ids):
    for sid in session_ids:
        row = db.query(KappaSession).filter(
            KappaSession.session_id == sid
        ).first()
        if row:
            db.delete(row)
    db.commit()


def test_picks_the_newest_unexpired_session():
    db = SessionLocal()
    ids = ["test_sess_old", "test_sess_new"]
    try:
        _add(db, "test_sess_old", 990026, "2026-09-29T09:00:00.000000Z",
             NOW - timedelta(days=2))
        _add(db, "test_sess_new", 990026, "2026-09-30T09:00:00.000000Z",
             NOW - timedelta(hours=1))
        found = find_live_session_for_user(990026, now=NOW)
        assert found["kappa_token"] == "token-test_sess_new"
    finally:
        _cleanup(db, *ids)
        db.close()


def test_skips_expired_sessions():
    db = SessionLocal()
    try:
        _add(db, "test_sess_dead", 990027, "2026-09-20T09:00:00.000000Z",
             NOW - timedelta(hours=1))
        assert find_live_session_for_user(990027, now=NOW) is None
    finally:
        _cleanup(db, "test_sess_dead")
        db.close()


def test_unparseable_expiry_is_treated_as_usable():
    db = SessionLocal()
    try:
        _add(db, "test_sess_weird", 990028, "not-a-date",
             NOW - timedelta(hours=1))
        found = find_live_session_for_user(990028, now=NOW)
        assert found["kappa_token"] == "token-test_sess_weird"
    finally:
        _cleanup(db, "test_sess_weird")
        db.close()


def test_unknown_user_has_no_session():
    assert find_live_session_for_user(990029, now=NOW) is None


def test_owner_of_dataset_reads_the_mapping(tmp_path, monkeypatch):
    import kappa_dataset_mapping as kdm

    mapping = tmp_path / "kappa_datasets.yaml"
    mapping.write_text(
        "datasets:\n"
        "  26:glioblastoma:current: 350\n"
        "  52:glioblastoma:current: 349\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kdm, "MAPPING_PATH", mapping)

    assert kdm.owner_of_dataset(350) == 26
    assert kdm.owner_of_dataset(349) == 52
    assert kdm.owner_of_dataset(999) is None
```

Before running, confirm the module-level constant holding the YAML path is really called `MAPPING_PATH`:

```bash
grep -n "MAPPING_PATH\|_load_mapping" -A 8 backend/kappa_dataset_mapping.py | head -25
```

If it has another name, use that name in the monkeypatch line above and in Step 3.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_session_lookup.py -q`
Expected: FAIL — `ImportError: cannot import name 'find_live_session_for_user'`.

- [ ] **Step 3: Add `owner_of_dataset` and move the caller**

In `backend/kappa_dataset_mapping.py`, after `datasets_of_user` (`:57`):

```python
def owner_of_dataset(dataset_id: int) -> Optional[int]:
    """Which Kappa user owns a dataset, per configs/kappa_datasets.yaml.
    None if the dataset id is not in the mapping at all.

    Lives here rather than in app.py because the delivery worker needs it and
    is started from app's lifespan — importing back into app would close a
    cycle.
    """
    for key, value in (_load_mapping().get("datasets") or {}).items():
        if int(value) == int(dataset_id):
            return int(str(key).split(":", 1)[0])
    return None
```

In `backend/app.py`, delete the `_dataset_owner` function at `:1372-1379` and replace its single call site inside `_resolve_longitudinal_records` (`:1404`):

```python
            from kappa_dataset_mapping import owner_of_dataset
            owner = owner_of_dataset(run.kappa_dataset_id)
```

Check nothing else referenced the old name:

```bash
grep -rn "_dataset_owner" backend/ || echo "no remaining references"
```

- [ ] **Step 4: Add `find_live_session_for_user`**

In `backend/kappa_auth.py`, after `get_session` (`:97`):

```python
def find_live_session_for_user(
    user_id: int, now: Optional[datetime] = None
) -> Optional[Dict[str, Any]]:
    """The newest session for this Kappa user whose token has not expired.

    Deferred uploads outlive the session that started the run: sessions expire
    after ~7 days and a re-login issues a NEW session_id, so a remembered
    session id is dead exactly when the retry needs it. The Kappa user id is
    stable, so that is what we search by.

    A row with a NULL or unparseable token_expiry is treated as usable — the
    upload attempt is the real test, and refusing to try would strand runs
    over a change in Kappa's date formatting.
    """
    if user_id is None:
        return None

    now = now or datetime.now(timezone.utc)

    db = SessionLocal()
    try:
        rows = db.query(KappaSession).filter(
            KappaSession.user_id == user_id
        ).order_by(KappaSession.created_at.desc()).all()

        for record in rows:
            expiry = _parse_expiry(record.token_expiry)
            if expiry is not None and expiry <= now:
                continue
            return {
                "kappa_token": record.kappa_token,
                "user_id": record.user_id,
                "user_type_id": record.user_type_id,
                "user_name": record.user_name,
                "first_name": record.first_name,
                "last_name": record.last_name,
                "token_expiry": record.token_expiry,
                "org_details": (
                    json.loads(record.org_details) if record.org_details else None
                ),
            }
        return None
    finally:
        db.close()


def _parse_expiry(value: Optional[str]) -> Optional[datetime]:
    """Kappa sends e.g. "2026-09-29T09:36:18.391277Z"; Python 3.12 parses the
    trailing Z directly. None means "cannot tell", not "expired"."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
```

Add whatever of these the module does not already import at the top of `backend/kappa_auth.py`:

```python
from datetime import datetime, timezone
from typing import Any, Dict, Optional
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_kappa_session_lookup.py -q`
Expected: 5 passed.

Run: `python -m pytest backend/ -q`
Expected: no new failures. `_resolve_longitudinal_records` has existing tests — they must still pass after the function move.

- [ ] **Step 6: Commit**

```bash
git add backend/kappa_dataset_mapping.py backend/kappa_auth.py backend/app.py \
        backend/test_kappa_session_lookup.py
git commit -m "feat(kappa): look up a live session by user, move owner_of_dataset

A deferred upload outlives the session that started the run, so it cannot
use a remembered session_id: re-login issues a new one. The Kappa user id
is stable, so find_live_session_for_user searches by that instead.

_dataset_owner moves from app.py to kappa_dataset_mapping.py — the worker
needs it and is started from app's lifespan, so importing back into app
would close a cycle.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The post-run upload records its outcome

**Files:**
- Modify: `backend/pipeline_monitor.py:65-112` (`_monitor_loop`) and `:168-215` (`_kappa_upload_safe`)
- Test: `backend/test_pipeline_monitor_delivery.py` (create)

**Interfaces:**
- Consumes: `kappa_delivery.classify`, `kappa_delivery.NO_SESSION`, `database.set_kappa_delivery`, `database.get_kappa_delivery`
- Produces: `PipelineMonitor._record_delivery(run_id: str, result: Optional[dict], exc: Optional[BaseException]) -> dict` — classifies, persists, returns the new state

**Background — the bug this kills.** `upload_results()` reports "Kappa unreachable" by **returning** `{"error": "Failed to resolve dataset_id"}`, not by raising. Today `_kappa_upload_safe` logs that at INFO as if it were a result. Error handling written only around exceptions is blind to code that returns its errors.

Second change: the uploader is currently built at the **top** of `_monitor_loop` (`:78`), holding the token as it was then. A three-hour run finishes with a three-hour-old token. Build it at completion instead.

- [ ] **Step 1: Write the failing test**

Create `backend/test_pipeline_monitor_delivery.py`:

```python
"""The post-run upload must record what happened — including the failures
that arrive as a return value rather than an exception."""
import pytest

import database as db_mod
import pipeline_monitor as pm
from database import SessionLocal, create_pipeline_run, get_kappa_delivery


class _Uploader:
    """Stands in for KappaUploader: returns or raises whatever it was given."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    async def upload_results(self):
        if self.exc:
            raise self.exc
        return self.result


def _make_run(db, run_id):
    run = create_pipeline_run(
        db, input_path="/in", output_path="/out",
        run_id=run_id, kappa_upload_status="pending", kappa_user_id=26,
    )
    run.status = "completed"
    db.commit()
    return run


def _cleanup(db, run_id):
    run = db_mod.get_pipeline_run(db, run_id)
    if run:
        db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_returned_error_is_recorded_as_pending():
    db = SessionLocal()
    run_id = "test_monitor_returned_error"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(result={"error": "Failed to resolve dataset_id"})
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        assert get_kappa_delivery(run)["reason"] == "network"
        assert run.kappa_upload_next_attempt is not None
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_exception_is_recorded_as_pending():
    db = SessionLocal()
    run_id = "test_monitor_exception"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(exc=RuntimeError("connection reset"))
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        assert "connection reset" in get_kappa_delivery(run)["last_error"]
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_full_success_is_recorded_as_done():
    db = SessionLocal()
    run_id = "test_monitor_success"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(result={
            "dataset_id": 350, "uploaded": 1, "total": 1,
            "sessions": [{"session": "sub-001_ses-001", "success": True,
                          "entity_id": "e1"}],
        })
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "done"
        assert run.kappa_upload_next_attempt is None
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_name_clash_is_recorded_as_needs_attention():
    db = SessionLocal()
    run_id = "test_monitor_clash"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(result={
            "dataset_id": 350, "uploaded": 0, "total": 1,
            "sessions": [{"session": "sub-003_ses-001", "success": False,
                          "error": "name_clash", "message": "уже есть"}],
        })
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "needs_attention"
        assert get_kappa_delivery(run)["blocked"][0]["reason"] == "name_clash"
    finally:
        _cleanup(db, run_id)
        db.close()


def test_missing_uploader_is_recorded_as_waiting_for_a_session():
    """No usable Kappa session when the run finished means no attempt was
    ever made. The run must stay pending, and say WHY, so the worker picks it
    up once someone logs in — and so the UI does not call it a network
    problem."""
    from kappa_delivery import NO_SESSION

    db = SessionLocal()
    run_id = "test_monitor_no_uploader"
    try:
        _make_run(db, run_id)
        pm.pipeline_monitor._record_delivery(run_id, NO_SESSION, None)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        detail = get_kappa_delivery(run)
        assert detail["reason"] == "no_session"
        assert detail["attempts"] == 0
    finally:
        _cleanup(db, run_id)
        db.close()


def test_run_without_upload_intent_is_left_alone():
    """A CLI run (NULL status) must never acquire delivery state, or the
    worker would start chasing runs nobody meant to upload."""
    db = SessionLocal()
    run_id = "test_monitor_no_intent"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
        )
        run.status = "completed"
        db.commit()

        verdict = pm.pipeline_monitor._record_delivery(
            run_id, {"error": "Failed to resolve dataset_id"}, None
        )
        assert verdict is None

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status is None
    finally:
        _cleanup(db, run_id)
        db.close()
```

Check that `pytest-asyncio` is available and how it is configured:

```bash
grep -rn "asyncio_mode\|pytest-asyncio" pytest.ini setup.cfg pyproject.toml requirements*.txt 2>/dev/null
python -c "import pytest_asyncio; print(pytest_asyncio.__version__)"
```

If `asyncio_mode` is not set to `auto` anywhere, the `@pytest.mark.asyncio` markers above are correct as written. If `pytest_asyncio` is not installed, install it (`pip install pytest-asyncio`) and add it to `requirements.txt`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_pipeline_monitor_delivery.py -q`
Expected: FAIL — the status stays `pending` with no detail written, so `test_full_success_is_recorded_as_done` and the reason assertions fail.

- [ ] **Step 3: Add `_record_delivery` and wire it in**

In `backend/pipeline_monitor.py`, add this method to `PipelineMonitor` just above `_kappa_upload_safe`:

```python
    def _record_delivery(self, run_id, result, exc):
        """Classify one upload attempt and persist the verdict.

        Kept separate from the upload itself so the decision is testable and
        so all three callers (post-run, background worker, manual retry) reach
        the same policy.
        """
        from datetime import datetime, timezone

        from database import get_kappa_delivery, set_kappa_delivery
        from kappa_delivery import classify

        db = SessionLocal()
        try:
            run = get_pipeline_run(db, run_id)
            if run is None or run.kappa_upload_status is None:
                # NULL status means this run never intended to upload (CLI,
                # no Kappa session at start) — nothing to record.
                return None
            state = get_kappa_delivery(run)
            verdict = classify(result, exc, state, datetime.now(timezone.utc))
            set_kappa_delivery(
                db, run_id, verdict["status"],
                verdict["next_attempt"], verdict["detail"],
            )
            logger.info(
                "Kappa delivery for %s: %s (%d/%d, reason=%s)",
                run_id, verdict["status"],
                verdict["detail"].get("delivered", 0),
                verdict["detail"].get("total", 0),
                verdict["detail"].get("reason"),
            )
            return verdict
        finally:
            db.close()
```

Now change `_kappa_upload_safe` (`:168`). Keep the existing WebSocket notification for entities exactly as it is; wrap the call so both the returned value and the exception reach `_record_delivery`, and broadcast a deferred message when the verdict is not `done`:

```python
    async def _kappa_upload_safe(self, uploader, run_id: str = None):
        """Обёртка для безопасного вызова upload_results"""
        results = None
        failure = None
        try:
            results = await uploader.upload_results()
            logger.info("Kappa upload results: %s", results)
        except Exception as e:
            failure = e
            logger.error("Kappa upload error: %s", e)

        verdict = self._record_delivery(run_id, results, failure) if run_id else None

        # Уведомляем фронт о завершении загрузки в Каппу
        if run_id and results and not results.get("error"):
            ...  # existing entity-notification body, unchanged
        
        if run_id and verdict and verdict["status"] != "done":
            await ws_manager.broadcast(run_id, {
                "type": "kappa_upload_deferred",
                "run_id": run_id,
                "status": verdict["status"],
                "detail": verdict["detail"],
            })
```

Note the added `and results and not results.get("error")` guard on the existing block: it currently runs `results.get("sessions", [])` on a dict that may be `None` after an exception.

Finally, rebuild the uploader at completion time instead of reusing the one from the top of the loop. In `_monitor_loop` (`:75-99`), replace the eager construction with a deferred one:

```python
        # Kappa uploader is built at completion, not here: a long run would
        # otherwise finish holding a token snapshotted hours earlier.
        try:
            while True:
                run = get_pipeline_run(db, run_id)

                if not run:
                    logger.error(f"Run {run_id} не найден в БД")
                    break

                if run.status in ["completed", "failed"]:
                    logger.info(f"Pipeline {run_id} завершён со статусом: {run.status}")
                    await self._send_update(run_id, output_path, db)

                    if kappa_session_id and lesion_type and run.status == "completed":
                        kappa_uploader = self._create_kappa_uploader(
                            run_id, output_path, kappa_session_id, lesion_type
                        )
                        if kappa_uploader:
                            logger.info("Starting Kappa upload for completed run %s", run_id)
                            asyncio.create_task(
                                self._kappa_upload_safe(kappa_uploader, run_id)
                            )
                        else:
                            # No usable session now — the background worker
                            # will retry once one exists.
                            self._record_delivery(run_id, NO_SESSION, None)
                    break
```

Delete the now-dead eager block at `:76-80` and add the import at the top of the module:

```python
from kappa_delivery import NO_SESSION
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_pipeline_monitor_delivery.py -q`
Expected: 6 passed.

Run: `python -m pytest backend/ -q`
Expected: no new failures. `backend/test_create_kappa_uploader_dataset.py` exercises `_create_kappa_uploader` directly and is unaffected by moving its call site.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_monitor.py backend/test_pipeline_monitor_delivery.py
git commit -m "fix(kappa): a failed upload after a run is now recorded, not just logged

upload_results() reports an unreachable Kappa by RETURNING
{'error': ...}, so the try/except wrapper logged it at INFO as if it were
a result. Three of the four ways delivery fails were invisible.

Every outcome now goes through classify() and is persisted, and the
uploader is built when the run completes rather than when monitoring
starts — a three-hour run was finishing with a three-hour-old token.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Record upload intent when the run starts

**Files:**
- Modify: `backend/app.py:462-476` (start endpoint), `backend/app.py:689-700` (requeue)
- Test: `backend/test_app_upload_intent.py` (create)

**Interfaces:**
- Consumes: `create_pipeline_run(..., kappa_upload_status, kappa_user_id)` from Task 1
- Produces: runs created with `kappa_upload_status='pending'` and `kappa_user_id` set whenever the request carried a Kappa session

**Background:** intent is recorded at birth so nothing downstream has to guess whether a run was meant to reach Kappa. A run started without a session gets NULL and the worker never looks at it — which is how CLI and `orchestrator.py` runs stay out of the queue by construction.

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_upload_intent.py`:

```python
"""A run started with a Kappa session is born owing delivery."""
import database as db_mod
from database import SessionLocal, create_pipeline_run


def _cleanup(db, run_id):
    run = db_mod.get_pipeline_run(db, run_id)
    if run:
        db.delete(run)
    db.commit()


def test_run_with_a_session_is_born_pending():
    db = SessionLocal()
    run_id = "test_intent_with_session"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_dataset_id=350, kappa_upload_status="pending",
            kappa_user_id=26,
        )
        assert run.kappa_upload_status == "pending"
        assert run.kappa_user_id == 26
    finally:
        _cleanup(db, run_id)
        db.close()


def test_run_without_a_session_has_no_upload_status():
    db = SessionLocal()
    run_id = "test_intent_no_session"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
        )
        assert run.kappa_upload_status is None
        assert run.kappa_user_id is None
    finally:
        _cleanup(db, run_id)
        db.close()
```

- [ ] **Step 2: Run the test to verify it passes already**

Run: `python -m pytest backend/test_app_upload_intent.py -q`
Expected: 2 passed — Task 1 supplied the parameters. This test pins the contract the endpoint changes below must satisfy.

- [ ] **Step 3: Set the intent in the start endpoint**

In `backend/app.py`, just before `create_pipeline_run` at `:469`:

```python
    # Upload intent, recorded at birth. A run started with a Kappa session
    # owes delivery from the moment it exists, so a failure later has
    # somewhere to be recorded; a run started without one never enters the
    # delivery queue at all (CLI and orchestrator runs stay out by
    # construction).
    upload_status = None
    upload_user_id = None
    if request.kappa_session_id:
        from kappa_auth import get_session as _get_kappa_session
        _session = _get_kappa_session(request.kappa_session_id)
        if _session:
            upload_status = "pending"
            upload_user_id = _session.get("user_id")

    run = create_pipeline_run(
        db,
        run_id=run_id,
        input_path=request.input_path,
        output_path=output_path,
        lesion_type=lesion_type,
        kappa_dataset_id=kappa_dataset_id,
        kappa_upload_status=upload_status,
        kappa_user_id=upload_user_id,
    )
```

- [ ] **Step 4: Do the same for requeue**

Find the `create_pipeline_run` call in the requeue endpoint at `backend/app.py:695` and give it the same treatment, reading `body.kappa_session_id` instead of `request.kappa_session_id`. Read the surrounding 40 lines first — the requeue path already resolves a scope and a dataset id, and the new arguments go beside `kappa_dataset_id` exactly as above.

- [ ] **Step 5: Verify the app still imports and the suite is green**

Run: `python -c "import sys; sys.path.insert(0, 'backend'); import app" `
Expected: no output, no traceback.

Run: `python -m pytest backend/ -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/test_app_upload_intent.py
git commit -m "feat(backend): record Kappa upload intent when a run starts

A run started with a Kappa session is created already owing delivery, so
a failure later has somewhere to be recorded. A run started without one
keeps a NULL status and never enters the queue — which is how CLI and
orchestrator runs stay out of it by construction rather than by a filter
somebody has to remember.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The background retry worker

**Files:**
- Create: `backend/kappa_delivery_worker.py`
- Modify: `backend/app.py:138-172` (lifespan)
- Test: `backend/test_kappa_delivery_worker.py` (create)

**Interfaces:**
- Consumes: `runs_due_for_delivery`, `set_kappa_delivery`, `get_kappa_delivery` (Task 1); `classify`, `NO_SESSION` (Task 2); `find_live_session_for_user`, `owner_of_dataset` (Task 3); `PipelineMonitor._create_kappa_uploader` (existing, `pipeline_monitor.py:114`)
- Produces:
  - `async deliver_one(run_id: str) -> Optional[dict]` — one attempt for one run, returns the verdict
  - `async tick(now: Optional[datetime] = None) -> int` — one pass, returns how many runs it touched
  - `async delivery_loop() -> None` — the forever loop
  - `start_delivery_worker() -> asyncio.Task`
  - `TICK_SECONDS: int = 60`, `BATCH_SIZE: int = 5`

**Background:** `_create_kappa_uploader` takes a **session id**, not a token, and looks the session up itself. The worker has a session dict, not an id, so it constructs `KappaUploader` directly — the same way `_create_kappa_uploader` does. Read `backend/pipeline_monitor.py:114-165` before writing this and mirror its argument list exactly, including `preprocessing_config_path`.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_delivery_worker.py`:

```python
"""The background worker: who it picks up, and what it does with no token."""
from datetime import datetime, timedelta, timezone

import pytest

import database as db_mod
import kappa_delivery_worker as worker
from database import (
    SessionLocal, create_pipeline_run, get_kappa_delivery, set_kappa_delivery,
)

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _make_run(db, run_id, **kwargs):
    run = create_pipeline_run(
        db, input_path="/in", output_path="/out", run_id=run_id,
        kappa_upload_status=kwargs.pop("upload_status", "pending"),
        kappa_user_id=kwargs.pop("user_id", 26),
        kappa_dataset_id=kwargs.pop("dataset_id", 350),
    )
    run.status = "completed"
    run.completed_at = NOW - timedelta(hours=1)
    db.commit()
    return run


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_no_live_session_defers_without_counting_an_attempt(monkeypatch):
    db = SessionLocal()
    run_id = "test_worker_no_session"
    try:
        _make_run(db, run_id)
        set_kappa_delivery(db, run_id, "pending", None, {"attempts": 3})
        monkeypatch.setattr(worker, "find_live_session_for_user",
                            lambda user_id, now=None: None)

        await worker.deliver_one(run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        detail = get_kappa_delivery(run)
        assert detail["reason"] == "no_session"
        assert detail["attempts"] == 3
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_successful_attempt_marks_the_run_done(monkeypatch):
    db = SessionLocal()
    run_id = "test_worker_success"
    try:
        _make_run(db, run_id)
        monkeypatch.setattr(
            worker, "find_live_session_for_user",
            lambda user_id, now=None: {
                "kappa_token": "fresh-token", "user_id": 26, "user_type_id": 1,
            },
        )

        class _Uploader:
            async def upload_results(self):
                return {"dataset_id": 350, "uploaded": 1, "total": 1,
                        "sessions": [{"session": "sub-001_ses-001",
                                      "success": True, "entity_id": "e1"}]}

        monkeypatch.setattr(worker, "build_uploader",
                            lambda run, session: _Uploader())

        await worker.deliver_one(run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "done"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_uploader_is_built_with_the_current_token(monkeypatch):
    """Regression guard: the worker must not reuse a token snapshotted when
    the run started — that token is what expired in the first place."""
    db = SessionLocal()
    run_id = "test_worker_fresh_token"
    seen = {}
    try:
        _make_run(db, run_id)
        monkeypatch.setattr(
            worker, "find_live_session_for_user",
            lambda user_id, now=None: {
                "kappa_token": "fresh-token", "user_id": 26, "user_type_id": 1,
            },
        )

        class _Uploader:
            async def upload_results(self):
                return {"error": "Failed to resolve dataset_id"}

        def _build(run, session):
            seen["token"] = session["kappa_token"]
            return _Uploader()

        monkeypatch.setattr(worker, "build_uploader", _build)

        await worker.deliver_one(run_id)
        assert seen["token"] == "fresh-token"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_tick_respects_the_batch_size(monkeypatch):
    db = SessionLocal()
    ids = [f"test_worker_batch_{i}" for i in range(7)]
    attempted = []
    try:
        for rid in ids:
            _make_run(db, rid)

        async def _fake_deliver(run_id):
            attempted.append(run_id)
            return None

        monkeypatch.setattr(worker, "deliver_one", _fake_deliver)
        touched = await worker.tick(now=NOW)

        assert touched == worker.BATCH_SIZE
        assert len(attempted) == worker.BATCH_SIZE
    finally:
        _cleanup(db, *ids)
        db.close()


@pytest.mark.asyncio
async def test_tick_skips_runs_that_are_not_due(monkeypatch):
    db = SessionLocal()
    run_id = "test_worker_not_due"
    attempted = []
    try:
        _make_run(db, run_id)
        set_kappa_delivery(db, run_id, "pending", NOW + timedelta(hours=1), {})

        async def _fake_deliver(rid):
            attempted.append(rid)
            return None

        monkeypatch.setattr(worker, "deliver_one", _fake_deliver)
        await worker.tick(now=NOW)

        assert run_id not in attempted
    finally:
        _cleanup(db, run_id)
        db.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_delivery_worker.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kappa_delivery_worker'`.

- [ ] **Step 3: Write the worker**

First read the uploader construction you are mirroring:

```bash
sed -n 114,166p backend/pipeline_monitor.py
```

Create `backend/kappa_delivery_worker.py`:

```python
"""Background delivery of results that did not reach Kappa the first time.

An asyncio task in the web process, started from app's lifespan beside the
orphaned-run reconciliation. No new process, container or scheduler —
pipeline_monitor already works this way.

Every attempt re-runs the existing KappaUploader, which dedups against the
dataset by study_hash. That is what makes retrying safe: an attempt is not
"send it again", it is "reconcile with what is actually there".
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from database import (
    SessionLocal,
    get_kappa_delivery,
    get_pipeline_run,
    runs_due_for_delivery,
    set_kappa_delivery,
)
from kappa_auth import find_live_session_for_user
from kappa_dataset_mapping import owner_of_dataset
from kappa_delivery import NO_SESSION, classify

logger = logging.getLogger(__name__)

TICK_SECONDS = 60
BATCH_SIZE = 5


def build_uploader(run, session: Dict[str, Any]):
    """A KappaUploader for this run, using the token we have right now.

    Mirrors PipelineMonitor._create_kappa_uploader, but takes a session dict
    rather than a session id: the worker has already resolved the account by
    user id, because the session id the run started with is long gone.
    """
    from kappa_uploader import KappaUploader
    from pipeline_manager import PipelineManager

    config_path = PipelineManager().get_config_file(run.output_path)
    if not config_path:
        return None

    return KappaUploader(
        run_id=run.run_id,
        output_path=run.output_path,
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        lesion_type=getattr(run, "lesion_type", None) or "glioblastoma",
        preprocessing_config_path=config_path,
        dataset_id=run.kappa_dataset_id,
    )


def _owner_of(run) -> Optional[int]:
    """The Kappa account this run belongs to. Runs created before this feature
    have no kappa_user_id, but the migration only backfilled rows that DO have
    a dataset id — so the mapping can answer for them."""
    if run.kappa_user_id is not None:
        return run.kappa_user_id
    if run.kappa_dataset_id is not None:
        return owner_of_dataset(run.kappa_dataset_id)
    return None


async def deliver_one(run_id: str) -> Optional[Dict[str, Any]]:
    """One delivery attempt for one run. Never raises."""
    db = SessionLocal()
    try:
        run = get_pipeline_run(db, run_id)
        if run is None or run.kappa_upload_status != "pending":
            return None

        state = get_kappa_delivery(run)
        now = datetime.now(timezone.utc)

        owner = _owner_of(run)
        session = find_live_session_for_user(owner) if owner else None

        if session is None:
            verdict = classify(NO_SESSION, None, state, now)
        else:
            result, exc = None, None
            try:
                uploader = build_uploader(run, session)
                if uploader is None:
                    result = {"error": "preprocessing config not found"}
                else:
                    result = await uploader.upload_results()
            except Exception as e:          # noqa: BLE001 - recorded, not raised
                exc = e
                logger.error("Deferred Kappa upload failed for %s: %s", run_id, e)
            verdict = classify(result, exc, state, now)

        set_kappa_delivery(
            db, run_id, verdict["status"],
            verdict["next_attempt"], verdict["detail"],
        )
        logger.info(
            "Deferred delivery %s: %s (%s/%s, reason=%s)",
            run_id, verdict["status"],
            verdict["detail"].get("delivered"), verdict["detail"].get("total"),
            verdict["detail"].get("reason"),
        )
        return verdict
    finally:
        db.close()


async def tick(now: Optional[datetime] = None) -> int:
    """One pass over the due runs. Returns how many were attempted."""
    now = now or datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        due = runs_due_for_delivery(db, now, limit=BATCH_SIZE)
        run_ids = [r.run_id for r in due]
    finally:
        db.close()

    for run_id in run_ids:
        await deliver_one(run_id)
    return len(run_ids)


async def delivery_loop() -> None:
    """Forever. One tick's failure must never kill the loop — a dead worker
    would silently stop every deferred upload in the system."""
    logger.info("Kappa delivery worker started (tick=%ss)", TICK_SECONDS)
    while True:
        try:
            await tick()
        except asyncio.CancelledError:
            logger.info("Kappa delivery worker cancelled")
            raise
        except Exception as e:              # noqa: BLE001
            logger.error("Kappa delivery tick failed: %s", e)
        await asyncio.sleep(TICK_SECONDS)


def start_delivery_worker() -> asyncio.Task:
    return asyncio.create_task(delivery_loop())
```

Confirm `PipelineManager.get_config_file` is the right method name — `_create_kappa_uploader` uses it, so copy exactly what you saw in the `sed` output above.

- [ ] **Step 4: Start it from the lifespan**

In `backend/app.py`, in `lifespan` just before `yield` (`:170`):

```python
    from kappa_delivery_worker import start_delivery_worker
    _delivery_task = start_delivery_worker()

    yield

    # Shutdown
    _delivery_task.cancel()
    logger.info("Завершение работы приложения")
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_kappa_delivery_worker.py -q`
Expected: 5 passed.

Run: `python -m pytest backend/ -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add backend/kappa_delivery_worker.py backend/app.py \
        backend/test_kappa_delivery_worker.py
git commit -m "feat(kappa): background worker that finishes deferred uploads

An asyncio task in the existing lifespan, beside the orphaned-run
reconciliation — no new process or scheduler. Every 60s it takes up to 5
due runs, finds a live token by Kappa user id, and re-runs the uploader.

Retrying is safe because the uploader dedups against the dataset: an
attempt is not 'send it again', it is 'reconcile with what is there'.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The manual retry endpoint persists its outcome

**Files:**
- Modify: `backend/app.py:2624-2660` (`retry_kappa_upload`)
- Test: `backend/test_app_retry_upload.py` (create)

**Interfaces:**
- Consumes: `pipeline_monitor._record_delivery` (Task 4)
- Produces: the endpoint's JSON response gains a `delivery` key holding the new state

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_retry_upload.py`:

```python
"""The manual retry button must leave the same trace as every other path."""
import sys

import pytest

sys.path.insert(0, "backend")

import database as db_mod  # noqa: E402
from database import SessionLocal, create_pipeline_run  # noqa: E402


def _cleanup(db, run_id):
    run = db_mod.get_pipeline_run(db, run_id)
    if run:
        db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_manual_retry_records_the_verdict(monkeypatch):
    import app
    import pipeline_monitor as pm

    db = SessionLocal()
    run_id = "test_retry_records"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="pending", kappa_user_id=26,
            kappa_dataset_id=350,
        )
        run.status = "completed"
        db.commit()

        class _Uploader:
            async def upload_results(self):
                return {"dataset_id": 350, "uploaded": 1, "total": 1,
                        "sessions": [{"session": "sub-001_ses-001",
                                      "success": True, "entity_id": "e1"}]}

        monkeypatch.setattr(
            pm.pipeline_monitor, "_create_kappa_uploader",
            lambda *a, **k: _Uploader(),
        )

        response = await app.retry_kappa_upload(run_id, session_id="sid")
        assert response["delivery"]["status"] == "done"

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "done"
    finally:
        _cleanup(db, run_id)
        db.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_retry_upload.py -q`
Expected: FAIL — `KeyError: 'delivery'`.

- [ ] **Step 3: Persist and return the verdict**

In `backend/app.py`, replace the last two lines of `retry_kappa_upload` (`:2657-2659`):

```python
    results = await uploader.upload_results()
    logger.info("Kappa retry-upload results for %s: %s", run_id, results)

    # Same policy as the post-run upload and the background worker — three
    # callers, one decision about what the run's delivery state now is.
    verdict = pipeline_monitor._record_delivery(run_id, results, None)
    return {**results, "delivery": verdict}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_app_retry_upload.py -q`
Expected: 1 passed.

Run: `python -m pytest backend/ -q`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/test_app_retry_upload.py
git commit -m "feat(backend): manual Kappa retry records its verdict too

The endpoint has existed since the 504 incident but left no trace: a
successful manual retry still showed the run as owing delivery. It now
goes through the same classify() path as the other two callers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Expose delivery state to the frontend

**Files:**
- Modify: `backend/models.py:354-368` (`PipelineRunHistoryItem`)
- Modify: `backend/app.py:803-846` (history endpoint), and add the summary endpoint beside the other Kappa routes (after `:2660`)
- Test: `backend/test_app_delivery_api.py` (create)

**Interfaces:**
- Consumes: `get_kappa_delivery` (Task 1)
- Produces:
  - `models.KappaDeliveryStatus` with `status: str`, `delivered: int`, `total: int`, `blocked: List[KappaBlockedSession]`, `next_attempt_at: Optional[datetime]`, `reason: Optional[str]`
  - `models.KappaBlockedSession` with `session: Optional[str]`, `reason: str`, `message: str`
  - `PipelineRunHistoryItem.kappa_upload: Optional[KappaDeliveryStatus]`
  - `GET /api/kappa/delivery/summary` → `{"pending": int, "needs_attention": int}`

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_delivery_api.py`:

```python
"""Delivery state reaching the frontend."""
import sys

import pytest

sys.path.insert(0, "backend")

import database as db_mod  # noqa: E402
from database import SessionLocal, create_pipeline_run, set_kappa_delivery  # noqa: E402


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_history_carries_delivery_state():
    import app

    db = SessionLocal()
    run_id = "test_api_history_delivery"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="needs_attention",
        )
        run.status = "completed"
        db.commit()
        set_kappa_delivery(db, run_id, "needs_attention", None, {
            "total": 2, "delivered": 1, "reason": "name_clash",
            "blocked": [{"session": "sub-002_ses-001",
                         "reason": "name_clash", "message": "уже есть"}],
        })

        response = await app.get_history(limit=100, offset=0, db=db)
        item = next(r for r in response.runs if r.run_id == run_id)

        assert item.kappa_upload.status == "needs_attention"
        assert item.kappa_upload.delivered == 1
        assert item.kappa_upload.total == 2
        assert item.kappa_upload.blocked[0].reason == "name_clash"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_history_item_without_delivery_has_none():
    import app

    db = SessionLocal()
    run_id = "test_api_history_no_delivery"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
        )
        run.status = "completed"
        db.commit()

        response = await app.get_history(limit=100, offset=0, db=db)
        item = next(r for r in response.runs if r.run_id == run_id)
        assert item.kappa_upload is None
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_summary_counts_by_status():
    import app

    db = SessionLocal()
    ids = ["test_api_sum_p1", "test_api_sum_p2", "test_api_sum_na"]
    try:
        for rid, status in [
            ("test_api_sum_p1", "pending"),
            ("test_api_sum_p2", "pending"),
            ("test_api_sum_na", "needs_attention"),
        ]:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out", run_id=rid,
                kappa_upload_status=status,
            )
            run.status = "completed"
            db.commit()

        summary = await app.kappa_delivery_summary(db=db)
        assert summary["pending"] >= 2
        assert summary["needs_attention"] >= 1
    finally:
        _cleanup(db, *ids)
        db.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_delivery_api.py -q`
Expected: FAIL — `AttributeError: 'PipelineRunHistoryItem' object has no attribute 'kappa_upload'`.

- [ ] **Step 3: Add the response models**

In `backend/models.py`, immediately before `PipelineRunHistoryItem` (`:354`):

```python
class KappaBlockedSession(BaseModel):
    """Одна сессия, которую повтор не починит."""
    session: Optional[str] = Field(None, description="BIDS-ключ сессии")
    reason: str = Field(..., description="name_clash | missing_files")
    message: str = Field("", description="Объяснение для оператора")


class KappaDeliveryStatus(BaseModel):
    """Состояние выгрузки прогона в Kappa."""
    status: str = Field(..., description="pending | done | needs_attention")
    delivered: int = Field(0, description="Сколько сессий уже в датасете")
    total: int = Field(0, description="Сколько сессий всего")
    blocked: List[KappaBlockedSession] = Field(default_factory=list)
    next_attempt_at: Optional[datetime] = Field(
        None, description="Когда будет следующая автоматическая попытка"
    )
    reason: Optional[str] = Field(
        None, description="network | no_session | name_clash | missing_files | stuck"
    )
```

Add the field to `PipelineRunHistoryItem`:

```python
    kappa_upload: Optional[KappaDeliveryStatus] = Field(
        None, description="Состояние выгрузки в Kappa; None — выгрузка не предполагалась"
    )
```

- [ ] **Step 4: Fill it in the history endpoint and add the summary route**

In `backend/app.py`, add a helper just above `get_history` (`:803`):

```python
def _delivery_status(run) -> Optional[KappaDeliveryStatus]:
    """Delivery state for the history list. None means this run never
    intended to upload (CLI, no Kappa session), which the UI shows as a dash
    rather than as a problem."""
    if not getattr(run, "kappa_upload_status", None):
        return None
    detail = get_kappa_delivery(run)
    return KappaDeliveryStatus(
        status=run.kappa_upload_status,
        delivered=detail.get("delivered", 0),
        total=detail.get("total", 0),
        blocked=[
            KappaBlockedSession(
                session=b.get("session"),
                reason=b.get("reason", "name_clash"),
                message=b.get("message", ""),
            )
            for b in (detail.get("blocked") or [])
        ],
        next_attempt_at=run.kappa_upload_next_attempt,
        reason=detail.get("reason"),
    )
```

Add `kappa_upload=_delivery_status(run),` to the `PipelineRunHistoryItem(...)` construction at `:824-841`, and extend the imports at the top of `app.py`: `get_kappa_delivery` from `database`, `KappaDeliveryStatus` and `KappaBlockedSession` from `models`.

Then add the summary endpoint after `retry_kappa_upload`:

```python
@app.get("/api/kappa/delivery/summary")
async def kappa_delivery_summary(db: Session = Depends(get_db)):
    """Сколько прогонов ещё не доехали до Kappa. Для предупреждения в истории."""
    from database import PipelineRun

    rows = (
        db.query(PipelineRun.kappa_upload_status, func.count(PipelineRun.run_id))
        .filter(PipelineRun.kappa_upload_status.in_(["pending", "needs_attention"]))
        .group_by(PipelineRun.kappa_upload_status)
        .all()
    )
    counts = {status: count for status, count in rows}
    return {
        "pending": counts.get("pending", 0),
        "needs_attention": counts.get("needs_attention", 0),
    }
```

Add `from sqlalchemy import func` to `app.py`'s imports if it is not already there.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_app_delivery_api.py -q`
Expected: 3 passed.

Run: `python -m pytest backend/ -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add backend/models.py backend/app.py backend/test_app_delivery_api.py
git commit -m "feat(api): history carries Kappa delivery state, add summary endpoint

Delivery state rides along inside the existing history response, so the
list costs no extra round trip. A NULL status becomes None, which the UI
shows as a dash rather than as a problem.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The operator sees it

**Files:**
- Modify: `frontend/src/services/api.js` (add beside `requeuePipelineRun` at `:338`)
- Modify: `frontend/src/components/ProgressMonitor.jsx:106-110` (WS handler) and its render
- Modify: `frontend/src/components/PipelineHistory.jsx` (columns at `:163-245`, plus state and the summary alert)

**Interfaces:**
- Consumes: `kappa_upload` on each history item and `GET /api/kappa/delivery/summary` (Task 8); the `kappa_upload_deferred` WebSocket message (Task 4)
- Produces: no backend-facing interfaces

**Background — two traps in this codebase.** First, `handleWebSocketMessage` in `ProgressMonitor.jsx:106` currently drops every message that is not `progress_update`/`status`, so `kappa_upload_complete` has been broadcast into the void since it was written; a new message type needs a handler, not just a sender. Second, `PipelineHistory.jsx` polls on a 5s interval that deliberately keeps `history` out of its dependency array (`:41-49`) — do not add state to those deps or you will recreate the runaway-polling bug the comment describes.

There is no JavaScript test harness in this repo. Verification is `npm run lint` plus the manual check in Task 10.

- [ ] **Step 1: Add the API helpers**

In `frontend/src/services/api.js`, after `requeuePipelineRun` (`:343`):

```javascript
/**
 * Повторить выгрузку результатов запуска в Kappa вручную.
 */
export const retryKappaUpload = async (runId) => {
  const response = await apiClient.post(
    `/kappa/retry-upload/${runId}`,
    null,
    { params: { session_id: localStorage.getItem('kappa_session_id') } },
  );
  return response.data;
};

/**
 * Сколько запусков ещё не доехали до Kappa.
 */
export const getKappaDeliverySummary = async () => {
  const response = await apiClient.get('/kappa/delivery/summary');
  return response.data;
};
```

Add both names to the default-export object at `:474`.

- [ ] **Step 2: Handle the deferred message in ProgressMonitor**

In `frontend/src/components/ProgressMonitor.jsx`, add state beside the others near `:30` and extend the handler at `:106`:

```javascript
  const [kappaDeferred, setKappaDeferred] = useState(null);
```

```javascript
  const handleWebSocketMessage = (data) => {
    if (data.type === 'progress_update' || data.type === 'status') {
      updateStatus(data);
    } else if (data.type === 'kappa_upload_deferred') {
      setKappaDeferred(data);
    }
  };
```

Render it just inside the returned fragment at `:339`, above the existing alerts:

```jsx
      {kappaDeferred && (
        <Alert
          type={kappaDeferred.status === 'needs_attention' ? 'error' : 'warning'}
          showIcon
          style={{ marginBottom: 16 }}
          message={
            kappaDeferred.status === 'needs_attention'
              ? 'Выгрузка в Kappa требует внимания'
              : 'Результаты пока не ушли в Kappa'
          }
          description={
            kappaDeferred.status === 'needs_attention'
              ? 'Автоматический повтор не поможет — посмотрите подробности в истории запусков.'
              : 'Сервис недоступен. Досылка выполняется автоматически, данные не потеряны.'
          }
        />
      )}
```

- [ ] **Step 3: Add the Kappa column and modal to PipelineHistory**

In `frontend/src/components/PipelineHistory.jsx`, extend the antd import at `:5` to include `Modal`, `Alert`, `List` and `Tooltip`, and add `CloudUploadOutlined` and `WarningOutlined` to the icon import at `:6`. Add `getKappaDeliverySummary, retryKappaUpload` to the api import at `:16`.

Add state and a summary fetch beside the existing state at `:20-25`:

```javascript
  const [deliverySummary, setDeliverySummary] = useState(null);
  const [deliveryDetail, setDeliveryDetail] = useState(null);
  const [retrying, setRetrying] = useState(false);
```

Fetch the summary inside the existing `fetchHistory` (`:54`), right after the history call succeeds, so it refreshes on the same schedule and adds no new interval:

```javascript
      try {
        setDeliverySummary(await getKappaDeliverySummary());
      } catch {
        setDeliverySummary(null);   // сводка не критична, история важнее
      }
```

Add a helper above the columns definition:

```javascript
  /** Человеческая подпись к состоянию выгрузки в Kappa. */
  const deliveryLabel = (d) => {
    if (d.status === 'done') return `Kappa ${d.delivered}/${d.total}`;
    if (d.status === 'needs_attention') {
      return d.reason === 'stuck'
        ? 'не удаётся выгрузить'
        : `${d.delivered} из ${d.total} · нужна проверка`;
    }
    if (d.reason === 'no_session') return 'ждёт входа в Kappa';
    return `${d.delivered} из ${d.total} · досылается`;
  };
```

Add the column to the `columns` array, after "Качество" (`:208-228`):

```javascript
    {
      title: 'Kappa',
      key: 'kappa_upload',
      width: 180,
      render: (_, record) => {
        const d = record.kappa_upload;
        if (!d) return <span style={{ color: '#bbb' }}>—</span>;
        const color = d.status === 'done'
          ? 'success'
          : d.status === 'needs_attention' ? 'error' : 'warning';
        const icon = d.status === 'needs_attention'
          ? <WarningOutlined />
          : <CloudUploadOutlined />;
        return (
          <Tooltip title="Показать подробности выгрузки">
            <Tag
              color={color}
              icon={icon}
              style={{ cursor: 'pointer' }}
              onClick={() => setDeliveryDetail(record)}
            >
              {deliveryLabel(d)}
            </Tag>
          </Tooltip>
        );
      },
    },
```

Add the modal inside the component's returned JSX, next to the `Card`:

```jsx
      <Modal
        open={!!deliveryDetail}
        onCancel={() => setDeliveryDetail(null)}
        title="Выгрузка в Kappa"
        footer={[
          <Button key="close" onClick={() => setDeliveryDetail(null)}>
            Закрыть
          </Button>,
          <Button
            key="retry"
            type="primary"
            loading={retrying}
            onClick={async () => {
              setRetrying(true);
              try {
                await retryKappaUpload(deliveryDetail.run_id);
                message.success('Повтор выполнен');
                setDeliveryDetail(null);
                fetchHistory();
              } catch (e) {
                message.error(
                  e?.response?.data?.detail || 'Не удалось повторить выгрузку',
                );
              } finally {
                setRetrying(false);
              }
            }}
          >
            Повторить сейчас
          </Button>,
        ]}
      >
        {deliveryDetail?.kappa_upload && (
          <>
            <p>
              Выгружено {deliveryDetail.kappa_upload.delivered} из{' '}
              {deliveryDetail.kappa_upload.total}.
              {deliveryDetail.kappa_upload.next_attempt_at && (
                <> Следующая попытка:{' '}
                  {new Date(
                    deliveryDetail.kappa_upload.next_attempt_at,
                  ).toLocaleString('ru-RU')}.
                </>
              )}
            </p>
            {deliveryDetail.kappa_upload.blocked.length > 0 && (
              <List
                size="small"
                header="Требуют внимания"
                dataSource={deliveryDetail.kappa_upload.blocked}
                renderItem={(b) => (
                  <List.Item>
                    <strong>{b.session}</strong>: {b.message || b.reason}
                  </List.Item>
                )}
              />
            )}
          </>
        )}
      </Modal>
```

And the summary alert, immediately above the `Table`:

```jsx
      {deliverySummary &&
        (deliverySummary.pending > 0 || deliverySummary.needs_attention > 0) && (
          <Alert
            type={deliverySummary.needs_attention > 0 ? 'error' : 'warning'}
            showIcon
            style={{ marginBottom: 16 }}
            message={
              `Ожидают выгрузки в Kappa: ${deliverySummary.pending}` +
              (deliverySummary.needs_attention > 0
                ? `, требуют внимания: ${deliverySummary.needs_attention}`
                : '')
            }
          />
        )}
```

- [ ] **Step 4: Lint**

Run: `cd frontend && npm run lint`
Expected: no new errors. If ESLint flags the empty `catch {}` block, give it a parameter and a `void` statement rather than removing the guard — a failing summary must never break the history list.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/api.js \
        frontend/src/components/ProgressMonitor.jsx \
        frontend/src/components/PipelineHistory.jsx
git commit -m "feat(frontend): show Kappa delivery state and let the operator retry

Three surfaces, because 'did not reach Kappa' is noticed at three
different moments: a warning on the progress screen right after the run,
a per-run tag with a counter in the history, and a summary alert for the
run that failed days ago and scrolled out of sight.

The retry endpoint has existed since the 504 incident and was never
called from the UI. Now it is.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: End-to-end verification against a live stack

**Files:** none — this task changes no code.

This feature's whole point is behaviour under a failure that unit tests simulate. Verify it for real before calling the work done.

- [ ] **Step 1: Run the full suite**

```bash
cd /home/ubuntu/mri_ai_service && source venv/bin/activate
python -m pytest backend/ -q
python -m pytest test_*.py -q
cd frontend && npm run lint
```

Expected: green, with no new failures versus the baseline from "Before You Start".

- [ ] **Step 2: Rebuild the image**

`backend/` and `frontend/` are baked into the `web` image — they are NOT bind-mounted, so a restart alone ships none of this.

```bash
docker compose --profile full up --build -d
```

- [ ] **Step 3: Confirm the migration ran**

```bash
sqlite3 backend/data/brain_lesion.db "PRAGMA table_info(pipeline_runs);" | grep kappa_upload
sqlite3 backend/data/brain_lesion.db \
  "SELECT kappa_upload_status, COUNT(*) FROM pipeline_runs GROUP BY 1;"
```

Expected: the three `kappa_upload_*` columns exist, and the backfill marked some recent completed runs `pending`.

- [ ] **Step 4: Break Kappa on purpose and run the pipeline**

Point the Kappa host at something unreachable, start a small run from the UI, and let it finish.

Expected: the run completes; the progress screen shows "Результаты пока не ушли в Kappa"; the history row shows a warning tag; `kappa_upload_status` is `pending` with a `next_attempt` a minute or two out.

- [ ] **Step 5: Restore Kappa and watch the worker finish the job**

Restore the host and watch:

```bash
docker compose logs -f web | grep -i "deferred delivery"
```

Expected: within about a minute the run's status becomes `done`, the history tag turns green, and the dataset in Kappa contains each patient **exactly once**.

- [ ] **Step 6: Report**

Report what you saw at each step — including anything that did not match. Do not mark this task complete on a partial verification.
