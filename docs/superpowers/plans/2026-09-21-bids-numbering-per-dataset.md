# BIDS Numbering Per Kappa Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Number BIDS subjects within the Kappa dataset a run uploads into — a fresh dataset starts at `sub-001`, a dataset that already holds data continues after its highest number — without changing any number already issued.

**Architecture:** The dataset is resolved once at run start and recorded on the run; it becomes both the numbering scope for Stage 01 and the upload target, so the two ends cannot disagree. The dataset itself is the authority for the next number: its entity names (`sub-001_ses-001`) set a floor that the local SQLite allocator never goes below. Lookups resolve a patient inside a dataset and then join that person's sessions across the datasets of the same Kappa account.

**Tech Stack:** Python 3.12 (uv venv), SQLite via stdlib `sqlite3` (allocator) and SQLAlchemy (backend), FastAPI, pytest, React 19 + Ant Design.

**Spec:** `docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md`

## Global Constraints

- **Never renumber an existing patient.** Numbers already issued are entity names in Kappa; migration moves them between scopes but never changes them.
- **Gaps are intentional.** An incomplete patient keeps its number so it can be completed later. Do not make numbering contiguous.
- **`utils/bids_allocator.py` stays dependency-free** (stdlib `sqlite3` only): it is imported both by the standalone Stage 01 subprocess and by the FastAPI backend.
- **A run must start even when Kappa is unreachable.** Never let a Kappa error abort a run that has a dataset in the mapping.
- **Deferred upload with automatic retry is out of scope** (Spec B).
- Comments and commit messages in English; conventional commits; tests with pytest.
- Tests must never write to real `configs/` files — point modules at `tmp_path` via `monkeypatch` (see `backend/test_preprocessing_version.py` for the pattern).
- **Do not `git checkout` branches while the docker stack is running** — `configs/`, `scripts/` and `utils/` are bind-mounted into the web container.

---

### Task 1: Scope-keyed allocator

**Files:**
- Modify: `utils/bids_allocator.py` (whole file — key changes from `lesion_type` to `scope`, add floor support)
- Test: `test_bids_allocator.py` (create, repo root — root `test_*.py` is where `utils/` modules are tested, e.g. `test_config_loader.py`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `dataset_scope(dataset_id: int) -> str` → `"ds:337"`
  - `local_scope(lesion_type: str) -> str` → `"local:glioblastoma"`
  - `pending_scope(run_id: str) -> str` → `"pending:<run_id>"`
  - `get_or_allocate(scope: str, original_patient_id: str, db_path=None) -> str`
  - `get_bids_id(scope, original_patient_id, db_path=None) -> Optional[str]`
  - `get_original_id(scope, bids_id, db_path=None) -> Optional[str]`
  - `get_allocations(scope, db_path=None) -> Dict[str, str]`
  - `set_floor(scope: str, floor: int, db_path=None) -> None` — raises the scope's floor; never lowers it
  - `rebind_scope(old_scope: str, new_scope: str, db_path=None) -> int` — moves rows, returns count

- [ ] **Step 1: Write the failing tests**

```python
# test_bids_allocator.py
import pytest
from utils.bids_allocator import (
    dataset_scope, local_scope, pending_scope,
    get_or_allocate, get_bids_id, get_original_id, get_allocations,
    set_floor, rebind_scope,
)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "alloc.db"


def test_each_scope_starts_at_one(db):
    """The whole point: a fresh dataset numbers from sub-001, even for a
    patient who already has a number somewhere else."""
    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-001"
    assert get_or_allocate(dataset_scope(349), "P001", db) == "sub-001"


def test_number_is_stable_within_a_scope(db):
    first = get_or_allocate(dataset_scope(337), "P001", db)
    get_or_allocate(dataset_scope(337), "P002", db)

    assert get_or_allocate(dataset_scope(337), "P001", db) == first


def test_numbers_increment_within_a_scope(db):
    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-001"
    assert get_or_allocate(dataset_scope(337), "P002", db) == "sub-002"


def test_floor_from_an_existing_dataset(db):
    """A dataset already holding sub-001..sub-007 must continue at sub-008,
    even though this machine's table knows nothing about those patients."""
    set_floor(dataset_scope(337), 7, db)

    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-008"


def test_floor_never_lowers(db):
    set_floor(dataset_scope(337), 7, db)
    set_floor(dataset_scope(337), 3, db)

    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-008"


def test_floor_below_allocated_numbers_is_harmless(db):
    get_or_allocate(dataset_scope(337), "P001", db)
    get_or_allocate(dataset_scope(337), "P002", db)
    set_floor(dataset_scope(337), 1, db)

    assert get_or_allocate(dataset_scope(337), "P003", db) == "sub-003"


def test_scopes_do_not_leak(db):
    get_or_allocate(dataset_scope(337), "P001", db)

    assert get_bids_id(dataset_scope(349), "P001", db) is None
    assert get_allocations(dataset_scope(349), db) == {}


def test_reverse_lookup_is_scoped(db):
    get_or_allocate(dataset_scope(337), "P001", db)
    get_or_allocate(dataset_scope(349), "P002", db)

    assert get_original_id(dataset_scope(337), "sub-001", db) == "P001"
    assert get_original_id(dataset_scope(349), "sub-001", db) == "P002"


def test_rebind_moves_a_pending_scope(db):
    """A run numbered while Kappa was unreachable keeps its numbers when the
    dataset finally gets created."""
    get_or_allocate(pending_scope("run-1"), "P001", db)
    get_or_allocate(pending_scope("run-1"), "P002", db)

    moved = rebind_scope(pending_scope("run-1"), dataset_scope(350), db)

    assert moved == 2
    assert get_bids_id(dataset_scope(350), "P001", db) == "sub-001"
    assert get_allocations(pending_scope("run-1"), db) == {}


def test_local_scope_is_separate_from_datasets(db):
    get_or_allocate(local_scope("glioblastoma"), "P001", db)

    assert get_bids_id(dataset_scope(337), "P001", db) is None


def test_concurrent_allocation_does_not_reuse_a_number(db):
    """Two Stage 01 processes starting at once must not take the same number.
    BEGIN IMMEDIATE serialises them."""
    import threading

    results = []
    def worker(pid):
        results.append(get_or_allocate(dataset_scope(337), pid, db))

    threads = [threading.Thread(target=worker, args=(f"P{i:03d}",)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(set(results)) == 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest test_bids_allocator.py -q`
Expected: FAIL — `ImportError: cannot import name 'dataset_scope'`

- [ ] **Step 3: Rewrite `utils/bids_allocator.py`**

Keep the module docstring's history but restate the scope rule. Replace the table and every `lesion_type` parameter:

```python
_TABLE = "bids_patient_allocation"
_FLOOR_TABLE = "bids_scope_floor"


def dataset_scope(dataset_id: int) -> str:
    """Numbering space of a Kappa dataset."""
    return f"ds:{dataset_id}"


def local_scope(lesion_type: str) -> str:
    """Numbering space for runs with no Kappa behind them (CLI)."""
    return f"local:{lesion_type}"


def pending_scope(run_id: str) -> str:
    """Temporary space for a run that had no dataset yet (Kappa unreachable).

    Bound to a newly created dataset with rebind_scope() at upload time. It is
    never merged into an existing dataset — those numbers are already taken.
    """
    return f"pending:{run_id}"
```

`_connect()` creates both tables:

```python
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            scope                TEXT NOT NULL,
            original_patient_id  TEXT NOT NULL,
            bids_id              TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            PRIMARY KEY (scope, original_patient_id),
            UNIQUE (scope, bids_id)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_FLOOR_TABLE} (
            scope TEXT PRIMARY KEY,
            floor INTEGER NOT NULL
        )
        """
    )
```

`_next_bids_id` respects the floor:

```python
def _next_bids_id(conn: sqlite3.Connection, scope: str) -> str:
    """Next sub-NNN for a scope: above both what we allocated and what the
    dataset already contains (the floor)."""
    rows = conn.execute(
        f"SELECT bids_id FROM {_TABLE} WHERE scope = ?", (scope,)
    ).fetchall()
    max_n = 0
    for (bids_id,) in rows:
        try:
            max_n = max(max_n, int(bids_id.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    floor_row = conn.execute(
        f"SELECT floor FROM {_FLOOR_TABLE} WHERE scope = ?", (scope,)
    ).fetchone()
    if floor_row:
        max_n = max(max_n, int(floor_row[0]))
    return f"sub-{max_n + 1:03d}"
```

New functions:

```python
def set_floor(scope: str, floor: int, db_path=None) -> None:
    """Raise the scope's floor to `floor` (never lowers it).

    The floor is what the Kappa dataset already contains. Persisting it means
    an offline run still numbers above what the dataset held when we last saw
    it, instead of restarting at sub-001.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT floor FROM {_FLOOR_TABLE} WHERE scope = ?", (scope,)
        ).fetchone()
        current = int(row[0]) if row else 0
        if floor > current:
            conn.execute(
                f"INSERT INTO {_FLOOR_TABLE} (scope, floor) VALUES (?, ?) "
                f"ON CONFLICT(scope) DO UPDATE SET floor = excluded.floor",
                (scope, int(floor)),
            )
        conn.commit()
    finally:
        conn.close()


def rebind_scope(old_scope: str, new_scope: str, db_path=None) -> int:
    """Move every allocation from one scope to another; returns the count."""
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            f"UPDATE {_TABLE} SET scope = ? WHERE scope = ?", (new_scope, old_scope)
        )
        conn.execute(f"DELETE FROM {_FLOOR_TABLE} WHERE scope = ?", (old_scope,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
```

Rename the `lesion_type` parameter to `scope` in `get_or_allocate`, `get_bids_id`, `get_original_id` and `get_allocations`, keeping their bodies otherwise unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest test_bids_allocator.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/bids_allocator.py test_bids_allocator.py
git commit -m "feat(bids): key subject allocation by scope, with a dataset floor"
```

---

### Task 2: Read the dataset's highest number from Kappa

**Files:**
- Create: `backend/kappa_dataset_resolver.py`
- Modify: `backend/kappa_uploader.py:132-176` (`_resolve_dataset_id` delegates to the new module)
- Test: `backend/test_kappa_dataset_resolver.py` (create)

**Interfaces:**
- Consumes: `kappa_client.get_dataset_entities`, `kappa_client.create_dataset`, `kappa_dataset_mapping.get_dataset_id/set_dataset_id` (all existing).
- Produces:
  - `highest_subject_number(entities: list) -> int`
  - `async resolve_or_create(token, user_id, user_type_id, lesion_type, preprocessing_id, create: bool = True) -> Optional[int]`
  - `async dataset_floor(token, user_id, user_type_id, dataset_id) -> Optional[int]`

- [ ] **Step 1: Write the failing tests**

```python
# backend/test_kappa_dataset_resolver.py
import pytest
import kappa_dataset_resolver as resolver


def test_highest_number_of_an_empty_dataset_is_zero():
    assert resolver.highest_subject_number([]) == 0


def test_highest_number_reads_entity_names():
    entities = [
        {"dsEntityName": "sub-001_ses-001"},
        {"dsEntityName": "sub-007_ses-002"},
        {"dsEntityName": "sub-003_ses-001"},
    ]

    assert resolver.highest_subject_number(entities) == 7


def test_highest_number_ignores_names_that_are_not_bids():
    """Someone may have uploaded by hand. Unparseable names must not crash the
    run or be read as a number."""
    entities = [{"dsEntityName": "notes.txt"}, {"dsEntityName": "sub-004_ses-001"},
                {"dsEntityName": None}, {}]

    assert resolver.highest_subject_number(entities) == 4


@pytest.mark.asyncio
async def test_resolve_returns_the_mapped_dataset_without_creating(monkeypatch):
    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: 337)
    async def fail_create(**kwargs):
        raise AssertionError("must not create when the mapping already has one")
    monkeypatch.setattr(resolver, "create_dataset", fail_create)

    got = await resolver.resolve_or_create(
        token="t", user_id=26, user_type_id=3,
        lesion_type="glioblastoma", preprocessing_id="1099b9cd")

    assert got == 337


@pytest.mark.asyncio
async def test_resolve_creates_and_registers_when_unmapped(monkeypatch):
    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: None)
    async def fake_create(**kwargs):
        return 350
    recorded = {}
    monkeypatch.setattr(resolver, "create_dataset", fake_create)
    monkeypatch.setattr(resolver, "set_dataset_id",
                        lambda u, l, p, d: recorded.update(user=u, lesion=l, prep=p, ds=d))

    got = await resolver.resolve_or_create(
        token="t", user_id=52, user_type_id=4,
        lesion_type="glioblastoma", preprocessing_id="1099b9cd")

    assert got == 350
    assert recorded == {"user": 52, "lesion": "glioblastoma",
                        "prep": "1099b9cd", "ds": 350}


@pytest.mark.asyncio
async def test_resolve_without_create_returns_none_when_unmapped(monkeypatch):
    """Run start uses create=False when Kappa is unreachable — it must report
    'no dataset' rather than raise."""
    monkeypatch.setattr(resolver, "get_dataset_id", lambda *a: None)

    got = await resolver.resolve_or_create(
        token="t", user_id=52, user_type_id=4, lesion_type="glioblastoma",
        preprocessing_id="1099b9cd", create=False)

    assert got is None


@pytest.mark.asyncio
async def test_floor_is_none_when_kappa_cannot_be_reached(monkeypatch):
    """None means 'unknown', which the caller must not confuse with 0 —
    0 would restart numbering at sub-001 in a populated dataset."""
    async def boom(**kwargs):
        raise OSError("connection refused")
    monkeypatch.setattr(resolver, "get_dataset_entities", boom)

    assert await resolver.dataset_floor("t", 26, 3, 337) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest backend/test_kappa_dataset_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kappa_dataset_resolver'`

- [ ] **Step 3: Write `backend/kappa_dataset_resolver.py`**

```python
"""
One place that answers "which Kappa dataset does this run belong to, and what
number comes next in it".

Both ends of a run need this: the start, to fix the numbering scope before
Stage 01 issues any id, and the upload, to send entities to the dataset those
ids were issued for. It used to live inside KappaUploader, which is why the
two ends could disagree when `current` was repointed mid-run.
"""

import logging
import re
from typing import Any, Optional

from kappa_client import create_dataset, get_dataset_entities
from kappa_dataset_mapping import get_dataset_id, set_dataset_id

logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"^sub-(\d+)")


def highest_subject_number(entities: list) -> int:
    """Highest sub-NNN among a dataset's entity names (0 if there are none).

    Entity names are the session keys we upload ("sub-001_ses-001"), so the
    dataset itself tells us what numbers are taken — including ones issued on
    another machine.
    """
    highest = 0
    for entity in entities or []:
        name = (entity or {}).get("dsEntityName") or ""
        match = _SUBJECT_RE.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


async def resolve_or_create(
    token: str,
    user_id: int,
    user_type_id: int,
    lesion_type: str,
    preprocessing_id: str,
    create: bool = True,
) -> Optional[int]:
    """The dataset for this (user, lesion, preprocessing), creating one if the
    mapping has none and `create` is set."""
    dataset_id = get_dataset_id(user_id, lesion_type, preprocessing_id)
    if dataset_id is not None:
        return dataset_id
    if not create:
        return None

    short_id = preprocessing_id[:8]
    new_id = await create_dataset(
        token=token,
        user_id=user_id,
        user_type_id=user_type_id,
        dataset_name=f"{lesion_type}_{short_id}",
        dataset_short_info=f"Lesion: {lesion_type}, Preprocessing: {preprocessing_id}",
        dataset_type=1,
        dataset_tags=f"Image Segmentation,mri,{lesion_type}",
    )
    if new_id is not None:
        set_dataset_id(user_id, lesion_type, preprocessing_id, new_id)
        logger.info("New dataset created: id=%d", new_id)
    return new_id


async def dataset_floor(
    token: str, user_id: int, user_type_id: int, dataset_id: int
) -> Optional[int]:
    """Highest subject number the dataset already holds, or None if Kappa
    could not be asked. None means unknown — never treat it as 0."""
    try:
        entities = await get_dataset_entities(
            token=token, user_id=user_id,
            user_type_id=user_type_id, dataset_id=dataset_id,
        )
    except Exception as exc:
        logger.warning("Could not read dataset %s from Kappa: %s", dataset_id, exc)
        return None
    if entities is None:
        return None
    return highest_subject_number(entities)
```

Then in `backend/kappa_uploader.py`, replace the body of `_resolve_dataset_id` with a call into the resolver, keeping the empty-dataset warning:

```python
    async def _resolve_dataset_id(self) -> Optional[int]:
        """Dataset for this run: the one fixed at start, else resolve/create."""
        if self.dataset_id is not None:
            return self.dataset_id
        from kappa_dataset_resolver import resolve_or_create
        if get_dataset_id(self.user_id, self.lesion_type, self.preprocessing_id) is None:
            await self._warn_if_empty_dataset_exists()
        return await resolve_or_create(
            token=self.token, user_id=self.user_id, user_type_id=self.user_type_id,
            lesion_type=self.lesion_type, preprocessing_id=self.preprocessing_id,
        )
```

Add `dataset_id: Optional[int] = None` to `KappaUploader.__init__` and store it as `self.dataset_id`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest backend/test_kappa_dataset_resolver.py backend/test_kappa_uploader_resolve.py -q`
Expected: PASS — new tests pass and the existing uploader-resolve tests still pass

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_dataset_resolver.py backend/test_kappa_dataset_resolver.py backend/kappa_uploader.py
git commit -m "feat(kappa): shared dataset resolver that can read a dataset's highest subject number"
```

---

### Task 3: Record the dataset on the run

**Files:**
- Modify: `backend/database.py` — `PipelineRun` model (add column next to `lesion_type`), `create_pipeline_run()` at line 100, migrations list at line ~290
- Test: `backend/test_pipeline_run_dataset_id.py` (create; mirror `backend/test_pipeline_run_parent_id.py`)

**Interfaces:**
- Consumes: nothing.
- Produces: `PipelineRun.kappa_dataset_id` (`Optional[int]`), `create_pipeline_run(..., kappa_dataset_id: Optional[int] = None)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/test_pipeline_run_dataset_id.py
from database import create_pipeline_run, get_pipeline_run


def test_run_stores_its_dataset(session):
    run = create_pipeline_run(
        session, input_path="/in", output_path="/out",
        lesion_type="glioblastoma", kappa_dataset_id=337)

    assert get_pipeline_run(session, run.run_id).kappa_dataset_id == 337


def test_dataset_is_optional(session):
    """CLI runs and runs started without a Kappa session have none."""
    run = create_pipeline_run(session, input_path="/in", output_path="/out")

    assert run.kappa_dataset_id is None
```

Use the same `session` fixture as `backend/test_pipeline_run_parent_id.py` — copy its fixture setup verbatim if it is defined locally rather than in `conftest.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `source venv/bin/activate && python -m pytest backend/test_pipeline_run_dataset_id.py -q`
Expected: FAIL — `TypeError: create_pipeline_run() got an unexpected keyword argument 'kappa_dataset_id'`

- [ ] **Step 3: Add the column, the migration and the parameter**

In the `PipelineRun` model, after `lesion_type`:

```python
    # The Kappa dataset this run numbers its subjects in and uploads to. Fixed
    # at start so numbering and upload cannot target different datasets.
    kappa_dataset_id = Column(Integer, nullable=True)
```

New migration beside `_migrate_add_lesion_type`, following it exactly:

```python
def _migrate_add_kappa_dataset_id():
    """Add kappa_dataset_id to pipeline_runs if it doesn't exist yet."""
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            __import__('sqlalchemy').text("PRAGMA table_info(pipeline_runs)")
        )]
        if 'kappa_dataset_id' not in cols:
            conn.execute(__import__('sqlalchemy').text(
                "ALTER TABLE pipeline_runs ADD COLUMN kappa_dataset_id INTEGER"
            ))
            conn.commit()
```

Call it next to the others (line ~292), and add the parameter to `create_pipeline_run`:

```python
def create_pipeline_run(
    db: Session,
    input_path: str,
    output_path: str,
    lesion_type: str = 'glioblastoma',
    parent_run_id: Optional[str] = None,
    kappa_dataset_id: Optional[int] = None,
) -> PipelineRun:
```

passing `kappa_dataset_id=kappa_dataset_id` into the `PipelineRun(...)` constructor.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest backend/ -q`
Expected: PASS — 169 existing + 2 new

- [ ] **Step 5: Commit**

```bash
git add backend/database.py backend/test_pipeline_run_dataset_id.py
git commit -m "feat(db): record the Kappa dataset on a pipeline run"
```

---

### Task 4: Fix the scope at run start and pass it to Stage 01

**Files:**
- Modify: `backend/app.py:420-480` (start endpoint), `backend/app.py:237-262` (`run_pipeline_background`), `backend/app.py:668-690` (resume inherits the parent's dataset)
- Modify: `backend/pipeline_manager.py:87-125` (`create_runtime_config`), `:312-380` (`start_pipeline` threads the scope)
- Modify: `orchestrator.py:62-72` (pass `--numbering-scope` to Stage 01)
- Modify: `scripts/01_reorganize_folders.py:548-575` (IDMapper), `:2285-2295` (main)
- Test: `backend/test_numbering_scope.py` (create), `test_orchestrator.py` (extend), `test_stage01_numbering_scope.py` (create)

**Interfaces:**
- Consumes: `dataset_scope`, `local_scope`, `pending_scope`, `set_floor` (Task 1); `resolve_or_create`, `dataset_floor` (Task 2); `create_pipeline_run(..., kappa_dataset_id=)` (Task 3).
- Produces:
  - `backend/numbering.py`: `async def scope_for_run(run_id: str, lesion_type: str, kappa_session_id: Optional[str]) -> tuple[str, Optional[int], Optional[str]]` returning `(scope, dataset_id, warning)`
  - `general.numbering_scope` in the runtime config
  - `IDMapper(lesion_type=None, db_path=None, scope=None)`

- [ ] **Step 1: Write the failing tests**

```python
# backend/test_numbering_scope.py
import pytest
import numbering


@pytest.mark.asyncio
async def test_no_kappa_session_means_local_scope():
    """A CLI-style run with no Kappa behind it keeps today's numbering."""
    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id=None)

    assert scope == "local:glioblastoma"
    assert dataset_id is None
    assert warning is None


@pytest.mark.asyncio
async def test_mapped_dataset_becomes_the_scope(monkeypatch):
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26, "user_type_id": 3})
    async def fake_resolve(**kwargs):
        return 337
    async def fake_floor(**kwargs):
        return 7
    recorded = {}
    monkeypatch.setattr(numbering, "resolve_or_create", fake_resolve)
    monkeypatch.setattr(numbering, "dataset_floor", fake_floor)
    monkeypatch.setattr(numbering, "set_floor",
                        lambda scope, floor: recorded.update(scope=scope, floor=floor))

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert (scope, dataset_id, warning) == ("ds:337", 337, None)
    assert recorded == {"scope": "ds:337", "floor": 7}


@pytest.mark.asyncio
async def test_unreachable_kappa_with_no_dataset_falls_back_to_pending(monkeypatch):
    """The run must still start. Numbers are issued in a temporary scope and
    bound to a dataset at upload time."""
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 52, "user_type_id": 4})
    async def boom(**kwargs):
        raise OSError("connection refused")
    monkeypatch.setattr(numbering, "resolve_or_create", boom)

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert scope == "pending:r1"
    assert dataset_id is None
    assert "Kappa" in warning


@pytest.mark.asyncio
async def test_unknown_floor_does_not_reset_numbering(monkeypatch):
    """Kappa unreachable while the dataset IS known: keep the dataset scope and
    leave the floor alone. Writing 0 would restart a populated dataset at 1."""
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26, "user_type_id": 3})
    async def fake_resolve(**kwargs):
        return 337
    async def no_floor(**kwargs):
        return None
    calls = []
    monkeypatch.setattr(numbering, "resolve_or_create", fake_resolve)
    monkeypatch.setattr(numbering, "dataset_floor", no_floor)
    monkeypatch.setattr(numbering, "set_floor", lambda *a: calls.append(a))

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert (scope, dataset_id) == ("ds:337", 337)
    assert calls == []
```

```python
# test_stage01_numbering_scope.py  (repo root)
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "stage01", Path(__file__).parent / "scripts" / "01_reorganize_folders.py")
stage01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage01)


def test_id_mapper_allocates_inside_its_scope(tmp_path):
    db = tmp_path / "alloc.db"
    mapper = stage01.IDMapper(scope="ds:337", db_path=db)
    other = stage01.IDMapper(scope="ds:349", db_path=db)

    assert mapper.get_new_id("P001") == "sub-001"
    assert other.get_new_id("P001") == "sub-001"


def test_id_mapper_is_stable_within_a_scope(tmp_path):
    db = tmp_path / "alloc.db"
    mapper = stage01.IDMapper(scope="ds:337", db_path=db)
    first = mapper.get_new_id("P001")

    fresh = stage01.IDMapper(scope="ds:337", db_path=db)
    assert fresh.get_new_id("P001") == first


def test_lesion_type_alone_still_works(tmp_path):
    """CLI runs pass only --lesion-type; they must keep numbering as before."""
    db = tmp_path / "alloc.db"
    mapper = stage01.IDMapper(lesion_type="glioblastoma", db_path=db)

    assert mapper.get_new_id("P001") == "sub-001"
```

Extend `test_orchestrator.py` with:

```python
# test_orchestrator.py — build_command(stage_name, config, project_root),
# imported at the top of that file already.
def test_numbering_scope_goes_to_stage_01():
    config = create_test_config()
    config["general"]["numbering_scope"] = "ds:337"

    cmd = build_command("stage_01_reorganize", config, Path("/project"))

    assert "--numbering-scope" in cmd
    assert "ds:337" in cmd


def test_numbering_scope_is_not_passed_to_other_stages():
    """Only Stage 01 issues subject ids; the flag would be an unknown argument
    elsewhere and would abort the stage."""
    config = create_test_config()
    config["general"]["numbering_scope"] = "ds:337"

    cmd = build_command("stage_06_segmentation", config, Path("/project"))

    assert "--numbering-scope" not in cmd


def test_stage_01_without_a_scope_is_unchanged():
    """CLI runs have no scope; the command must stay exactly as it is today."""
    config = create_test_config()

    cmd = build_command("stage_01_reorganize", config, Path("/project"))

    assert "--numbering-scope" not in cmd
```

`create_test_config()` is the helper already in `test_orchestrator.py`; if its
config has no `stage_06_segmentation` entry, add one there in the same shape as
the existing stages.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest backend/test_numbering_scope.py test_stage01_numbering_scope.py test_orchestrator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'numbering'`, `IDMapper() got an unexpected keyword argument 'scope'`

- [ ] **Step 3: Implement**

`backend/numbering.py`:

```python
"""
Decide the numbering scope for a run, once, at start.

Stage 01 issues subject ids before anything is uploaded, so the scope has to be
fixed before the pipeline starts. Resolving it here — rather than at upload —
is also what keeps numbering and upload pointed at the same dataset.
"""

import logging
from typing import Optional, Tuple

from kappa_auth import get_session
from kappa_dataset_resolver import dataset_floor, resolve_or_create
from pipeline_monitor import PREPROCESSING_CONFIG  # backend/pipeline_monitor.py:19
from preprocessing_version import compute_preprocessing_id
from utils.bids_allocator import dataset_scope, local_scope, pending_scope, set_floor

logger = logging.getLogger(__name__)


async def scope_for_run(
    run_id: str, lesion_type: str, kappa_session_id: Optional[str]
) -> Tuple[str, Optional[int], Optional[str]]:
    """(scope, dataset_id, warning) for a run about to start.

    Never raises: a run must start even when Kappa is down.
    """
    if not kappa_session_id:
        return local_scope(lesion_type), None, None

    session = get_session(kappa_session_id)
    if not session:
        return local_scope(lesion_type), None, None

    preprocessing_id = compute_preprocessing_id(str(PREPROCESSING_CONFIG))
    try:
        dataset_id = await resolve_or_create(
            token=session["kappa_token"], user_id=session["user_id"],
            user_type_id=session["user_type_id"], lesion_type=lesion_type,
            preprocessing_id=preprocessing_id,
        )
    except Exception as exc:
        logger.warning("Kappa unreachable at run start: %s", exc)
        return (pending_scope(run_id), None,
                "Kappa недоступна: датасет будет создан при выгрузке")

    if dataset_id is None:
        return (pending_scope(run_id), None,
                "Датасет в Kappa не создан: будет создан при выгрузке")

    scope = dataset_scope(dataset_id)
    floor = await dataset_floor(
        token=session["kappa_token"], user_id=session["user_id"],
        user_type_id=session["user_type_id"], dataset_id=dataset_id,
    )
    if floor is not None:
        set_floor(scope, floor)
    return scope, dataset_id, None
```

If importing `PREPROCESSING_CONFIG` from `pipeline_monitor` creates a cycle, move the constant to `backend/config.py` and import it from there in both modules — do not re-derive the path in two places.

In `backend/app.py`'s start endpoint, before `create_pipeline_run`:

```python
    run_id_for_scope = str(uuid.uuid4())
    numbering_scope, kappa_dataset_id, scope_warning = await numbering.scope_for_run(
        run_id_for_scope, request.lesion_type or "glioblastoma", request.kappa_session_id)
```

`create_pipeline_run` currently generates the run id itself; add a `run_id` parameter so the pending scope matches the real run, or resolve the scope after creating the run and update the row — either is fine, but the scope string and the run id must agree.

Pass `kappa_dataset_id=kappa_dataset_id` to `create_pipeline_run` and `numbering_scope=numbering_scope` through `run_pipeline_background` → `pipeline_manager.start_pipeline` → `create_runtime_config`, which writes:

```python
        # Numbering space for Stage 01: the Kappa dataset this run uploads to.
        config['general']['numbering_scope'] = numbering_scope
```

Return `scope_warning` in `PipelineStartResponse.message` when set, so the UI shows it.

In the resume path (`backend/app.py:668`), inherit the parent's dataset:

```python
        kappa_dataset_id=original_run.kappa_dataset_id,
```

and derive the scope from it rather than resolving again.

`orchestrator.py`, inside the existing stage-name branch, after `--lesion-type`:

```python
    if stage_name == 'stage_01_reorganize':
        scope = config['general'].get('numbering_scope')
        if scope:
            cmd.extend(['--numbering-scope', scope])
```

`scripts/01_reorganize_folders.py`:

```python
    def __init__(self, lesion_type: Optional[str] = None, db_path=None,
                 scope: Optional[str] = None):
        ...
        self._scope = scope
```

```python
        scope = self._scope
        if scope is None and self._lesion_type:
            # CLI runs pass only --lesion-type; keep their existing numbering.
            from utils.bids_allocator import local_scope
            scope = local_scope(self._lesion_type)

        if scope:
            from utils.bids_allocator import get_or_allocate
            new_id = get_or_allocate(scope, original_id, self._db_path)
        else:
            self._patient_counter += 1
            new_id = f"sub-{self._patient_counter:03d}"
```

Add `--numbering-scope` to the argument parser and pass it at line ~2291:

```python
        id_mapper = IDMapper(lesion_type=args.lesion_type, scope=args.numbering_scope)
```

Update the `get_allocations` call below it to use the same scope.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest backend/ test_orchestrator.py test_stage01_numbering_scope.py test_bids_allocator.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/numbering.py backend/test_numbering_scope.py backend/app.py \
        backend/pipeline_manager.py orchestrator.py test_orchestrator.py \
        scripts/01_reorganize_folders.py test_stage01_numbering_scope.py
git commit -m "feat(bids): fix the numbering scope at run start and pass it to stage 01"
```

---

### Task 5: Upload into the run's dataset, and bind a pending scope

**Files:**
- Modify: `backend/pipeline_monitor.py:118-150` (`_create_kappa_uploader` passes the run's dataset id)
- Modify: `backend/kappa_uploader.py:60-80` (`upload_results` binds a pending scope once the dataset exists)
- Test: `backend/test_kappa_uploader_dataset.py` (create)

**Interfaces:**
- Consumes: `PipelineRun.kappa_dataset_id` (Task 3), `rebind_scope`, `pending_scope`, `dataset_scope` (Task 1).
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
# backend/test_kappa_uploader_dataset.py
import pytest
from kappa_uploader import KappaUploader


def _uploader(**kwargs):
    return KappaUploader(
        run_id="r1", output_path="/tmp/out", token="t", user_id=26,
        user_type_id=3, lesion_type="glioblastoma",
        preprocessing_config_path="configs/preprocessing_config.yaml", **kwargs)


@pytest.mark.asyncio
async def test_run_dataset_is_used_without_resolving(monkeypatch):
    """The dataset was chosen at start; resolving again could pick a different
    one if `current` moved meanwhile."""
    uploader = _uploader(dataset_id=337)
    monkeypatch.setattr("kappa_uploader.get_dataset_id",
                        lambda *a: (_ for _ in ()).throw(AssertionError("resolved again")))

    assert await uploader._resolve_dataset_id() == 337


@pytest.mark.asyncio
async def test_pending_scope_is_rebound_to_the_created_dataset(monkeypatch, tmp_path):
    """A run numbered while Kappa was down keeps its numbers: the temporary
    scope is renamed to the dataset that finally got created."""
    from utils.bids_allocator import get_bids_id, get_or_allocate, pending_scope
    db = tmp_path / "alloc.db"
    get_or_allocate(pending_scope("r1"), "P001", db)

    uploader = _uploader()
    monkeypatch.setattr(uploader, "_allocation_db", db, raising=False)
    async def fake_resolve():
        return 350
    monkeypatch.setattr(uploader, "_resolve_dataset_id", fake_resolve)

    await uploader._bind_pending_scope(350)

    assert get_bids_id("ds:350", "P001", db) == "sub-001"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest backend/test_kappa_uploader_dataset.py -q`
Expected: FAIL — `KappaUploader.__init__() got an unexpected keyword argument 'dataset_id'` (if Task 2 is not yet merged) and `no attribute '_bind_pending_scope'`

- [ ] **Step 3: Implement**

In `backend/pipeline_monitor.py::_create_kappa_uploader`, read the run and pass its dataset:

```python
            from database import SessionLocal as DBSessionLocal, get_pipeline_run
            db = DBSessionLocal()
            try:
                run = get_pipeline_run(db, run_id)
                dataset_id = run.kappa_dataset_id if run else None
            finally:
                db.close()

            uploader = KappaUploader(..., dataset_id=dataset_id)
```

In `KappaUploader`, add:

```python
    def _bind_pending_scope(self, dataset_id: int) -> None:
        """Move numbers issued under pending:<run_id> onto the real dataset.

        Only ever called for a dataset created for this run — a pending scope is
        never merged into a dataset that already holds numbers.
        """
        from utils.bids_allocator import dataset_scope, pending_scope, rebind_scope
        moved = rebind_scope(pending_scope(self.run_id), dataset_scope(dataset_id),
                             getattr(self, "_allocation_db", None))
        if moved:
            logger.info("Bound %d pending allocations of run %s to dataset %d",
                        moved, self.run_id, dataset_id)
```

and call it in `upload_results()` right after `dataset_id` is resolved, before the duplicate check.

Then warn on a name clash — the spec's answer to two machines numbering into one
dataset. After `existing_hashes` is fetched, also collect the entity names, and
for every session whose key is already present with a *different* study hash:

```python
        # Same sub-NNN, different study: two machines numbered into this dataset
        # independently. We cannot renumber (the id is in the file names), so
        # make it loud rather than silent.
        if session_key in existing_names and study_hash not in existing_hashes:
            logger.warning(
                "Name clash in dataset %d: %s already exists with a different "
                "study — numbers were issued twice for this dataset",
                dataset_id, session_key,
            )
            results.append({
                "session": session_key,
                "success": False,
                "error": "name_clash",
                "message": (
                    f"В датасете уже есть {session_key} с другими данными — "
                    f"номер выдан дважды, загрузка пропущена"
                ),
            })
            continue
```

Add a test for it in `backend/test_kappa_uploader_dataset.py`:

```python
@pytest.mark.asyncio
async def test_name_clash_is_reported_and_not_uploaded(monkeypatch):
    """Two machines can issue the same number while offline. The upload must
    refuse that session loudly instead of creating a second sub-001."""
    uploader = _uploader(dataset_id=337)
    monkeypatch.setattr(uploader, "_discover_sessions",
                        lambda: {"sub-001_ses-001": {"files": []}})
    monkeypatch.setattr(uploader, "_compute_study_hash", lambda data: "hash-B")
    async def existing_hashes(dataset_id):
        return {"hash-A"}
    async def existing_names(dataset_id):
        return {"sub-001_ses-001"}
    monkeypatch.setattr(uploader, "_get_existing_study_hashes", existing_hashes)
    monkeypatch.setattr(uploader, "_get_existing_entity_names", existing_names)

    report = await uploader.upload_results()

    assert report["sessions"][0]["error"] == "name_clash"
```

Add `_get_existing_entity_names(dataset_id)` beside `_get_existing_study_hashes`,
reading `dsEntityName` from the same `get_dataset_entities` response.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest backend/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_monitor.py backend/kappa_uploader.py backend/test_kappa_uploader_dataset.py
git commit -m "feat(kappa): upload into the dataset fixed at run start; bind pending numbering"
```

---

### Task 6: Scope patient lookups to the dataset, then join by the real patient

**Files:**
- Modify: `backend/kappa_dataset_mapping.py` (add `datasets_of_user`)
- Modify: `backend/patient_registry.py:197-222` (`find_by_bids_id`, `find_by_bids_subject`)
- Modify: `backend/app.py:1340-1380` and `:1430-1460` (both longitudinal endpoints)
- Test: `backend/test_longitudinal_scoping.py` (create), `backend/test_kappa_dataset_mapping.py` (extend)

**Interfaces:**
- Consumes: `PipelineRun.kappa_dataset_id` (Task 3).
- Produces:
  - `datasets_of_user(user_id: int) -> set[int]`
  - `find_by_bids_id(bids_id: str, dataset_ids: Optional[set[int]] = None)`
  - `find_by_bids_subject(subject: str, dataset_ids: Optional[set[int]] = None)`
  - `GET /api/longitudinal/{patient_id}?lesion_type=&run_id=` (new optional `run_id`)
  - `GET /api/longitudinal/{patient_id}/diff?lesion_type=&run_id=`

- [ ] **Step 1: Write the failing tests**

```python
# backend/test_kappa_dataset_mapping.py  (append)
def test_datasets_of_user(tmp_path, monkeypatch):
    import kappa_dataset_mapping as m
    path = tmp_path / "kappa_datasets.yaml"
    path.write_text(
        "datasets:\n"
        "  26:glioblastoma:current: 337\n"
        "  26:glioblastoma:3a183dc7: 249\n"
        "  52:glioblastoma:current: 349\n"
        "lesion_types: []\n")
    monkeypatch.setattr(m, "MAPPING_FILE", path)

    assert m.datasets_of_user(26) == {337, 249}
    assert m.datasets_of_user(52) == {349}
    assert m.datasets_of_user(99) == set()
```

```python
# backend/test_longitudinal_scoping.py
"""Two people can now be called sub-001. These tests pin down that the
registry never merges them, and never splits one person across datasets."""
from patient_registry import find_by_bids_subject, register_patient


def _register(session_key, original_id, dataset_id, lesion_type="multiple_sclerosis"):
    register_patient(
        study_hash=f"{original_id}-{session_key}",
        bids_id=session_key,
        original_patient_id=original_id,
        lesion_type=lesion_type,
        kappa_dataset_id=dataset_id,
    )


def test_same_number_in_two_datasets_stays_two_people(registry_db):
    _register("sub-001_ses-001", "P100", 158)
    _register("sub-001_ses-001", "P200", 338)

    found = find_by_bids_subject("sub-001", dataset_ids={158})

    assert {r["original_patient_id"] for r in found} == {"P100"}


def test_one_person_across_two_datasets_is_joined(registry_db):
    _register("sub-003_ses-001", "P100", 158)
    _register("sub-001_ses-002", "P100", 338)

    found = find_by_bids_subject("sub-003", dataset_ids={158, 338})
    sessions = set()
    for record in found:
        sessions.update(
            r["bids_id"] for r in find_by_patient_id(record["original_patient_id"]))

    assert sessions == {"sub-003_ses-001", "sub-001_ses-002"}


def test_no_filter_keeps_the_old_behaviour(registry_db):
    _register("sub-001_ses-001", "P100", 158)
    _register("sub-001_ses-001", "P200", 338)

    assert len(find_by_bids_subject("sub-001")) == 2
```

Import `find_by_patient_id` alongside the others. Use the registry fixture from `backend/conftest.py` (`ensure_tables` against a temp DB); if none exists, add a `registry_db` fixture there that points `database.SessionLocal` at `tmp_path`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest backend/test_longitudinal_scoping.py backend/test_kappa_dataset_mapping.py -q`
Expected: FAIL — `find_by_bids_subject() got an unexpected keyword argument 'dataset_ids'`, `no attribute 'datasets_of_user'`

- [ ] **Step 3: Implement**

`backend/kappa_dataset_mapping.py`:

```python
def datasets_of_user(user_id: int) -> set:
    """Every dataset id mapped to this Kappa user.

    Keys are "{user_id}:{lesion_type}:{preprocessing_id}", so the prefix is the
    owner. Used to keep one person's timeline together across their datasets
    without reaching into anyone else's.
    """
    datasets = _load_mapping().get("datasets", {})
    prefix = f"{user_id}:"
    return {int(v) for k, v in datasets.items() if str(k).startswith(prefix)}
```

`backend/patient_registry.py` — add the filter to both finders:

```python
def find_by_bids_id(bids_id: str, dataset_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    """Records for a BIDS id. `dataset_ids` narrows to those Kappa datasets —
    required now that sub-001 is only unique within a dataset."""
    db = SessionLocal()
    try:
        query = db.query(PatientRegistry).filter(PatientRegistry.bids_id == bids_id)
        if dataset_ids is not None:
            query = query.filter(PatientRegistry.kappa_dataset_id.in_(dataset_ids))
        return [_to_dict(r) for r in query.all()]
    finally:
        db.close()
```

and the same `dataset_ids` filter inside `find_by_bids_subject` (it filters in Python; apply the dataset check in the same comprehension, treating `kappa_dataset_id is None` as "include", since an un-uploaded run exists only on this machine).

`backend/app.py` — both longitudinal endpoints gain `run_id: Optional[str] = None` and resolve the scope before looking up:

```python
    allowed_datasets = None
    if run_id:
        run = get_pipeline_run(db, run_id)
        if run and run.kappa_dataset_id:
            from kappa_dataset_mapping import datasets_of_user
            owner = _dataset_owner(run.kappa_dataset_id)      # user id from the mapping
            allowed_datasets = datasets_of_user(owner) if owner else {run.kappa_dataset_id}
            all_records = find_by_bids_id(patient_id, {run.kappa_dataset_id}) \
                or find_by_bids_subject(patient_id, {run.kappa_dataset_id})
            # Resolve to the real person, then widen to the account's datasets.
            if all_records:
                original = all_records[0]["original_patient_id"]
                all_records = [r for r in find_by_patient_id(original)
                               if r.get("kappa_dataset_id") in allowed_datasets
                               or r.get("kappa_dataset_id") is None]
```

Keep the existing unscoped resolution as the fallback when `run_id` is absent, so old links keep working. Add the small helper:

```python
def _dataset_owner(dataset_id: int) -> Optional[int]:
    """Which Kappa user owns a dataset, per configs/kappa_datasets.yaml."""
    from kappa_dataset_mapping import _load_mapping
    for key, value in (_load_mapping().get("datasets") or {}).items():
        if int(value) == int(dataset_id):
            return int(str(key).split(":", 1)[0])
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest backend/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_dataset_mapping.py backend/patient_registry.py backend/app.py \
        backend/test_longitudinal_scoping.py backend/test_kappa_dataset_mapping.py
git commit -m "feat(registry): resolve a patient within a dataset, then join across the account"
```

---

### Task 7: Pass the run through to the timeline on the frontend

**Files:**
- Modify: `frontend/src/services/api.js:99-115` (`getLongitudinalReport`, `getLongitudinalDiff`)
- Modify: `frontend/src/components/LongitudinalTimeline.jsx:1-25`
- Modify: `frontend/src/components/ClinicalReportContent.jsx:689`

**Interfaces:**
- Consumes: the `run_id` query parameter from Task 6.
- Produces: nothing for later tasks.

- [ ] **Step 1: Make the change**

`api.js` — both functions take and forward `runId`:

```js
export const getLongitudinalReport = async (patientId, lesionType = 'multiple_sclerosis', runId = null) => {
  const response = await apiClient.get(`/longitudinal/${patientId}`, {
    params: { lesion_type: lesionType, ...(runId ? { run_id: runId } : {}) },
  });
  return response.data;
};
```

`LongitudinalTimeline.jsx` — accept `runId` and pass it into both calls:

```js
const LongitudinalTimeline = ({ patientId, lesionType, runId = null }) => {
  ...
      getLongitudinalReport(patientId, lesionType, runId),
      getLongitudinalDiff(patientId, lesionType, runId),
```

`ClinicalReportContent.jsx:689` — it already has `runId` in scope:

```jsx
            <LongitudinalTimeline patientId={patientId} lesionType="multiple_sclerosis" runId={runId} />
```

- [ ] **Step 2: Verify the build and lint are clean**

Run: `cd frontend && npx eslint src/services/api.js src/components/LongitudinalTimeline.jsx src/components/ClinicalReportContent.jsx && npm run build`
Expected: no NEW errors (`ClinicalReportContent.jsx` has pre-existing warnings; compare against `git stash` if unsure), build succeeds

- [ ] **Step 3: Verify in the browser**

Rebuild the image (`docker compose --profile full build web && docker compose --profile full up -d`), open an MS clinical report from run history and confirm the timeline still renders with its sessions. Check the request in DevTools carries `run_id`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.js frontend/src/components/LongitudinalTimeline.jsx \
        frontend/src/components/ClinicalReportContent.jsx
git commit -m "feat(frontend): send the run with longitudinal requests so the patient is unambiguous"
```

---

### Task 8: Migrate existing allocations into dataset scopes

**Files:**
- Create: `scripts/migrate_bids_allocation_scopes.py`
- Test: `test_migrate_bids_allocation_scopes.py` (repo root)

**Interfaces:**
- Consumes: the scope helpers from Task 1.
- Produces: `find_conflicts(db_path) -> list[str]`, `migrate(db_path, dry_run: bool = True) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# test_migrate_bids_allocation_scopes.py
import sqlite3
import pytest
from scripts.migrate_bids_allocation_scopes import find_conflicts, migrate


def _db(tmp_path, registry_rows, allocation_rows):
    path = tmp_path / "brain_lesion.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE patient_registry (
        bids_id TEXT, original_patient_id TEXT, lesion_type TEXT, kappa_dataset_id INTEGER)""")
    con.executemany("INSERT INTO patient_registry VALUES (?,?,?,?)", registry_rows)
    con.execute("""CREATE TABLE bids_patient_allocation (
        lesion_type TEXT, original_patient_id TEXT, bids_id TEXT, created_at TEXT,
        PRIMARY KEY (lesion_type, original_patient_id))""")
    con.executemany("INSERT INTO bids_patient_allocation VALUES (?,?,?,'2026-01-01')",
                    allocation_rows)
    con.commit(); con.close()
    return path


def test_uploaded_patients_move_to_their_dataset(tmp_path):
    db = _db(tmp_path,
             [("sub-005_ses-001", "P100", "glioblastoma", 249)],
             [("glioblastoma", "P100", "sub-005")])

    result = migrate(db, dry_run=False)

    con = sqlite3.connect(db)
    row = con.execute("SELECT scope, bids_id FROM bids_patient_allocation "
                      "WHERE original_patient_id='P100'").fetchone()
    assert row == ("ds:249", "sub-005")
    assert result["moved_to_datasets"] == 1


def test_never_uploaded_patients_go_to_the_local_scope(tmp_path):
    db = _db(tmp_path, [], [("glioblastoma", "P900", "sub-300")])

    migrate(db, dry_run=False)

    con = sqlite3.connect(db)
    row = con.execute("SELECT scope, bids_id FROM bids_patient_allocation "
                      "WHERE original_patient_id='P900'").fetchone()
    assert row == ("local:glioblastoma", "sub-300")


def test_no_number_changes(tmp_path):
    db = _db(tmp_path,
             [("sub-005_ses-001", "P100", "glioblastoma", 249),
              ("sub-007_ses-001", "P200", "glioblastoma", 266)],
             [("glioblastoma", "P100", "sub-005"), ("glioblastoma", "P200", "sub-007")])

    migrate(db, dry_run=False)

    con = sqlite3.connect(db)
    got = dict(con.execute("SELECT original_patient_id, bids_id FROM bids_patient_allocation"))
    assert got == {"P100": "sub-005", "P200": "sub-007"}


def test_conflict_inside_one_dataset_aborts(tmp_path):
    """Two people under one number in the same dataset cannot be split
    automatically — stop and report instead of guessing."""
    db = _db(tmp_path,
             [("sub-001_ses-001", "P100", "glioblastoma", 249),
              ("sub-001_ses-001", "P200", "glioblastoma", 249)],
             [("glioblastoma", "P100", "sub-001")])

    conflicts = find_conflicts(db)
    assert conflicts

    with pytest.raises(SystemExit):
        migrate(db, dry_run=False)


def test_collision_across_datasets_is_not_a_conflict(tmp_path):
    """This is the real sub-003 case: same number, different datasets, two
    people — exactly what dataset scoping is meant to separate."""
    db = _db(tmp_path,
             [("sub-003_ses-001", "P100", "glioblastoma", 133),
              ("sub-003_ses-001", "P200", "glioblastoma", 249)],
             [])

    assert find_conflicts(db) == []


def test_dry_run_changes_nothing(tmp_path):
    db = _db(tmp_path,
             [("sub-005_ses-001", "P100", "glioblastoma", 249)],
             [("glioblastoma", "P100", "sub-005")])

    migrate(db, dry_run=True)

    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(bids_patient_allocation)")]
    assert "scope" not in cols
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest test_migrate_bids_allocation_scopes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.migrate_bids_allocation_scopes'`

- [ ] **Step 3: Write the migration**

`scripts/migrate_bids_allocation_scopes.py`:

```python
"""
Move BIDS allocations from per-lesion-type numbering to per-dataset numbering.

Numbers are never changed: a number already issued is an entity name in Kappa.
Rows are only re-filed — into the dataset they were uploaded to, or into the
local scope when they were never uploaded. The old table is kept under
bids_patient_allocation_legacy so the change can be undone.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "backend" / "data" / "brain_lesion.db"


def _subject(bids_id: str) -> str:
    """'sub-001_ses-002' -> 'sub-001'."""
    return (bids_id or "").split("_", 1)[0]


def find_conflicts(db_path) -> list:
    """Problems that make an automatic move unsafe, as readable lines.

    Only collisions INSIDE one dataset matter. The same number in two different
    datasets is exactly what this migration separates, not a conflict.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT kappa_dataset_id, bids_id, original_patient_id "
            "FROM patient_registry WHERE kappa_dataset_id IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    by_number, by_person = {}, {}
    for dataset_id, bids_id, original in rows:
        by_number.setdefault((dataset_id, _subject(bids_id)), set()).add(original)
        by_person.setdefault((dataset_id, original), set()).add(_subject(bids_id))

    conflicts = []
    for (dataset_id, subject), people in sorted(by_number.items()):
        if len(people) > 1:
            conflicts.append(
                f"dataset {dataset_id}: {subject} is held by {len(people)} patients "
                f"({', '.join(sorted(people))})")
    for (dataset_id, original), subjects in sorted(by_person.items()):
        if len(subjects) > 1:
            conflicts.append(
                f"dataset {dataset_id}: patient {original} holds {len(subjects)} numbers "
                f"({', '.join(sorted(subjects))})")
    return conflicts


def migrate(db_path, dry_run: bool = True) -> dict:
    """Re-file every allocation into its scope. Aborts on any conflict."""
    conflicts = find_conflicts(db_path)
    if conflicts:
        print("Конфликты внутри датасета — миграция остановлена:")
        for line in conflicts:
            print("  -", line)
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        legacy = conn.execute(
            "SELECT lesion_type, original_patient_id, bids_id, created_at "
            "FROM bids_patient_allocation"
        ).fetchall()
        registry = {
            (lesion, original): dataset_id
            for dataset_id, original, lesion in conn.execute(
                "SELECT kappa_dataset_id, original_patient_id, lesion_type "
                "FROM patient_registry WHERE kappa_dataset_id IS NOT NULL")
        }
        floors = {}
        for dataset_id, bids_id in conn.execute(
            "SELECT kappa_dataset_id, bids_id FROM patient_registry "
            "WHERE kappa_dataset_id IS NOT NULL"
        ):
            try:
                number = int(_subject(bids_id).split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            floors[dataset_id] = max(floors.get(dataset_id, 0), number)

        planned = []
        for lesion, original, bids_id, created_at in legacy:
            dataset_id = registry.get((lesion, original))
            scope = f"ds:{dataset_id}" if dataset_id else f"local:{lesion}"
            planned.append((scope, original, bids_id, created_at))

        result = {
            "moved_to_datasets": sum(1 for s, *_ in planned if s.startswith("ds:")),
            "moved_to_local": sum(1 for s, *_ in planned if s.startswith("local:")),
            "floors": len(floors),
        }
        if dry_run:
            print("Сухой прогон:", result)
            return result

        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ALTER TABLE bids_patient_allocation "
                     "RENAME TO bids_patient_allocation_legacy")
        conn.execute("""
            CREATE TABLE bids_patient_allocation (
                scope                TEXT NOT NULL,
                original_patient_id  TEXT NOT NULL,
                bids_id              TEXT NOT NULL,
                created_at           TEXT NOT NULL,
                PRIMARY KEY (scope, original_patient_id),
                UNIQUE (scope, bids_id)
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bids_scope_floor (
                scope TEXT PRIMARY KEY,
                floor INTEGER NOT NULL
            )""")
        conn.executemany(
            "INSERT INTO bids_patient_allocation "
            "(scope, original_patient_id, bids_id, created_at) VALUES (?,?,?,?)",
            planned)
        conn.executemany(
            "INSERT OR REPLACE INTO bids_scope_floor (scope, floor) VALUES (?,?)",
            [(f"ds:{dataset_id}", floor) for dataset_id, floor in floors.items()])
        conn.commit()
        print("Готово:", result)
        return result
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--apply", action="store_true",
                        help="Without it the script only reports what it would do")
    args = parser.parse_args()
    migrate(args.db, dry_run=not args.apply)


if __name__ == "__main__":
    main()
```

`scripts/__init__.py` already exists, so `from scripts.migrate_bids_allocation_scopes import ...` works from the repo root.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest test_migrate_bids_allocation_scopes.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Dry-run against a copy of the real database**

```bash
cp backend/data/brain_lesion.db /tmp/brain_lesion_copy.db
source venv/bin/activate
python scripts/migrate_bids_allocation_scopes.py --db /tmp/brain_lesion_copy.db
python scripts/migrate_bids_allocation_scopes.py --db /tmp/brain_lesion_copy.db --apply
```

Expected: no conflicts; 113 GBM + 4 MS allocations land in `ds:*`; the rest of the 317 GBM numbers land in `local:glioblastoma`; spot-check that a known patient kept its number.

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_bids_allocation_scopes.py test_migrate_bids_allocation_scopes.py
git commit -m "feat(bids): migrate existing allocations into per-dataset scopes"
```

---

### Task 9: Document the change

**Files:**
- Modify: `CLAUDE.md` (Gotchas), `KNOWN_ISSUES.md` (KI-057), `research`-style DEVLOG not needed here

- [ ] **Step 1: Write the notes**

`CLAUDE.md`, under Gotchas:

```markdown
- Номера пациентов (`sub-XXX`) выдаются в рамках датасета Kappa, а не глобально.
  Один и тот же человек в двух датасетах — это два разных номера; искать его
  нужно по датасету запуска (`pipeline_runs.kappa_dataset_id`), а склеивать
  сессии — по `original_patient_id`. Источник правды о занятых номерах —
  сам датасет в Kappa, локальная таблица лишь кэш.
```

`KNOWN_ISSUES.md`, in KI-057: note that recommendation 1 ("проверять доступность датасета до запуска пайплайна") is implemented — the dataset is resolved at run start, so a permissions error surfaces before the pipeline runs.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md KNOWN_ISSUES.md
git commit -m "docs: per-dataset subject numbering; KI-057 recommendation 1 implemented"
```

---

## Verification before merge

1. `source venv/bin/activate && python -m pytest backend/ -q` — all green.
2. `python -m pytest $(ls test_*.py | grep -v test_stage05_fixes.py) -q` — no NEW failures. Ten stage-01/03/04 tests and `test_stage05_fixes.py` already fail on `main`; compare against a clean worktree rather than assuming.
3. `cd frontend && npm run build` — succeeds.
4. Manual, with the stack rebuilt: a run into a dataset that already holds data continues after its last number; a run into a fresh dataset starts at `sub-001`; an MS clinical report timeline still shows every session.
5. Back up `backend/data/brain_lesion.db` before running the migration for real.
