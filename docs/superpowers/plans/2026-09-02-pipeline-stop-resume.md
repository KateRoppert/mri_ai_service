# Pipeline Stop & Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator stop a running pipeline from the UI, keeping results already produced, and resume a stopped run without recomputing them.

**Architecture:** An in-memory registry in `PipelineManager` maps `run_id` to the running `Popen`, so a stop endpoint can reach it and kill the whole process group via the existing `_kill_process_tree`. Stopping is immediate; the truncated files it leaves are handled by replacing the stages' "file exists ⇒ done" test with a completeness check that compares the gzip trailer against the size implied by the NIfTI header. Resume reuses the existing `requeue` endpoint, extended to compare the stopped run's retained config snapshot against current settings before proceeding.

**Tech Stack:** FastAPI, SQLAlchemy (SQLite), pytest, React 19 + Ant Design, nibabel, numpy.

**Spec:** `docs/superpowers/specs/2026-09-02-pipeline-stop-resume-design.md`

## Global Constraints

- Python comments and docstrings in English; UI copy in Russian (project convention).
- Tests live beside the code: `backend/test_*.py` for backend, `tests/preprocessing_steps/` for pipeline helpers (that directory is in `.gitignore` — add with `git add -f`).
- SQLite migrations follow the existing pattern in `backend/database.py`: a `_migrate_add_<column>()` function guarded by `PRAGMA table_info`, called from `init_db()`.
- Never write a bare `except:` — catch specific exceptions or `Exception`.
- Do not raise stage worker counts; RAM limits are tuned (KI-042).
- Frontend: run `npm run lint` in `frontend/` before committing frontend changes.

---

### Task 1: NIfTI completeness check

A truncated `.nii.gz` passes `nibabel.load()` — the header is intact and only the data is cut short (verified during design). Detection therefore has to look at the end of the file, not the beginning.

**Files:**
- Create: `utils/nifti_integrity.py`
- Test: `tests/preprocessing_steps/test_nifti_integrity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `is_complete_nifti(path: Path | str) -> bool` — True when the file exists and its data section is fully present. Used by Task 2.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for utils/nifti_integrity.py."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.nifti_integrity import is_complete_nifti


@pytest.fixture
def complete_nifti(tmp_path):
    """A small but structurally real .nii.gz."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    path = tmp_path / "complete.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def test_complete_file_passes(complete_nifti):
    assert is_complete_nifti(complete_nifti) is True


def test_truncated_file_fails(complete_nifti, tmp_path):
    # The exact failure a killed process leaves behind: the write stopped
    # partway. Note nibabel.load() still succeeds on this — the header
    # survived — which is why existence and header checks are insufficient.
    truncated = tmp_path / "truncated.nii.gz"
    blob = complete_nifti.read_bytes()
    truncated.write_bytes(blob[: len(blob) // 2])

    assert is_complete_nifti(truncated) is False


def test_empty_file_fails(tmp_path):
    empty = tmp_path / "empty.nii.gz"
    empty.touch()

    assert is_complete_nifti(empty) is False


def test_missing_file_fails(tmp_path):
    assert is_complete_nifti(tmp_path / "nope.nii.gz") is False


def test_garbage_file_fails(tmp_path):
    garbage = tmp_path / "garbage.nii.gz"
    garbage.write_bytes(b"this is not a nifti file at all")

    assert is_complete_nifti(garbage) is False


def test_uncompressed_nifti_supported(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    path = tmp_path / "plain.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

    assert is_complete_nifti(path) is True


def test_truncated_uncompressed_nifti_fails(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    path = tmp_path / "plain.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    blob = path.read_bytes()
    path.write_bytes(blob[: len(blob) // 2])

    assert is_complete_nifti(path) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && python -m pytest tests/preprocessing_steps/test_nifti_integrity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.nifti_integrity'`

- [ ] **Step 3: Write the implementation**

```python
"""
Is a NIfTI file complete, or was it truncated mid-write?

Pipeline stages decide "already processed, skip it" from the presence of an
output file. That test is wrong for any file whose writer died partway
through — a stopped run, a full disk, an OOM kill — because a truncated
.nii.gz still has a valid header and still satisfies exists(). nibabel.load()
does not help either: it reads the header and returns happily.

The end of the file is where the truth is. A gzip stream records the
uncompressed size of its payload in the last four bytes, and the NIfTI header
states how many bytes of image data there should be; if the two disagree, the
file is not whole.
"""

import logging
import struct
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# NIfTI-1 puts image data at byte 352 when vox_offset is left at 0.
_NIFTI1_HEADER_BYTES = 352

# gzip's ISIZE field is the payload size modulo 2**32, so this check cannot
# distinguish a 4 GiB file from an empty one. Our volumes are ~30 MB; refuse
# to guess above the limit rather than return a confident wrong answer.
_ISIZE_MODULUS = 2 ** 32

# Header extensions legitimately sit between the header and the data, so the
# trailer may exceed the computed minimum by a little. A real truncation is
# off by megabytes, never by kilobytes.
_EXTENSION_SLACK_BYTES = 65536


def is_complete_nifti(path: Union[Path, str]) -> bool:
    """
    True if `path` is a NIfTI file whose image data is fully present.

    Errs toward False: an unreadable or unrecognisable file is reported
    incomplete. Recomputing a good file costs minutes; trusting a bad one
    puts a wrong number in a clinical report.
    """
    path = Path(path)

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
    except OSError as e:
        logger.debug("cannot stat %s: %s", path, e)
        return False

    try:
        img = nib.load(str(path))
        header = img.header
        data_bytes = int(np.prod(img.shape)) * header.get_data_dtype().itemsize
        # vox_offset is 0 in files written without extensions; the data then
        # starts right after the fixed-size header.
        data_start = int(header["vox_offset"]) or _NIFTI1_HEADER_BYTES
    except Exception as e:
        # Unreadable header — corrupt, empty, or not a NIfTI at all.
        logger.debug("cannot read NIfTI header of %s: %s", path, e)
        return False

    expected = data_start + data_bytes

    if path.name.endswith(".gz"):
        if expected >= _ISIZE_MODULUS:
            logger.warning(
                "%s is larger than gzip's 4 GiB ISIZE field can describe — "
                "cannot verify completeness, assuming complete", path,
            )
            return True
        try:
            with open(path, "rb") as f:
                f.seek(-4, 2)
                isize = struct.unpack("<I", f.read(4))[0]
        except (OSError, struct.error) as e:
            logger.debug("cannot read gzip trailer of %s: %s", path, e)
            return False
        complete = expected <= isize <= expected + _EXTENSION_SLACK_BYTES
    else:
        complete = path.stat().st_size >= expected

    if not complete:
        logger.warning("incomplete NIfTI (truncated write?): %s", path)
    return complete
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest tests/preprocessing_steps/test_nifti_integrity.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add utils/nifti_integrity.py
git add -f tests/preprocessing_steps/test_nifti_integrity.py
git commit -m "feat(utils): detect truncated NIfTI files

Stages treat an output file's existence as proof it is complete, which is
wrong for any file whose writer died partway through. A truncated .nii.gz
keeps a valid header, so nibabel.load() succeeds on it and exists() has
never had a chance.

Checks the end of the file instead: gzip records its payload size in the
last four bytes, and the NIfTI header implies what that size should be.
Measured at 0.076 ms per file, against 56 ms for a full data read."
```

---

### Task 2: Use the completeness check in stage skip logic

**Files:**
- Modify: `scripts/05_preprocessing.py` (`check_subject_processed`, ~line 244)
- Modify: `scripts/04_assess_quality.py:350` and `:476`
- Modify: `scripts/06_segmentation.py:662`
- Test: `tests/preprocessing_steps/test_skip_completeness.py`

**Interfaces:**
- Consumes: `is_complete_nifti(path) -> bool` from Task 1.
- Produces: nothing new; changes existing skip behaviour.

- [ ] **Step 1: Read the three call sites**

```bash
sed -n '244,300p' scripts/05_preprocessing.py
sed -n '345,355p' scripts/04_assess_quality.py
sed -n '470,480p' scripts/04_assess_quality.py
sed -n '655,670p' scripts/06_segmentation.py
```

Note for each whether the checked path is a NIfTI (`.nii`/`.nii.gz`) or a JSON report — Stage 04 writes JSON reports, which the NIfTI check does not apply to. Stage 04's skip therefore keeps `exists()` for its report file but must additionally verify the *input* volume it would otherwise skip over.

- [ ] **Step 2: Write the failing test**

```python
"""Stage skip logic must not accept a truncated file as finished work."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

importlib = __import__("importlib")
stage05 = importlib.import_module("05_preprocessing")


def _write_nifti(path, truncate=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(8, 8, 8).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    if truncate:
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])


def test_complete_subject_is_skipped(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    for base in (in_dir, out_dir):
        _write_nifti(base / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz")

    is_processed, missing = stage05.check_subject_processed(
        in_dir, out_dir, "sub-001", "ses-001", ["t1"]
    )

    assert is_processed is True
    assert missing == []


def test_truncated_output_is_not_skipped(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_nifti(in_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz")
    _write_nifti(
        out_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz",
        truncate=True,
    )

    is_processed, missing = stage05.check_subject_processed(
        in_dir, out_dir, "sub-001", "ses-001", ["t1"]
    )

    assert is_processed is False, "a truncated output must be recomputed"
    assert "t1" in missing
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/preprocessing_steps/test_skip_completeness.py -v`
Expected: `test_truncated_output_is_not_skipped` FAILS — current code returns True because the file exists.

- [ ] **Step 4: Change Stage 05's check**

In `scripts/05_preprocessing.py`, add the import near the other `utils` imports:

```python
from utils.nifti_integrity import is_complete_nifti
```

Then in `check_subject_processed`, replace the existence test on each output file (around line 288, `if not output_file.exists():`) with:

```python
        # A file that exists may still be truncated — a run stopped mid-write
        # leaves exactly that. Treat incomplete output as not done, or the
        # damaged volume is carried into segmentation and the report.
        if not is_complete_nifti(output_file):
            missing_on_output.append(modality)
            continue
```

- [ ] **Step 5: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/preprocessing_steps/test_skip_completeness.py -v`
Expected: PASS, 2 tests

- [ ] **Step 6: Apply the same change to stages 04 and 06**

`scripts/06_segmentation.py:662` — the checked path is a mask NIfTI:

```python
            if is_complete_nifti(session.output_mask_path):
```

`scripts/04_assess_quality.py` — the skip is keyed on a JSON report, which this check does not cover; leave `report_file.exists()` as is, and add above it a guard on the input volume so a truncated input is not silently assessed:

```python
        if not is_complete_nifti(nifti_path):
            logger.warning("Skipping %s — input volume is incomplete", nifti_path.name)
            return False
```

Add `from utils.nifti_integrity import is_complete_nifti` to both files.

- [ ] **Step 7: Run the full stage test suites**

Run: `source venv/bin/activate && python -m pytest tests/preprocessing_steps/ -q`
Expected: PASS, all tests including the ones from Task 1

- [ ] **Step 8: Commit**

```bash
git add scripts/05_preprocessing.py scripts/04_assess_quality.py scripts/06_segmentation.py
git add -f tests/preprocessing_steps/test_skip_completeness.py
git commit -m "fix(stages): do not skip work whose output is truncated

skip_existing accepted any file that exists as finished work. A run killed
mid-write leaves a truncated volume that passes that test, so the next run
skipped the patient and carried the damaged file into segmentation and the
clinical report. Stages now require the output to be complete, not merely
present."
```

---

### Task 3: `stopped` run state

**Files:**
- Modify: `backend/models.py:11-16` (`PipelineStatus`)
- Modify: `backend/database.py` (model columns + migration, following `_migrate_add_parent_run_id`)
- Test: `backend/test_stopped_status_migration.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PipelineStatus.STOPPED = "stopped"`; `PipelineRun.stopped_at_stage: int | None`; `PipelineRun.stopped_by: str | None`. Used by Tasks 5 and 7.

- [ ] **Step 1: Write the failing test**

```python
"""The stopped status and its columns must exist and survive migration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import PipelineStatus


def test_stopped_status_exists():
    assert PipelineStatus.STOPPED.value == "stopped"


def test_stopped_is_distinct_from_failed():
    # A stopped run is a deliberate act, not a defect. Conflating them would
    # corrupt any reading of how often the pipeline actually fails.
    assert PipelineStatus.STOPPED != PipelineStatus.FAILED


def test_migration_adds_columns(tmp_path, monkeypatch):
    import sqlalchemy

    db_file = tmp_path / "test.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db_file}")

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE pipeline_runs (run_id VARCHAR PRIMARY KEY, status VARCHAR)"
        ))
        conn.commit()

    import database
    monkeypatch.setattr(database, "engine", engine)
    database._migrate_add_stop_columns()

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]

    assert "stopped_at_stage" in cols
    assert "stopped_by" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_stopped_status_migration.py -v`
Expected: FAIL — `AttributeError: STOPPED`

- [ ] **Step 3: Add the status and columns**

`backend/models.py`:

```python
class PipelineStatus(str, Enum):
    """Статусы выполнения pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # Deliberately ended by an operator — distinct from FAILED, which is a
    # defect. Mixing them would make the failure rate meaningless.
    STOPPED = "stopped"
```

`backend/database.py`, on the `PipelineRun` model beside `config_path`:

```python
    stopped_at_stage = Column(Integer, nullable=True)
    stopped_by = Column(String, nullable=True)
```

and the migration, following the existing pattern:

```python
def _migrate_add_stop_columns():
    """Add stopped_at_stage / stopped_by to pipeline_runs if not present."""
    import sqlalchemy
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]
        if 'stopped_at_stage' not in cols:
            conn.execute(sqlalchemy.text(
                "ALTER TABLE pipeline_runs ADD COLUMN stopped_at_stage INTEGER"
            ))
        if 'stopped_by' not in cols:
            conn.execute(sqlalchemy.text(
                "ALTER TABLE pipeline_runs ADD COLUMN stopped_by VARCHAR"
            ))
        conn.commit()
```

Call it from `init_db()` beside the existing migrations.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_stopped_status_migration.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add backend/models.py backend/database.py backend/test_stopped_status_migration.py
git commit -m "feat(db): add stopped run state

A stopped run is not a failed one: how often runs fail is a health signal,
how often an operator stops them is not, and one status for both would make
the history unreadable. Records which stage was interrupted and who stopped
it — the latter matters now that more than one person can start runs."
```

---

### Task 4: Process registry

**Files:**
- Modify: `backend/pipeline_manager.py` (registry + `start_pipeline` ~line 236)
- Modify: `backend/app.py` (`run_pipeline_background`, deregister in `finally`)
- Test: `backend/test_process_registry.py`

**Interfaces:**
- Consumes: nothing.
- Produces: on `PipelineManager` — `register_process(run_id: str, process: subprocess.Popen) -> None`, `get_process(run_id: str) -> subprocess.Popen | None`, `unregister_process(run_id: str) -> None`. Used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
"""The registry is how a stop request reaches a running process."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def test_registered_process_is_retrievable():
    mgr = PipelineManager()
    proc = MagicMock()

    mgr.register_process("run-1", proc)

    assert mgr.get_process("run-1") is proc


def test_unknown_run_returns_none():
    assert PipelineManager().get_process("never-registered") is None


def test_unregister_removes_it():
    mgr = PipelineManager()
    mgr.register_process("run-1", MagicMock())

    mgr.unregister_process("run-1")

    assert mgr.get_process("run-1") is None


def test_unregister_is_idempotent():
    # Called from a finally block that also runs when the process never
    # started; it must not raise there.
    PipelineManager().unregister_process("never-registered")


def test_registry_is_shared_across_instances():
    # app.py constructs its own PipelineManager; the stop endpoint must see
    # the process that the start path registered.
    PipelineManager().register_process("run-shared", MagicMock())

    assert PipelineManager().get_process("run-shared") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_process_registry.py -v`
Expected: FAIL — `AttributeError: 'PipelineManager' object has no attribute 'register_process'`

- [ ] **Step 3: Implement the registry**

In `backend/pipeline_manager.py`, at module level:

```python
# run_id -> Popen of the orchestrator driving that run.
#
# Module-level, not per-instance: app.py builds its own PipelineManager per
# request path, and the stop endpoint has to see what the start path
# registered. In-memory is sufficient because the pipeline is a child of the
# backend process inside the same container — a backend restart kills the
# run too, so no run ever outlives this dict.
_RUNNING_PROCESSES: Dict[str, subprocess.Popen] = {}
```

On the class:

```python
    def register_process(self, run_id: str, process: subprocess.Popen) -> None:
        """Record a started run so a stop request can find it."""
        _RUNNING_PROCESSES[run_id] = process
        logger.info("Registered process for run %s (pid=%s)", run_id, process.pid)

    def get_process(self, run_id: str) -> Optional[subprocess.Popen]:
        """The running process for `run_id`, or None if it is not running."""
        return _RUNNING_PROCESSES.get(run_id)

    def unregister_process(self, run_id: str) -> None:
        """Forget a run. Safe to call for a run that was never registered."""
        if _RUNNING_PROCESSES.pop(run_id, None) is not None:
            logger.info("Unregistered process for run %s", run_id)
```

In `start_pipeline`, right after the `Popen(...)` call succeeds:

```python
            self.register_process(run_id, process)
```

- [ ] **Step 4: Deregister when the run ends**

In `backend/app.py::run_pipeline_background`, wrap the existing body so every exit path clears the entry:

```python
    finally:
        # Normal completion, failure and timeout all land here.
        pipeline_manager.unregister_process(run_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_process_registry.py -v`
Expected: PASS, 5 tests

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/app.py backend/test_process_registry.py
git commit -m "feat(backend): track running pipeline processes

The Popen object lived only inside run_pipeline_background, so nothing
outside could reach a running pipeline — the timeout path was the only
place that could kill one. A module-level registry makes the process
reachable by run_id."
```

---

### Task 5: Stop endpoint

**Files:**
- Modify: `backend/app.py` (new endpoint; `_kill_process_tree` already exists at ~line 162)
- Test: `backend/test_app_stop_endpoint.py`

**Interfaces:**
- Consumes: registry from Task 4; `PipelineStatus.STOPPED` and columns from Task 3.
- Produces: `POST /api/pipeline-runs/{run_id}/stop` returning `{run_id, status, stopped_at_stage}`. Used by Task 8.

- [ ] **Step 1: Write the failing tests**

```python
"""Stop endpoint: kill the run, record why, never lie about the outcome."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _run(run_id="run-1", status="running", current_stage=5):
    return SimpleNamespace(
        run_id=run_id,
        status=status,
        current_stage=current_stage,
        output_path="/out",
        input_path="/in",
        lesion_type="glioblastoma",
    )


def test_404_when_run_unknown():
    with patch("app.get_pipeline_run", return_value=None):
        response = client.post("/api/pipeline-runs/nope/stop")

    assert response.status_code == 404


def test_409_when_run_already_finished():
    with patch("app.get_pipeline_run", return_value=_run(status="completed")):
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 409


def test_kills_process_group_and_records_state():
    proc = MagicMock()
    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=proc), \
         patch("app._kill_process_tree") as kill, \
         patch("app.update_pipeline_run") as update:
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    kill.assert_called_once_with(proc)
    kwargs = update.call_args.kwargs
    assert kwargs["status"] == "stopped"
    assert kwargs["stopped_at_stage"] == 5


def test_marks_stopped_even_when_process_already_gone():
    # After a backend restart the DB can still say "running" while no process
    # exists. The process is provably dead, so stop corrects the record
    # instead of failing.
    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=None), \
         patch("app._kill_process_tree") as kill, \
         patch("app.update_pipeline_run") as update:
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    kill.assert_not_called()
    assert update.call_args.kwargs["status"] == "stopped"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_app_stop_endpoint.py -v`
Expected: FAIL — 404 for every case (route not registered)

- [ ] **Step 3: Implement the endpoint**

In `backend/app.py`, beside the requeue endpoint:

```python
@app.post("/api/pipeline-runs/{run_id}/stop")
async def stop_pipeline_run(
    run_id: str,
    db: Session = Depends(get_db),
):
    """
    Остановить выполняющийся запуск, сохранив уже полученные результаты.

    Kills the whole process group immediately — see the design spec for why
    draining the current stage was rejected. Files being written at that
    moment are left truncated; the stages' completeness check (Task 2) makes
    sure they are recomputed rather than skipped on the next run.
    """
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    if run.status not in (PipelineStatus.PENDING, PipelineStatus.RUNNING):
        raise HTTPException(
            status_code=409,
            detail=f"Запуск уже завершён (статус: {run.status})",
        )

    process = pipeline_manager.get_process(run_id)
    if process is not None:
        _kill_process_tree(process)
        pipeline_manager.unregister_process(run_id)
        logger.info("Stopped run %s at stage %s", run_id, run.current_stage)
    else:
        # No process for a run the DB calls active: the backend was restarted,
        # which killed the pipeline with it. Correct the record rather than
        # reporting an error for something already true.
        logger.warning(
            "Run %s marked active but no process found — recording as stopped",
            run_id,
        )

    update_pipeline_run(
        db,
        run_id,
        status=PipelineStatus.STOPPED.value,
        stopped_at_stage=run.current_stage,
        completed_at=datetime.utcnow(),
    )

    # Push the new state to every open tab watching this run.
    await ws_manager.broadcast(run_id, {
        "type": "status",
        "status": PipelineStatus.STOPPED.value,
        "stopped_at_stage": run.current_stage,
    })

    return {
        "run_id": run_id,
        "status": PipelineStatus.STOPPED.value,
        "stopped_at_stage": run.current_stage,
    }
```

`ws_manager` is the module-level `WebSocketManager` already imported in
`app.py`; `broadcast(run_id, message)` is its existing fan-out method
(`backend/websocket_manager.py:52`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_app_stop_endpoint.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Verify against a real run**

```bash
docker compose --profile full up -d
# start a run through the UI, then:
curl -s -X POST http://localhost:8000/api/pipeline-runs/<run_id>/stop | python3 -m json.tool
ps aux | grep -c "[o]rchestrator.py"   # expected: 0 — the whole group is gone
```

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/test_app_stop_endpoint.py
git commit -m "feat(api): endpoint to stop a running pipeline

Kills the process group via the existing _kill_process_tree, records the
stage that was interrupted, and pushes the new state over WebSocket so open
tabs update without polling.

A run the database calls active but which has no process (the backend was
restarted, taking the pipeline with it) is marked stopped rather than
returning an error — the process is provably gone, so the record was wrong."
```

---

### Task 6: Retain and compare the config snapshot

**Files:**
- Modify: `backend/pipeline_manager.py` (`cleanup_runtime_config` ~line 1077, `create_runtime_config` ~line 116)
- Create: `backend/config_diff.py`
- Test: `backend/test_config_diff.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `diff_configs(snapshot: dict, current: dict) -> list[dict]` returning `[{"setting": str, "was": str, "now": str}, ...]` in operator-facing wording. Used by Task 7.

- [ ] **Step 1: Write the failing tests**

```python
"""Differences between a stopped run's settings and the current ones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_diff import diff_configs


def _config(method="bet", reference="t1"):
    return {
        "steps": [
            {"name": "skull_stripping", "enabled": True,
             "params": {"method": method, "reference_modality": reference}},
        ]
    }


def test_identical_configs_have_no_differences():
    assert diff_configs(_config(), _config()) == []


def test_changed_skull_stripper_is_reported():
    result = diff_configs(_config(method="bet"), _config(method="hdbet"))

    assert len(result) == 1
    assert result[0]["was"] == "bet"
    assert result[0]["now"] == "hdbet"
    assert "череп" in result[0]["setting"].lower()


def test_changed_reference_modality_is_reported():
    result = diff_configs(_config(reference="t1"), _config(reference="t1c"))

    assert len(result) == 1
    assert result[0]["was"] == "t1"


def test_multiple_differences_all_reported():
    result = diff_configs(_config("bet", "t1"), _config("hdbet", "t1c"))

    assert len(result) == 2


def test_missing_section_does_not_crash():
    assert diff_configs({}, _config()) != []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_config_diff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config_diff'`

- [ ] **Step 3: Implement the comparison**

```python
"""
Compare a stopped run's saved settings against the current ones.

Resuming adopts whatever the configuration says at that moment, so a run
stopped to change a setting and then resumed would process its remaining
patients differently from the ones already done — with nothing in the output
recording the split. This turns that difference into something the operator
sees before deciding.

Only settings that change results are compared; worker counts and paths do
not belong here.
"""

from typing import Any, Dict, List

# Config path -> operator-facing label. Keeping the list explicit (rather
# than diffing the whole document) keeps the dialog about things that alter
# results, not about incidental noise.
_WATCHED = {
    ("skull_stripping", "method"): "Удаление черепа",
    ("skull_stripping", "reference_modality"): "Опорная модальность (удаление черепа)",
    ("skull_stripping", "fallback_method"): "Запасной инструмент удаления черепа",
    ("registration", "registration_type"): "Тип регистрации",
    ("registration", "reference_modality"): "Опорная модальность (регистрация)",
    ("bias_correction", "shrink_factor"): "Коррекция поля: shrink factor",
    ("resampling", "output_resolution"): "Разрешение ресемплинга",
}


def _step_params(config: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    for step in (config or {}).get("steps", []) or []:
        if step.get("name") == step_name:
            return step.get("params") or {}
    return {}


def diff_configs(snapshot: Dict[str, Any],
                 current: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Result-affecting settings that differ, in the operator's words.

    Returns [] when nothing relevant changed, which is the signal to resume
    without asking.
    """
    differences: List[Dict[str, str]] = []

    for (step_name, param), label in _WATCHED.items():
        was = _step_params(snapshot, step_name).get(param)
        now = _step_params(current, step_name).get(param)
        if was != now:
            differences.append({
                "setting": label,
                "was": "не задано" if was is None else str(was),
                "now": "не задано" if now is None else str(now),
            })

    return differences
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_config_diff.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Keep the snapshot for stopped runs**

In `backend/pipeline_manager.py::cleanup_runtime_config`, keep the file when the run was stopped — the snapshot is what a resume compares against:

```python
    def cleanup_runtime_config(self, run_id: str, keep_for_debug: bool = False,
                               keep_as_snapshot: bool = False):
        """
        Remove a run's runtime config.

        keep_as_snapshot: retain it because the run was stopped and may be
        resumed — resume compares these settings against the current ones to
        catch a configuration change between the two halves of the work.
        """
        if keep_for_debug or keep_as_snapshot:
            return
```

In the stop endpoint from Task 5, record the snapshot path on the run so resume can find it:

```python
    update_pipeline_run(
        db,
        run_id,
        status=PipelineStatus.STOPPED.value,
        stopped_at_stage=run.current_stage,
        config_path=str(pipeline_manager.runtime_config_path(run_id)),
        completed_at=datetime.utcnow(),
    )
```

Add the accessor to `PipelineManager`:

```python
    def runtime_config_path(self, run_id: str) -> Path:
        """Where this run's runtime config lives."""
        return self.pipeline_root / "runtime_configs" / f"config_{run_id}.yaml"
```

- [ ] **Step 6: Run the backend suite**

Run: `cd backend && source ../venv/bin/activate && python -m pytest -q`
Expected: no new failures (`test_preprocessing_version.py::test_dataset_mapping` fails for an unrelated local config reason)

- [ ] **Step 7: Commit**

```bash
git add backend/config_diff.py backend/pipeline_manager.py backend/app.py backend/test_config_diff.py
git commit -m "feat(backend): keep and compare a stopped run's settings

create_runtime_config re-reads pipeline_config.yaml on every start and the
snapshot was deleted afterwards, so resuming silently adopted whatever the
settings said at that moment. A run stopped to change a setting and then
resumed would process its remaining patients differently from those already
done, with nothing recording the split.

Stopped runs keep their snapshot, and diff_configs reports result-affecting
differences in the operator's words."
```

---

### Task 7: Resume with a configuration guard

**Files:**
- Modify: `backend/app.py` (`requeue_pipeline_run` ~line 389)
- Test: `backend/test_app_resume_guard.py`

**Interfaces:**
- Consumes: `diff_configs` (Task 6), `PipelineStatus.STOPPED` (Task 3).
- Produces: `POST /api/pipeline-runs/{run_id}/requeue` accepting `{"use_snapshot": bool}`; returns 409 with `{"differences": [...]}` when settings changed and no choice was given. Used by Task 9.

- [ ] **Step 1: Write the failing tests**

```python
"""Resuming a stopped run must not silently mix two configurations."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _stopped_run(run_id="run-1"):
    return SimpleNamespace(
        run_id=run_id,
        status="stopped",
        input_path="/in",
        output_path="/out",
        lesion_type="glioblastoma",
        config_path="/configs/config_run-1.yaml",
        current_stage=5,
    )


def test_stopped_run_can_be_resumed():
    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={}), \
         patch("app.diff_configs", return_value=[]), \
         patch("app.create_pipeline_run", return_value=SimpleNamespace(
             run_id="new", input_path="/in", output_path="/out",
             lesion_type="glioblastoma", created_at=None)):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 200


def test_changed_settings_block_resume_and_are_reported():
    differences = [{"setting": "Удаление черепа", "was": "bet", "now": "hdbet"}]

    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={}), \
         patch("app.diff_configs", return_value=differences):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 409
    assert response.json()["detail"]["differences"] == differences


def test_use_snapshot_resumes_despite_differences():
    differences = [{"setting": "Удаление черепа", "was": "bet", "now": "hdbet"}]

    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={}), \
         patch("app.diff_configs", return_value=differences), \
         patch("app.create_pipeline_run", return_value=SimpleNamespace(
             run_id="new", input_path="/in", output_path="/out",
             lesion_type="glioblastoma", created_at=None)):
        response = client.post(
            "/api/pipeline-runs/run-1/requeue", json={"use_snapshot": True}
        )

    assert response.status_code == 200


def test_running_run_still_rejected():
    running = _stopped_run()
    running.status = "running"

    with patch("app.get_pipeline_run", return_value=running):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 409
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_app_resume_guard.py -v`
Expected: FAIL — the endpoint takes no body and does not compare configs

- [ ] **Step 3: Extend the endpoint**

Add a request model in `backend/models.py`:

```python
class RequeueRequest(BaseModel):
    """Опции повторного запуска."""
    use_snapshot: bool = Field(
        False,
        description="Использовать настройки остановленного запуска, а не текущие",
    )
```

Add a snapshot loader to `backend/app.py`:

```python
def load_config_snapshot(config_path: Optional[str]) -> dict:
    """Settings a stopped run was started with, or {} if not retained."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not read config snapshot %s: %s", path, e)
        return {}
```

In `requeue_pipeline_run`, after the existing active-path check and before creating the new run:

```python
    # A stopped run may be resumed, but not blindly: resuming adopts current
    # settings, so a configuration changed since the stop would split the
    # output between two behaviours with nothing recording it.
    if original_run.status == PipelineStatus.STOPPED and not body.use_snapshot:
        snapshot = load_config_snapshot(original_run.config_path)
        if snapshot:
            with open(PREPROCESSING_CONFIG, encoding="utf-8") as f:
                current = yaml.safe_load(f) or {}
            differences = diff_configs(snapshot, current)
            if differences:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "settings_changed",
                        "message": "Настройки изменились с момента остановки",
                        "differences": differences,
                    },
                )
```

Change the signature to accept the body:

```python
async def requeue_pipeline_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    body: RequeueRequest = RequeueRequest(),
    db: Session = Depends(get_db),
):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && source ../venv/bin/activate && python -m pytest test_app_resume_guard.py test_app_requeue_endpoint.py -v`
Expected: PASS — both the new tests and the existing requeue tests

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/models.py backend/test_app_resume_guard.py
git commit -m "feat(api): resume a stopped run, guarded against setting changes

Resume reuses requeue, which already reruns on the same paths and relies on
skip_existing. What it lacked was any notice that the settings changed since
the stop — the case an operator hits precisely when they stopped because
something about the run was wrong.

Identical settings resume without a prompt; differences return 409 with the
list, and use_snapshot resumes on the saved configuration instead."
```

---

### Task 8: Stop button in the UI

**Files:**
- Modify: `frontend/src/services/api.js`
- Modify: `frontend/src/components/ProgressMonitor.jsx` (buttons near line 273)

**Interfaces:**
- Consumes: `POST /api/pipeline-runs/{run_id}/stop` (Task 5).
- Produces: nothing for later tasks.

- [ ] **Step 1: Add the API call**

In `frontend/src/services/api.js`, beside the other pipeline calls:

```javascript
/**
 * Остановить выполняющийся запуск. Возвращает сводку: сколько пациентов
 * успели обработаться и на каком этапе прервались.
 */
export const stopPipelineRun = async (runId) => {
  const response = await api.post(`/pipeline-runs/${runId}/stop`);
  return response.data;
};
```

- [ ] **Step 2: Add the button and confirmation**

In `ProgressMonitor.jsx`, import `Modal` and `message` from `antd` and `stopPipelineRun` from the API module, then add beside the existing buttons:

```jsx
  const [stopping, setStopping] = useState(false);

  const handleStop = () => {
    Modal.confirm({
      title: 'Остановить обработку?',
      // The action is irreversible and may land an hour into a run, so the
      // dialog states what survives and what is lost rather than asking a
      // bare yes/no.
      content: (
        <div>
          <p>Уже обработанные пациенты сохранятся, их результаты останутся доступны.</p>
          <p>Текущий этап будет прерван — эти пациенты обработаются заново при возобновлении.</p>
        </div>
      ),
      okText: 'Остановить',
      okButtonProps: { danger: true },
      cancelText: 'Отмена',
      onOk: async () => {
        setStopping(true);
        try {
          const result = await stopPipelineRun(runId);
          message.success(
            `Обработка остановлена на этапе ${result.stopped_at_stage ?? '—'}`
          );
          setStatus('stopped');
        } catch (error) {
          message.error(
            error.response?.data?.detail || 'Не удалось остановить обработку'
          );
        } finally {
          setStopping(false);
        }
      },
    });
  };
```

and in the render, alongside the existing action buttons:

```jsx
        {(status === 'running' || status === 'pending') && (
          <Button danger onClick={handleStop} loading={stopping}>
            Остановить
          </Button>
        )}
```

- [ ] **Step 3: Lint**

Run: `cd frontend && npm run lint`
Expected: no new errors

- [ ] **Step 4: Verify by hand**

```bash
docker compose --profile full up -d
```
Start a run in the UI, press **Остановить**, confirm. Expected: dialog appears; after confirming the status becomes stopped, progress stops advancing, and `ps aux | grep [o]rchestrator.py` returns nothing.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/api.js frontend/src/components/ProgressMonitor.jsx
git commit -m "feat(ui): stop button for a running pipeline

Shown only while a run is active. Confirms first, stating what survives and
what will be recomputed — the action is irreversible and may land an hour
into a run."
```

---

### Task 9: Resume button and the differences dialog

**Files:**
- Modify: `frontend/src/services/api.js`
- Modify: `frontend/src/components/PipelineHistory.jsx`

**Interfaces:**
- Consumes: `POST /api/pipeline-runs/{run_id}/requeue` with `{use_snapshot}` (Task 7).
- Produces: nothing for later tasks.

- [ ] **Step 1: Extend the API call**

```javascript
/**
 * Возобновить остановленный запуск. Если настройки изменились с момента
 * остановки, бэкенд отвечает 409 со списком различий — их показывают
 * пользователю, и он выбирает, продолжать ли на сохранённых настройках.
 */
export const resumePipelineRun = async (runId, useSnapshot = false) => {
  const response = await api.post(`/pipeline-runs/${runId}/requeue`, {
    use_snapshot: useSnapshot,
  });
  return response.data;
};
```

- [ ] **Step 2: Add the button and the differences dialog**

```jsx
  const handleResume = async (runId) => {
    try {
      await resumePipelineRun(runId);
      message.success('Обработка возобновлена');
      refresh();
    } catch (error) {
      const detail = error.response?.data?.detail;
      if (error.response?.status === 409 && detail?.differences) {
        Modal.confirm({
          title: 'Настройки изменились с момента остановки',
          content: (
            <div>
              <p>После остановки изменилось следующее:</p>
              <ul>
                {detail.differences.map((d) => (
                  <li key={d.setting}>
                    {d.setting}: <b>{d.was}</b> → <b>{d.now}</b>
                  </li>
                ))}
              </ul>
              <p>
                Если продолжить на новых настройках, часть пациентов будет
                обработана иначе, чем остальные.
              </p>
            </div>
          ),
          okText: 'На прежних настройках',
          cancelText: 'Отмена',
          onOk: async () => {
            await resumePipelineRun(runId, true);
            message.success('Обработка возобновлена на сохранённых настройках');
            refresh();
          },
        });
      } else {
        message.error(detail || 'Не удалось возобновить обработку');
      }
    }
  };
```

and in the row actions:

```jsx
        {run.status === 'stopped' && (
          <Button size="small" onClick={() => handleResume(run.run_id)}>
            Возобновить
          </Button>
        )}
```

- [ ] **Step 3: Show the stopped state distinctly**

Wherever the history renders a status tag, add the stopped case so it does not read as a failure:

```jsx
  stopped: { color: 'orange', label: 'Остановлен' },
```

- [ ] **Step 4: Lint**

Run: `cd frontend && npm run lint`
Expected: no new errors

- [ ] **Step 5: Verify both paths by hand**

1. Stop a run, press **Возобновить** without touching anything — expected: resumes with no dialog, already-processed patients are skipped.
2. Stop a run, change `method` in `configs/preprocessing_config.yaml` from `hdbet` to `bet`, press **Возобновить** — expected: dialog lists "Удаление черепа: hdbet → bet".

- [ ] **Step 6: Commit**

```bash
git add frontend/src/services/api.js frontend/src/components/PipelineHistory.jsx
git commit -m "feat(ui): resume a stopped run

Resumes directly when nothing changed. When the backend reports that
settings differ from the stopped run's, the dialog lists them and offers to
continue on the saved ones — the case where resuming blindly would split the
output between two behaviours."
```

---

### Task 10: End-to-end verification and documentation

**Files:**
- Modify: `KNOWN_ISSUES.md` (KI-052 gains a note that stop reuses the group-kill)
- Modify: `CLAUDE.md` (stop/resume in the pipeline description)

- [ ] **Step 1: Full-cycle check on real data**

```bash
docker compose --profile full up -d
```

Start a run on a multi-patient folder. Once patient 2 is in progress, press **Остановить**. Then:

```bash
# nothing left running
ps aux | grep -c "[o]rchestrator.py"          # expected 0
# results of finished patients survived
ls <output>/preprocessed/                      # expected: completed patients present
# the interrupted patient's output is detected as incomplete
source venv/bin/activate && python -c "
from utils.nifti_integrity import is_complete_nifti
from pathlib import Path
for f in Path('<output>/preprocessed').rglob('*.nii.gz'):
    ok = is_complete_nifti(f)
    if not ok:
        print('incomplete (will be recomputed):', f)
"
```

- [ ] **Step 2: Resume and confirm no double work**

Press **Возобновить**. Expected in `{output}/logs/`: completed patients logged as skipped, the interrupted one recomputed, remaining ones processed.

- [ ] **Step 3: Run every test suite**

```bash
source venv/bin/activate
python -m pytest tests/preprocessing_steps/ -q
cd backend && python -m pytest -q && cd ..
cd services/gbm-seg/src && python -m pytest test_model_selection.py -q && cd ../../..
cd frontend && npm run lint
```

- [ ] **Step 4: Update the docs**

In `CLAUDE.md`, under the pipeline description, add that a run can be stopped from the UI and resumed, that stopping is immediate, and that resume compares settings against the stopped run's snapshot.

In `KNOWN_ISSUES.md`, append to KI-052 that the group-kill it introduced is now also the mechanism behind the stop endpoint.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md KNOWN_ISSUES.md
git commit -m "docs: stop and resume in project docs"
```
