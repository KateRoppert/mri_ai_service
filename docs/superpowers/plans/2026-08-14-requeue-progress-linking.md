# Requeue Progress Linking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a doctor requeues a pipeline run from the incomplete-patients review queue, the active-run screen switches to show the new run's live progress, with a banner linking back to the original run it continues from.

**Architecture:** A nullable `parent_run_id` column on `pipeline_runs`, set only by the requeue endpoint, exposed through the existing pipeline-status response. The frontend reuses its existing "a new run just started" handler for requeues too (same shape of response, same effect — activate it), and `ProgressMonitor` renders a small banner when the run it's tracking has a parent, linking to the history tab where the original run's full report set already lives (unchanged — everything is read from the shared `output_path`, not `run_id`).

**Tech Stack:** Python/FastAPI + SQLAlchemy backend (`backend/`), React 19 + Ant Design v6 frontend (`frontend/`), pytest.

## Global Constraints

- No merging/combining the two runs' stage timelines into one view — switch + banner-link only (per design decision during brainstorming).
- Only the immediate `parent_run_id` is tracked and shown — no full ancestor-chain resolution.
- No changes to any report-fetching endpoint — they're already keyed by `output_path`, not `run_id`.
- The DB migration must be safe against an existing database with real data (same pattern as the existing `_migrate_add_lesion_type` — `ALTER TABLE` guarded by a `PRAGMA table_info` check, called from `init_db()`).
- No frontend automated test framework exists in this project — frontend verification is manual browser testing only.

---

### Task 1: Backend — `parent_run_id` column, migration, `create_pipeline_run`

**Files:**
- Modify: `backend/database.py` (`PipelineRun` model at line 30, `init_db`/`_migrate_add_lesion_type` at lines 247-263, `create_pipeline_run` at line 92)
- Test: `backend/test_pipeline_run_parent_id.py` (new file)

**Interfaces:**
- Produces: `PipelineRun.parent_run_id: Optional[str]` (ORM attribute), `create_pipeline_run(db, input_path, output_path, lesion_type='glioblastoma', parent_run_id: Optional[str] = None) -> PipelineRun` — the new `parent_run_id` parameter is keyword-only in practice (all existing callers use keyword args for `lesion_type` already) and defaults to `None`, so every existing call site (`/api/pipeline/start`) is unaffected.

- [ ] **Step 1: Write the failing tests**

Create `backend/test_pipeline_run_parent_id.py`:

```python
"""
Tests for parent_run_id: links a requeued pipeline run back to the run it
continues from, so the frontend can show a "this continues run X" banner.
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent))

import database
from database import Base, create_pipeline_run


@pytest.fixture
def session():
    """Isolated in-memory SQLite session with the schema created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()


def test_create_pipeline_run_sets_parent_run_id(session):
    run = create_pipeline_run(
        session, input_path="/in", output_path="/out", parent_run_id="orig-run-id",
    )
    assert run.parent_run_id == "orig-run-id"


def test_create_pipeline_run_parent_run_id_defaults_to_none(session):
    run = create_pipeline_run(session, input_path="/in", output_path="/out")
    assert run.parent_run_id is None


def test_migrate_add_parent_run_id_adds_column_to_old_table():
    # Simulate a pre-migration database: a pipeline_runs table with no
    # parent_run_id column at all (raw CREATE TABLE, bypassing the ORM model
    # which already declares the column).
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE pipeline_runs "
            "(run_id VARCHAR PRIMARY KEY, input_path VARCHAR, output_path VARCHAR, status VARCHAR)"
        ))
        conn.commit()

    original_engine = database.engine
    database.engine = engine
    try:
        database._migrate_add_parent_run_id()
        with engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(pipeline_runs)"))]
        assert "parent_run_id" in cols
    finally:
        database.engine = original_engine


def test_migrate_add_parent_run_id_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE pipeline_runs "
            "(run_id VARCHAR PRIMARY KEY, input_path VARCHAR, output_path VARCHAR, status VARCHAR)"
        ))
        conn.commit()

    original_engine = database.engine
    database.engine = engine
    try:
        database._migrate_add_parent_run_id()
        database._migrate_add_parent_run_id()  # must not raise (e.g. "duplicate column")
    finally:
        database.engine = original_engine
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest test_pipeline_run_parent_id.py -v`
Expected: all 4 FAIL — `test_create_pipeline_run_sets_parent_run_id` and `test_create_pipeline_run_parent_run_id_defaults_to_none` with `TypeError: create_pipeline_run() got an unexpected keyword argument 'parent_run_id'` (the first one) or `AttributeError: 'PipelineRun' object has no attribute 'parent_run_id'` (the second one, since the column doesn't exist yet on the model); the two migration tests with `AttributeError: module 'database' has no attribute '_migrate_add_parent_run_id'`.

- [ ] **Step 3: Implement**

In `backend/database.py`, add the column to `PipelineRun` (currently lines 30-59), right after the existing `lesion_type` line:

```python
    # Тип поражения (glioblastoma / multiple_sclerosis)
    lesion_type = Column(String, nullable=True, default='glioblastoma')

    # Если этот запуск — requeue (повторная обработка после ручной правки),
    # здесь лежит run_id исходного запуска. NULL для обычных запусков.
    parent_run_id = Column(String, nullable=True)
```

Add the migration function right after `_migrate_add_lesion_type` (currently lines 253-263), and call it from `init_db`:

```python
def init_db():
    """Инициализировать базу данных и применить миграции колонок."""
    Base.metadata.create_all(bind=engine)
    _migrate_add_lesion_type()
    _migrate_add_parent_run_id()


def _migrate_add_lesion_type():
    """Add lesion_type column to pipeline_runs if it doesn't exist yet."""
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            __import__('sqlalchemy').text("PRAGMA table_info(pipeline_runs)")
        )]
        if 'lesion_type' not in cols:
            conn.execute(__import__('sqlalchemy').text(
                "ALTER TABLE pipeline_runs ADD COLUMN lesion_type TEXT DEFAULT 'glioblastoma'"
            ))
            conn.commit()


def _migrate_add_parent_run_id():
    """Add parent_run_id column to pipeline_runs if it doesn't exist yet."""
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            __import__('sqlalchemy').text("PRAGMA table_info(pipeline_runs)")
        )]
        if 'parent_run_id' not in cols:
            conn.execute(__import__('sqlalchemy').text(
                "ALTER TABLE pipeline_runs ADD COLUMN parent_run_id VARCHAR"
            ))
            conn.commit()
```

Update `create_pipeline_run` (currently lines 92-127) to accept and set the new field:

```python
def create_pipeline_run(
    db: Session,
    input_path: str,
    output_path: str,
    lesion_type: str = 'glioblastoma',
    parent_run_id: Optional[str] = None,
) -> PipelineRun:
    """Создать новый запуск pipeline"""
    run_id = str(uuid.uuid4())

    run = PipelineRun(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        status="pending",
        created_at=datetime.now(timezone.utc),
        lesion_type=lesion_type,
        parent_run_id=parent_run_id,
    )
```

(The rest of the function body — stage-execution creation, commit, return — is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest test_pipeline_run_parent_id.py -v`
Expected: PASS (4/4).

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: same baseline as before this task (86 passed, 1 pre-existing unrelated failure in `test_preprocessing_version.py::test_dataset_mapping`) plus the 4 new tests. Run `git checkout -- configs/preprocessing_versions.json` afterward (known harmless test-pollution side effect) before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/database.py backend/test_pipeline_run_parent_id.py
git commit -m "feat(backend): parent_run_id links a requeued run back to its original"
```

---

### Task 2: Backend — wire `parent_run_id` into requeue and status endpoints

**Files:**
- Modify: `backend/app.py` (`requeue_pipeline_run` at line 365, `get_pipeline_status` at line 429)
- Modify: `backend/models.py` (`PipelineStatusResponse` at line 234)
- Test: `backend/test_app_requeue_endpoint.py` (extend), `backend/test_app_pipeline_status_endpoint.py` (new file)

**Interfaces:**
- Consumes: `create_pipeline_run(..., parent_run_id=...)` from Task 1.
- Produces: `PipelineStatusResponse.parent_run_id: Optional[str]` — consumed by Task 3's `ProgressMonitor.jsx`.

- [ ] **Step 1: Write the failing test for the requeue endpoint**

Add to `backend/test_app_requeue_endpoint.py`, after `test_creates_new_run_with_same_paths_and_does_not_run_pipeline_synchronously`:

```python
def test_requeue_passes_parent_run_id_to_create_pipeline_run():
    original = _fake_original_run(run_id="orig-run")
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.create_pipeline_run", return_value=new_run) as mock_create, \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    _, kwargs = mock_create.call_args
    assert kwargs["parent_run_id"] == "orig-run"
```

- [ ] **Step 2: Write the failing tests for the status endpoint**

Create `backend/test_app_pipeline_status_endpoint.py`:

```python
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _fake_run(run_id="run-1", status="completed", parent_run_id=None):
    return SimpleNamespace(
        run_id=run_id,
        input_path="/in",
        output_path="/out",
        status=status,
        current_stage=7,
        overall_progress=100.0,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error_message=None,
        lesion_type="glioblastoma",
        parent_run_id=parent_run_id,
    )


def test_status_response_includes_parent_run_id_when_set():
    run = _fake_run(parent_run_id="orig-run")

    with patch("app.get_pipeline_run", return_value=run), \
         patch("app.get_stage_executions", return_value=[]):
        response = client.get("/api/pipeline/status/run-1")

    assert response.status_code == 200
    assert response.json()["parent_run_id"] == "orig-run"


def test_status_response_parent_run_id_none_for_ordinary_run():
    run = _fake_run(parent_run_id=None)

    with patch("app.get_pipeline_run", return_value=run), \
         patch("app.get_stage_executions", return_value=[]):
        response = client.get("/api/pipeline/status/run-1")

    assert response.status_code == 200
    assert response.json()["parent_run_id"] is None
```

- [ ] **Step 3: Run all three new tests to confirm they fail**

Run: `cd backend && python3 -m pytest test_app_requeue_endpoint.py::test_requeue_passes_parent_run_id_to_create_pipeline_run test_app_pipeline_status_endpoint.py -v`
Expected: `test_requeue_passes_parent_run_id_to_create_pipeline_run` FAILS with `KeyError: 'parent_run_id'` (not yet passed as a kwarg). The two status-endpoint tests FAIL with `KeyError: 'parent_run_id'` on the response JSON (field doesn't exist yet on `PipelineStatusResponse`).

- [ ] **Step 4: Implement**

In `backend/models.py`, add to `PipelineStatusResponse` (currently lines 234-244), after `lesion_type`:

```python
class PipelineStatusResponse(BaseModel):
    """Ответ с текущим статусом pipeline"""
    run_id: str = Field(..., description="ID запуска")
    status: PipelineStatus = Field(..., description="Общий статус pipeline")
    current_stage: Optional[int] = Field(None, description="Номер текущего этапа (1-6)")
    overall_progress: float = Field(0.0, ge=0.0, le=100.0, description="Общий прогресс")
    stages: List[StageProgress] = Field(default_factory=list, description="Детали по каждому этапу")
    created_at: datetime = Field(..., description="Время запуска")
    completed_at: Optional[datetime] = Field(None, description="Время завершения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    lesion_type: Optional[str] = Field(None, description="Тип поражения (glioblastoma / multiple_sclerosis)")
    parent_run_id: Optional[str] = Field(None, description="ID исходного запуска, если это requeue")
```

In `backend/app.py`, `requeue_pipeline_run` (currently lines 397-402), pass the new argument:

```python
    run = create_pipeline_run(
        db,
        input_path=original_run.input_path,
        output_path=original_run.output_path,
        lesion_type=original_run.lesion_type or "glioblastoma",
        parent_run_id=run_id,
    )
```

In `get_pipeline_status` (currently lines 483-493), include the field:

```python
    return PipelineStatusResponse(
        run_id=run.run_id,
        status=PipelineStatus(run.status),
        current_stage=run.current_stage,
        overall_progress=run.overall_progress,
        stages=stages,
        created_at=run.created_at,
        completed_at=run.completed_at,
        error=run.error_message,
        lesion_type=getattr(run, "lesion_type", None),
        parent_run_id=getattr(run, "parent_run_id", None),
    )
```

- [ ] **Step 5: Run the new tests to confirm they pass**

Run: `cd backend && python3 -m pytest test_app_requeue_endpoint.py test_app_pipeline_status_endpoint.py -v`
Expected: PASS (all tests in both files).

- [ ] **Step 6: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: baseline from Task 1 (86 + 4 = 90 passed, 1 pre-existing unrelated failure) plus this task's 3 new tests = 93 passed, 1 pre-existing failure. Run `git checkout -- configs/preprocessing_versions.json` afterward before committing.

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/models.py backend/test_app_requeue_endpoint.py backend/test_app_pipeline_status_endpoint.py
git commit -m "feat(backend): requeue endpoint sets parent_run_id, status endpoint exposes it"
```

---

### Task 3: Frontend — switch active run on requeue, show banner link back

**Files:**
- Modify: `frontend/src/components/IncompletePatients.jsx` (`handleRequeue`, currently lines 47-67; component props, currently line 12)
- Modify: `frontend/src/App.jsx` (state declarations ~lines 23-38, `Tabs` ~lines 173-237, `ProgressMonitor` mount ~lines 192-196, history-triggered `IncompletePatients` mount ~lines 266-276)
- Modify: `frontend/src/components/ProgressMonitor.jsx` (component props at line 19, state ~lines 20-30, `fetchInitialStatus` ~lines 65-73, render ~lines 191-206, own `IncompletePatients` mount ~lines 288-293)

**Interfaces:**
- Consumes: `parent_run_id` field on the `getPipelineStatus()` response (Task 2).
- Produces: `IncompletePatients`'s new `onRequeued` prop — `(response: {run_id, status, created_at, lesion_type}) => void`, called with the exact response object `requeuePipelineRun()` already returns (same shape `handlePipelineStarted` already consumes — this task reuses that existing handler rather than writing a near-duplicate one).

- [ ] **Step 1: `IncompletePatients.jsx` — add the `onRequeued` callback**

Read the file first to confirm current line numbers (may have drifted slightly from a prior session). Add `onRequeued` to the destructured props (currently line 12):

```jsx
const IncompletePatients = ({ runId, visible, onClose, canRequeue = true, onRequeued }) => {
```

In `handleRequeue` (currently lines 47-67), call it right after `onClose()`:

```jsx
  const handleRequeue = async () => {
    setRequeuing(true);
    try {
      const result = await requeuePipelineRun(runId);
      message.success(
        `Обработка перезапущена (run_id: ${result.run_id.substring(0, 8)}...). ` +
        'Отслеживайте прогресс во вкладке «История запусков».'
      );
      onClose();
      if (onRequeued) {
        onRequeued(result);
      }
    } catch (err) {
      console.error('Ошибка перезапуска:', err);
      const detail = err.response?.data?.detail;
      if (err.response?.status === 409 && detail) {
        message.error(detail);
      } else {
        message.error('Не удалось перезапустить обработку');
      }
    } finally {
      setRequeuing(false);
    }
  };
```

(Only the `if (onRequeued) { onRequeued(result); }` block after `onClose();` is new — everything else in this function is unchanged.)

- [ ] **Step 2: `App.jsx` — controlled tabs + wire `onRequeued` and `onSwitchToHistory`**

Read the file first to confirm current line numbers. Add one new state variable alongside the existing ones (near line 23):

```jsx
  const [activeTabKey, setActiveTabKey] = useState('pipeline');
```

Add a handler near `handlePipelineStarted` (it can go directly below it):

```jsx
  const handleSwitchToHistoryTab = () => {
    setActiveTabKey('history');
  };
```

Change the `<Tabs>` element (currently starts around line 173) from uncontrolled to controlled — replace `defaultActiveKey="pipeline"` with:

```jsx
            <Tabs
              activeKey={activeTabKey}
              onChange={setActiveTabKey}
              size="large"
```

Pass the two new props into `<ProgressMonitor>` (currently lines 192-196):

```jsx
                        <ProgressMonitor
                          runId={activeRun.runId}
                          lesionType={activeRun.lesionType}
                          onComplete={handlePipelineComplete}
                          onRequeued={handlePipelineStarted}
                          onSwitchToHistory={handleSwitchToHistoryTab}
                        />
```

(`handlePipelineStarted` is reused as-is — it already does exactly `setActiveRun({runId: response.run_id, status: response.status, createdAt: response.created_at, lesionType: response.lesion_type || 'glioblastoma'})`, which is precisely "activate this run," whether it came from starting fresh or from a requeue.)

Pass `onRequeued={handlePipelineStarted}` into the history-triggered `<IncompletePatients>` (currently lines 266-276):

```jsx
            {showHistoryIncompletePatients && (
              <IncompletePatients
                runId={historyIncompletePatientsRunId}
                visible={showHistoryIncompletePatients}
                onClose={() => setShowHistoryIncompletePatients(false)}
                canRequeue={
                  historyIncompletePatientsStatus === 'completed' ||
                  historyIncompletePatientsStatus === 'failed'
                }
                onRequeued={handlePipelineStarted}
              />
            )}
```

- [ ] **Step 3: `ProgressMonitor.jsx` — accept the new props, track `parentRunId`, render the banner**

Read the file first to confirm current line numbers. Add the two new props to the component signature (currently line 19):

```jsx
const ProgressMonitor = ({ runId, onComplete, lesionType = 'glioblastoma', onRequeued, onSwitchToHistory }) => {
```

Add a new state variable alongside the existing ones (near line 30):

```jsx
  const [parentRunId, setParentRunId] = useState(null);
```

In `fetchInitialStatus` (currently lines 65-73), capture it — do NOT add this to `updateStatus`, since that function also runs on every WebSocket message, which doesn't carry `parent_run_id` and would otherwise reset it back to `undefined` on the next live update:

```jsx
  const fetchInitialStatus = async () => {
    try {
      const data = await getPipelineStatus(runId);
      updateStatus(data);
      setParentRunId(data.parent_run_id || null);
    } catch (error) {
      // Не показываем ошибку — WebSocket подхватит обновления
      console.warn('Начальный запрос статуса не удался, ожидаем WebSocket:', error);
    }
  };
```

Render the banner. In the returned JSX (currently starting at line 191, the `<Card ...>` block), insert it as the first child, right after the opening `<Card ...>` tag and before the "Общий прогресс" div:

```jsx
      {parentRunId && (
        <Alert
          type="info"
          showIcon
          message={`Это повторный запуск после ручной правки (исходный запуск: ${parentRunId.substring(0, 8)}...)`}
          action={
            <Button size="small" onClick={onSwitchToHistory}>
              Перейти к истории
            </Button>
          }
          style={{ marginBottom: 16 }}
        />
      )}
```

`Alert` and `Button` are already imported in this file (used elsewhere for the error alert and stage-progress buttons) — no new imports needed.

Pass `onRequeued` into this component's own `<IncompletePatients>` mount (currently lines 288-293):

```jsx
      <IncompletePatients
        runId={runId}
        visible={showIncompletePatients}
        onClose={() => setShowIncompletePatients(false)}
        canRequeue={status === 'completed' || status === 'failed'}
        onRequeued={onRequeued}
      />
```

- [ ] **Step 4: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors from the three modified files (baseline at time of writing: 21 pre-existing problems in unrelated files — confirm the count/file-list is unchanged).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/IncompletePatients.jsx frontend/src/App.jsx frontend/src/components/ProgressMonitor.jsx
git commit -m "feat(frontend): requeue switches the active-run screen, with a banner linking back to the original"
```

---

### Task 4: Manual verification

**Files:** none modified — verification only.

- [ ] **Step 1: Rebuild and restart the stack**

```bash
docker compose --profile full up --build
```

- [ ] **Step 2: Verify requeue from the active-run screen**

Start a run with at least one incomplete patient (or reuse an existing incomplete run). Once stage 1 completes, open "Пациенты, требующие внимания" from the active-run screen, fix a session so it becomes complete, and click "Запустить обработку". Confirm: the modal closes, and the SAME screen (still on "Запуск обработки" tab) now shows a fresh `ProgressMonitor` tracking the NEW run (progress resets to reflect the new run's own stages), with a blue info banner at the top: "Это повторный запуск после ручной правки (исходный запуск: `<8 chars>`...)" and a "Перейти к истории" button.

- [ ] **Step 3: Verify the banner's link**

Click "Перейти к истории". Confirm the tab switches to "История запусков" without a page reload, and the original run's row (now completed) is visible there with all its report buttons intact (quality report, clinical report, visualization, etc. — unchanged from before this feature).

- [ ] **Step 4: Verify requeue from the history tab**

From "История запусков", find a completed run with an incomplete/reconsiderable session, open its "Неполные пациенты" button, fix something, click "Запустить обработку". Confirm: the app switches you to the "Запуск обработки" tab automatically, showing the new run's live progress with the same banner pointing back at the run you just requeued from.

- [ ] **Step 5: Verify an ordinary (non-requeue) run shows no banner**

Start a brand-new pipeline run from the form (not a requeue). Confirm no banner appears on its `ProgressMonitor` — `parent_run_id` is `None` for it.

- [ ] **Step 6: Report findings**

Note any issues found — this is real, unscripted verification.
