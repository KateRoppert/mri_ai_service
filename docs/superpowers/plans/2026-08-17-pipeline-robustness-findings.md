# Pipeline Robustness Findings (KI-052..055) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four robustness gaps found while investigating the full BO dataset run (KNOWN_ISSUES.md KI-052..055), in priority order, each independently testable and committed.

**Architecture:** Four independent backend fixes (one touches a pipeline preprocessing step, three touch `backend/`), plus one small frontend addition (KI-054's report needs a place to view it). No shared code between the four — they can be reviewed and verified one at a time.

**Tech Stack:** Python/FastAPI backend (`backend/`), a preprocessing script (`scripts/preprocessing_steps/reorient.py`), React 19 + Ant Design v6 frontend (`frontend/`), pytest.

## Global Constraints

- KI-052 and KI-053 are fully verifiable via `pytest` — no Docker/browser needed (confirmed with the project owner).
- KI-054's backend aggregation is `pytest`-verifiable; its new frontend modal needs manual browser verification, folded into the single end-to-end pass at the end of this plan (not after Task 3/4 individually).
- KI-054 covers only the 5 stages that already write `{stage}_incomplete_data.json` (01, 03, 04, 05, 06). Stages 07/08 don't write this report at all yet — out of scope for this plan, left as a follow-up (no losses were observed there in the investigated run).
- KI-054's aggregator does not cross-reference patient identity between stages (some report BIDS `sub-XXX`, others report the bare original id) — each entry is shown exactly as its own stage reports it.
- KI-055 must not change `merge_sessions()`'s or `get_incomplete_patients()`'s behavior for existing callers that don't pass the new parameters (both new parameters default so old call sites keep working unmodified).
- All new tests follow this project's existing pytest conventions in the file/directory they're added to (see each task for the exact precedent file).

---

### Task 1: KI-052 — timeout scales with dataset size; kill the whole process group, not just the orchestrator

**Files:**
- Modify: `backend/config.py` (`pipeline_timeout_seconds`, currently line 44)
- Modify: `backend/pipeline_manager.py` (`start_pipeline`'s `subprocess.Popen` call, currently lines 204-210; add `estimate_pipeline_timeout` method)
- Modify: `backend/app.py` (imports, `run_pipeline_background`'s `process.communicate()` call at line 201, the `except subprocess.TimeoutExpired:` block at lines 251-262; add `_kill_process_tree` helper)
- Test: `backend/test_pipeline_manager_timeout.py` (new file), `backend/test_app_kill_process_tree.py` (new file)

**Interfaces:**
- Produces: `PipelineManager.estimate_pipeline_timeout(input_path: str) -> int`; `_kill_process_tree(process: subprocess.Popen) -> None` (module-level function in `backend/app.py`).

- [ ] **Step 1: Write the failing tests for `estimate_pipeline_timeout`**

Create `backend/test_pipeline_manager_timeout.py`:

```python
"""
KI-052: the pipeline timeout was a flat 7200s constant for the whole
multi-stage run regardless of dataset size — too short for a full
~175-patient clinical dataset (stage 5 alone measured 56 min / 161
patients on real BO data). It must scale with how many patients are
actually being processed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from pipeline_manager import PipelineManager


def test_estimate_pipeline_timeout_scales_with_patient_count(tmp_path):
    for i in range(5):
        (tmp_path / f"patient-{i}").mkdir()
    (tmp_path / ".hidden").mkdir()  # must not be counted as a patient
    (tmp_path / "a_file.txt").write_text("x")  # not a directory, must not be counted

    pm = PipelineManager()
    timeout = pm.estimate_pipeline_timeout(str(tmp_path))

    expected = settings.pipeline_timeout_base_seconds + settings.pipeline_timeout_per_patient_seconds * 5
    assert timeout == expected


def test_estimate_pipeline_timeout_returns_base_for_missing_path(tmp_path):
    pm = PipelineManager()
    timeout = pm.estimate_pipeline_timeout(str(tmp_path / "does-not-exist"))
    assert timeout == settings.pipeline_timeout_base_seconds


def test_estimate_pipeline_timeout_base_for_empty_directory(tmp_path):
    pm = PipelineManager()
    timeout = pm.estimate_pipeline_timeout(str(tmp_path))
    assert timeout == settings.pipeline_timeout_base_seconds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest test_pipeline_manager_timeout.py -v`
Expected: all 3 FAIL — `AttributeError: 'Settings' object has no attribute 'pipeline_timeout_base_seconds'` (the config fields don't exist yet) or `AttributeError: 'PipelineManager' object has no attribute 'estimate_pipeline_timeout'`.

- [ ] **Step 3: Write the failing test for `_kill_process_tree`**

Create `backend/test_app_kill_process_tree.py`:

```python
"""
KI-052: on timeout, the backend called process.kill() on the orchestrator
process only. orchestrator.py runs each pipeline stage as ITS OWN child
subprocess — killing only the orchestrator leaves that grandchild running,
orphaned, invisible to the DB/UI, for however long it takes to finish on
its own. Confirmed on real data: a run the DB marked "failed" at a 2h
timeout kept producing real segmentation output for almost 2 more hours.

This test spawns a real two-level process tree (a shell that backgrounds
a sleep and waits on it — same shape as orchestrator.py spawning a stage
subprocess) to prove the fix actually reaches the grandchild, not just
mocks the call.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import _kill_process_tree


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_kill_process_tree_kills_grandchild_not_just_direct_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    process = subprocess.Popen(
        ["bash", "-c", f"sleep 60 & echo $! > {pid_file}; wait"],
        start_new_session=True,
    )
    # give the shell time to background the sleep and write its pid
    for _ in range(20):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    grandchild_pid = int(pid_file.read_text().strip())
    assert _process_alive(grandchild_pid), "test setup failed: grandchild never started"

    _kill_process_tree(process)
    process.wait(timeout=5)

    time.sleep(0.2)
    assert not _process_alive(grandchild_pid), (
        "grandchild (stage subprocess) survived — only the direct child was killed"
    )


def test_kill_process_tree_does_not_raise_if_already_dead(tmp_path):
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait(timeout=5)
    # process has already exited on its own — killing its group must not raise
    _kill_process_tree(process)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest test_app_kill_process_tree.py -v`
Expected: both FAIL with `ImportError: cannot import name '_kill_process_tree' from 'app'`.

- [ ] **Step 5: Implement**

In `backend/config.py`, replace the single timeout field (currently line 44):

```python
    # Таймауты — база + за каждого пациента во входной директории, а не
    # единая константа: 7200с хватало на демо-прогоны (5-20 пациентов), но
    # не на полный клинический датасет (161-175 пациентов на BO реально
    # заняли больше — стадия 5 сама по себе 56 минут на 161 пациента).
    pipeline_timeout_base_seconds: int = 1200      # 20 минут — фиксированные накладные расходы этапов
    pipeline_timeout_per_patient_seconds: int = 90  # покрывает препроцессинг+сегментацию+остальное на пациента
```

In `backend/pipeline_manager.py`, add `estimate_pipeline_timeout` as a new method (place it right before `start_pipeline`, currently at line 159):

```python
    def estimate_pipeline_timeout(self, input_path: str) -> int:
        """
        Estimate a generous timeout for the whole multi-stage pipeline,
        scaled by how many patients are in the input directory — see
        KI-052 in KNOWN_ISSUES.md. Counts non-hidden top-level
        subdirectories, mirroring DatasetScanner.scan_dataset's own simple
        patient-counting convention in scripts/01_reorganize_folders.py
        (not imported here — that scanner also handles nested single-patient
        layouts, which the backend doesn't need just to size a timeout).
        """
        path = Path(input_path)
        if not path.is_dir():
            return settings.pipeline_timeout_base_seconds

        patient_count = sum(
            1 for entry in path.iterdir()
            if entry.is_dir() and not entry.name.startswith('.')
        )
        return settings.pipeline_timeout_base_seconds + settings.pipeline_timeout_per_patient_seconds * patient_count
```

Update `start_pipeline`'s `subprocess.Popen` call (currently lines 204-210) to give the orchestrator its own process group:

```python
            # Запускаем процесс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.pipeline_root),
                start_new_session=True,  # own process group — see KI-052 / _kill_process_tree in app.py
            )
```

In `backend/app.py`, add `import signal` near the existing `import subprocess`/`import os` (currently lines 16-18), and add `_kill_process_tree` as a module-level function right before `run_pipeline_background` (currently starts at line 160):

```python
def _kill_process_tree(process: subprocess.Popen) -> None:
    """
    Kill the whole process group of a subprocess, not just the direct
    child. orchestrator.py runs each pipeline stage as its own child
    subprocess; process.kill() only signals the immediate orchestrator.py
    process, leaving a currently-running stage subprocess orphaned and
    still executing — invisible to the DB/UI — for however long it takes
    to finish on its own (see KI-052 in KNOWN_ISSUES.md). Requires the
    process to have been started with start_new_session=True so it has
    its own group to kill (see PipelineManager.start_pipeline).
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        # Already exited on its own — nothing to kill.
        pass
```

Update `run_pipeline_background` (currently line 201) to use the scaled timeout:

```python
    timeout = pipeline_manager.estimate_pipeline_timeout(input_path)
    logger.info(f"Таймаут для run_id {run_id}: {timeout}s (input_path={input_path})")

    # Ждём завершения процесса
    try:
        stdout, stderr = process.communicate(timeout=timeout)
```

Update the `except subprocess.TimeoutExpired:` block (currently lines 251-262) to kill the whole tree:

```python
    except subprocess.TimeoutExpired:
        # Таймаут
        logger.error(f"Таймаут выполнения pipeline для run_id: {run_id}")
        _kill_process_tree(process)

        update_pipeline_run(
            db,
            run_id,
            status="failed",
            error_message="Превышено время ожидания выполнения",
            completed_at=datetime.utcnow()
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest test_pipeline_manager_timeout.py test_app_kill_process_tree.py -v`
Expected: PASS (5/5). The process-tree test takes a little over a second (spawns real processes with small sleeps) — that's expected, not a hang.

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: same baseline as before this task (find the current count by running it once beforehand if unsure — as of this plan's writing, 103 passed / 1 pre-existing unrelated failure in `test_preprocessing_version.py::test_dataset_mapping`) plus this task's 5 new tests. Run `git checkout -- configs/preprocessing_versions.json` afterward (known harmless test-pollution side effect) before committing.

- [ ] **Step 8: Commit**

```bash
git add backend/config.py backend/pipeline_manager.py backend/app.py backend/test_pipeline_manager_timeout.py backend/test_app_kill_process_tree.py
git commit -m "fix(backend): scale pipeline timeout with dataset size, kill whole process group on timeout (KI-052)"
```

---

### Task 2: KI-053 — reorient.py must not crash with a cryptic error on a degenerate affine

**Files:**
- Modify: `scripts/preprocessing_steps/reorient.py` (`reorient_to_standard`, currently lines 38-39; `check_orientation`, currently lines 150-151)
- Test: `tests/preprocessing_steps/test_reorient_orientation_guard.py` (new file)

**Interfaces:** none consumed from or produced for other tasks — fully self-contained.

- [ ] **Step 1: Write the failing tests**

Create `tests/preprocessing_steps/test_reorient_orientation_guard.py`, following the same module-loading convention already used by `tests/preprocessing_steps/test_apply_transform_interpolator.py`:

```python
"""
KI-053: nib.aff2axcodes(affine) returns None for an axis when the affine
matrix is degenerate/non-orthogonal — reorient.py did ''.join(axcodes)
straight afterward, crashing with a cryptic "sequence item 2: expected
str instance, NoneType found" instead of a clear, actionable message.
Reproduced twice on real data: BO-214's t1c series (2026-08-14 and
2026-08-17 runs) — t1/t2/t2fl of the same patient reoriented fine, only
t1c's affine was degenerate.
"""
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPTS_DIR / "preprocessing_steps" / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorient = _load_module("reorient.py", "reorient_orientation_guard")


class TestReorientToStandardOrientationGuard:
    def test_returns_clear_error_when_axcodes_contains_none(self, tmp_path):
        fake_img = MagicMock()
        fake_img.affine = MagicMock()
        with patch.object(reorient.nib, "load", return_value=fake_img), \
             patch.object(reorient.nib, "aff2axcodes", return_value=(None, 'P', 'S')):
            result = reorient.reorient_to_standard(
                tmp_path / "sub-214_ses-001_t1c.nii.gz",
                tmp_path / "out.nii.gz",
            )

        assert result["success"] is False
        assert "degenerate" in result["error"].lower()
        assert "sub-214_ses-001_t1c.nii.gz" in result["error"]

    def test_still_succeeds_for_a_normal_orientation(self, tmp_path):
        fake_img = MagicMock()
        fake_img.affine = MagicMock()
        fake_img.header = MagicMock()
        fake_reoriented = MagicMock()
        fake_reoriented.affine = fake_img.affine
        fake_reoriented.get_fdata.return_value = MagicMock()
        with patch.object(reorient.nib, "load", return_value=fake_img), \
             patch.object(reorient.nib, "aff2axcodes", return_value=('L', 'P', 'S')), \
             patch.object(reorient.nib, "save"):
            result = reorient.reorient_to_standard(
                tmp_path / "sub-001_ses-001_t1.nii.gz",
                tmp_path / "out.nii.gz",
                target_orientation="LPS",
            )

        assert result["success"] is True
        assert result["original_orientation"] == "LPS"
        assert result["transformation_applied"] is False


class TestCheckOrientationGuard:
    def test_raises_clear_error_when_axcodes_contains_none(self, tmp_path):
        fake_img = MagicMock()
        fake_img.affine = MagicMock()
        with patch.object(reorient.nib, "load", return_value=fake_img), \
             patch.object(reorient.nib, "aff2axcodes", return_value=(None, 'P', 'S')):
            with pytest.raises(ValueError, match="degenerate"):
                reorient.check_orientation(tmp_path / "sub-214_ses-001_t1c.nii.gz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/mri_ai_service && python3 -m pytest tests/preprocessing_steps/test_reorient_orientation_guard.py -v`
Expected: `test_returns_clear_error_when_axcodes_contains_none` FAILS — the actual error message is the cryptic `sequence item 2: expected str instance, NoneType found`, not one containing "degenerate". `test_raises_clear_error_when_axcodes_contains_none` FAILS — currently raises `TypeError`, not `ValueError`. `test_still_succeeds_for_a_normal_orientation` may already PASS (it exercises the unchanged success path) — that's fine, it's a regression guard, not meant to fail.

- [ ] **Step 3: Implement**

In `scripts/preprocessing_steps/reorient.py`, update `reorient_to_standard` (currently lines 37-39):

```python
        # Get original orientation
        original_orientation = nib.aff2axcodes(img.affine)
        if any(axis is None for axis in original_orientation):
            raise ValueError(
                f"Cannot determine orientation from affine matrix (degenerate or "
                f"non-orthogonal affine) for {input_path.name}: aff2axcodes returned {original_orientation!r}"
            )
        original_orientation_str = ''.join(original_orientation)
```

Update `check_orientation` (currently lines 149-151):

```python
    # Get orientation
    orientation = nib.aff2axcodes(img.affine)
    if any(axis is None for axis in orientation):
        raise ValueError(
            f"Cannot determine orientation from affine matrix (degenerate or "
            f"non-orthogonal affine) for {image_path.name}: aff2axcodes returned {orientation!r}"
        )
    orientation_str = ''.join(orientation)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/mri_ai_service && python3 -m pytest tests/preprocessing_steps/test_reorient_orientation_guard.py -v`
Expected: PASS (3/3).

- [ ] **Step 5: Run the full stage05/preprocessing_steps test suites**

Run: `cd /home/ubuntu/mri_ai_service && python3 -m pytest tests/preprocessing_steps/ tests/stage05/ -q`
Expected: all pass, no regressions from the two changed functions.

- [ ] **Step 6: Commit**

```bash
git add scripts/preprocessing_steps/reorient.py tests/preprocessing_steps/test_reorient_orientation_guard.py
git commit -m "fix(pipeline): reorient.py raises a clear error on a degenerate affine instead of a cryptic TypeError (KI-053)"
```

---

### Task 3: KI-054 backend — aggregate per-stage loss reports, new endpoint

**Files:**
- Modify: `backend/pipeline_manager.py` (add `_LOSS_REPORT_FILES` module constant, `get_pipeline_losses` method, `_describe_loss_reason` static method)
- Modify: `backend/models.py` (add `PipelineLoss`, `PipelineLossesResponse`)
- Modify: `backend/app.py` (add endpoint, import the two new models)
- Test: `backend/test_pipeline_manager_losses.py` (new file)

**Interfaces:**
- Produces: `PipelineManager.get_pipeline_losses(output_path: str) -> List[Dict[str, Any]]` (each dict: `{stage, patient_id, session_id, reason}`), `GET /api/pipeline-runs/{run_id}/losses` — consumed by Task 4's frontend.

- [ ] **Step 1: Write the failing tests**

Create `backend/test_pipeline_manager_losses.py`:

```python
"""
KI-054: stages 01/03/04/05/06 already write their own
{stage}_incomplete_data.json into their output directories — nobody
reads them. This aggregates them into one flat "who got lost, where, why"
list. Confirmed real shapes on BO data:
  - stage 01: {"patient_id": "sub-133", "incomplete_sessions": [{"session_id": "ses-001", "missing": [...], "available": [...]}]}
  - stage 05: {"patient_id": "214", "incomplete_sessions": [{"session_id": "001", "reason": "modality_mismatch", "missing_in_output": [...]}]}
  - stage 06: {"patient_id": "214", "incomplete_sessions": [{"session_id": "001", "reason": "patient_missing_in_output"}]}
Stages 07/08 don't write this report at all yet (out of scope here).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _write_loss_report(output_dir: Path, relative_path: str, content: dict):
    report_file = output_dir / relative_path
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(content), encoding="utf-8")


class TestGetPipelineLosses:
    def test_aggregates_losses_from_multiple_stages(self, tmp_path):
        _write_loss_report(tmp_path, "bids_organized/incomplete_data/01_reorganize_folders_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "sub-133", "incomplete_sessions": [
                    {"session_id": "ses-001", "missing": ["t1", "t1c"]}
                ]}
            ]
        })
        _write_loss_report(tmp_path, "segmentation/incomplete_data/segmentation_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "214", "incomplete_sessions": [
                    {"session_id": "001", "reason": "patient_missing_in_output"}
                ]}
            ]
        })

        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))

        assert len(losses) == 2
        by_stage = {loss["stage"]: loss for loss in losses}
        assert by_stage["01_reorganize"]["patient_id"] == "sub-133"
        assert by_stage["01_reorganize"]["session_id"] == "ses-001"
        assert "t1" in by_stage["01_reorganize"]["reason"]
        assert by_stage["06_segmentation"]["patient_id"] == "214"
        assert by_stage["06_segmentation"]["reason"] == "patient_missing_in_output"

    def test_missing_report_files_are_skipped_not_errors(self, tmp_path):
        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))
        assert losses == []

    def test_reason_synthesized_from_reason_and_missing_in_output(self, tmp_path):
        _write_loss_report(tmp_path, "preprocessed/incomplete_data/preprocessing_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "214", "incomplete_sessions": [
                    {"session_id": "001", "reason": "modality_mismatch", "missing_in_output": ["t1c"]}
                ]}
            ]
        })
        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))

        assert len(losses) == 1
        assert "modality_mismatch" in losses[0]["reason"]
        assert "t1c" in losses[0]["reason"]

    def test_reason_synthesized_from_missing_only_when_no_reason_field(self, tmp_path):
        _write_loss_report(tmp_path, "bids_organized/incomplete_data/01_reorganize_folders_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "sub-050", "incomplete_sessions": [
                    {"session_id": "ses-001", "missing": ["t2fl"], "available": ["t1", "t1c", "t2"]}
                ]}
            ]
        })
        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))

        assert len(losses) == 1
        assert "t2fl" in losses[0]["reason"]

    def test_malformed_json_report_is_skipped_not_fatal(self, tmp_path):
        report_file = tmp_path / "bids_organized" / "incomplete_data" / "01_reorganize_folders_incomplete_data.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("{not valid json", encoding="utf-8")

        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))
        assert losses == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest test_pipeline_manager_losses.py -v`
Expected: all 5 FAIL with `AttributeError: 'PipelineManager' object has no attribute 'get_pipeline_losses'`.

- [ ] **Step 3: Implement**

In `backend/pipeline_manager.py`, add a module-level constant near the other module-level constants (`_BIDS_PATIENT_ID_PATTERN` etc.):

```python
# Per-stage "who got lost and why" reports — already written by stages
# 01/03/04/05/06 into their own output directories, but never read by
# anything (KI-054 in KNOWN_ISSUES.md). Stages 07/08 don't write this
# report yet — not covered here.
_LOSS_REPORT_FILES = [
    ("01_reorganize", "bids_organized/incomplete_data/01_reorganize_folders_incomplete_data.json"),
    ("03_convert", "nifti/incomplete_data/03_convert_to_nifti_incomplete_data.json"),
    ("04_quality", "quality_reports/incomplete_data/04_assess_quality_incomplete_data.json"),
    ("05_preprocessing", "preprocessed/incomplete_data/preprocessing_incomplete_data.json"),
    ("06_segmentation", "segmentation/incomplete_data/segmentation_incomplete_data.json"),
]
```

Add the two new methods to `PipelineManager` (place them right after `get_incomplete_patients`, i.e. after the method currently ending at line 624):

```python
    def get_pipeline_losses(self, output_path: str) -> List[Dict[str, Any]]:
        """
        Aggregate the per-stage {stage}_incomplete_data.json reports that
        stages 01/03/04/05/06 already write into their own output
        directories, into one flat "who got lost, at which stage, and why"
        list for the doctor (KI-054 in KNOWN_ISSUES.md). Stages 07/08
        don't write this report yet — a known gap, not covered here.

        Each stage's report uses a different shape (patient_id sometimes
        BIDS-style "sub-XXX", sometimes the bare original id; the "why" is
        sometimes an explicit reason string, sometimes only missing/
        available modality lists) — this reports each entry exactly as
        that stage describes it, without cross-referencing identities
        across stages.
        """
        losses: List[Dict[str, Any]] = []
        base = Path(output_path)

        for stage_label, relative_path in _LOSS_REPORT_FILES:
            report_file = base / relative_path
            if not report_file.exists():
                continue
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            for patient_entry in report.get('incomplete_data', []):
                patient_id = patient_entry.get('patient_id', '')
                for session_entry in patient_entry.get('incomplete_sessions', []):
                    losses.append({
                        "stage": stage_label,
                        "patient_id": patient_id,
                        "session_id": session_entry.get('session_id', ''),
                        "reason": self._describe_loss_reason(session_entry),
                    })

        return losses

    @staticmethod
    def _describe_loss_reason(session_entry: Dict[str, Any]) -> str:
        """Build a human-readable reason string from whichever fields a
        given stage's incomplete_data.json entry happens to carry."""
        reason = session_entry.get('reason')
        missing = session_entry.get('missing_in_output') or session_entry.get('missing')
        if reason and missing:
            return f"{reason} (нет: {', '.join(missing)})"
        if reason:
            return reason
        if missing:
            return f"не хватает модальностей: {', '.join(missing)}"
        return "причина не указана в отчёте этапа"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest test_pipeline_manager_losses.py -v`
Expected: PASS (5/5).

- [ ] **Step 5: Add the response models and endpoint**

In `backend/models.py`, add right after `MergeSessionsResponse` (currently ends around line 190):

```python
class PipelineLoss(BaseModel):
    """Один потерянный пациент/сессия на одном из этапов пайплайна"""
    stage: str = Field(..., description="Этап пайплайна, на котором пациент был потерян")
    patient_id: str = Field(..., description="ID пациента, как его сообщает данный этап (формат может отличаться между этапами)")
    session_id: str = Field(..., description="ID сессии")
    reason: str = Field(..., description="Причина потери, как её описывает данный этап")


class PipelineLossesResponse(BaseModel):
    """Агрегированный отчёт о пациентах, потерянных на любом этапе пайплайна"""
    total: int = Field(..., description="Количество потерянных пациентов/сессий по всем этапам")
    losses: List[PipelineLoss] = Field(..., description="Список потерь")
```

In `backend/app.py`, add the two new model names to the existing `from models import (...)` block (currently lines 61-66, alongside `MergeSessionsRequest`/`MergeSessionsResponse`):

```python
    MergeSessionsRequest,
    MergeSessionsResponse,
    PipelineLossesResponse,
```

Add the endpoint, right after the `merge_sessions` route (currently ends at line 1014):

```python
@app.get("/api/pipeline-runs/{run_id}/losses", response_model=PipelineLossesResponse)
async def get_pipeline_losses(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Агрегированный отчёт о пациентах, потерянных на любом этапе пайплайна (KI-054)"""
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    losses = pipeline_manager.get_pipeline_losses(run.output_path)
    return PipelineLossesResponse(total=len(losses), losses=losses)
```

- [ ] **Step 6: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: baseline from Task 1 plus this task's 5 new tests, no regressions. Run `git checkout -- configs/preprocessing_versions.json` afterward before committing.

- [ ] **Step 7: Commit**

```bash
git add backend/pipeline_manager.py backend/models.py backend/app.py backend/test_pipeline_manager_losses.py
git commit -m "feat(backend): aggregate per-stage loss reports into GET .../losses (KI-054)"
```

---

### Task 4: KI-054 frontend — show the losses report from run history

**Files:**
- Create: `frontend/src/components/PipelineLosses.jsx`
- Modify: `frontend/src/services/api.js` (add `getPipelineLosses`)
- Modify: `frontend/src/components/PipelineHistory.jsx` (new button + prop)
- Modify: `frontend/src/App.jsx` (new state, handler, mount)

**Interfaces:**
- Consumes: `GET /api/pipeline-runs/{run_id}/losses` (Task 3).
- Produces: none consumed elsewhere — last piece of KI-054.

- [ ] **Step 1: `api.js` — add `getPipelineLosses`**

Read the file first to confirm current line numbers. Add after `getIncompletePatients` (or any convenient spot near the other run-scoped GETs):

```js
/**
 * Агрегированный отчёт о пациентах, потерянных на любом этапе пайплайна
 */
export const getPipelineLosses = async (runId) => {
  const response = await apiClient.get(`/pipeline-runs/${runId}/losses`);
  return response.data;
};
```

Add `getPipelineLosses` to the default-export object at the bottom of the file.

- [ ] **Step 2: Create `PipelineLosses.jsx`**

```jsx
/**
 * Модальное окно с агрегированным отчётом о пациентах, потерянных на
 * любом этапе пайплайна (не только на этапе 1, как IncompletePatients) —
 * для расследования постфактум завершённого/упавшего запуска.
 */
import { useState, useEffect } from 'react';
import { Modal, Table, Tag, Spin, Alert } from 'antd';
import { getPipelineLosses } from '../services/api';

const STAGE_LABELS = {
  '01_reorganize': 'Этап 1: Анонимизация и стандартизация',
  '03_convert': 'Этап 2: Конвертация в NIfTI',
  '04_quality': 'Этап 3: Оценка качества',
  '05_preprocessing': 'Этап 4: Предобработка',
  '06_segmentation': 'Этап 5: Сегментация',
};

const PipelineLosses = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [losses, setLosses] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (visible && runId) {
      fetchLosses();
    }
  }, [visible, runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchLosses = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getPipelineLosses(runId);
      setLosses(data.losses || []);
    } catch (err) {
      console.error('Ошибка загрузки отчёта о потерянных пациентах:', err);
      setError('Не удалось загрузить отчёт');
    } finally {
      setLoading(false);
    }
  };

  const columns = [
    {
      title: 'Этап',
      dataIndex: 'stage',
      key: 'stage',
      render: (stage) => <Tag color="orange">{STAGE_LABELS[stage] || stage}</Tag>,
    },
    { title: 'Пациент', dataIndex: 'patient_id', key: 'patient_id' },
    { title: 'Сессия', dataIndex: 'session_id', key: 'session_id' },
    { title: 'Причина', dataIndex: 'reason', key: 'reason' },
  ];

  return (
    <Modal
      title="Потерянные пациенты по этапам"
      open={visible}
      onCancel={onClose}
      width={800}
      footer={null}
    >
      {error && <Alert type="error" description={error} showIcon style={{ marginBottom: 16 }} />}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
        </div>
      ) : (
        <Table
          columns={columns}
          dataSource={losses}
          rowKey={(r, idx) => `${r.stage}_${r.patient_id}_${r.session_id}_${idx}`}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: 'Потерянных пациентов не найдено' }}
        />
      )}
    </Modal>
  );
};

export default PipelineLosses;
```

- [ ] **Step 3: `PipelineHistory.jsx` — new button**

Read the file first to confirm current line numbers. Add `onShowPipelineLosses` to the destructured props (currently line 17):

```jsx
const PipelineHistory = ({ onShowVisualization, onShowQualityReport, onShowClinicalReport, onShowIncompletePatients, onShowPipelineLosses }) => {
```

Add the button right after the existing "Неполные пациенты" button (currently lines 247-255), same gate condition (losses can occur starting at stage 1, same as the incomplete-patients queue):

```jsx
          {record.current_stage >= 1 && record.status !== 'pending' && (
            <Button
              type="link"
              size="small"
              onClick={() => onShowPipelineLosses(record.run_id)}
            >
              Потерянные пациенты
            </Button>
          )}
```

- [ ] **Step 4: `App.jsx` — wire it up**

Read the file first to confirm current line numbers. Add the import (alongside `import IncompletePatients from './components/IncompletePatients';`, currently line 15):

```jsx
import PipelineLosses from './components/PipelineLosses';
```

Add state (alongside `historyIncompletePatientsRunId`/`showHistoryIncompletePatients`, currently lines 35-37):

```jsx
  const [historyPipelineLossesRunId, setHistoryPipelineLossesRunId] = useState(null);
  const [showHistoryPipelineLosses, setShowHistoryPipelineLosses] = useState(false);
```

Add a handler (alongside `handleShowHistoryIncompletePatients`, currently lines 100-107):

```jsx
  /**
   * Показать отчёт о потерянных пациентах по всем этапам
   */
  const handleShowHistoryPipelineLosses = (runId) => {
    setHistoryPipelineLossesRunId(runId);
    setShowHistoryPipelineLosses(true);
  };
```

Pass the new prop into `<PipelineHistory>` (currently around line 234, alongside `onShowIncompletePatients`):

```jsx
                      onShowPipelineLosses={handleShowHistoryPipelineLosses}
```

Mount the modal (alongside the `showHistoryIncompletePatients` block, currently lines 278-289):

```jsx
            {showHistoryPipelineLosses && (
              <PipelineLosses
                runId={historyPipelineLossesRunId}
                visible={showHistoryPipelineLosses}
                onClose={() => setShowHistoryPipelineLosses(false)}
              />
            )}
```

- [ ] **Step 5: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors from the four touched/created files.

- [ ] **Step 6: Build check**

Run: `cd frontend && npm run build`
Expected: builds successfully.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/services/api.js frontend/src/components/PipelineLosses.jsx frontend/src/components/PipelineHistory.jsx frontend/src/App.jsx
git commit -m "feat(frontend): show the per-stage patient-losses report from run history (KI-054)"
```

---

### Task 5: KI-055 — merged sessions stop showing after a real requeue

**Files:**
- Modify: `backend/pipeline_manager.py` (`merge_sessions`, currently lines 651-737; `get_incomplete_patients`, currently lines 553-624)
- Modify: `backend/app.py` (`merge_sessions` and `get_incomplete_patients` endpoints)
- Test: `backend/test_incomplete_patients_api.py`

**Interfaces:**
- Consumes: none new.
- Produces: `merge_sessions(..., run_id: Optional[str] = None)`, `get_incomplete_patients(..., current_run_id: Optional[str] = None)` — both parameters default so every existing call site keeps working unmodified.

- [ ] **Step 1: Write the failing tests**

Add to `backend/test_incomplete_patients_api.py`, as a new class right after `TestGetIncompletePatientsMerged`:

```python
class TestMergeSessionsRunIdScoping:
    def test_merge_stores_merged_at_run_id(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {"original_date": "20230201", "status": "incomplete", "series": {}, "excluded_series": []},
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002", run_id="run-A")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        donor = mapping["patients"]["sub-001"]["sessions"]["ses-002"]
        assert donor["merged_at_run_id"] == "run-A"

    def test_merged_session_visible_when_queried_from_the_same_run_id(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {"original_date": "20230201", "status": "incomplete", "series": {}, "excluded_series": []},
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002", run_id="run-A")

        result = pm.get_incomplete_patients(str(tmp_path), current_run_id="run-A")
        keys = {(r["patient_id"], r["session_id"]) for r in result}
        assert ("sub-001", "ses-002") in keys

    def test_merged_session_hidden_when_queried_from_a_different_run_id(self, tmp_path):
        """After a real requeue (new run_id, stage 1 re-ran), the merged
        donor entry from the OLD run_id must stop cluttering the queue."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {"original_date": "20230201", "status": "incomplete", "series": {}, "excluded_series": []},
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002", run_id="run-A")

        result = pm.get_incomplete_patients(str(tmp_path), current_run_id="run-B")
        keys = {(r["patient_id"], r["session_id"]) for r in result}
        assert ("sub-001", "ses-002") not in keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k TestMergeSessionsRunIdScoping`
Expected: `test_merge_stores_merged_at_run_id` FAILS with `TypeError: merge_sessions() got an unexpected keyword argument 'run_id'`. `test_merged_session_hidden_when_queried_from_a_different_run_id` would fail on the assertion (the session is currently ALWAYS included once merged, regardless of any run_id) once the `run_id` TypeError above is fixed enough to reach it — run this step again after Step 3's `merge_sessions` signature change alone if needed to confirm the visibility logic itself is what's still missing.

- [ ] **Step 3: Implement**

In `backend/pipeline_manager.py`'s `merge_sessions` (currently lines 651-654 for the signature, 726-727 for where `status`/`merged_into_session_id` are set):

```python
    def merge_sessions(
        self, output_path: str, patient_id: str,
        primary_session_id: str, donor_session_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
```

(Update the docstring's `Returns`/`Raises` sections only if you want to mention `run_id` — not required, keep the change minimal.)

```python
        donor_session_data['status'] = 'merged'
        donor_session_data['merged_into_session_id'] = primary_session_id
        donor_session_data['merged_at_run_id'] = run_id
```

In `get_incomplete_patients` (currently lines 553 for the signature, 583-594 for the loop and filter):

```python
    def get_incomplete_patients(
        self, output_path: str, lesion_type: str = 'glioblastoma',
        current_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
```

```python
        for patient_id, patient_data in mapping_data.get('patients', {}).items():
            for session_id, session_data in patient_data.get('sessions', {}).items():
                status = session_data.get('status')
                has_alternatives = bool(session_data.get('excluded_series'))
                manually_reviewed = session_data.get('manually_reviewed', False)
                # A merged donor is visible only from the SAME run_id it was
                # merged under (or when current_run_id isn't given at all,
                # preserving old callers' behavior) — see KI-055 in
                # KNOWN_ISSUES.md. Once a real requeue produces a NEW
                # run_id and stage 1 has re-run, the merge has already
                # served its "confirm this happened" purpose and shouldn't
                # keep cluttering the queue.
                is_visible_merge = status == 'merged' and (
                    current_run_id is None
                    or session_data.get('merged_at_run_id') == current_run_id
                )
                needs_review = (
                    status == 'incomplete'
                    or (status == 'complete' and has_alternatives)
                    or status == 'discarded'
                    or is_visible_merge
                    or manually_reviewed
                )
```

(The `if manually_reviewed and status == 'complete' and not has_alternatives:` block and the `results.append({...})` block below are unchanged.)

In `backend/app.py`, update the two call sites. The `get_incomplete_patients` endpoint (currently line 924):

```python
    sessions = pipeline_manager.get_incomplete_patients(
        run.output_path, lesion_type=run.lesion_type or 'glioblastoma', current_run_id=run_id,
    )
```

The `merge_sessions` endpoint (currently lines 1001-1006):

```python
        result = pipeline_manager.merge_sessions(
            output_path=run.output_path,
            patient_id=patient_id,
            primary_session_id=request.primary_session_id,
            donor_session_id=request.donor_session_id,
            run_id=run_id,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k "TestMergeSessionsRunIdScoping or TestGetIncompletePatientsMerged or TestMergeSessions"`
Expected: PASS (all — the pre-existing `TestGetIncompletePatientsMerged`/`TestMergeSessions` tests call `get_incomplete_patients`/`merge_sessions` without the new parameters, so `current_run_id` defaults to `None`, preserving the old "always visible" behavior for them — confirm none of those regress).

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: baseline from Task 3 plus this task's 3 new tests, no regressions. Run `git checkout -- configs/preprocessing_versions.json` afterward before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/app.py backend/test_incomplete_patients_api.py
git commit -m "fix(backend): merged sessions stop appearing once viewed from a new run_id after a real requeue (KI-055)"
```

---

### Task 6: Manual end-to-end verification

**Files:** none modified — verification only. Folds together the frontend part of KI-054 (only piece not covered by Task 1-5's automated tests) with a sanity pass over all four fixes together.

- [ ] **Step 1: Rebuild and restart the stack**

```bash
docker compose --profile full up --build
```

- [ ] **Step 2: Verify KI-054's UI**

From "История запусков", open a completed/failed run that had real losses (e.g. the BO full-dataset run investigated for this plan). Click "Потерянные пациенты" — confirm the modal shows one row per lost patient/session, grouped visibly by stage tag, with a readable reason per row. Confirm a run with no losses shows the empty-state message instead of an empty table with no explanation.

- [ ] **Step 3: Verify KI-052/053 don't regress anything visible**

Start an ordinary small demo run end-to-end (a handful of patients) — confirm it still completes normally and quickly (the new timeout formula gives a small dataset a lower, not higher, ceiling than before only if patient_count is small enough that `1200 + 90*n < 7200` — for n up to ~66 patients the new timeout is actually SHORTER than the old flat 7200s; call this out explicitly if the demo dataset has more than ~66 patients, otherwise no visible behavior change is expected for a normal demo run).

- [ ] **Step 4: Verify KI-055's behavior with a real merge + requeue**

Using a patient with 2+ sessions in the review queue, merge one into the other, assign a pulled candidate to complete it, then requeue (per the existing `feat/requeue-progress-linking`/`feat/manual-session-merge` flow). Once the requeue's own stage 1 completes, reopen "Пациенты, требующие внимания" for the NEW run — confirm the merged donor session (from the OLD run) is no longer listed.

- [ ] **Step 5: Report findings**

Note any issues found — this is real, unscripted verification across all four fixes together.
