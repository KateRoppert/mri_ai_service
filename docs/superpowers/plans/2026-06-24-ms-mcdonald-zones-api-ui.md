# MS McDonald Zone Classification — API + UI Plan (2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `*_mcdonald_report.json` (produced by plan 2a's `MSZoneAnalyzer`) through the backend API, the Kappa validation flow, and a new section in the MS clinical report UI.

**Architecture:** New Pydantic response models + `pipeline_manager.get_mcdonald_reports()` + `GET /api/mcdonald-reports/{run_id}`, modeled directly on the existing lobar-report plumbing (`get_lobar_reports` / `/api/lobar-reports/{run_id}`). `KappaUploader` gets a third report type alongside `lobar_report`/`lesion_stats`. `ClinicalReportContent.jsx` gets a new "Локализация очагов (McDonald)" section in the MS render path, fed by either the local API or the Kappa-normalized path (single render code path, per the existing `normalizeKappaEntity` pattern).

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy, pytest, React, antd, axios.

**Depends on:** `docs/superpowers/plans/2026-06-23-ms-mcdonald-zones-pipeline.md` (plan 2a) — this plan consumes the exact `*_mcdonald_report.json` schema that `MSZoneAnalyzer.analyze_mask()` produces:
```json
{
  "mask_file": "...", "patient_id": "...", "session_id": "...",
  "total_lesion_count": 5,
  "zones": {
    "periventricular":  {"lesion_count": 2, "total_volume_cm3": 1.5},
    "juxtacortical":     {"lesion_count": 1, "total_volume_cm3": 0.3},
    "infratentorial":    {"lesion_count": 1, "total_volume_cm3": 0.2},
    "deep_white_matter": {"lesion_count": 1, "total_volume_cm3": 0.1},
    "spinal_cord": {"supported": false}
  },
  "lesion_zones_by_label": {"1": "periventricular", "2": "periventricular", "3": "juxtacortical", "4": "infratentorial", "5": "deep_white_matter"}
}
```
(`patient_id`/`session_id` are injected by `08_anatomical_analysis.py::process_one_mask` after `analyze_mask()` returns — same as for `lobar_report.json` and `lesion_stats_report.json` today.)

## Global Constraints

- Zero behavior change to glioblastoma — no GBM code path is touched in this plan.
- Spinal cord stays explicitly "unsupported" end-to-end — the UI must show it as such, never hide it.
- Conventional commits, one commit per task minimum.
- Every touched `.py` file passes `python -m py_compile` before commit.
- This project has no frontend test framework configured yet (no vitest/jest in `frontend/package.json`) — frontend changes are verified manually via the dev server per Task 6, not via new automated UI tests. Do not introduce a test framework as a side effect of this plan.

---

### Task 1: Pydantic response models + `pipeline_manager.get_mcdonald_reports()`

**Files:**
- Modify: `backend/models.py` (add after `LobarReportListResponse`, line 104)
- Modify: `backend/pipeline_manager.py` (add after `get_lobar_reports`, line ~465)
- Test: `backend/test_pipeline_manager_mcdonald.py`

**Interfaces:**
- Produces: `models.McDonaldZoneResult`, `models.McDonaldSpinalCordStatus`, `models.McDonaldReportResponse`, `models.McDonaldReportListResponse`.
- Produces: `PipelineManager.get_mcdonald_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]` — same contract shape as `get_lobar_reports`.

- [ ] **Step 1: Write the failing test**

Create `backend/test_pipeline_manager_mcdonald.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _write_report(seg_dir: Path, subject: str, session: str, data: dict):
    out_dir = seg_dir / subject / session / "anat" / "multiple_sclerosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{subject}_{session}_t1_mcdonald_report.json"
    report_path.write_text(json.dumps(data), encoding="utf-8")
    return report_path


def test_returns_none_when_segmentation_dir_missing(tmp_path):
    manager = PipelineManager()
    result = manager.get_mcdonald_reports(str(tmp_path / "does_not_exist"))
    assert result is None


def test_returns_none_when_no_mcdonald_reports_found(tmp_path):
    seg_dir = tmp_path / "segmentation"
    seg_dir.mkdir()
    manager = PipelineManager()
    result = manager.get_mcdonald_reports(str(tmp_path))
    assert result is None


def test_loads_all_mcdonald_reports(tmp_path):
    seg_dir = tmp_path / "segmentation"
    seg_dir.mkdir()
    report_data = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001",
        "session_id": "ses-001",
        "total_lesion_count": 2,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 1, "total_volume_cm3": 0.2},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular", "2": "juxtacortical"},
    }
    _write_report(seg_dir, "sub-001", "ses-001", report_data)

    manager = PipelineManager()
    result = manager.get_mcdonald_reports(str(tmp_path))

    assert result is not None
    assert len(result) == 1
    assert result[0]["total_lesion_count"] == 2
    assert result[0]["zones"]["spinal_cord"] == {"supported": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_mcdonald.py -v`
Expected: `AttributeError: 'PipelineManager' object has no attribute 'get_mcdonald_reports'`

- [ ] **Step 3: Add the Pydantic models — `backend/models.py`**

Insert immediately after `LobarReportListResponse` (currently ending at line 104, right before the `# МОДЕЛИ ДЛЯ ЗАПУСКА PIPELINE` comment):

```python
class McDonaldZoneResult(BaseModel):
    """Результат классификации очагов МС по одной McDonald-зоне"""
    lesion_count: int = Field(..., description="Количество очагов в зоне")
    total_volume_cm3: float = Field(..., description="Суммарный объём очагов в зоне, см³")

class McDonaldSpinalCordStatus(BaseModel):
    """Зона spinal cord не реализована в этой версии — явный статус, не пропуск поля"""
    supported: bool = Field(False, description="Всегда False — зона не поддерживается")

class McDonaldReportResponse(BaseModel):
    """Отчёт о McDonald-классификации очагов МС для одной маски"""
    mask_file: str = Field(..., description="Имя файла маски")
    patient_id: str = Field(..., description="ID пациента")
    session_id: str = Field(..., description="ID сессии")
    total_lesion_count: int = Field(..., description="Всего очагов")
    zones: Dict[str, Union[McDonaldZoneResult, McDonaldSpinalCordStatus]] = Field(
        ..., description="Результаты по зонам, включая spinal_cord как {'supported': false}"
    )
    lesion_zones_by_label: Dict[str, str] = Field(default_factory=dict, description="Зона для каждого очага по его label")

class McDonaldReportListResponse(BaseModel):
    """Список McDonald-отчётов"""
    total: int = Field(..., description="Количество отчётов")
    reports: List[McDonaldReportResponse] = Field(..., description="Список отчётов")
```

`backend/models.py:6` currently reads `from typing import Optional, List, Dict` — it does not import `Union`. Change it to:

```python
from typing import Optional, List, Dict, Union
```

- [ ] **Step 4: Add `PipelineManager.get_mcdonald_reports` — `backend/pipeline_manager.py`**

Insert immediately after `get_lobar_reports` (ends around line 465, right before `get_lesion_stats_reports`):

```python
    def get_mcdonald_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить все отчёты о McDonald-классификации очагов МС из segmentation/

        Структура: segmentation/sub-XXX/ses-XXX/anat/multiple_sclerosis/*_mcdonald_report.json

        Returns:
            Список словарей с отчётами
        """
        seg_dir = Path(output_path) / "segmentation"

        if not seg_dir.exists():
            logger.warning(f"Директория сегментации не найдена: {seg_dir}")
            return None

        report_files = list(seg_dir.rglob("*_mcdonald_report.json"))

        if not report_files:
            logger.warning(f"McDonald-отчёты не найдены в {seg_dir}")
            return None

        logger.info(f"Найдено {len(report_files)} McDonald-отчётов")

        reports = []
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                reports.append(report_data)
                logger.info(f"McDonald-отчёт загружен: {report_file.name}, очагов: {report_data.get('total_lesion_count')}")
            except Exception as e:
                logger.error(f"Ошибка чтения McDonald-отчёта {report_file}: {e}")
                continue

        if not reports:
            logger.warning("Не удалось загрузить ни одного McDonald-отчёта")
            return None

        logger.info(f"Успешно загружено {len(reports)} McDonald-отчётов")
        return reports
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_mcdonald.py -v`
Expected: `3 passed`

- [ ] **Step 6: Validate the new Pydantic models against a real sample dict**

Add this test to the same file (it would have failed at Step 1 too, but is written now since it depends on `models.py` from Step 3):

```python
def test_mcdonald_response_model_validates_real_shape():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    from models import McDonaldReportResponse

    sample = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001",
        "session_id": "ses-001",
        "total_lesion_count": 2,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 1, "total_volume_cm3": 0.2},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular", "2": "juxtacortical"},
    }

    parsed = McDonaldReportResponse(**sample)
    assert parsed.zones["periventricular"].lesion_count == 1
    assert parsed.zones["spinal_cord"].supported is False
```

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_mcdonald.py -v`
Expected: `4 passed` — this confirms the `Union[McDonaldZoneResult, McDonaldSpinalCordStatus]` dict typing correctly discriminates the two shapes (Pydantic v2 picks the matching union member per key; a model mismatch here would surface as a validation error, not a silent wrong type).

- [ ] **Step 7: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile backend/models.py backend/pipeline_manager.py`
Expected: no output, exit code 0

- [ ] **Step 8: Commit**

```bash
git add backend/models.py backend/pipeline_manager.py backend/test_pipeline_manager_mcdonald.py
git commit -m "feat(backend): add McDonald report models and pipeline_manager.get_mcdonald_reports"
```

---

### Task 2: `GET /api/mcdonald-reports/{run_id}` endpoint

**Files:**
- Modify: `backend/app.py` (add after `get_lobar_reports` route, line ~759)
- Test: `backend/test_app_mcdonald_endpoint.py`

**Interfaces:**
- Produces: `GET /api/mcdonald-reports/{run_id}` → `McDonaldReportListResponse`, mirroring `/api/lobar-reports/{run_id}`'s gating (404 if run not found, 400 if `current_stage < 7`, 404 if no reports).

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_mcdonald_endpoint.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app, pipeline_manager


client = TestClient(app)


def _fake_run(current_stage=8, output_path="/fake/output"):
    return SimpleNamespace(current_stage=current_stage, output_path=output_path)


def test_404_when_run_not_found():
    with patch("app.get_pipeline_run", return_value=None):
        response = client.get("/api/mcdonald-reports/nonexistent-run")
    assert response.status_code == 404


def test_400_when_stage_not_completed():
    with patch("app.get_pipeline_run", return_value=_fake_run(current_stage=3)):
        response = client.get("/api/mcdonald-reports/some-run")
    assert response.status_code == 400


def test_404_when_no_reports_found():
    with patch("app.get_pipeline_run", return_value=_fake_run()):
        with patch.object(pipeline_manager, "get_mcdonald_reports", return_value=None):
            response = client.get("/api/mcdonald-reports/some-run")
    assert response.status_code == 404


def test_200_with_reports():
    sample_report = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001",
        "session_id": "ses-001",
        "total_lesion_count": 1,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular"},
    }
    with patch("app.get_pipeline_run", return_value=_fake_run()):
        with patch.object(pipeline_manager, "get_mcdonald_reports", return_value=[sample_report]):
            response = client.get("/api/mcdonald-reports/some-run")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["reports"][0]["zones"]["spinal_cord"]["supported"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_app_mcdonald_endpoint.py -v`
Expected: `404` instead of `200` for `test_200_with_reports` (and similar) — route doesn't exist yet, FastAPI returns 404 for unmatched paths. (If `app.py` fails to import at all due to missing config/DB setup in this environment, fix the import error first — this project's other backend modules already import cleanly per `test_kappa_uploader.py`'s existing pattern, so `app.py` should too.)

- [ ] **Step 3: Write the implementation — `backend/app.py`**

Insert immediately after the `get_lobar_reports` route (ends at line 759, right before `@app.get("/api/lesion-stats/{run_id}"...)`:

```python
@app.get("/api/mcdonald-reports/{run_id}", response_model=McDonaldReportListResponse)
async def get_mcdonald_reports(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить отчёты о McDonald-классификации очагов МС (зональная локализация)
    """
    logger.info(f"Запрос McDonald-отчётов для run_id: {run_id}")

    run = get_pipeline_run(db, run_id)

    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    if run.current_stage < 7:
        raise HTTPException(
            status_code=400,
            detail="Anatomical analysis stage not yet completed"
        )

    reports = pipeline_manager.get_mcdonald_reports(run.output_path)

    if not reports:
        raise HTTPException(status_code=404, detail="McDonald reports not found (MS only)")

    logger.info(f"Найдено {len(reports)} McDonald-отчётов для run_id: {run_id}")

    return McDonaldReportListResponse(
        total=len(reports),
        reports=reports
    )
```

In `backend/app.py`, the `from models import (...)` block (lines 23-42) currently has `LobarReportListResponse,` at line 37. Change that line to:

```python
    LobarReportListResponse,
    McDonaldReportListResponse,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_app_mcdonald_endpoint.py -v`
Expected: `4 passed`

- [ ] **Step 5: Run the full backend test suite to confirm no regressions**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_mcdonald.py test_app_mcdonald_endpoint.py test_mask_service.py test_patient_registry.py test_preprocessing_version.py -v`
Expected: all pass

- [ ] **Step 6: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile backend/app.py`
Expected: no output, exit code 0

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/test_app_mcdonald_endpoint.py
git commit -m "feat(backend): add GET /api/mcdonald-reports/{run_id} endpoint"
```

---

### Task 3: Kappa validation flow — surface `mcdonald_report` in `dsEntityInfo`

**Files:**
- Modify: `backend/kappa_uploader.py:202-297` (`_discover_sessions`)
- Modify: `backend/kappa_uploader.py:483-587` (`_build_entity_info`)
- Test: `backend/test_kappa_uploader_mcdonald.py`

**Interfaces:**
- Produces: `KappaUploader._discover_sessions()` results now include a `"mcdonald_report"` key per session (a `Path` or `None`, alongside the existing `"lobar_report"`/`"lesion_stats_report"` keys).
- Produces: `KappaUploader._build_entity_info()`'s returned dict now includes an `"mcdonald_report"` key (mirroring `"lesion_stats"`) when present.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_uploader_mcdonald.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kappa_uploader import KappaUploader

PROJECT_ROOT = Path(__file__).parent.parent
REAL_PREPROCESSING_CONFIG = PROJECT_ROOT / "configs" / "preprocessing_config.yaml"


def _make_uploader(output_path: Path) -> KappaUploader:
    return KappaUploader(
        run_id="test-run",
        output_path=str(output_path),
        token="dummy-token",
        user_id=1,
        user_type_id=1,
        lesion_type="multiple_sclerosis",
        preprocessing_config_path=str(REAL_PREPROCESSING_CONFIG),
    )


def _build_session_dirs(output_path: Path, subject: str, session: str):
    preproc_dir = output_path / "preprocessed" / subject / session / "anat"
    preproc_dir.mkdir(parents=True, exist_ok=True)
    (preproc_dir / f"{subject}_{session}_t1.nii.gz").write_bytes(b"")

    seg_dir = output_path / "segmentation" / subject / session / "anat" / "multiple_sclerosis"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / f"{subject}_{session}_t1_segmask.nii.gz").write_bytes(b"")
    return seg_dir


def test_discover_sessions_finds_mcdonald_report(tmp_path):
    seg_dir = _build_session_dirs(tmp_path, "sub-001", "ses-001")
    mcdonald_data = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001", "session_id": "ses-001",
        "total_lesion_count": 1,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular"},
    }
    report_path = seg_dir / "sub-001_ses-001_t1_mcdonald_report.json"
    report_path.write_text(json.dumps(mcdonald_data), encoding="utf-8")

    uploader = _make_uploader(tmp_path)
    sessions = uploader._discover_sessions()

    assert "sub-001_ses-001" in sessions
    assert sessions["sub-001_ses-001"]["mcdonald_report"] == report_path


def test_build_entity_info_includes_mcdonald_report(tmp_path):
    seg_dir = _build_session_dirs(tmp_path, "sub-001", "ses-001")
    mcdonald_data = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001", "session_id": "ses-001",
        "total_lesion_count": 1,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular"},
    }
    (seg_dir / "sub-001_ses-001_t1_mcdonald_report.json").write_text(
        json.dumps(mcdonald_data), encoding="utf-8"
    )

    uploader = _make_uploader(tmp_path)
    sessions = uploader._discover_sessions()
    info = uploader._build_entity_info("sub-001_ses-001", sessions["sub-001_ses-001"])

    assert info["mcdonald_report"]["total_lesion_count"] == 1
    assert info["mcdonald_report"]["zones"]["spinal_cord"] == {"supported": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_kappa_uploader_mcdonald.py -v`
Expected: `KeyError: 'mcdonald_report'` in `test_discover_sessions_finds_mcdonald_report` (the key doesn't exist in the session dict yet)

- [ ] **Step 3: Update `_discover_sessions`**

In `backend/kappa_uploader.py`, add `"mcdonald_report": None,` to the `sessions.setdefault(...)` dict (line 225-233):

```python
            sessions.setdefault(session_key, {
                "preprocessed": [],
                "masks": [],
                "quality_reports": [],
                "volume_report": None,
                "lobar_report": None,
                "lesion_stats_report": None,
                "mcdonald_report": None,
                "lesion_labels_mask": None,
            })
```

Add a new discovery loop right after the existing "Находим lesion_stats reports" loop (after line 277):

```python
        # Находим mcdonald reports (МС — классификация очагов по McDonald-зонам)
        for mr in sorted(segmentation_dir.rglob("*_mcdonald_report.json")):
            session_key = self._extract_session_key(mr)
            if session_key and session_key in sessions:
                sessions[session_key]["mcdonald_report"] = mr
```

- [ ] **Step 4: Update `_build_entity_info`**

Add this block right after the existing "Lesion stats report" block (after line 585, before `return info`):

```python
        # McDonald zone report (МС — классификация очагов по зонам)
        if session_data.get("mcdonald_report"):
            try:
                with open(session_data["mcdonald_report"], "r") as f:
                    mr = json.load(f)
                info["mcdonald_report"] = {
                    "total_lesion_count": mr.get("total_lesion_count"),
                    "zones": mr.get("zones", {}),
                    "lesion_zones_by_label": mr.get("lesion_zones_by_label", {}),
                }
            except Exception as e:
                logger.warning("Failed to read mcdonald report: %s", e)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_kappa_uploader_mcdonald.py -v`
Expected: `2 passed`

- [ ] **Step 6: Run the full backend test suite**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_mcdonald.py test_app_mcdonald_endpoint.py test_kappa_uploader_mcdonald.py -v`
Expected: all pass

- [ ] **Step 7: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile backend/kappa_uploader.py`
Expected: no output, exit code 0

- [ ] **Step 8: Commit**

```bash
git add backend/kappa_uploader.py backend/test_kappa_uploader_mcdonald.py
git commit -m "feat(kappa): surface mcdonald_report in dsEntityInfo for MS validation flow"
```

---

### Task 4: Frontend API client function

**Files:**
- Modify: `frontend/src/services/api.js:66-77`

**Interfaces:**
- Produces: `getMcdonaldReports(runId) -> Promise<{total, reports}>`

- [ ] **Step 1: Add the function**

In `frontend/src/services/api.js`, right after `getLobarReports` (ends at line 69), add:

```js
/**
 * Локализация очагов МС по McDonald-зонам (periventricular/juxtacortical/infratentorial)
 */
export const getMcdonaldReports = async (runId) => {
  const response = await apiClient.get(`/mcdonald-reports/${runId}`);
  return response.data;
};
```

- [ ] **Step 2: Add it to the default export object**

Find the `export default { ... getVolumeReports, getLobarReports, getLesionStatsReports, ... }` block (around line 306-308) and add `getMcdonaldReports,` next to `getLobarReports,`.

- [ ] **Step 3: Verify the file still parses**

Run: `cd /home/ubuntu/mri_ai_service/frontend && node --check src/services/api.js 2>&1 || npx eslint src/services/api.js`
Expected: no syntax errors reported. (`node --check` validates plain JS syntax; if the project's build uses JSX/ESM transforms that `node --check` chokes on for unrelated reasons, fall back to `npx eslint src/services/api.js` instead — either confirms the file is syntactically valid.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.js
git commit -m "feat(frontend): add getMcdonaldReports API client function"
```

---

### Task 5: Render the McDonald zone section in `ClinicalReportContent.jsx`

**Files:**
- Modify: `frontend/src/components/ClinicalReportContent.jsx`

**Interfaces:**
- Consumes: `getMcdonaldReports` (Task 4).
- Produces: a new "Локализация очагов (McDonald)" section visible in the MS render path, fed identically whether the data source is the local API or Kappa.

- [ ] **Step 1: Add zone display metadata and a state slot**

After the existing `LESION_SIZE_BANDS` constant (line 25-29), add:

```jsx
// McDonald zone display metadata (RU labels, colors). spinal_cord is rendered
// separately as an explicit "unsupported" tag, not as a regular zone row.
const MCDONALD_ZONES = [
  { key: 'periventricular',  label: 'Перивентрикулярная',  color: '#1890ff' },
  { key: 'juxtacortical',     label: 'Юкстакортикальная',    color: '#52c41a' },
  { key: 'infratentorial',    label: 'Инфратенториальная',   color: '#fa8c16' },
  { key: 'deep_white_matter', label: 'Глубокое белое вещество', color: '#8c8c8c' },
];
```

In the component body, add a new state slot right after `lesionStatsReports` (line 115):

```jsx
  const [mcdonaldReports, setMcdonaldReports] = useState([]);
```

- [ ] **Step 2: Extend `normalizeKappaEntity`**

Inside `normalizeKappaEntity` (starts at line 56), add a new normalized array after `lobarReports` (after line 87, before `volumeReports`):

```jsx
  const mcdonaldReports = info.mcdonald_report
    ? [{
        patient_id,
        session_id,
        total_lesion_count: info.mcdonald_report.total_lesion_count,
        zones: info.mcdonald_report.zones || {},
      }]
    : [];
```

Update the function's `return` statement (line 107) to include it:

```jsx
  return { volumeReports, lobarReports, lesionStatsReports, mcdonaldReports };
```

- [ ] **Step 3: Wire the Kappa effect, the reset effect, and `fetchAllData`**

In the Kappa-source `useEffect` (lines 120-129), destructure and set the new field:

```jsx
  useEffect(() => {
    if (kappaEntityInfo) {
      const { volumeReports: v, lobarReports: l, lesionStatsReports: s, mcdonaldReports: m } =
        normalizeKappaEntity(kappaEntityInfo);
      setVolumeReports(v);
      setLobarReports(l);
      setLesionStatsReports(s);
      setMcdonaldReports(m);
      setLoaded(true);
    }
  }, [kappaEntityInfo]);
```

In the reset `useEffect` (lines 141-149), add `setMcdonaldReports([]);` next to `setLesionStatsReports([]);`.

In `fetchAllData` (lines 159-183), add the fetch alongside the existing `lesion-stats` fetch:

```jsx
  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const fetches = [
        getVolumeReports(runId).catch(() => ({ reports: [] })),
        getLobarReports(runId).catch(() => ({ reports: [] })),
      ];
      if (lesionType === 'multiple_sclerosis') {
        fetches.push(getLesionStatsReports(runId).catch(() => ({ reports: [] })));
        fetches.push(getMcdonaldReports(runId).catch(() => ({ reports: [] })));
      }
      const results = await Promise.all(fetches);
      setVolumeReports(sortByPatientSession(results[0].reports));
      setLobarReports(results[1].reports || []);
      if (lesionType === 'multiple_sclerosis') {
        setLesionStatsReports(sortByPatientSession(results[2]?.reports));
        setMcdonaldReports(sortByPatientSession(results[3]?.reports));
      }
      setLoaded(true);
    } catch (err) {
      console.error('Ошибка загрузки отчёта:', err);
      setError('Не удалось загрузить данные');
    } finally {
      setLoading(false);
    }
  };
```

Add `getMcdonaldReports` to the existing `import { getVolumeReports, getLobarReports, getLesionStatsReports, getLongitudinalReport } from '../services/api';` line at the top of the file.

- [ ] **Step 4: Add the McDonald table columns and data mapper**

Add this alongside the existing `lobarColumns`/`getLobarTableData` (after line 423):

```jsx
  const mcdonaldColumns = [
    {
      title: 'Зона',
      dataIndex: 'label',
      key: 'label',
      width: 220,
      render: (text, record) => (
        <Space>
          <div style={{
            width: 12, height: 12, borderRadius: 2,
            backgroundColor: record.color,
            border: '1px solid rgba(0,0,0,0.1)',
          }} />
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: 'Очагов',
      dataIndex: 'lesion_count',
      key: 'count',
      width: 100,
      align: 'right',
    },
    {
      title: 'Объём (см³)',
      dataIndex: 'total_volume_cm3',
      key: 'volume',
      width: 120,
      align: 'right',
      render: (val) => val.toFixed(3),
    },
  ];

  const getMcdonaldTableData = (report) => {
    if (!report?.zones) return [];
    return MCDONALD_ZONES.map(({ key, label, color }) => ({
      key,
      label,
      color,
      lesion_count: report.zones[key]?.lesion_count ?? 0,
      total_volume_cm3: report.zones[key]?.total_volume_cm3 ?? 0,
    }));
  };
```

- [ ] **Step 5: Render the section in the MS render path**

In the MS render path (starts at line 443), insert a new section between the "Объёмы всех очагов" `<Collapse>` block (ends at line 515) and the "Динамика между сессиями" `<Divider>` (line 518). The MS branch currently maps over `lesionStatsReports`; find the matching `mcdonaldReports` entry by `patient_id`/`session_id`, same pattern as the GLIO branch's `lobar` lookup (line 536-538):

```jsx
              {/* Локализация очагов (McDonald) */}
              {(() => {
                const mcdonald = mcdonaldReports.find(
                  mr => mr.patient_id === stats.patient_id && mr.session_id === stats.session_id
                );
                if (!mcdonald) return null;
                return (
                  <>
                    <Divider orientation="left" style={{ fontSize: 14 }}>
                      <Space><EnvironmentOutlined /> Локализация очагов (McDonald)</Space>
                    </Divider>
                    <Table
                      columns={mcdonaldColumns}
                      dataSource={getMcdonaldTableData(mcdonald)}
                      pagination={false}
                      size="small"
                      bordered
                      style={{ marginBottom: 12, maxWidth: 500 }}
                    />
                    <Tag color="default" style={{ marginBottom: 16 }}>
                      Спинной мозг: классификация не поддерживается в этой версии
                    </Tag>
                  </>
                );
              })()}

```

Insert this block right before the existing `{/* Динамика между сессиями */}` comment (line 518), inside the same `.map((stats, idx) => ...)` callback (so `stats.patient_id`/`stats.session_id` are in scope).

- [ ] **Step 6: Manual verification (no automated frontend tests in this project — see Task 6)**

This step is intentionally deferred to Task 6, which runs the actual dev server.

- [ ] **Step 7: Compile-check via the project's lint config**

Run: `cd /home/ubuntu/mri_ai_service/frontend && npx eslint src/components/ClinicalReportContent.jsx src/services/api.js`
Expected: no errors (warnings about pre-existing code are fine; this step exists to catch syntax mistakes introduced by the edits above, like unmatched JSX tags).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ClinicalReportContent.jsx
git commit -m "feat(frontend): render McDonald zone localization section for MS reports"
```

---

### Task 6: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full touched backend test suite**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_mcdonald.py test_app_mcdonald_endpoint.py test_kappa_uploader_mcdonald.py -v`
Expected: all pass

- [ ] **Step 2: Start the backend and frontend dev servers**

Run (in separate terminals, or background):
```bash
cd /home/ubuntu/mri_ai_service && docker compose --profile full up --build
```
Wait for the web service to report ready (per the project's standard startup, see CLAUDE.md "Quick start").

- [ ] **Step 3: Exercise the new endpoint directly**

Run a pipeline for `data/MS_5/P000915` through at least Stage 08 (multiple_sclerosis), then:

```bash
curl -s http://localhost:8000/api/mcdonald-reports/<run_id> | python -m json.tool
```

Expected: JSON with `total` and `reports[].zones.spinal_cord == {"supported": false}`.

- [ ] **Step 4: Exercise the UI in a browser**

Open the frontend, navigate to the MS pipeline run's clinical report, and confirm:
- A new "Локализация очагов (McDonald)" section appears below "Объёмы всех очагов".
- It lists 4 zone rows (periventricular, juxtacortical, infratentorial, deep white matter) with counts and volumes.
- A grey "Спинной мозг: классификация не поддерживается" tag is visible — not silently absent.
- The existing glioblastoma report (a GBM run) renders exactly as before — no regression in the unrelated render path.

- [ ] **Step 5: Push**

```bash
git push
```

## Self-Review Notes

- **Spec coverage:** Completes the "(A) McDonald-классификация" feature end-to-end (pipeline from plan 2a + API/Kappa/UI here). The design spec's requirement that spinal cord be "явно помечается... не пропускается молча" is implemented at three layers: `MSZoneAnalyzer` (plan 2a) emits `{"supported": false}`, the Pydantic model (`McDonaldSpinalCordStatus`) types it explicitly rather than as a loose dict, and the UI renders a visible tag rather than omitting the zone.
- **Type consistency:** `McDonaldReportResponse.zones` (Task 1) is a `Dict[str, Union[McDonaldZoneResult, McDonaldSpinalCordStatus]]`; the frontend's `getMcdonaldTableData` (Task 5) only ever reads `report.zones[key]` for the four `McDonaldZoneResult`-shaped keys (never `spinal_cord`, which is handled separately as a static tag) — no shape confusion between the two union members at the call site.
- **No placeholders:** every step shows literal before/after code, full function bodies, or an exact command with expected output.
