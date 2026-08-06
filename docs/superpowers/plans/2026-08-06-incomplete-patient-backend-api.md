# Piece C-backend: Incomplete-Patient Review API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the doctor-facing UI (a later, separate plan) a backend to read incomplete sessions from a pipeline run, manually relabel an unrecognized series into a required modality, discard a session, and trigger reprocessing — all without a database, reading/writing the `dataset_mapping.json` piece B already produces.

**Architecture:** No DB migration. `PipelineManager` (already the layer `app.py` calls for run-scoped file reads, e.g. `get_mcdonald_reports`) gains a `get_incomplete_patients(output_path)` reader and a `relabel_series`/`discard_session` mutator, all operating on `<output_path>/bids_organized/dataset_mapping.json`. The DICOM copy+anonymize logic currently private to `FileOrganizer.copy_series()` in `scripts/01_reorganize_folders.py` is extracted into `utils/dicom_file_ops.py` so both stage 01 and the backend call the same function — no duplicated anonymization logic. Requeueing reuses the existing `PipelineManager.create_runtime_config()`/`start_pipeline()` pair (already how the web UI starts any run) pointed at the same paths; `skip_existing: true` (already the base config default) means only the newly-fixed patient actually gets reprocessed.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, pytest, SQLAlchemy (only for the existing `pipeline_runs` lookup — no schema change).

## Global Constraints

- No new DB tables/columns. Status lives in `dataset_mapping.json` only (see design spec `docs/superpowers/specs/2026-08-06-incomplete-patient-workflow-design.md`, "Backend API" section).
- `dataset_mapping.json` path is always `Path(run.output_path) / "bids_organized" / "dataset_mapping.json"` (confirmed via `pipeline_config.yaml`'s `output_structure.stage_01: bids_organized`).
- One list endpoint serves the whole review screen (no separate per-session detail endpoint) — matches the design's "one screen" decision.
- The shared `utils/dicom_file_ops.py` module must be the only place DICOM copy+anonymize logic lives; `FileOrganizer.copy_series()` becomes a thin wrapper around it, not a parallel implementation.
- Follow existing `backend/app.py` conventions exactly: Pydantic response models in `backend/models.py`, `@app.get/post("/api/...")` route style, `get_pipeline_run(db, run_id)` + `HTTPException(status_code=404, ...)` pattern for missing runs (see `get_mcdonald_reports` at `backend/app.py:803-833` as the reference).
- No frontend work in this plan — deferred to a separate C-frontend plan.

---

### Task 1: Extract `utils/dicom_file_ops.py` — shared DICOM file helpers

**Files:**
- Create: `utils/dicom_file_ops.py`
- Modify: `scripts/01_reorganize_folders.py:45-84` (remove local defs, import from the new module), `scripts/01_reorganize_folders.py:1118-1178` (`FileOrganizer.copy_series` delegates to the new function)
- Test: `tests/stage01/test_dicom_file_ops.py` (new)

**Interfaces:**
- Produces: `utils.dicom_file_ops.is_dicom_file(path) -> bool`, `utils.dicom_file_ops.find_dicom_files(directory, recursive=True) -> List[Path]`, `utils.dicom_file_ops.MODALITY_BIDS_SUFFIX: Dict[str, str]`, `utils.dicom_file_ops.copy_and_anonymize_series(source_files, target_dir, patient_id, session_id, modality, metadata_extractor=None, logger=None) -> int` — this last one is new (Task 2/3's future backend code and `FileOrganizer.copy_series` both call it).

- [ ] **Step 1: Write the failing test**

```python
# tests/stage01/test_dicom_file_ops.py
import logging
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from utils.dicom_file_ops import (
    is_dicom_file, find_dicom_files, MODALITY_BIDS_SUFFIX, copy_and_anonymize_series,
)

LOGGER = logging.getLogger("test_dicom_file_ops")


class TestModalityBidsSuffix:
    def test_known_modalities(self):
        assert MODALITY_BIDS_SUFFIX == {
            't1': 'T1w', 't1c': 'T1wCE', 't2': 'T2w', 't2fl': 'FLAIR',
        }


class TestIsDicomFile:
    def test_dcm_extension_trusted_without_reading(self, tmp_path):
        f = tmp_path / "not_really_dicom.dcm"
        f.write_bytes(b"garbage")
        assert is_dicom_file(f) is True

    def test_extensionless_file_checked_by_magic_bytes(self, tmp_path):
        f = tmp_path / "23831328"
        f.write_bytes(b"\x00" * 128 + b"DICM" + b"restofdata")
        assert is_dicom_file(f) is True

    def test_non_dicom_extensionless_file_rejected(self, tmp_path):
        f = tmp_path / "README"
        f.write_bytes(b"just some text, not a dicom file at all")
        assert is_dicom_file(f) is False


class TestFindDicomFiles:
    def test_finds_dcm_files_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.dcm").write_bytes(b"x")
        (tmp_path / "sub" / "b.dcm").write_bytes(b"x")
        (tmp_path / "readme.txt").write_bytes(b"not dicom")
        found = find_dicom_files(tmp_path)
        assert len(found) == 2
        assert all(f.suffix == ".dcm" for f in found)


class TestCopyAndAnonymizeSeries:
    def test_plain_copy_without_metadata_extractor(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "IM-0001.dcm").write_bytes(b"fake dicom content")
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        source_files = find_dicom_files(source_dir)
        copied = copy_and_anonymize_series(
            source_files, target_dir, "sub-001", "ses-001", "t1c",
        )

        assert copied == 1
        assert (target_dir / "sub-001_ses-001_T1wCE_0001.dcm").exists()

    def test_returns_zero_and_logs_on_write_failure(self, tmp_path, caplog):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "IM-0001.dcm").write_bytes(b"fake dicom content")
        # target_dir intentionally NOT created -> shutil.copy2 raises FileNotFoundError
        target_dir = tmp_path / "does_not_exist"

        source_files = find_dicom_files(source_dir)
        copied = copy_and_anonymize_series(
            source_files, target_dir, "sub-001", "ses-001", "t1",
        )
        assert copied == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/stage01/test_dicom_file_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.dicom_file_ops'`

- [ ] **Step 3: Create the shared module**

Create `utils/dicom_file_ops.py` — move `is_dicom_file`, `find_dicom_files`, `MODALITY_BIDS_SUFFIX` verbatim from `scripts/01_reorganize_folders.py:45-84` (exact current code, do not alter), then add the new function:

```python
"""Shared DICOM file operations used by both stage 01 (01_reorganize_folders.py)
and the backend API (incomplete-patient manual relabel endpoint)."""
import shutil
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import pydicom

if TYPE_CHECKING:
    from scripts.metadata_extractor import MetadataExtractor

MODALITY_BIDS_SUFFIX: Dict[str, str] = {
    't1':   'T1w',
    't1c':  'T1wCE',
    't2':   'T2w',
    't2fl': 'FLAIR',
}


def is_dicom_file(path: Path) -> bool:
    """
    True if `path` is a DICOM file, independent of extension (KI-029).

    Real clinical exports are not always named ``*.dcm`` — some vendors
    write extensionless files (e.g. ``23831328``), others use ``.IMA`` or
    ``.dicom``. Extension alone is unreliable both ways: it misses files
    without ``.dcm``, and (in theory) could match a non-DICOM file someone
    renamed to ``.dcm``.

    Fast path: files already named ``*.dcm`` are trusted directly, so
    datasets that already use this convention (the common case) pay no
    extra I/O. Anything else is checked via the standard DICOM Part-10
    magic marker: a 128-byte preamble followed by ``b'DICM'`` at offset
    128. This is authoritative — unrelated files (README.txt, DICOMDIR,
    thumbnails) will not carry it by coincidence — and correctly
    recognizes extensionless DICOM and other extensions uniformly.
    """
    if path.suffix.lower() == '.dcm':
        return True
    try:
        with open(path, 'rb') as f:
            header = f.read(132)
    except OSError:
        return False
    return len(header) == 132 and header[128:132] == b'DICM'


def find_dicom_files(directory: Path, recursive: bool = True) -> List[Path]:
    """All DICOM files under `directory`, any extension or none (see is_dicom_file)."""
    pattern = '**/*' if recursive else '*'
    return sorted(f for f in directory.glob(pattern) if f.is_file() and is_dicom_file(f))


def copy_and_anonymize_series(
    source_files: List[Path],
    target_dir: Path,
    patient_id: str,
    session_id: str,
    modality: str,
    metadata_extractor: Optional['MetadataExtractor'] = None,
    logger=None,
) -> int:
    """
    Copy DICOM files to target_dir with BIDS naming, anonymizing each file
    if metadata_extractor is provided (else a plain copy). Caller is
    responsible for calling find_dicom_files() to build source_files and
    for creating target_dir beforehand.

    Returns:
        Number of files successfully copied.
    """
    bids_suffix = MODALITY_BIDS_SUFFIX.get(modality, modality.upper())
    copied = 0
    for idx, source_file in enumerate(source_files, 1):
        target_name = f"{patient_id}_{session_id}_{bids_suffix}_{idx:04d}.dcm"
        target_path = target_dir / target_name
        try:
            if metadata_extractor:
                dcm = pydicom.dcmread(str(source_file), force=True)
                removed = metadata_extractor.anonymize_dicom(dcm)
                dcm.save_as(str(target_path))
                if idx == 1 and logger:
                    logger.info(
                        f"    Anonymized: removed {len(removed)} tags: "
                        f"{', '.join(removed)}"
                    )
            else:
                shutil.copy2(source_file, target_path)
            copied += 1
        except Exception as e:
            if logger:
                logger.error(f"Failed to process {source_file}: {e}")
    return copied
```

In `scripts/01_reorganize_folders.py`, remove the local `is_dicom_file`/`find_dicom_files`/`MODALITY_BIDS_SUFFIX` definitions (lines 45-84) and replace with an import near the existing `from utils.config_loader import ...` line (line 37):

```python
from utils.dicom_file_ops import is_dicom_file, find_dicom_files, MODALITY_BIDS_SUFFIX, copy_and_anonymize_series
```

Replace `FileOrganizer.copy_series` (lines 1118-1178) with a thin wrapper:

```python
    def copy_series(
        self,
        source_dir: Path,
        target_dir: Path,
        patient_id: str,
        session_id: str,
        modality: str
    ) -> int:
        """
        Copy DICOM series to target directory with BIDS naming.
        If metadata_extractor is provided, extracts metadata from the first
        file and anonymizes all files by removing configured tags before saving.

        Returns:
            Number of files copied
        """
        source_files = find_dicom_files(source_dir)

        if not source_files:
            self.logger.warning(f"No DICOM files in {source_dir}")
            return 0

        if self.dry_run:
            count = len(source_files)
            self.files_would_copy += count
            self.logger.debug(f"    [DRY RUN] Would copy {count} files for {modality}")
            return count

        # Extract and save metadata from first file before anonymization
        if self.metadata_extractor and self.metadata_dir:
            self._extract_and_save_metadata(
                source_files[0], patient_id, session_id, modality
            )

        copied = copy_and_anonymize_series(
            source_files, target_dir, patient_id, session_id, modality,
            metadata_extractor=self.metadata_extractor, logger=self.logger,
        )
        self.files_copied += copied
        return copied
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/stage01/test_dicom_file_ops.py -v`
Expected: PASS (6/6)

- [ ] **Step 5: Run full stage01 suite — this is a refactor, must show ZERO behavior change**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: same pass count as before this task (104, per the piece-B ledger), no regressions. If any test that constructs a `FileOrganizer` and calls `copy_series` fails, the extraction introduced a behavior change — stop and fix before proceeding, do not paper over it.

- [ ] **Step 6: Commit**

```bash
git add utils/dicom_file_ops.py scripts/01_reorganize_folders.py tests/stage01/test_dicom_file_ops.py
git commit -m "refactor(stage01): extract DICOM copy/anonymize logic to utils/dicom_file_ops.py"
```

---

### Task 2: `PipelineManager.get_incomplete_patients()` + `GET /api/incomplete-patients/{run_id}`

**Files:**
- Modify: `backend/pipeline_manager.py` (new method, place near `get_mcdonald_reports`/`get_lesion_stats_reports`, currently around line 468-551)
- Modify: `backend/models.py` (new Pydantic models, place near `McDonaldReportResponse`/`McDonaldReportListResponse`, currently around line 115-129)
- Modify: `backend/app.py` (new endpoint, place near `get_mcdonald_reports`, currently around line 803-833; add new model names to the `from models import (...)` block at line 37)
- Test: `backend/test_incomplete_patients_api.py` (new)

**Interfaces:**
- Consumes: `dataset_mapping.json`'s per-session shape from piece B — `patients[patient_id]['sessions'][session_id]` has `original_date`, `status` (`"complete"`/`"incomplete"`), `series` (dict of modality -> series info), `unrecognized_series` (list of `{original_path, series_description, slice_count}`).
- Produces: `PipelineManager.get_incomplete_patients(output_path: str) -> List[Dict]`; `GET /api/incomplete-patients/{run_id}` returning `IncompletePatientsResponse`. Task 3/4 reuse `get_incomplete_patients`'s path-resolution logic (`Path(output_path) / "bids_organized" / "dataset_mapping.json"`) — keep it as a small reusable private helper `_dataset_mapping_path(output_path)` on `PipelineManager` rather than inlining the path twice.

- [ ] **Step 1: Write the failing test**

```python
# backend/test_incomplete_patients_api.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _write_mapping(output_dir: Path, patients: dict):
    bids_dir = output_dir / "bids_organized"
    bids_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"patients": patients, "output_dir": str(bids_dir), "created_at": "x", "updated_at": "x"}
    (bids_dir / "dataset_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")


class TestGetIncompletePatients:
    def test_returns_only_incomplete_non_discarded_sessions(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "complete",
                        "series": {"t1": {}, "t1c": {}, "t2": {}, "t2fl": {}},
                        "unrecognized_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230115",
                        "status": "incomplete",
                        "series": {"t1c": {}},
                        "unrecognized_series": [
                            {"original_path": "/raw/weird", "series_description": "xyz", "slice_count": 20}
                        ],
                    },
                },
            },
            "sub-002": {
                "original_id": "P2",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "discarded",
                        "series": {"t1c": {}},
                        "unrecognized_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))

        assert len(result) == 1
        assert result[0]["patient_id"] == "sub-001"
        assert result[0]["session_id"] == "ses-002"
        assert result[0]["status"] == "incomplete"
        assert result[0]["available"] == ["t1c"]
        assert len(result[0]["unrecognized_series"]) == 1

    def test_missing_mapping_file_returns_empty_list(self, tmp_path):
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))
        assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest backend/test_incomplete_patients_api.py -v`
Expected: FAIL — `AttributeError: 'PipelineManager' object has no attribute 'get_incomplete_patients'`

- [ ] **Step 3: Implement `PipelineManager.get_incomplete_patients`**

Add to `backend/pipeline_manager.py`, near `get_mcdonald_reports`:

```python
    def _dataset_mapping_path(self, output_path: str) -> Path:
        return Path(output_path) / "bids_organized" / "dataset_mapping.json"

    def get_incomplete_patients(self, output_path: str) -> List[Dict[str, Any]]:
        """
        Read dataset_mapping.json and return every session whose status is
        not "complete" and not "discarded" — the doctor-review queue for a run.
        """
        mapping_file = self._dataset_mapping_path(output_path)
        if not mapping_file.exists():
            return []

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        results: List[Dict[str, Any]] = []
        for patient_id, patient_data in mapping_data.get('patients', {}).items():
            for session_id, session_data in patient_data.get('sessions', {}).items():
                status = session_data.get('status')
                if status not in ('incomplete',):
                    continue
                results.append({
                    "patient_id": patient_id,
                    "original_id": patient_data.get('original_id', ''),
                    "session_id": session_id,
                    "date": session_data.get('original_date', ''),
                    "status": status,
                    "available": sorted(session_data.get('series', {}).keys()),
                    "unrecognized_series": session_data.get('unrecognized_series', []),
                })
        return results
```

(Note: filtering on `status not in ('incomplete',)` rather than `status not in ('complete', 'discarded')` — this is deliberately the allow-list form, so any future status value other than exactly `"incomplete"` is excluded by default rather than silently included. Confirm `json` and `Path`/`Dict`/`Any`/`List` are already imported at the top of `pipeline_manager.py` — they are, per the existing `get_mcdonald_reports` method's own usage of `Path`, `Dict`, `List`, `Any`, `logger`.)

- [ ] **Step 4: Add Pydantic models**

Add to `backend/models.py`, near `McDonaldReportResponse`/`McDonaldReportListResponse`:

```python
class UnrecognizedSeriesInfo(BaseModel):
    """Серия, найденная на этапе 01, но не распознанная как требуемая модальность"""
    original_path: str = Field(..., description="Путь к исходной DICOM-серии")
    series_description: str = Field(..., description="ProtocolName | SeriesDescription")
    slice_count: int = Field(..., description="Число DICOM-файлов в серии")


class IncompletePatientSession(BaseModel):
    """Одна неполная сессия, требующая внимания врача"""
    patient_id: str = Field(..., description="BIDS ID пациента (sub-XXX)")
    original_id: str = Field(..., description="Исходный ID пациента из источника данных")
    session_id: str = Field(..., description="BIDS ID сессии (ses-XXX)")
    date: str = Field(..., description="Дата сессии YYYYMMDD")
    status: str = Field(..., description="Статус сессии (сейчас всегда 'incomplete')")
    available: List[str] = Field(..., description="Модальности, которые уже есть")
    unrecognized_series: List[UnrecognizedSeriesInfo] = Field(
        default_factory=list, description="Нераспознанные серии — кандидаты на ручную переразметку"
    )


class IncompletePatientsResponse(BaseModel):
    """Список неполных сессий текущего запуска"""
    total: int = Field(..., description="Количество неполных сессий")
    sessions: List[IncompletePatientSession] = Field(..., description="Список неполных сессий")
```

- [ ] **Step 5: Add the endpoint**

In `backend/app.py`, add `IncompletePatientsResponse` to the `from models import (...)` block (line 37), then add near `get_mcdonald_reports` (~line 803):

```python
@app.get("/api/incomplete-patients/{run_id}", response_model=IncompletePatientsResponse)
async def get_incomplete_patients(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить список неполных сессий (не хватает требуемых модальностей)
    для данного запуска — очередь ручного review для врача
    """
    logger.info(f"Запрос неполных пациентов для run_id: {run_id}")

    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    sessions = pipeline_manager.get_incomplete_patients(run.output_path)

    return IncompletePatientsResponse(
        total=len(sessions),
        sessions=sessions
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest backend/test_incomplete_patients_api.py -v`
Expected: PASS (2/2)

- [ ] **Step 7: Run full backend test suite**

Run: `cd backend && python3 -m pytest -q`
Expected: all pass, no regressions (record the baseline count observed).

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline_manager.py backend/models.py backend/app.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): GET /api/incomplete-patients/{run_id} — doctor review queue"
```

---

### Task 3: `POST /api/incomplete-patients/{run_id}/{patient_id}/{session_id}/relabel`

**Files:**
- Modify: `backend/pipeline_manager.py` (new `relabel_series` method)
- Modify: `backend/models.py` (new request/response models)
- Modify: `backend/app.py` (new endpoint)
- Test: `backend/test_incomplete_patients_api.py` (extend from Task 2)

**Interfaces:**
- Consumes: Task 1's `utils.dicom_file_ops.copy_and_anonymize_series` and `find_dicom_files`; Task 2's `_dataset_mapping_path`.
- Produces: the actual mutation — moves a series from `unrecognized_series` into `series[modality]`, recomputes status, and (if now complete) physically moves the session directory from `_incomplete/sub-XXX/ses-YYY` to `sub-XXX/ses-YYY`.

- [ ] **Step 1: Write the failing test**

```python
# append to backend/test_incomplete_patients_api.py
import shutil


class TestRelabelSeries:
    def _make_dicom_series(self, series_dir: Path, n_files=2):
        series_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake dicom bytes")
        return series_dir

    def test_relabel_completes_session_and_moves_to_main_tree(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-001" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series")

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "unrecognized_series": [
                            {
                                "original_path": str(raw_series),
                                "series_description": "xyz | xyz",
                                "slice_count": 2,
                            }
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-001",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t1c",
        )

        assert result["status"] == "complete"
        # session moved out of _incomplete/
        assert not incomplete_dir.exists()
        assert (bids_dir / "sub-001" / "ses-001" / "anat" / "t1c").exists()
        copied_files = list((bids_dir / "sub-001" / "ses-001" / "anat" / "t1c").glob("*.dcm"))
        assert len(copied_files) == 2

        # dataset_mapping.json updated on disk
        mapping = json.loads((bids_dir / "dataset_mapping.json").read_text())
        session = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        assert session["status"] == "complete"
        assert "t1c" in session["series"]
        assert session["unrecognized_series"] == []

    def test_relabel_leaves_session_incomplete_if_still_missing_modalities(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-002" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series_2")

        _write_mapping(tmp_path, {
            "sub-002": {
                "original_id": "P2",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "unrecognized_series": [
                            {"original_path": str(raw_series), "series_description": "xyz", "slice_count": 2}
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-002",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t2",
        )

        assert result["status"] == "incomplete"
        # still under _incomplete/ — not yet complete
        assert (bids_dir / "_incomplete" / "sub-002" / "ses-001" / "anat" / "t2").exists()
        assert not (bids_dir / "sub-002").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest backend/test_incomplete_patients_api.py -v -k Relabel`
Expected: FAIL — `AttributeError: 'PipelineManager' object has no attribute 'relabel_series'`

- [ ] **Step 3: Implement `PipelineManager.relabel_series`**

`pipeline_manager.py` currently has zero imports from `utils/`/`scripts/` (confirmed — only stdlib + `from config import settings`), and its production runtime (`web.Dockerfile`: `WORKDIR /app`, `ENV PYTHONPATH=/app`, `ENTRYPOINT ["python3", "backend/app.py"]`) resolves `utils.*` imports only because `/app` (project root) is on `PYTHONPATH` — that is NOT guaranteed when running backend tests locally (`cd backend && python3 -m pytest`, no such env var set). Add an explicit, environment-independent sys.path guard near the top of `backend/pipeline_manager.py`, before the new imports:

```python
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dicom_file_ops import find_dicom_files, copy_and_anonymize_series
from utils.config_loader import load_lesion_type_config
```

(`Path` is already imported in this file; `import shutil` is also already present at the top per the existing `create_runtime_config`-adjacent code — confirm before adding a duplicate.)

```python
    def relabel_series(
        self,
        output_path: str,
        patient_id: str,
        session_id: str,
        original_path: str,
        modality: str,
        lesion_type: str = 'glioblastoma',
    ) -> Dict[str, Any]:
        """
        Manually assign a modality to a series the automatic detector
        couldn't classify (piece B's unrecognized_series). Copies it into
        the correct BIDS location, updates dataset_mapping.json, and moves
        the whole session out of _incomplete/ into the main tree if this
        was the last missing modality.

        Returns:
            Dict with the updated session's "status" and "available" modalities.

        Raises:
            ValueError if the patient/session/series isn't found in dataset_mapping.json.
        """
        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        session_data = mapping_data['patients'][patient_id]['sessions'][session_id]

        unrecognized = session_data.get('unrecognized_series', [])
        matches = [u for u in unrecognized if u['original_path'] == original_path]
        if not matches:
            raise ValueError(
                f"No unrecognized series with original_path={original_path!r} "
                f"in {patient_id}/{session_id}"
            )
        series_entry = matches[0]

        bids_dir = Path(output_path) / "bids_organized"
        was_incomplete = session_data.get('status') == 'incomplete'
        current_root = (bids_dir / "_incomplete") if was_incomplete else bids_dir
        target_dir = current_root / patient_id / session_id / "anat" / modality
        target_dir.mkdir(parents=True, exist_ok=True)

        source_files = find_dicom_files(Path(original_path))
        copy_and_anonymize_series(
            source_files, target_dir, patient_id, session_id, modality,
            metadata_extractor=None, logger=logger,
        )

        # Move the series entry from unrecognized_series to series[modality]
        session_data['unrecognized_series'] = [
            u for u in unrecognized if u['original_path'] != original_path
        ]
        session_data['series'][modality] = {
            'original_path': original_path,
            'slice_count': len(source_files),
            'series_description': series_entry['series_description'],
        }

        # Recompute completeness — same lesion-type-aware required set as
        # CompletenessChecker.check_session() in 01_reorganize_folders.py
        # (piece B), not a hardcoded glioblastoma-only set — MS requires
        # t1/t2/t2fl, no t1c.
        try:
            required = set(load_lesion_type_config(lesion_type)['required_modalities'])
        except KeyError:
            required = {'t1', 't1c', 't2', 't2fl'}
        is_complete = required.issubset(session_data['series'].keys())
        session_data['status'] = 'complete' if is_complete else 'incomplete'

        # If now complete and it was previously under _incomplete/, move the whole session
        if is_complete and was_incomplete:
            source_session_dir = bids_dir / "_incomplete" / patient_id / session_id
            target_session_dir = bids_dir / patient_id / session_id
            target_session_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_session_dir), str(target_session_dir))
            # Clean up now-empty _incomplete/<patient_id> if this was its only session
            parent_dir = bids_dir / "_incomplete" / patient_id
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        return {
            'status': session_data['status'],
            'available': sorted(session_data['series'].keys()),
        }
```

- [ ] **Step 4: Add Pydantic models**

Add to `backend/models.py`:

```python
class RelabelSeriesRequest(BaseModel):
    """Запрос на ручную переразметку нераспознанной серии"""
    original_path: str = Field(..., description="Путь к исходной DICOM-серии (из unrecognized_series)")
    modality: str = Field(..., description="Модальность, назначаемая врачом: t1, t1c, t2 или t2fl")


class RelabelSeriesResponse(BaseModel):
    """Результат переразметки"""
    status: str = Field(..., description="Статус сессии после переразметки: complete или incomplete")
    available: List[str] = Field(..., description="Модальности, доступные после переразметки")
```

- [ ] **Step 5: Add the endpoint**

In `backend/app.py`, add `RelabelSeriesRequest`/`RelabelSeriesResponse` to the models import, then:

```python
@app.post(
    "/api/incomplete-patients/{run_id}/{patient_id}/{session_id}/relabel",
    response_model=RelabelSeriesResponse,
)
async def relabel_series(
    run_id: str,
    patient_id: str,
    session_id: str,
    request: RelabelSeriesRequest,
    db: Session = Depends(get_db)
):
    """Вручную назначить модальность нераспознанной серии"""
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    try:
        result = pipeline_manager.relabel_series(
            output_path=run.output_path,
            patient_id=patient_id,
            session_id=session_id,
            original_path=request.original_path,
            modality=request.modality,
            lesion_type=run.lesion_type or 'glioblastoma',
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(f"Переразметка {patient_id}/{session_id}: {request.original_path} -> {request.modality}")
    return RelabelSeriesResponse(**result)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python3 -m pytest backend/test_incomplete_patients_api.py -v -k Relabel`
Expected: PASS (2/2)

- [ ] **Step 7: Run full backend test suite**

Run: `cd backend && python3 -m pytest -q`
Expected: all pass, no regressions.

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline_manager.py backend/models.py backend/app.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): POST .../relabel — manual modality assignment for unrecognized series"
```

---

### Task 4: `POST .../discard` + `POST /api/pipeline-runs/{run_id}/requeue`

**Files:**
- Modify: `backend/pipeline_manager.py` (new `discard_session` method)
- Modify: `backend/models.py` (new response model for discard; requeue reuses `PipelineStartResponse` if its shape fits, else a small new model — check `PipelineStartResponse`'s fields in `backend/models.py` first and reuse if it matches "run_id + status" shape, since `PipelineStartRequest`/`PipelineStartResponse` already exist per `backend/app.py:135` and the `/api/pipeline/start` endpoint at line 291)
- Modify: `backend/app.py` (two new endpoints)
- Test: `backend/test_incomplete_patients_api.py` (extend)

**Interfaces:**
- Consumes: Task 3's `_dataset_mapping_path`; existing `PipelineManager.create_runtime_config()`/`start_pipeline()` (already used by `/api/pipeline/start`, read that endpoint's body at `backend/app.py:291-358` first to mirror exactly how it builds a `run_id`, calls these two methods, and records the run in the DB — the requeue endpoint should do the same thing, just with `input_path`/`output_path` taken from the EXISTING run being requeued instead of from a new user-submitted request).

- [ ] **Step 1: Write the failing test for discard**

```python
# append to backend/test_incomplete_patients_api.py
class TestDiscardSession:
    def test_discard_sets_status(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "unrecognized_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.discard_session(str(tmp_path), "sub-001", "ses-001")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        assert mapping["patients"]["sub-001"]["sessions"]["ses-001"]["status"] == "discarded"

    def test_discarded_session_excluded_from_incomplete_list(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {}, "unrecognized_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.discard_session(str(tmp_path), "sub-001", "ses-001")
        assert pm.get_incomplete_patients(str(tmp_path)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest backend/test_incomplete_patients_api.py -v -k Discard`
Expected: FAIL — `AttributeError: 'PipelineManager' object has no attribute 'discard_session'`

- [ ] **Step 3: Implement `PipelineManager.discard_session`**

Add to `backend/pipeline_manager.py`:

```python
    def discard_session(self, output_path: str, patient_id: str, session_id: str) -> None:
        """Mark a session as intentionally excluded from review — stays out of
        get_incomplete_patients() without deleting any data."""
        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        session_data = mapping_data['patients'][patient_id]['sessions'][session_id]
        session_data['status'] = 'discarded'

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Add the discard endpoint**

In `backend/app.py`:

```python
@app.post("/api/incomplete-patients/{run_id}/{patient_id}/{session_id}/discard")
async def discard_session(
    run_id: str,
    patient_id: str,
    session_id: str,
    db: Session = Depends(get_db)
):
    """Пометить сессию как намеренно исключённую из очереди review"""
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    try:
        pipeline_manager.discard_session(run.output_path, patient_id, session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(f"Сессия {patient_id}/{session_id} отброшена (run_id={run_id})")
    return {"status": "discarded"}
```

**Context already traced (no further reading needed):** `/api/pipeline/start` (`backend/app.py:291-347`) does NOT call `create_runtime_config()`/`start_pipeline()` directly — it calls `database.create_pipeline_run(db, input_path, output_path, lesion_type)` (creates a fresh `PipelineRun` row with a new UUID `run_id`, status `"pending"`, plus 7 `StageExecution` rows), then schedules `run_pipeline_background` as a `BackgroundTasks` task (that function is what actually calls `pipeline_manager.start_pipeline(...)` and updates status as the subprocess progresses), then separately kicks off `pipeline_monitor.start_monitoring(run.run_id, run.output_path, kappa_session_id, lesion_type)` via `asyncio.create_task`. The requeue endpoint reuses this exact same sequence, just sourcing `input_path`/`output_path`/`lesion_type` from the run being requeued (via `get_pipeline_run(db, run_id)`) instead of a request body, and passing `kappa_session_id=None` (no new Kappa session context for a requeue — matches `pipeline_monitor.start_monitoring`'s existing `Optional[str]` parameter).

- [ ] **Step 5: Add the requeue endpoint**

In `backend/app.py`, near `/api/pipeline/start`:

```python
@app.post("/api/pipeline-runs/{run_id}/requeue", response_model=PipelineStartResponse)
async def requeue_pipeline_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Перезапускает pipeline на тех же input/output путях, что и исходный
    запуск. skip_existing (базовый конфиг) гарантирует, что реально
    обработается только дособранный вручную пациент.
    """
    original_run = get_pipeline_run(db, run_id)
    if not original_run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    run = create_pipeline_run(
        db,
        input_path=original_run.input_path,
        output_path=original_run.output_path,
        lesion_type=original_run.lesion_type or "glioblastoma",
    )

    background_tasks.add_task(
        run_pipeline_background,
        run.run_id,
        run.input_path,
        run.output_path,
        db,
        lesion_type=run.lesion_type,
    )

    asyncio.create_task(pipeline_monitor.start_monitoring(
        run.run_id, run.output_path, None, run.lesion_type
    ))

    logger.info(f"Requeue: новый run_id {run.run_id} на тех же путях, что и {run_id}")

    return PipelineStartResponse(
        run_id=run.run_id,
        status=PipelineStatus.PENDING,
        message="Pipeline перезапущен (skip_existing обработает только новые/изменённые данные)",
        created_at=run.created_at,
        lesion_type=run.lesion_type,
    )
```

Confirm `create_pipeline_run` is already imported into `app.py`'s `from database import (...)` block (line 62) — it must be, since `/api/pipeline/start` already uses it; no new import needed.

- [ ] **Step 6: Write and run a test for the requeue endpoint**

Follow the existing test pattern for `/api/pipeline/start` (find it in `backend/test_*.py` — likely `backend/test_app*.py` or similar; search for `PipelineStartRequest` usage in tests) and mirror it for requeue: mock/stub `start_pipeline` (it launches a real subprocess — tests must not actually run the full pipeline), assert a new `pipeline_runs` row is created with the same `input_path`/`output_path` as the original run.

Run: `cd backend && python3 -m pytest -k requeue -v`
Expected: PASS.

- [ ] **Step 7: Run full backend test suite**

Run: `cd backend && python3 -m pytest -q`
Expected: all pass, no regressions.

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline_manager.py backend/models.py backend/app.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): discard session + requeue endpoints"
```

---

### Task 5: Real end-to-end verification against real BO data

**Files:** none modified — verification only.

- [ ] **Step 1: Start the backend locally (or via the running docker container) and exercise the full flow against real data**

Using a scratch copy of a real incomplete BO run (e.g. re-run `01_reorganize_folders.py` on `data/clinical_dicom/BO/BO-181` into a scratch output dir, as done in piece B's Task 4), then:

1. Manually insert (or use existing tooling to insert) a `pipeline_runs` row pointing `output_path` at that scratch dir.
2. `curl GET /api/incomplete-patients/{run_id}` — confirm the real incomplete BO-181 session appears with correct `available: ["t1c"]`.
3. If BO-181's data happens to have any genuinely unrecognized series (check the real `dataset_mapping.json` from that run), exercise `POST .../relabel` against one and confirm the file lands in the right place and `status` recomputes correctly. If BO-181 has none (it may not, since its single series is `t1c`-only and correctly classified), pick a different real incomplete patient from the earlier BO investigation that does have an unrecognized series, or construct one isolated real-DICOM fixture the same way Task 3's tests did but pointed at real bytes instead of fake ones — whichever is faster.
4. `POST .../discard` on a different session, confirm it disappears from the list endpoint.

- [ ] **Step 2: Clean up any scratch directories and DB rows created for this verification**

- [ ] **Step 3: Final full-suite run across both touched areas**

Run: `python3 -m pytest tests/stage01/ -q && cd backend && python3 -m pytest -q`
Expected: all pass.
