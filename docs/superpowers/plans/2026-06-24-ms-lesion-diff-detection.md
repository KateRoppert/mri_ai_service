# MS New/Growing Lesion Detection — Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify each MS lesion in a session as new, growing, stable, or resolved relative to the patient's previous session, and surface this per consecutive session pair in the longitudinal panel.

**Architecture:** Sessions of the same patient are independently registered to the same standard-space atlas/grid (Stage 05), so their `*_segmask_labels.nii.gz` (per-lesion labeled masks, already written by Stage 08 for every MS session) are directly voxel-comparable — no inter-session registration needed. A new `backend/lesion_diff.py` compares two labeled masks by per-component overlap (with the same dilation tolerance used for McDonald zones). The backend resolves, for a given patient, every consecutive pair of MS sessions' label-mask files (via a new `pipeline_manager.get_segmask_label_path`) and exposes the classification through a new endpoint, consumed by `LongitudinalTimeline.jsx`.

**Tech Stack:** Python 3.12, nibabel, numpy, scipy.ndimage, FastAPI, Pydantic v2, pytest, React, antd, axios.

**Depends on:** Stage 08 already writes `*_segmask_labels.nii.gz` for every `multiple_sclerosis` mask today (pre-existing behavior, not introduced by this plan or by plans 1/2a/2b — confirmed in `08_anatomical_analysis.py`'s `process_one_mask`, the renamed file from the Stage 08 refactor plan). This plan does not depend on plans 2a/2b's McDonald-zone work landing first; it only needs the Stage 08 refactor plan (for the renamed file/module names referenced below) and can be implemented in parallel with 2a/2b.

## Global Constraints

- Zero behavior change to glioblastoma or to existing per-session reports (`lobar_report.json`, `lesion_stats_report.json`, `mcdonald_report.json`) — this plan only adds a new cross-session comparison, it never modifies how a single session is analyzed.
- Components below the noise threshold (5 mm³ — same value as `lesion_stats.MIN_LESION_VOLUME_MM3`) must be excluded from diff classification, same as they're excluded from `lesion_count` today. This constant is intentionally duplicated as a literal in `backend/lesion_diff.py` rather than imported across the scripts/↔backend boundary — see Task 1 Step 3's comment for why.
- Growth threshold is config-driven (`configs/ms_longitudinal_config.yaml`), not hardcoded — it is an explicit placeholder default pending clinician input (per the design spec), and must be easy to change without touching code.
- Conventional commits, one commit per task minimum.
- Every touched `.py` file passes `python -m py_compile` before commit.

**Deviation from the committed design spec, flagged explicitly:** `docs/superpowers/specs/2026-06-23-ms-clinical-report-v2-design.md` says the growth threshold is "стора[ится] в `configs/ms_zones_config.yaml`". This plan instead creates a dedicated `configs/ms_longitudinal_config.yaml`. Reason: `ms_zones_config.yaml` (introduced by plan 2a) is McDonald-zone atlas configuration; the growth threshold is an unrelated, independent concern (cross-session diffing, not spatial zone classification), and conflating the two files would reintroduce exactly the "не лить всё в кучу" problem this whole branch exists to fix. No other part of the spec is affected.

---

### Task 1: Core diff algorithm — `backend/lesion_diff.py`

**Files:**
- Create: `configs/ms_longitudinal_config.yaml`
- Create: `backend/lesion_diff.py`
- Test: `backend/test_lesion_diff.py`

**Interfaces:**
- Produces: `lesion_diff.compare_labeled_masks(prev_path: Path, curr_path: Path, *, growth_threshold_relative: float = 0.20, growth_threshold_absolute_cm3: float = 0.03, dilation_voxels: int = 1, min_lesion_volume_mm3: float = 5.0) -> dict`:
  ```python
  {
    "new_count": int, "growing_count": int, "stable_count": int, "resolved_count": int,
    "lesions": [
      {"label": int, "status": "new"|"growing"|"stable"|"resolved",
       "volume_cm3": float, "previous_volume_cm3": float},
      ...
    ],
  }
  ```
  Raises `ValueError` if `prev_path` and `curr_path` have mismatched array shapes (caller's responsibility to catch and skip — see Task 3).

- [ ] **Step 1: Write `configs/ms_longitudinal_config.yaml`**

```yaml
# MS longitudinal lesion-diff config (Этап 5.6, part B).
# Growth threshold: a matched lesion (overlapping a prior-session lesion,
# within dilation_voxels tolerance) is classified "growing" if EITHER
# condition holds — relative-only would miss large lesions growing by a
# clinically significant absolute amount but a small percentage; absolute-only
# would miss small lesions growing by a large percentage. This is a
# provisional default pending clinician input — McDonald 2017 itself only
# defines "new T2 lesion" as a criterion, not a growth threshold for
# existing ones.
growth_threshold_relative: 0.20
growth_threshold_absolute_cm3: 0.03

# Voxel dilation tolerance for matching a lesion across sessions (registration
# jitter). Same value/rationale as McDonald zone classification's
# dilation_voxels, kept as an independent knob here since the two features
# can be tuned separately.
dilation_voxels: 1
```

- [ ] **Step 2: Write the failing test**

Create `backend/test_lesion_diff.py`:

```python
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lesion_diff import compare_labeled_masks


def _save_labeled(path: Path, labels: dict, shape=(20, 20, 20)):
    """labels: {label_int: (slice_x, slice_y, slice_z)}"""
    data = np.zeros(shape, dtype=np.int16)
    for label, (sx, sy, sz) in labels.items():
        data[sx, sy, sz] = label
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_lesion_with_no_prior_overlap_is_new(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 3), slice(0, 3), slice(0, 3))})       # far away
    _save_labeled(curr_path, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})  # new lesion

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1)

    assert result["new_count"] == 1
    statuses = {l["status"] for l in result["lesions"]}
    assert "new" in statuses


def test_prior_lesion_absent_in_current_is_resolved(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 3), slice(0, 3), slice(0, 3))})
    _save_labeled(curr_path, {})  # empty — lesion resolved

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1)

    assert result["resolved_count"] == 1
    resolved = [l for l in result["lesions"] if l["status"] == "resolved"][0]
    assert resolved["volume_cm3"] == 0.0
    assert resolved["previous_volume_cm3"] > 0.0


def test_matched_lesion_growing_by_relative_threshold(tmp_path):
    # Small lesion: 3x3x3=27 voxels (0.027 cm3) -> grows to 5x5x5=125 voxels
    # (0.125 cm3). Relative growth = (0.125-0.027)/0.027 ≈ 3.6 >> 0.20.
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(5, 8), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(5, 10), slice(5, 10), slice(5, 10))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["growing_count"] == 1
    assert result["new_count"] == 0


def test_matched_lesion_growing_by_absolute_threshold_with_small_relative_growth(tmp_path):
    # Large lesion: 10x10x10=1000 voxels (1.0 cm3) -> grows to 1100 voxels by
    # adding a 10x10x1 slab (100 voxels = 0.1 cm3). Relative growth = 10% < 20%,
    # but absolute growth 0.1 cm3 >= 0.03 cm3 threshold -> still "growing".
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})
    _save_labeled(curr_path, {1: (slice(0, 10), slice(0, 10), slice(0, 11))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["growing_count"] == 1
    assert result["stable_count"] == 0


def test_matched_lesion_with_negligible_change_is_stable(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})
    _save_labeled(curr_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["stable_count"] == 1
    assert result["growing_count"] == 0


def test_sub_threshold_components_excluded_from_both_sides(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    # A single 1mm voxel = 0.001 cm3, far below the 5mm3 (0.005 cm3) floor.
    _save_labeled(prev_path, {1: (slice(0, 1), slice(0, 1), slice(0, 1))})
    _save_labeled(curr_path, {1: (slice(15, 16), slice(15, 16), slice(15, 16))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1, min_lesion_volume_mm3=5.0)

    assert result["lesions"] == []
    assert result["new_count"] == 0
    assert result["resolved_count"] == 0


def test_current_lesion_overlapping_two_prior_lesions_sums_previous_volume(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    # Two separate prior lesions, each 3x3x3=27 voxels (0.027 cm3), adjacent
    # but not touching (gap at x=8). A merged current lesion spans both.
    _save_labeled(prev_path, {
        1: (slice(0, 3), slice(0, 3), slice(0, 3)),
        2: (slice(8, 11), slice(0, 3), slice(0, 3)),
    })
    _save_labeled(curr_path, {1: (slice(0, 11), slice(0, 3), slice(0, 3))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    matched = [l for l in result["lesions"] if l["status"] in ("growing", "stable")][0]
    assert matched["previous_volume_cm3"] == pytest.approx(0.027 + 0.027, abs=1e-4)


def test_shape_mismatch_raises_value_error(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr_different_shape.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.int16), np.eye(4)), str(prev_path))
    nib.save(nib.Nifti1Image(np.zeros((22, 22, 22), dtype=np.int16), np.eye(4)), str(curr_path))

    with pytest.raises(ValueError):
        compare_labeled_masks(prev_path, curr_path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_lesion_diff.py -v`
Expected: `ModuleNotFoundError: No module named 'lesion_diff'`

- [ ] **Step 4: Write the implementation — `backend/lesion_diff.py`**

```python
"""
Cross-session MS lesion diffing: classifies each lesion in the current
session as new, growing, stable, or resolved relative to the patient's
previous session.

Both sessions' labeled masks (*_segmask_labels.nii.gz, written by Stage 08
for every multiple_sclerosis mask) are already on the same voxel grid —
each session is independently registered to the same standard-space atlas
during preprocessing — so no inter-session registration is needed here,
just a direct per-component overlap comparison.

MIN_LESION_VOLUME_MM3 intentionally duplicates the value in
scripts/lesion_stats.py rather than importing it: backend/ and scripts/ are
two independently-deployed layers (a long-running FastAPI service vs. a
per-run pipeline subprocess) and the project does not cross-import between
them anywhere today. It is the same clinical noise-floor concept in both
places — keep the two literals in sync if the value ever changes.
"""

import logging
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

MIN_LESION_VOLUME_MM3 = 5.0


def compare_labeled_masks(
    prev_path: Path,
    curr_path: Path,
    *,
    growth_threshold_relative: float = 0.20,
    growth_threshold_absolute_cm3: float = 0.03,
    dilation_voxels: int = 1,
    min_lesion_volume_mm3: float = MIN_LESION_VOLUME_MM3,
) -> Dict:
    prev_img = nib.load(str(prev_path))
    curr_img = nib.load(str(curr_path))
    prev_data = np.asarray(prev_img.dataobj)
    curr_data = np.asarray(curr_img.dataobj)

    if prev_data.shape != curr_data.shape:
        raise ValueError(
            f"Shape mismatch comparing {prev_path.name} ({prev_data.shape}) "
            f"vs {curr_path.name} ({curr_data.shape}) — sessions are not on "
            f"the same grid; cannot diff without registration."
        )

    voxel_vol_cm3 = float(np.prod(np.abs(np.diag(curr_img.affine[:3, :3])))) / 1000.0
    min_cm3 = min_lesion_volume_mm3 / 1000.0
    structure = np.ones((3, 3, 3))

    def _volume_cm3(data: np.ndarray, label: int) -> float:
        return round(int((data == label).sum()) * voxel_vol_cm3, 4)

    prev_labels = sorted(
        int(l) for l in np.unique(prev_data) if l != 0 and _volume_cm3(prev_data, int(l)) >= min_cm3
    )
    curr_labels = sorted(
        int(l) for l in np.unique(curr_data) if l != 0 and _volume_cm3(curr_data, int(l)) >= min_cm3
    )
    prev_labels_set = set(prev_labels)

    lesions = []
    matched_prev_labels = set()

    for label in curr_labels:
        curr_mask = (curr_data == label)
        curr_vol = _volume_cm3(curr_data, label)

        dilated = curr_mask
        for _ in range(dilation_voxels):
            dilated = binary_dilation(dilated, structure=structure)

        overlapping_prev = sorted(
            int(l) for l in np.unique(prev_data[dilated]) if l != 0 and int(l) in prev_labels_set
        )

        if not overlapping_prev:
            lesions.append({
                "label": label, "status": "new",
                "volume_cm3": curr_vol, "previous_volume_cm3": 0.0,
            })
            continue

        matched_prev_labels.update(overlapping_prev)
        prev_vol = round(sum(_volume_cm3(prev_data, l) for l in overlapping_prev), 4)

        growth_abs = curr_vol - prev_vol
        growth_rel = (growth_abs / prev_vol) if prev_vol > 0 else float("inf")
        is_growing = (growth_rel >= growth_threshold_relative) or (growth_abs >= growth_threshold_absolute_cm3)
        status = "growing" if is_growing else "stable"

        lesions.append({
            "label": label, "status": status,
            "volume_cm3": curr_vol, "previous_volume_cm3": prev_vol,
        })

    for label in prev_labels:
        if label not in matched_prev_labels:
            lesions.append({
                "label": label, "status": "resolved",
                "volume_cm3": 0.0, "previous_volume_cm3": _volume_cm3(prev_data, label),
            })

    counts = {"new": 0, "growing": 0, "stable": 0, "resolved": 0}
    for lesion in lesions:
        counts[lesion["status"]] += 1

    return {
        "new_count": counts["new"],
        "growing_count": counts["growing"],
        "stable_count": counts["stable"],
        "resolved_count": counts["resolved"],
        "lesions": lesions,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_lesion_diff.py -v`
Expected: `8 passed`

- [ ] **Step 6: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile backend/lesion_diff.py`
Expected: no output, exit code 0

- [ ] **Step 7: Commit**

```bash
git add configs/ms_longitudinal_config.yaml backend/lesion_diff.py backend/test_lesion_diff.py
git commit -m "feat(backend): add lesion_diff — new/growing/stable/resolved classification across sessions"
```

---

### Task 2: `pipeline_manager.get_segmask_label_path`

**Files:**
- Modify: `backend/pipeline_manager.py` (add after `get_mcdonald_reports`, if plan 2b already landed — otherwise after `get_lesion_stats_reports`)
- Test: `backend/test_pipeline_manager_segmask_label_path.py`

**Interfaces:**
- Produces: `PipelineManager.get_segmask_label_path(self, output_path: str, subject_id: str, session_id: str) -> Optional[Path]` — `subject_id`/`session_id` are BIDS-style (`"sub-001"`, `"ses-002"`).

- [ ] **Step 1: Write the failing test**

Create `backend/test_pipeline_manager_segmask_label_path.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def test_returns_none_when_directory_missing(tmp_path):
    manager = PipelineManager()
    result = manager.get_segmask_label_path(str(tmp_path), "sub-001", "ses-001")
    assert result is None


def test_returns_none_when_no_labels_file(tmp_path):
    seg_dir = tmp_path / "segmentation" / "sub-001" / "ses-001" / "anat" / "multiple_sclerosis"
    seg_dir.mkdir(parents=True)
    (seg_dir / "sub-001_ses-001_t1_segmask.nii.gz").write_bytes(b"")  # no _labels file
    manager = PipelineManager()
    result = manager.get_segmask_label_path(str(tmp_path), "sub-001", "ses-001")
    assert result is None


def test_finds_labels_file(tmp_path):
    seg_dir = tmp_path / "segmentation" / "sub-001" / "ses-001" / "anat" / "multiple_sclerosis"
    seg_dir.mkdir(parents=True)
    labels_path = seg_dir / "sub-001_ses-001_t1_segmask_labels.nii.gz"
    labels_path.write_bytes(b"")
    manager = PipelineManager()
    result = manager.get_segmask_label_path(str(tmp_path), "sub-001", "ses-001")
    assert result == labels_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_segmask_label_path.py -v`
Expected: `AttributeError: 'PipelineManager' object has no attribute 'get_segmask_label_path'`

- [ ] **Step 3: Write the implementation**

Add to `backend/pipeline_manager.py`:

```python
    def get_segmask_label_path(self, output_path: str, subject_id: str, session_id: str) -> Optional[Path]:
        """
        Locate the per-lesion labeled mask (*_segmask_labels.nii.gz) Stage 08
        writes for a specific MS session — used for cross-session lesion diffing.

        Args:
            output_path: pipeline run's output directory.
            subject_id: BIDS subject, e.g. "sub-001".
            session_id: BIDS session, e.g. "ses-001".
        """
        seg_dir = Path(output_path) / "segmentation" / subject_id / session_id / "anat" / "multiple_sclerosis"
        if not seg_dir.exists():
            return None
        matches = list(seg_dir.glob("*_segmask_labels.nii.gz"))
        return matches[0] if matches else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_pipeline_manager_segmask_label_path.py -v`
Expected: `3 passed`

- [ ] **Step 5: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile backend/pipeline_manager.py`
Expected: no output, exit code 0

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/test_pipeline_manager_segmask_label_path.py
git commit -m "feat(backend): add PipelineManager.get_segmask_label_path for lesion diffing"
```

---

### Task 3: `GET /api/longitudinal/{patient_id}/diff` endpoint

**Files:**
- Modify: `backend/models.py` (add `LesionDiffEntry`, `LongitudinalDiffPair`, `LongitudinalDiffResponse`)
- Modify: `backend/app.py` (add the endpoint near the existing `/api/longitudinal/{patient_id}` route, which ends around line 862)
- Test: `backend/test_app_longitudinal_diff_endpoint.py`

**Interfaces:**
- Consumes: `lesion_diff.compare_labeled_masks` (Task 1), `pipeline_manager.get_segmask_label_path` (Task 2).
- Produces: `GET /api/longitudinal/{patient_id}/diff?lesion_type=multiple_sclerosis` → `LongitudinalDiffResponse`:
  ```json
  {
    "patient_id": "P000915",
    "pairs": [
      {"from_session_id": "ses-001", "to_session_id": "ses-002",
       "new_count": 1, "growing_count": 0, "stable_count": 2, "resolved_count": 0,
       "lesions": [{"label": 1, "status": "stable", "volume_cm3": 0.5, "previous_volume_cm3": 0.48}]}
    ]
  }
  ```
  Pairs where either session's label-mask file can't be found are skipped (logged, not raised) — a single missing file shouldn't blank out diffs that are computable from the other available sessions.

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_longitudinal_diff_endpoint.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app, pipeline_manager

client = TestClient(app)


def _fake_record(bids_id, scan_date, run_id):
    return {"bids_id": bids_id, "scan_date": scan_date, "pipeline_run_id": run_id, "lesion_type": "multiple_sclerosis"}


def _fake_run(output_path="/fake/output"):
    return SimpleNamespace(output_path=output_path)


def test_404_when_no_sessions_found():
    with patch("app.find_by_patient_id", return_value=[]), \
         patch("app.find_by_bids_id", return_value=[]), \
         patch("app.find_by_bids_subject", return_value=[]):
        response = client.get("/api/longitudinal/P999/diff")
    assert response.status_code == 404


def test_404_when_only_one_session():
    records = [_fake_record("sub-001_ses-001", "2022-01-01", "run-1")]
    with patch("app.find_by_patient_id", return_value=records), \
         patch("app.get_pipeline_run", return_value=_fake_run()):
        response = client.get("/api/longitudinal/P000915/diff")
    assert response.status_code == 404


def test_200_with_two_sessions_computes_one_pair(tmp_path):
    records = [
        _fake_record("sub-001_ses-001", "2022-01-18", "run-1"),
        _fake_record("sub-001_ses-002", "2023-03-25", "run-2"),
    ]
    fake_diff_result = {
        "new_count": 1, "growing_count": 0, "stable_count": 0, "resolved_count": 0,
        "lesions": [{"label": 1, "status": "new", "volume_cm3": 0.5, "previous_volume_cm3": 0.0}],
    }
    with patch("app.find_by_patient_id", return_value=records), \
         patch("app.get_pipeline_run", return_value=_fake_run()), \
         patch.object(pipeline_manager, "get_segmask_label_path", return_value=Path("/fake/labels.nii.gz")), \
         patch("app.compare_labeled_masks", return_value=fake_diff_result):
        response = client.get("/api/longitudinal/sub-001/diff?lesion_type=multiple_sclerosis")

    assert response.status_code == 200
    body = response.json()
    assert len(body["pairs"]) == 1
    assert body["pairs"][0]["from_session_id"] == "ses-001"
    assert body["pairs"][0]["to_session_id"] == "ses-002"
    assert body["pairs"][0]["new_count"] == 1


def test_pair_skipped_when_label_file_missing():
    records = [
        _fake_record("sub-001_ses-001", "2022-01-18", "run-1"),
        _fake_record("sub-001_ses-002", "2023-03-25", "run-2"),
    ]
    with patch("app.find_by_patient_id", return_value=records), \
         patch("app.get_pipeline_run", return_value=_fake_run()), \
         patch.object(pipeline_manager, "get_segmask_label_path", return_value=None):
        response = client.get("/api/longitudinal/sub-001/diff?lesion_type=multiple_sclerosis")

    assert response.status_code == 200
    assert response.json()["pairs"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_app_longitudinal_diff_endpoint.py -v`
Expected: 404 for the routes that don't exist yet (or import errors for `app.find_by_patient_id`/`app.compare_labeled_masks` not being patchable names — fixed by Step 3's implementation importing them into `app.py`'s module namespace).

- [ ] **Step 3: Add the Pydantic models — `backend/models.py`**

Add after `LongitudinalResponse` (end of file, or wherever that model currently sits):

```python
class LesionDiffEntry(BaseModel):
    """Классификация одного очага между двумя сессиями"""
    label: int = Field(..., description="Label очага в labeled-маске")
    status: str = Field(..., description="new | growing | stable | resolved")
    volume_cm3: float = Field(..., description="Объём в текущей сессии, см³ (0 если resolved)")
    previous_volume_cm3: float = Field(..., description="Объём в предыдущей сессии, см³ (0 если new)")

class LongitudinalDiffPair(BaseModel):
    """Диагностика очагов между двумя соседними по времени сессиями"""
    from_session_id: str = Field(..., description="Более ранняя сессия")
    to_session_id: str = Field(..., description="Более поздняя сессия")
    new_count: int
    growing_count: int
    stable_count: int
    resolved_count: int
    lesions: List[LesionDiffEntry]

class LongitudinalDiffResponse(BaseModel):
    """Диагностика очагов по всем парам соседних сессий пациента"""
    patient_id: str
    pairs: List[LongitudinalDiffPair]
```

- [ ] **Step 4: Add the endpoint — `backend/app.py`**

Add `LongitudinalDiffResponse` to the `from models import (...)` block (next to `LongitudinalResponse`). Add these two imports near the top of the file, alongside the existing `import json` / `import logging` block:

```python
from lesion_diff import compare_labeled_masks
from patient_registry import find_by_patient_id, find_by_bids_id, find_by_bids_subject
```

(`find_by_patient_id`/`find_by_bids_id`/`find_by_bids_subject` are currently imported *inside* `get_longitudinal` via a local `from patient_registry import (...)` statement — leave that local import as-is; this module-level import is what makes `patch("app.find_by_patient_id", ...)` in the tests above actually intercept the call. Local imports inside a function shadow the module-level name when called, so both call sites — the existing local import inside `get_longitudinal` and this new module-level one — independently resolve to the same underlying functions; no conflict.)

Add the new route immediately after `get_longitudinal` (which ends around line 862, right before the next `@app.get(...)`):

```python
def _split_bids_id(bids_id: str) -> tuple:
    """'sub-001_ses-002' -> ('sub-001', 'ses-002')"""
    idx = bids_id.find("_ses-")
    if idx == -1:
        return bids_id, ""
    return bids_id[:idx], bids_id[idx + 1:]


@app.get("/api/longitudinal/{patient_id}/diff", response_model=LongitudinalDiffResponse)
async def get_longitudinal_diff(
    patient_id: str,
    lesion_type: str = "multiple_sclerosis",
    db: Session = Depends(get_db)
):
    """
    Детекция новых/растущих/разрешившихся очагов между каждой парой соседних
    по времени сессий пациента (МС). Не использует ту же фильтрацию по
    pipeline_run_id, что get_longitudinal — резолвит output_path на лету.
    """
    all_records = find_by_patient_id(patient_id)
    if not all_records:
        all_records = find_by_bids_id(patient_id)
    if not all_records:
        all_records = find_by_bids_subject(patient_id)

    records = [r for r in all_records if r.get("lesion_type") == lesion_type]
    if not records:
        raise HTTPException(status_code=404, detail="No sessions found for this patient/lesion_type")

    sortable = sorted(records, key=lambda r: r.get("scan_date") or "")
    seen_bids_ids = set()
    ordered = []
    for r in sortable:
        bid = r.get("bids_id", "")
        if bid and bid not in seen_bids_ids:
            seen_bids_ids.add(bid)
            ordered.append(r)

    if len(ordered) < 2:
        raise HTTPException(status_code=404, detail="Not enough sessions for diff analysis (need >= 2)")

    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "ms_longitudinal_config.yaml"
    with open(config_path, "r") as f:
        diff_config = yaml.safe_load(f)

    pairs = []
    for prev_record, curr_record in zip(ordered[:-1], ordered[1:]):
        prev_run = get_pipeline_run(db, prev_record.get("pipeline_run_id"))
        curr_run = get_pipeline_run(db, curr_record.get("pipeline_run_id"))
        if not prev_run or not curr_run:
            continue

        prev_subject, prev_session = _split_bids_id(prev_record.get("bids_id", ""))
        curr_subject, curr_session = _split_bids_id(curr_record.get("bids_id", ""))

        prev_label_path = pipeline_manager.get_segmask_label_path(prev_run.output_path, prev_subject, prev_session)
        curr_label_path = pipeline_manager.get_segmask_label_path(curr_run.output_path, curr_subject, curr_session)
        if not prev_label_path or not curr_label_path:
            logger.warning(
                f"Skipping diff {prev_session}->{curr_session} for {patient_id}: "
                f"missing labeled mask (prev={prev_label_path}, curr={curr_label_path})"
            )
            continue

        try:
            result = compare_labeled_masks(
                prev_label_path, curr_label_path,
                growth_threshold_relative=diff_config.get("growth_threshold_relative", 0.20),
                growth_threshold_absolute_cm3=diff_config.get("growth_threshold_absolute_cm3", 0.03),
                dilation_voxels=diff_config.get("dilation_voxels", 1),
            )
        except ValueError as e:
            logger.warning(f"Skipping diff {prev_session}->{curr_session} for {patient_id}: {e}")
            continue

        pairs.append(LongitudinalDiffPair(
            from_session_id=prev_session,
            to_session_id=curr_session,
            new_count=result["new_count"],
            growing_count=result["growing_count"],
            stable_count=result["stable_count"],
            resolved_count=result["resolved_count"],
            lesions=[LesionDiffEntry(**l) for l in result["lesions"]],
        ))

    return LongitudinalDiffResponse(patient_id=patient_id, pairs=pairs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_app_longitudinal_diff_endpoint.py -v`
Expected: `4 passed`

- [ ] **Step 6: Run the full backend test suite for regressions**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_lesion_diff.py test_pipeline_manager_segmask_label_path.py test_app_longitudinal_diff_endpoint.py -v`
Expected: all pass

- [ ] **Step 7: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile backend/app.py backend/models.py`
Expected: no output, exit code 0

- [ ] **Step 8: Commit**

```bash
git add backend/models.py backend/app.py backend/test_app_longitudinal_diff_endpoint.py
git commit -m "feat(backend): add GET /api/longitudinal/{patient_id}/diff endpoint"
```

---

### Task 4: Frontend API client function

**Files:**
- Modify: `frontend/src/services/api.js:79-87`

**Interfaces:**
- Produces: `getLongitudinalDiff(patientId, lesionType) -> Promise<{patient_id, pairs}>`

- [ ] **Step 1: Add the function**

Right after `getLongitudinalReport` (ends at line 87), add:

```js
/**
 * Детекция новых/растущих/разрешившихся очагов между сессиями (МС)
 */
export const getLongitudinalDiff = async (patientId, lesionType = 'multiple_sclerosis') => {
  const response = await apiClient.get(`/longitudinal/${patientId}/diff`, {
    params: { lesion_type: lesionType },
  });
  return response.data;
};
```

- [ ] **Step 2: Add it to the default export object**

Add `getLongitudinalDiff,` next to `getLongitudinalReport,` in the default-export block (if `getLongitudinalReport` is even listed there — check; the default export block seen earlier only lists `getVolumeReports, getLobarReports, getLesionStatsReports,` explicitly, other functions may be consumed via named imports only). Run:

Run: `grep -n "getLongitudinalReport" /home/ubuntu/mri_ai_service/frontend/src/services/api.js`

If it appears only in its own `export const` (not in the default-export object), do not add it there either — match whatever the existing convention is for that function, for consistency.

- [ ] **Step 3: Verify the file still parses**

Run: `cd /home/ubuntu/mri_ai_service/frontend && npx eslint src/services/api.js`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.js
git commit -m "feat(frontend): add getLongitudinalDiff API client function"
```

---

### Task 5: Render new/growing indicators in `LongitudinalTimeline.jsx`

**Files:**
- Modify: `frontend/src/components/LongitudinalTimeline.jsx`

**Interfaces:**
- Consumes: `getLongitudinalDiff` (Task 4).
- Produces: a new table column showing new/growing lesion counts for each row (each row already represents one session; the new column reflects the diff *from the previous row to this one*, consistent with the existing `Δ объём` column's semantics).

- [ ] **Step 1: Fetch diff data alongside the existing points**

Change the imports (line 4) from:

```jsx
import { getLongitudinalReport } from '../services/api';
```

to:

```jsx
import { getLongitudinalReport, getLongitudinalDiff } from '../services/api';
```

Add a new state slot after `error` (line 9):

```jsx
  const [diffPairs, setDiffPairs] = useState([]);
```

Change the existing `useEffect` (lines 11-21) to also fetch the diff data:

```jsx
  useEffect(() => {
    if (!patientId) return;
    setLoading(true);
    Promise.all([
      getLongitudinalReport(patientId, lesionType),
      getLongitudinalDiff(patientId, lesionType).catch(() => ({ pairs: [] })),
    ])
      .then(([reportResp, diffResp]) => {
        setData(reportResp.points);
        setDiffPairs(diffResp.pairs || []);
      })
      .catch(err => {
        if (err.response?.status === 404) setData([]);
        else setError('Не удалось загрузить динамику');
      })
      .finally(() => setLoading(false));
  }, [patientId, lesionType]);
```

(`getLongitudinalDiff` is allowed to fail independently via its own `.catch` — a missing/incomplete diff must not block the existing volume/count timeline, which has worked standalone since before this plan.)

- [ ] **Step 2: Add the new column**

Add this column to the `columns` array (after the existing `'delta'` column, which ends at line 55):

```jsx
    {
      title: 'Новые / растущие',
      key: 'diff',
      align: 'center',
      render: (_, record, idx) => {
        if (idx === 0) return '—';
        const prevSessionId = data[idx - 1].session_id;
        const pair = diffPairs.find(
          p => p.from_session_id === prevSessionId && p.to_session_id === record.session_id
        );
        if (!pair) return <span style={{ color: '#bbb' }}>н/д</span>;
        if (pair.new_count === 0 && pair.growing_count === 0) {
          return <Tag color="green">стабильно</Tag>;
        }
        return (
          <Space size={4}>
            {pair.new_count > 0 && <Tag color="red">{pair.new_count} новых</Tag>}
            {pair.growing_count > 0 && <Tag color="orange">{pair.growing_count} растёт</Tag>}
          </Space>
        );
      },
    },
```

Add `Space` to the existing `import { Table, Spin, Alert, Tag } from 'antd';` line (line 3): `import { Table, Spin, Alert, Tag, Space } from 'antd';`.

- [ ] **Step 3: Confirm the empty-state guard still makes sense**

The component currently returns `null` if `data.length < 2` (line 25). This is unaffected — diff data is only ever rendered alongside row data that already passed that guard. No change needed here, just confirm by reading: `if (!data || data.length < 2) return null;` stays exactly where it is, before the new column's logic ever runs.

- [ ] **Step 4: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service/frontend && npx eslint src/components/LongitudinalTimeline.jsx`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/LongitudinalTimeline.jsx
git commit -m "feat(frontend): show new/growing lesion counts in longitudinal timeline"
```

---

### Task 6: End-to-end verification on real longitudinal data + push

**Files:** none (verification only)

- [ ] **Step 1: Run the full touched backend test suite**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m pytest test_lesion_diff.py test_pipeline_manager_segmask_label_path.py test_app_longitudinal_diff_endpoint.py -v`
Expected: all pass

- [ ] **Step 2: Run the pipeline for both sessions of `data/MS_5/P000915`**

This is the only real multi-session MS case in the repo (`2022-01-18` and `2023-03-25`). Run the full pipeline (through Stage 08) for each session separately via `orchestrator.py`, ensuring both end up registered in the patient registry under the same patient with `lesion_type=multiple_sclerosis`.

- [ ] **Step 3: Exercise the new endpoint directly**

```bash
curl -s "http://localhost:8000/api/longitudinal/P000915/diff?lesion_type=multiple_sclerosis" | python -m json.tool
```

Expected: one entry in `pairs`, `from_session_id`/`to_session_id` matching the two real session dates' BIDS session IDs, with `lesions` populated (not empty, given this is real clinical longitudinal data — some lesion-level change between 2022 and 2023 is expected, even if "stable" for most).

- [ ] **Step 4: Exercise the UI in a browser**

Open the frontend, navigate to `P000915`'s MS clinical report, expand "Динамика между сессиями", and confirm the new "Новые / растущие" column shows real values (not "н/д") for the second row.

- [ ] **Step 5: Push**

```bash
git push
```

## Self-Review Notes

- **Spec coverage:** Implements design spec section "(B) Детекция новых/растущих очагов" in full, including the explicitly-flagged-as-provisional growth threshold (now in its own config file — see the Global Constraints deviation note — rather than conflated with the McDonald-zone atlas config).
- **Type consistency:** `compare_labeled_masks`'s return dict keys (`new_count`, `growing_count`, `stable_count`, `resolved_count`, `lesions[].{label,status,volume_cm3,previous_volume_cm3}`) match `LesionDiffEntry`/`LongitudinalDiffPair`'s field names exactly (Task 3's endpoint does `LesionDiffEntry(**l)` and `result["new_count"]` etc. — verified against Task 1's actual dict construction, not assumed).
- **No placeholders:** every step shows literal before/after code or an exact command with expected output. The one explicitly-acknowledged provisional value (growth threshold default) is config, not code, and is documented as provisional in both the YAML comment and this plan's Global Constraints — that is the spec's own intent, not an implementation gap.
