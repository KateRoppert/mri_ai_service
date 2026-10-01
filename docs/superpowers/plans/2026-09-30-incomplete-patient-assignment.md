# Manual Modality Assignment for Incomplete Patients — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The doctor sees which modalities the algorithm selected, can swap any of them for any other series (including ones the detector could not classify), saves the whole set at once, and the patient is reprocessed on the next run.

**Architecture:** The API accepts the *desired final set*, not a list of operations, so the whole thing can be validated before a single file is written. Validation and diffing live in a pure module with no disk access; only the applier touches files, and it writes `dataset_mapping.json` exactly once at the end.

**Tech Stack:** Python 3.12, FastAPI, pytest, React 19 + Ant Design.

**Spec:** `docs/superpowers/specs/2026-09-30-incomplete-patient-assignment-design.md`

## Global Constraints

- **The required modality set always comes from `configs/lesion_types.yaml`**, via `load_lesion_type_config(lesion_type)['required_modalities']`. Never hardcode a modality list — not in Python, not in JSX. `glioblastoma` is `[t1, t1c, t2, t2fl]`; `multiple_sclerosis` is `[t1, t2, t2fl]` with **no `t1c`**.
- **`dataset_mapping.json` is written at most once per request**, after every file operation has succeeded. If anything raises, it is not written at all.
- **Only `incomplete` and `complete` sessions are editable.** `discarded` and `merged` are refused.
- **`bids_organized/` is never deleted.** It is the pipeline's input and holds the corrected assignment.
- **`patient_id` / `session_id` are validated against `_BIDS_PATIENT_ID_PATTERN` (`^sub-\d+$`) and `_BIDS_SESSION_ID_PATTERN` (`^ses-\d+$`) before any path is built.**
- Session data read from the mapping may be sparse — existing fixtures contain `"series": {"t1": {}}`. Read every field with a default; never index directly.
- Code comments in English. Operator-facing strings in Russian.
- Tests live beside the code as `backend/test_*.py`. Use `test_`-prefixed identifiers for anything written to the shared temp DB.
- Commit style: conventional commits, ending with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `backend/session_assignment.py` (new) | Validating a desired set and diffing it against the current one. Pure: dicts in, dicts out, no disk, no config loading. |
| `backend/session_artifacts.py` (new) | Deleting one session's per-stage outputs, and purging everything flagged for reprocessing. Knows the folder layout, nothing else. |
| `backend/pipeline_manager.py` | Gains `apply_assignment` (the only thing that touches files) and returns `selected`/`required` from `get_incomplete_patients`. |
| `backend/models.py` | `SelectedModality`, the two new fields on `IncompletePatientSession`, and the assignment request/response. |
| `backend/app.py` | The `PUT .../assignment` endpoint; requeue purges flagged sessions first. |
| `frontend/src/components/IncompletePatientDetail.jsx` | Two lists, checkboxes, save/cancel. Stops knowing what a modality is. |
| `frontend/src/services/api.js` | `saveAssignment()`. |

Dependency order is strict: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.

## Before You Start

```bash
cd /home/ubuntu/mri_ai_service
git checkout feat/incomplete-patients-assignment   # already exists, from main
source venv/bin/activate
python -m pytest backend/ -q        # record the number; it must not drop
```

**Do not `git add -A`.** `configs/kappa_datasets.yaml` and `pipeline_config.yaml` are intentionally dirty and must stay uncommitted. Stage by explicit path.

---

### Task 1: Return the selected set and the required set

**Files:**
- Modify: `backend/models.py` (near `IncompletePatientSession`, ~line 144)
- Modify: `backend/pipeline_manager.py:760-845` (`get_incomplete_patients`)
- Test: `backend/test_incomplete_patients_api.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `models.SelectedModality` with `modality: str`, `series_description: str`, `original_path: str`, `slice_count: int`
  - `IncompletePatientSession.selected: List[SelectedModality]`, `.required: List[str]`
  - `get_incomplete_patients()` dicts gain `"selected"` and `"required"`

**Background:** the protocol name is already stored at
`session_data['series'][modality]['series_description']`; it has simply never
been returned. `required` is what lets the frontend stop hardcoding modalities.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_incomplete_patients_api.py`:

```python
class TestSelectedAndRequired:
    def test_returns_what_the_algorithm_selected_with_protocol_names(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {
                            "t1": {
                                "original_path": "/raw/p1/t1",
                                "slice_count": 176,
                                "series_description": "t1_mprage_sag",
                            },
                        },
                        "excluded_series": [],
                    },
                },
            },
        })
        sessions = PipelineManager().get_incomplete_patients(
            str(tmp_path), lesion_type="glioblastoma",
        )
        selected = sessions[0]["selected"]
        assert len(selected) == 1
        assert selected[0]["modality"] == "t1"
        assert selected[0]["series_description"] == "t1_mprage_sag"
        assert selected[0]["slice_count"] == 176

    def test_required_comes_from_the_lesion_type_not_a_hardcoded_list(self, tmp_path):
        """MS does not use t1c. A hardcoded four-modality list offers the
        doctor a slot that lesion type has no concept of."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        ms = PipelineManager().get_incomplete_patients(
            str(tmp_path), lesion_type="multiple_sclerosis",
        )
        assert ms[0]["required"] == ["t1", "t2", "t2fl"]

        gbm = PipelineManager().get_incomplete_patients(
            str(tmp_path), lesion_type="glioblastoma",
        )
        assert gbm[0]["required"] == ["t1", "t1c", "t2", "t2fl"]

    def test_sparse_series_entries_do_not_crash(self, tmp_path):
        """Older mappings (and every existing fixture) store bare {} for a
        selected modality. Reading it must not require fields it lacks."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        selected = PipelineManager().get_incomplete_patients(
            str(tmp_path), lesion_type="glioblastoma",
        )[0]["selected"]
        assert selected[0] == {
            "modality": "t1", "series_description": "",
            "original_path": "", "slice_count": 0,
        }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_incomplete_patients_api.py::TestSelectedAndRequired -q`
Expected: FAIL — `KeyError: 'selected'`.

- [ ] **Step 3: Return the two new fields**

In `backend/pipeline_manager.py`, inside `get_incomplete_patients`, replace the
result dict construction (the block starting `available = sorted(...)`):

```python
                available = sorted(session_data.get('series', {}).keys())
                series = session_data.get('series', {}) or {}
                # Ordered by `required` so the screen lists modalities in the
                # lesion type's own order rather than alphabetically.
                selected = [
                    {
                        "modality": modality,
                        "series_description": (series[modality] or {}).get(
                            'series_description', ''),
                        "original_path": (series[modality] or {}).get(
                            'original_path', ''),
                        "slice_count": (series[modality] or {}).get(
                            'slice_count', 0),
                    }
                    for modality in sorted(required)
                    if modality in series
                ]
                results.append({
                    "patient_id": patient_id,
                    "original_id": patient_data.get('original_id', ''),
                    "session_id": session_id,
                    "date": session_data.get('original_date', ''),
                    "status": status,
                    "available": available,
                    "missing": sorted(required - set(available)),
                    "selected": selected,
                    "required": sorted(required),
                    "excluded_series": session_data.get('excluded_series', []),
                    "merged_into_session_id": session_data.get('merged_into_session_id'),
                })
```

- [ ] **Step 4: Add the response models**

In `backend/models.py`, immediately before `class IncompletePatientSession`:

```python
class SelectedModality(BaseModel):
    """Модальность, которую алгоритм (или врач) уже отобрал."""
    modality: str = Field(..., description="t1 | t1c | t2 | t2fl")
    series_description: str = Field("", description="Имя серии из протокола")
    original_path: str = Field("", description="Путь к исходной DICOM-серии")
    slice_count: int = Field(0, description="Число DICOM-файлов в серии")
```

And add to `IncompletePatientSession`:

```python
    selected: List[SelectedModality] = Field(
        default_factory=list, description="Что уже отобрано, с именами из протокола"
    )
    required: List[str] = Field(
        default_factory=list,
        description="Обязательные модальности ДЛЯ ЭТОГО типа поражения",
    )
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_incomplete_patients_api.py -q`
Expected: all pass, including the pre-existing classes.

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 6: Commit**

```bash
git add backend/models.py backend/pipeline_manager.py \
        backend/test_incomplete_patients_api.py
git commit -m "feat(review): return the selected set and the lesion type's required set

The queue showed only what was missing, so a modality the detector picked
wrongly was invisible — only an empty slot was actionable. The protocol
name of each selection was already stored and simply never returned.

required comes from configs/lesion_types.yaml so the frontend can stop
hardcoding four glioblastoma modalities at a multiple-sclerosis run.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Deleting one session's outputs

**Files:**
- Create: `backend/session_artifacts.py`
- Test: `backend/test_session_artifacts.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `STAGE_DIRS: tuple[str, ...]`
  - `delete_session_artifacts(output_path, patient_id, session_id) -> list[str]` — the paths removed
  - `purge_sessions_marked_for_reprocess(output_path) -> dict[str, list[str]]` — `{"sub-001/ses-001": [removed paths]}`

**Background:** every stage writes per-patient output as
`{stage_dir}/{sub-XXX}/{ses-YYY}/`, confirmed against a real run.
`bids_organized/` follows the same shape but is the pipeline's *input* and must
survive.

- [ ] **Step 1: Write the failing test**

Create `backend/test_session_artifacts.py`:

```python
"""Deleting one session's stage outputs so it gets rebuilt on the next run."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from session_artifacts import (
    STAGE_DIRS, delete_session_artifacts, purge_sessions_marked_for_reprocess,
)


def _make_run(root: Path, sessions=(("sub-001", "ses-001"), ("sub-002", "ses-001"))):
    """A run directory with per-stage output for each session, plus the
    bids_organized input that must survive."""
    for patient, session in sessions:
        for stage in STAGE_DIRS:
            d = root / stage / patient / session / "anat"
            d.mkdir(parents=True, exist_ok=True)
            (d / "file.nii.gz").write_bytes(b"x")
        bids = root / "bids_organized" / patient / session / "anat" / "t1"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "0001.dcm").write_bytes(b"x")
    return root


def test_removes_the_session_from_every_stage(tmp_path):
    _make_run(tmp_path)
    removed = delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    assert len(removed) == len(STAGE_DIRS)
    for stage in STAGE_DIRS:
        assert not (tmp_path / stage / "sub-001" / "ses-001").exists()


def test_keeps_the_pipeline_input(tmp_path):
    """bids_organized holds the corrected assignment — deleting it would
    throw away the very thing the doctor just fixed."""
    _make_run(tmp_path)
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    assert (tmp_path / "bids_organized" / "sub-001" / "ses-001").exists()


def test_leaves_other_patients_alone(tmp_path):
    _make_run(tmp_path)
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    for stage in STAGE_DIRS:
        assert (tmp_path / stage / "sub-002" / "ses-001").exists()


def test_removes_the_patient_directory_when_it_empties(tmp_path):
    _make_run(tmp_path, sessions=(("sub-001", "ses-001"),))
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    for stage in STAGE_DIRS:
        assert not (tmp_path / stage / "sub-001").exists()


def test_keeps_the_patient_directory_when_another_session_remains(tmp_path):
    _make_run(tmp_path, sessions=(("sub-001", "ses-001"), ("sub-001", "ses-002")))
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    for stage in STAGE_DIRS:
        assert (tmp_path / stage / "sub-001" / "ses-002").exists()


@pytest.mark.parametrize("patient,session", [
    ("../..", "ses-001"),
    ("sub-001", "../../etc"),
    ("sub-001/../..", "ses-001"),
    ("", "ses-001"),
])
def test_refuses_ids_that_could_escape_the_run_directory(tmp_path, patient, session):
    """This is a delete driven by values read out of a JSON file. Escaping
    the run directory must be impossible by construction, not by luck."""
    _make_run(tmp_path)
    with pytest.raises(ValueError):
        delete_session_artifacts(str(tmp_path), patient, session)
    assert (tmp_path / "nifti" / "sub-001" / "ses-001").exists()


def test_purge_clears_the_flags_it_acted_on(tmp_path):
    _make_run(tmp_path)
    bids = tmp_path / "bids_organized"
    mapping = {
        "patients": {
            "sub-001": {"original_id": "P1", "sessions": {
                "ses-001": {"status": "complete", "needs_reprocess": True},
            }},
            "sub-002": {"original_id": "P2", "sessions": {
                "ses-001": {"status": "complete"},
            }},
        }
    }
    (bids / "dataset_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    purged = purge_sessions_marked_for_reprocess(str(tmp_path))

    assert list(purged) == ["sub-001/ses-001"]
    assert not (tmp_path / "nifti" / "sub-001" / "ses-001").exists()
    assert (tmp_path / "nifti" / "sub-002" / "ses-001").exists()

    after = json.loads((bids / "dataset_mapping.json").read_text(encoding="utf-8"))
    assert after["patients"]["sub-001"]["sessions"]["ses-001"]["needs_reprocess"] is False


def test_purge_is_a_no_op_without_a_mapping(tmp_path):
    assert purge_sessions_marked_for_reprocess(str(tmp_path)) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_session_artifacts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'session_artifacts'`.

- [ ] **Step 3: Write the module**

Create `backend/session_artifacts.py`:

```python
"""Removing one session's stage outputs so the next run rebuilds it.

`skip_existing` makes every stage skip a patient whose output already
exists. That is what makes a correction pointless on its own: the doctor
fixes the modality set and the pipeline skips right past it. Deleting the
session's outputs is how the correction reaches the data.

What is deleted is only ever output. `bids_organized/` holds the corrected
assignment and is the pipeline's input, so it stays.
"""
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Every stage writes per-patient output as {stage}/{sub-XXX}/{ses-YYY}/.
STAGE_DIRS = (
    "nifti",
    "preprocessed",
    "quality_reports",
    "segmentation",
    "transformations",
)

_PATIENT_RE = re.compile(r"^sub-\d+$")
_SESSION_RE = re.compile(r"^ses-\d+$")


def delete_session_artifacts(
    output_path: str, patient_id: str, session_id: str
) -> List[str]:
    """Delete one session's output under every stage directory.

    Returns the paths actually removed. Raises ValueError for identifiers
    that are not plain BIDS ids — these come out of a JSON file, and a
    delete must not be steerable by its contents.
    """
    if not _PATIENT_RE.match(patient_id or ""):
        raise ValueError(f"Invalid patient_id: {patient_id!r}")
    if not _SESSION_RE.match(session_id or ""):
        raise ValueError(f"Invalid session_id: {session_id!r}")

    base = Path(output_path)
    removed: List[str] = []

    for stage in STAGE_DIRS:
        session_dir = base / stage / patient_id / session_id
        if not session_dir.is_dir():
            continue
        shutil.rmtree(session_dir)
        removed.append(str(session_dir))

        # Drop the patient directory too once its last session is gone, so
        # the tree does not accumulate empty shells.
        patient_dir = base / stage / patient_id
        if patient_dir.is_dir() and not any(patient_dir.iterdir()):
            patient_dir.rmdir()

    if removed:
        logger.info(
            "Удалены результаты %s/%s для переобработки: %d папок",
            patient_id, session_id, len(removed),
        )
    return removed


def purge_sessions_marked_for_reprocess(output_path: str) -> Dict[str, List[str]]:
    """Delete the outputs of every session flagged needs_reprocess, and clear
    the flags. Returns {"sub-001/ses-001": [removed paths]}.

    A session with no outputs yet simply yields an empty list — the flag is
    set whenever the assignment changed, without asking whether the session
    was ever processed, so a no-op here is expected rather than exceptional.
    """
    mapping_file = Path(output_path) / "bids_organized" / "dataset_mapping.json"
    if not mapping_file.exists():
        return {}

    try:
        with open(mapping_file, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.error("Не удалось прочитать %s: %s", mapping_file, exc)
        return {}

    purged: Dict[str, List[str]] = {}
    changed = False

    for patient_id, patient in (mapping.get("patients") or {}).items():
        for session_id, session in (patient.get("sessions") or {}).items():
            if not session.get("needs_reprocess"):
                continue
            try:
                purged[f"{patient_id}/{session_id}"] = delete_session_artifacts(
                    output_path, patient_id, session_id
                )
            except ValueError as exc:
                logger.error("Пропущена сессия с некорректным идентификатором: %s", exc)
                continue
            session["needs_reprocess"] = False
            changed = True

    if changed:
        with open(mapping_file, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, ensure_ascii=False)

    return purged
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_session_artifacts.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/session_artifacts.py backend/test_session_artifacts.py
git commit -m "feat(review): delete one session's outputs so it can be rebuilt

skip_existing makes every stage skip a patient whose output already
exists, which is what makes a corrected modality set pointless on its own
— the pipeline walks straight past the fix. Removing the session's outputs
is how the correction reaches the data.

Only output is removed; bids_organized holds the corrected assignment and
is the input. Identifiers are validated before any path is built: this
delete is driven by values read out of a JSON file and must not be
steerable by them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Validating and diffing a desired set

**Files:**
- Create: `backend/session_assignment.py`
- Test: `backend/test_session_assignment.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class AssignmentError(ValueError)`
  - `EDITABLE_STATUSES: frozenset[str]` — `{"incomplete", "complete"}`
  - `validate(session_data: dict, assignments: dict[str, str], required: Sequence[str]) -> None`
  - `@dataclass(frozen=True) Changes` with `assign: dict[str, str]`, `clear: tuple[str, ...]`, `unchanged: tuple[str, ...]`, and `is_empty() -> bool`
  - `plan_changes(session_data: dict, assignments: dict[str, str]) -> Changes`

**Background:** this module is deliberately pure — dicts in, dicts out, no
disk, no config loading. It is what lets the whole desired set be checked
before anything is written, which is the property that makes a save button
honest.

- [ ] **Step 1: Write the failing test**

Create `backend/test_session_assignment.py`:

```python
"""Validating a desired modality set, and working out what has to change."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from session_assignment import AssignmentError, plan_changes, validate

REQUIRED = ["t1", "t1c", "t2", "t2fl"]


def _session(series=None, excluded=(), status="incomplete"):
    return {
        "status": status,
        "series": series or {},
        "excluded_series": [
            {"original_path": p, "series_description": "d",
             "slice_count": 10, "detected_modality": None,
             "reason": "unrecognized"}
            for p in excluded
        ],
    }


def test_accepts_a_set_drawn_from_this_session():
    session = _session(series={"t1": {"original_path": "/raw/a"}},
                       excluded=["/raw/b"])
    validate(session, {"t1": "/raw/a", "t1c": "/raw/b"}, REQUIRED)


def test_rejects_a_path_from_outside_this_session():
    """A stale screen can send a path belonging to another patient."""
    session = _session(series={"t1": {"original_path": "/raw/a"}})
    with pytest.raises(AssignmentError, match="не принадлежит"):
        validate(session, {"t1": "/raw/somebody-else"}, REQUIRED)


def test_rejects_a_modality_this_lesion_type_does_not_use():
    """MS has no t1c. Offering the slot at all was the bug; accepting it
    would write a modality nothing downstream expects."""
    session = _session(excluded=["/raw/b"])
    with pytest.raises(AssignmentError, match="t1c"):
        validate(session, {"t1c": "/raw/b"}, ["t1", "t2", "t2fl"])


def test_rejects_one_series_assigned_to_two_modalities():
    session = _session(excluded=["/raw/b"])
    with pytest.raises(AssignmentError, match="дважды"):
        validate(session, {"t1": "/raw/b", "t2": "/raw/b"}, REQUIRED)


@pytest.mark.parametrize("status", ["discarded", "merged"])
def test_refuses_sessions_whose_fate_is_already_decided(status):
    """Discarding and merging are decisions already taken. Quietly
    re-opening them would undo a choice nobody asked to undo."""
    session = _session(excluded=["/raw/b"], status=status)
    with pytest.raises(AssignmentError, match=status):
        validate(session, {"t1": "/raw/b"}, REQUIRED)


def test_plan_leaves_an_unchanged_modality_alone():
    """Without this every save would re-copy every DICOM in the session."""
    session = _session(series={"t1": {"original_path": "/raw/a"}})
    changes = plan_changes(session, {"t1": "/raw/a"})

    assert changes.assign == {}
    assert changes.clear == ()
    assert changes.unchanged == ("t1",)
    assert changes.is_empty()


def test_plan_reports_a_replacement():
    session = _session(series={"t1": {"original_path": "/raw/a"}},
                       excluded=["/raw/b"])
    changes = plan_changes(session, {"t1": "/raw/b"})

    assert changes.assign == {"t1": "/raw/b"}
    assert changes.clear == ()
    assert not changes.is_empty()


def test_plan_reports_a_new_fill():
    session = _session(excluded=["/raw/b"])
    changes = plan_changes(session, {"t2": "/raw/b"})

    assert changes.assign == {"t2": "/raw/b"}


def test_plan_reports_a_clearing():
    """"This is not the t1c" is a legitimate thing for a doctor to say."""
    session = _session(series={"t1": {"original_path": "/raw/a"},
                               "t2": {"original_path": "/raw/c"}})
    changes = plan_changes(session, {"t1": "/raw/a"})

    assert changes.clear == ("t2",)
    assert changes.unchanged == ("t1",)
    assert not changes.is_empty()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_session_assignment.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'session_assignment'`.

- [ ] **Step 3: Write the module**

Create `backend/session_assignment.py`:

```python
"""Checking a desired modality set, and working out what has to change.

Pure on purpose: dicts in, dicts out, no disk and no config loading. The
API takes the desired FINAL set rather than a list of operations, and this
is what makes that worth doing — the whole set can be checked before a
single file is written, so the mistakes a doctor can actually make (a stale
path from an old screen, a modality this lesion type has no concept of)
cannot half-apply.
"""
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

# A discarded or merged session has already been decided about. Re-opening
# it through this path would silently undo that decision.
EDITABLE_STATUSES = frozenset({"incomplete", "complete"})


class AssignmentError(ValueError):
    """The desired set cannot be applied. Nothing has been written."""


def known_paths(session_data: Mapping[str, Any]) -> set:
    """Every series this session knows about, selected or not."""
    paths = {
        (entry or {}).get("original_path")
        for entry in (session_data.get("series") or {}).values()
    }
    paths |= {
        entry.get("original_path")
        for entry in (session_data.get("excluded_series") or [])
    }
    paths.discard(None)
    paths.discard("")
    return paths


def validate(
    session_data: Mapping[str, Any],
    assignments: Mapping[str, str],
    required: Sequence[str],
) -> None:
    """Raise AssignmentError if the desired set cannot be applied."""
    status = session_data.get("status")
    if status not in EDITABLE_STATUSES:
        raise AssignmentError(
            f"Сессию в состоянии «{status}» редактировать нельзя"
        )

    allowed = set(required)
    for modality in assignments:
        if modality not in allowed:
            raise AssignmentError(
                f"Модальность {modality} не входит в обязательные для этого "
                f"типа поражения: {sorted(allowed)}"
            )

    available = known_paths(session_data)
    for modality, path in assignments.items():
        if path not in available:
            raise AssignmentError(
                f"Серия {path!r} не принадлежит этой сессии"
            )

    seen: Dict[str, str] = {}
    for modality, path in assignments.items():
        if path in seen:
            raise AssignmentError(
                f"Серия {path!r} назначена дважды: {seen[path]} и {modality}"
            )
        seen[path] = modality


@dataclass(frozen=True)
class Changes:
    """What applying the desired set actually requires."""
    assign: Dict[str, str]          # modality -> path to copy in
    clear: Tuple[str, ...]          # modalities to empty
    unchanged: Tuple[str, ...]      # modalities to leave completely alone

    def is_empty(self) -> bool:
        return not self.assign and not self.clear


def plan_changes(
    session_data: Mapping[str, Any], assignments: Mapping[str, str]
) -> Changes:
    """Diff the desired set against the current one."""
    current = {
        modality: (entry or {}).get("original_path")
        for modality, entry in (session_data.get("series") or {}).items()
    }

    assign: Dict[str, str] = {}
    unchanged = []
    for modality, path in assignments.items():
        if current.get(modality) == path:
            unchanged.append(modality)
        else:
            assign[modality] = path

    clear = tuple(sorted(set(current) - set(assignments)))
    return Changes(assign=assign, clear=clear, unchanged=tuple(sorted(unchanged)))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_session_assignment.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/session_assignment.py backend/test_session_assignment.py
git commit -m "feat(review): validate and diff a desired modality set

The API will take the desired final set rather than a list of operations,
and this module is what makes that worth doing: the whole set is checked
before anything is written, so a stale path or a modality this lesion type
has no concept of cannot half-apply.

Pure by design — dicts in, dicts out — so the rules are testable without a
filesystem, and diffing keeps an unchanged modality untouched instead of
re-copying every DICOM in the session on every save.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Applying the set to disk

**Files:**
- Modify: `backend/pipeline_manager.py` (add `apply_assignment` after `relabel_series`, ~line 1163)
- Test: `backend/test_session_assignment_apply.py` (create)

**Interfaces:**
- Consumes: `session_assignment.validate`, `.plan_changes`, `.AssignmentError`
- Produces:
  - `PipelineManager.apply_assignment(output_path, patient_id, session_id, assignments: dict[str, str], lesion_type: str) -> dict` returning `{"status", "selected", "excluded_series", "needs_reprocess"}`

**Background:** read `relabel_series` (`backend/pipeline_manager.py:1016`)
before writing this — the copy guard, the `_incomplete/` move and the
"previous occupant returns to excluded_series" rule all come from there and
must behave identically.

- [ ] **Step 1: Write the failing test**

Create `backend/test_session_assignment_apply.py`:

```python
"""Applying a desired set: what lands on disk, and what must not."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import pipeline_manager
from pipeline_manager import PipelineManager
from session_assignment import AssignmentError


def _run_dir(tmp_path, series, excluded, status="incomplete"):
    bids = tmp_path / "bids_organized"
    (bids / "_incomplete").mkdir(parents=True, exist_ok=True)
    mapping = {"patients": {"sub-001": {"original_id": "P1", "sessions": {
        "ses-001": {
            "original_date": "20230101",
            "status": status,
            "series": series,
            "excluded_series": excluded,
        },
    }}}}
    (bids / "dataset_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8")
    return tmp_path


def _mapping(tmp_path):
    return json.loads(
        (tmp_path / "bids_organized" / "dataset_mapping.json").read_text(
            encoding="utf-8")
    )


def _excluded(path):
    return {"original_path": path, "series_description": "d",
            "slice_count": 3, "detected_modality": None,
            "reason": "unrecognized"}


def test_a_rejected_set_leaves_the_mapping_byte_identical(tmp_path):
    """The whole point of validating first: a bad request must not apply
    half of itself."""
    _run_dir(tmp_path, {"t1": {"original_path": "/raw/a"}}, [_excluded("/raw/b")])
    before = (tmp_path / "bids_organized" / "dataset_mapping.json").read_bytes()

    with pytest.raises(AssignmentError):
        PipelineManager().apply_assignment(
            str(tmp_path), "sub-001", "ses-001",
            {"t1": "/raw/does-not-belong"}, "glioblastoma",
        )

    after = (tmp_path / "bids_organized" / "dataset_mapping.json").read_bytes()
    assert after == before


def test_an_unchanged_set_copies_nothing_and_flags_nothing(tmp_path, monkeypatch):
    _run_dir(tmp_path, {"t1": {"original_path": "/raw/a"}}, [])
    copied = []
    monkeypatch.setattr(
        pipeline_manager, "copy_and_anonymize_series",
        lambda *a, **k: copied.append(a) or 0,
    )

    result = PipelineManager().apply_assignment(
        str(tmp_path), "sub-001", "ses-001", {"t1": "/raw/a"}, "glioblastoma",
    )

    assert copied == []
    assert result["needs_reprocess"] is False


def test_clearing_a_modality_makes_the_session_incomplete_again(tmp_path):
    """A doctor is allowed to say "this is not the t1c"."""
    _run_dir(
        tmp_path,
        {"t1": {"original_path": "/raw/a"}, "t1c": {"original_path": "/raw/b"},
         "t2": {"original_path": "/raw/c"}, "t2fl": {"original_path": "/raw/d"}},
        [], status="complete",
    )
    (tmp_path / "bids_organized" / "sub-001" / "ses-001" / "anat" / "t1c").mkdir(
        parents=True)

    result = PipelineManager().apply_assignment(
        str(tmp_path), "sub-001", "ses-001",
        {"t1": "/raw/a", "t2": "/raw/c", "t2fl": "/raw/d"}, "glioblastoma",
    )

    assert result["status"] == "incomplete"
    assert result["needs_reprocess"] is True
    # The cleared series is not lost — it goes back to the pool.
    paths = [e["original_path"] for e in result["excluded_series"]]
    assert "/raw/b" in paths
    # And the session moved back out of the main tree.
    assert (tmp_path / "bids_organized" / "_incomplete" / "sub-001" / "ses-001").exists()


def test_the_flag_is_set_whenever_the_set_changed(tmp_path, monkeypatch):
    _run_dir(tmp_path, {}, [_excluded("/raw/b")])
    monkeypatch.setattr(
        pipeline_manager, "find_dicom_files", lambda p: [Path("/raw/b/1.dcm")])
    monkeypatch.setattr(
        pipeline_manager, "copy_and_anonymize_series", lambda *a, **k: 1)
    monkeypatch.setattr(
        PipelineManager, "_build_metadata_extractor", lambda self: object())

    result = PipelineManager().apply_assignment(
        str(tmp_path), "sub-001", "ses-001", {"t1": "/raw/b"}, "glioblastoma",
    )

    assert result["needs_reprocess"] is True
    assert _mapping(tmp_path)["patients"]["sub-001"]["sessions"]["ses-001"][
        "needs_reprocess"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_session_assignment_apply.py -q`
Expected: FAIL — `AttributeError: 'PipelineManager' object has no attribute 'apply_assignment'`.

- [ ] **Step 3: Read the code this mirrors**

```bash
sed -n 1016,1163p backend/pipeline_manager.py
```

Note three behaviours to reproduce exactly: the target directory is cleared
before copying (a replacement with fewer files otherwise leaves the previous
occupant's tail behind), a short copy raises without touching the mapping, and
a session that becomes complete moves out of `_incomplete/`.

- [ ] **Step 4: Write `apply_assignment`**

Add to `backend/pipeline_manager.py`, after `relabel_series`:

```python
    def apply_assignment(
        self,
        output_path: str,
        patient_id: str,
        session_id: str,
        assignments: Dict[str, str],
        lesion_type: str = 'glioblastoma',
    ) -> Dict[str, Any]:
        """Apply a desired modality set to a session, all at once.

        Takes the FINAL set the doctor wants, not a list of operations, so
        the whole thing can be checked before a single file is written and
        the order edits were made in cannot change the outcome.

        dataset_mapping.json is written exactly once, at the end. If any copy
        fails it is not written at all, so the file every stage reads either
        describes the new set completely or is untouched.
        """
        from session_assignment import AssignmentError, plan_changes, validate

        if not _BIDS_PATIENT_ID_PATTERN.match(patient_id):
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")

        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        try:
            session_data = mapping_data['patients'][patient_id]['sessions'][session_id]
        except KeyError:
            raise ValueError(f"No such session: {patient_id}/{session_id}")

        try:
            required = list(load_lesion_type_config(lesion_type)['required_modalities'])
        except KeyError:
            required = ['t1', 't1c', 't2', 't2fl']

        # Everything that can be judged without touching the disk, judged now.
        validate(session_data, assignments, required)
        changes = plan_changes(session_data, assignments)

        bids_dir = Path(output_path) / "bids_organized"
        was_incomplete = session_data.get('status') == 'incomplete'
        current_root = (bids_dir / "_incomplete") if was_incomplete else bids_dir

        series = session_data.setdefault('series', {})
        excluded = list(session_data.get('excluded_series', []))

        def _to_excluded(modality: str, entry: Dict[str, Any], reason: str):
            excluded.append({
                'original_path': (entry or {}).get('original_path', ''),
                'series_description': (entry or {}).get('series_description', ''),
                'slice_count': (entry or {}).get('slice_count', 0),
                'detected_modality': modality,
                'reason': reason,
            })

        for modality in changes.clear:
            _to_excluded(modality, series.pop(modality), 'cleared_by_doctor')
            stale_dir = current_root / patient_id / session_id / "anat" / modality
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)

        for modality, original_path in changes.assign.items():
            source_entry = next(
                (e for e in excluded if e['original_path'] == original_path), None
            )
            target_dir = current_root / patient_id / session_id / "anat" / modality
            target_dir.mkdir(parents=True, exist_ok=True)
            # Clear first: a replacement with FEWER files than the previous
            # occupant would otherwise overwrite only the first N and leave
            # the old tail behind, so the directory would silently hold a mix
            # of two series while the mapping calls it one.
            for stale_file in target_dir.iterdir():
                if stale_file.is_file():
                    stale_file.unlink()

            metadata_extractor = self._build_metadata_extractor()
            if metadata_extractor is None:
                raise ValueError(
                    "Anonymization config (configs/dicom_tags.yaml) not found — "
                    "refusing to copy patient DICOM data without anonymizing it"
                )
            source_files = find_dicom_files(Path(original_path))
            copied = copy_and_anonymize_series(
                source_files, target_dir, patient_id, session_id, modality,
                metadata_extractor=metadata_extractor, logger=logger,
            )
            if copied != len(source_files):
                raise ValueError(
                    f"Copy failed: only {copied}/{len(source_files)} files copied "
                    f"for {patient_id}/{session_id}/{modality} (source: {original_path})"
                )

            previous = series.get(modality)
            if previous is not None:
                _to_excluded(modality, previous, 'replaced_by_manual_relabel')
            excluded = [e for e in excluded if e['original_path'] != original_path]
            series[modality] = {
                'original_path': original_path,
                'slice_count': len(source_files),
                'series_description': (source_entry or {}).get(
                    'series_description', ''),
            }

        session_data['excluded_series'] = excluded
        is_complete = set(required).issubset(series.keys())
        session_data['status'] = 'complete' if is_complete else 'incomplete'
        session_data['manually_reviewed'] = True
        if not changes.is_empty():
            session_data['needs_reprocess'] = True

        # Move the session between _incomplete/ and the main tree if its
        # completeness flipped. Only the incomplete -> complete direction
        # existed before; clearing a modality needs the other one.
        session_src = current_root / patient_id / session_id
        target_root = bids_dir if is_complete else (bids_dir / "_incomplete")
        session_dst = target_root / patient_id / session_id
        if session_src != session_dst and session_src.is_dir():
            session_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(session_src), str(session_dst))
            leftover = current_root / patient_id
            if leftover.is_dir() and not any(leftover.iterdir()):
                leftover.rmdir()

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        return {
            'status': session_data['status'],
            'selected': [
                {
                    'modality': m,
                    'series_description': (series[m] or {}).get('series_description', ''),
                    'original_path': (series[m] or {}).get('original_path', ''),
                    'slice_count': (series[m] or {}).get('slice_count', 0),
                }
                for m in sorted(required) if m in series
            ],
            'excluded_series': excluded,
            'needs_reprocess': bool(session_data.get('needs_reprocess')),
        }
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_session_assignment_apply.py -q`
Expected: 4 passed.

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/test_session_assignment_apply.py
git commit -m "feat(review): apply a whole modality set in one operation

The set is validated before any file is touched and dataset_mapping.json is
written exactly once at the end, so the file every stage reads either
describes the new set completely or is untouched.

Clearing a modality without replacing it is now possible — "this is not the
t1c" is a legitimate thing for a doctor to say — which also required the
incomplete <- complete move that never existed, only its opposite.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The assignment endpoint

**Files:**
- Modify: `backend/models.py` (after `RelabelSeriesResponse`, ~line 177)
- Modify: `backend/app.py` (after the relabel endpoint, ~line 1333)
- Test: `backend/test_app_assignment_endpoint.py` (create)

**Interfaces:**
- Consumes: `PipelineManager.apply_assignment`
- Produces:
  - `models.AssignmentRequest` with `assignments: Dict[str, str]`
  - `models.AssignmentResponse` with `status: str`, `selected: List[SelectedModality]`, `excluded_series: List[ExcludedSeriesInfo]`, `needs_reprocess: bool`, `kappa_warning: Optional[str]`
  - `PUT /api/incomplete-patients/{run_id}/{patient_id}/{session_id}/assignment`

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_assignment_endpoint.py`:

```python
"""The assignment endpoint: rejections are 400, not 500."""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, "backend")


@pytest.mark.asyncio
async def test_a_bad_set_is_a_client_error(monkeypatch):
    """AssignmentError means the doctor sent something impossible, not that
    the service broke. A 500 would also hide the reason from the screen."""
    import app
    import pipeline_manager as pm
    from models import AssignmentRequest
    from session_assignment import AssignmentError

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())

    def _reject(*a, **k):
        raise AssignmentError("Серия '/raw/x' не принадлежит этой сессии")

    monkeypatch.setattr(pm.pipeline_manager, "apply_assignment", _reject)

    with pytest.raises(HTTPException) as caught:
        await app.save_assignment(
            "run-1", "sub-001", "ses-001",
            AssignmentRequest(assignments={"t1": "/raw/x"}), db=None,
        )
    assert caught.value.status_code == 400
    assert "не принадлежит" in caught.value.detail


@pytest.mark.asyncio
async def test_a_successful_save_returns_the_new_state(monkeypatch):
    import app
    import pipeline_manager as pm
    from models import AssignmentRequest

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"
        kappa_dataset_id = None

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(
        pm.pipeline_manager, "apply_assignment",
        lambda *a, **k: {
            "status": "complete",
            "selected": [{"modality": "t1", "series_description": "d",
                          "original_path": "/raw/a", "slice_count": 3}],
            "excluded_series": [],
            "needs_reprocess": True,
        },
    )

    result = await app.save_assignment(
        "run-1", "sub-001", "ses-001",
        AssignmentRequest(assignments={"t1": "/raw/a"}), db=None,
    )
    assert result.status == "complete"
    assert result.needs_reprocess is True
    assert result.selected[0].modality == "t1"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_assignment_endpoint.py -q`
Expected: FAIL — `ImportError: cannot import name 'AssignmentRequest'`.

- [ ] **Step 3: Add the models**

In `backend/models.py`, after `RelabelSeriesResponse`:

```python
class AssignmentRequest(BaseModel):
    """Желаемый итоговый набор модальностей сессии.

    Именно набор, а не список действий: тогда его можно проверить целиком
    до первой записи на диск, и порядок правок не влияет на результат.
    """
    assignments: Dict[str, str] = Field(
        ..., description="модальность -> original_path выбранной серии"
    )


class AssignmentResponse(BaseModel):
    """Состояние сессии после сохранения набора."""
    status: str = Field(..., description="complete | incomplete")
    selected: List[SelectedModality] = Field(default_factory=list)
    excluded_series: List[ExcludedSeriesInfo] = Field(default_factory=list)
    needs_reprocess: bool = Field(
        False, description="Набор изменился — пациент будет переобработан"
    )
    kappa_warning: Optional[str] = Field(
        None, description="Предупреждение, если старая версия уже в Kappa"
    )
```

- [ ] **Step 4: Add the endpoint**

In `backend/app.py`, after the relabel endpoint, and add
`AssignmentRequest, AssignmentResponse` to the `from models import (...)` block:

```python
@app.put(
    "/api/incomplete-patients/{run_id}/{patient_id}/{session_id}/assignment",
    response_model=AssignmentResponse,
)
async def save_assignment(
    run_id: str,
    patient_id: str,
    session_id: str,
    request: AssignmentRequest,
    db: Session = Depends(get_db),
):
    """Сохранить набор модальностей сессии целиком."""
    from session_assignment import AssignmentError

    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    try:
        result = pipeline_monitor.pipeline_manager.apply_assignment(
            output_path=run.output_path,
            patient_id=patient_id,
            session_id=session_id,
            assignments=request.assignments,
            lesion_type=getattr(run, "lesion_type", None) or "glioblastoma",
        )
    except AssignmentError as e:
        # Набор невозможен — это ошибка запроса, а не сбой сервиса. И текст
        # должен дойти до экрана: врачу нужно знать, что именно не так.
        raise HTTPException(status_code=400, detail=str(e))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(
        "Набор модальностей сохранён: %s/%s -> %s",
        patient_id, session_id, sorted(request.assignments),
    )
    return AssignmentResponse(**result)
```

Use `pipeline_monitor.pipeline_manager` (the instance `app.py` already holds)
rather than constructing a new `PipelineManager`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_app_assignment_endpoint.py -q`
Expected: 2 passed.

Run: `python -c "import sys; sys.path.insert(0,'backend'); import app"`
Expected: no traceback.

- [ ] **Step 6: Commit**

```bash
git add backend/models.py backend/app.py backend/test_app_assignment_endpoint.py
git commit -m "feat(review): endpoint that saves a whole modality set

Takes the desired final set, answers with the session's complete new state
so the screen re-renders without a second request. A set that cannot be
applied is a 400 carrying the reason, not a 500 that hides it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Requeue rebuilds the corrected sessions

**Files:**
- Modify: `backend/app.py` (requeue endpoint, ~line 609)
- Test: `backend/test_app_requeue_endpoint.py` (extend)

**Interfaces:**
- Consumes: `session_artifacts.purge_sessions_marked_for_reprocess`
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_app_requeue_endpoint.py`:

```python
def test_requeue_rebuilds_sessions_whose_set_changed():
    """Without this the correction changes nothing: skip_existing sees the
    old outputs and walks straight past the patient."""
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()
    purged = []

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=new_run), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               side_effect=lambda out: purged.append(out) or {"sub-001/ses-001": []}):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    assert purged == ["/out"], "помеченные сессии не очищены перед запуском"


def test_a_failed_purge_does_not_block_the_run():
    """Cleanup is housekeeping. Refusing to start the run because a stale
    folder could not be removed would be a worse outcome than the mess."""
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=new_run), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               side_effect=OSError("disk is read-only")):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
```

These follow the file's existing pattern: `TestClient` plus `patch(...)` over
`app.*`, with `_fake_original_run` / `_fake_new_run` already defined at the top
of the file.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_requeue_endpoint.py -q`
Expected: FAIL — `assert [] == ['/out']`, because nothing purges yet.

- [ ] **Step 3: Purge before launching**

In `backend/app.py`, in `requeue_pipeline_run`, immediately before
`background_tasks.add_task(...)`:

```python
    # Сессии, у которых врач поменял набор модальностей, надо пересчитать.
    # skip_existing пропускает всё, у чего уже есть результаты, поэтому без
    # удаления исправление просто не дошло бы до данных.
    try:
        from session_artifacts import purge_sessions_marked_for_reprocess
        purged = purge_sessions_marked_for_reprocess(original_run.output_path)
        if purged:
            logger.info("Переобработка: очищено сессий — %d", len(purged))
    except Exception as e:  # noqa: BLE001 — запуск важнее уборки
        logger.error("Не удалось очистить помеченные сессии: %s", e)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_app_requeue_endpoint.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/test_app_requeue_endpoint.py
git commit -m "feat(review): requeue rebuilds the sessions whose set changed

skip_existing walks past any patient that already has output, so without
this a corrected modality set never reached the data.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Warn when Kappa already holds the old version

**Files:**
- Modify: `backend/app.py` (`save_assignment` from Task 5)
- Test: `backend/test_app_assignment_endpoint.py` (extend)

**Interfaces:**
- Consumes: `patient_registry.find_by_bids_id`, `AssignmentResponse.kappa_warning`
- Produces: nothing new.

**Background:** `_compute_study_hash` hashes `PatientID:StudyInstanceUID` — the
DICOM study's identity, not the images. A reprocessed patient therefore has the
same hash, so the uploader treats it as already delivered and Kappa keeps the
mask computed from the wrong set. Replacing it is a separate task; this one
makes sure the doctor is told.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_app_assignment_endpoint.py`:

```python
@pytest.mark.asyncio
async def test_warns_when_the_old_version_is_already_in_kappa(monkeypatch):
    """study_hash covers the DICOM study, not the images, so a reprocessed
    patient looks like a duplicate and Kappa keeps the old mask. Silence
    here would mean a correction that looks applied and is not."""
    import app
    import pipeline_manager as pm
    from models import AssignmentRequest

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"
        kappa_dataset_id = 351

    seen = {}

    def _find(bids_id, dataset_ids=None):
        seen["dataset_ids"] = dataset_ids
        return [{"bids_id": bids_id, "kappa_entity_id": "e1"}]

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app, "find_by_bids_id", _find)
    monkeypatch.setattr(
        pm.pipeline_manager, "apply_assignment",
        lambda *a, **k: {"status": "complete", "selected": [],
                         "excluded_series": [], "needs_reprocess": True},
    )

    result = await app.save_assignment(
        "run-1", "sub-001", "ses-001",
        AssignmentRequest(assignments={"t1": "/raw/a"}), db=None,
    )

    assert result.kappa_warning is not None
    assert "Kappa" in result.kappa_warning
    # sub-NNN is unique only WITHIN a dataset; an unscoped lookup would
    # report another account's patient as this one.
    assert seen["dataset_ids"] == {351}


@pytest.mark.asyncio
async def test_no_warning_when_nothing_changed(monkeypatch):
    import app
    import pipeline_manager as pm
    from models import AssignmentRequest

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"
        kappa_dataset_id = 351

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(
        pm.pipeline_manager, "apply_assignment",
        lambda *a, **k: {"status": "complete", "selected": [],
                         "excluded_series": [], "needs_reprocess": False},
    )

    result = await app.save_assignment(
        "run-1", "sub-001", "ses-001",
        AssignmentRequest(assignments={"t1": "/raw/a"}), db=None,
    )
    assert result.kappa_warning is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_assignment_endpoint.py -q`
Expected: FAIL — `kappa_warning` is None.

- [ ] **Step 3: Add the warning**

In `save_assignment`, before building the response:

```python
    # Переобработка даёт тот же study_hash (он считается от
    # PatientID:StudyInstanceUID, а не от изображений), поэтому загрузчик
    # сочтёт пациента дубликатом и в Kappa останется старая маска. Молчать
    # об этом нельзя: исправление выглядело бы применённым, не будучи им.
    kappa_warning = None
    if result.get("needs_reprocess") and run.kappa_dataset_id:
        session_key = f"{patient_id}_{session_id}"
        # Со скоупом по датасету: sub-NNN уникален только внутри датасета,
        # и неквалифицированный поиск выдал бы чужого пациента.
        records = find_by_bids_id(session_key, {run.kappa_dataset_id}) or []
        if any(r.get("kappa_entity_id") for r in records):
            kappa_warning = (
                "Этот пациент уже выгружен в Kappa. После переобработки там "
                "останется прежняя версия — она не обновится сама."
            )
            kappa_run_log.append(
                run.output_path,
                f"Набор модальностей {session_key} изменён врачом; "
                f"в Kappa остаётся прежняя версия",
            )

    return AssignmentResponse(**result, kappa_warning=kappa_warning)
```

Add `import kappa_run_log` at the top of the endpoint body, and make sure
`find_by_bids_id` is the module-level name already imported in `app.py` (it is —
`_resolve_longitudinal_records` uses it, and tests patch `app.find_by_bids_id`).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_app_assignment_endpoint.py -q`
Expected: 4 passed.

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/test_app_assignment_endpoint.py
git commit -m "feat(review): say when Kappa will keep the old version

study_hash covers the DICOM study, not the images, so a reprocessed patient
looks like a duplicate to the uploader and Kappa keeps the mask computed
from the wrong modality set. Locally corrected, remotely stale, silently.

Replacing the entity is a separate task; this makes sure nobody is left
believing the correction reached Kappa. The registry lookup is scoped to
the run's dataset — sub-NNN is unique only within one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The screen

**Files:**
- Modify: `frontend/src/services/api.js` (near `relabelSeries`, ~line 302)
- Modify: `frontend/src/components/IncompletePatientDetail.jsx` (whole modality area, lines 13-195)

**Interfaces:**
- Consumes: `session.selected`, `session.required`, `session.excluded_series`, `PUT .../assignment`
- Produces: no backend-facing interfaces.

**Background — the trap:** `MODALITY_OPTIONS` at line 13 is hardcoded to
glioblastoma's four. Deleting it is part of the work, not a nicety: it is why a
multiple-sclerosis run is offered `t1c`. After this the component must contain
no modality names at all.

- [ ] **Step 1: Add the API helper**

In `frontend/src/services/api.js`, after `relabelSeries`:

```javascript
/**
 * Сохранить итоговый набор модальностей сессии целиком.
 * assignments: { t1: '<original_path>', t2: '<original_path>', ... }
 */
export const saveAssignment = async (runId, patientId, sessionId, assignments) => {
  const response = await apiClient.put(
    `/incomplete-patients/${runId}/${patientId}/${sessionId}/assignment`,
    { assignments },
  );
  return response.data;
};
```

Add `saveAssignment` to the default-export object.

- [ ] **Step 2: Rebuild the modality area**

In `IncompletePatientDetail.jsx`: delete `MODALITY_OPTIONS`, and replace the
"Модальности" block and the "Неотобранные серии" list with a draft-based pair
of lists.

First extend `REASON_LABELS` (line 20) with the two reasons this work
introduces — without them the raw codes would be rendered to the doctor:

```javascript
const REASON_LABELS = {
  unrecognized: 'алгоритм не распознал',
  lost_deduplication: 'алгоритм распознал, но выбрал другую копию',
  replaced_by_manual_relabel: 'заменена вручную ранее',
  from_other_session: 'перенесена из другой сессии пациента',
  cleared_by_doctor: 'снята врачом',
  currently_selected: 'сейчас выбрана для другой модальности',
};
```

Draft state, seeded from the session and reset whenever it changes:

```javascript
  // Черновик: модальность -> original_path. На диск ничего не уходит,
  // пока не нажата «Сохранить».
  const [draft, setDraft] = useState({});
  useEffect(() => {
    const initial = {};
    (session.selected || []).forEach((s) => { initial[s.modality] = s.original_path; });
    setDraft(initial);
  }, [session]);

  const isDirty = () => {
    const initial = {};
    (session.selected || []).forEach((s) => { initial[s.modality] = s.original_path; });
    const keys = new Set([...Object.keys(initial), ...Object.keys(draft)]);
    return [...keys].some((k) => initial[k] !== draft[k]);
  };
```

Look up a series' description by path, across both lists, so a selected row can
be rendered from the draft alone:

```javascript
  const describe = (path) => {
    const fromSelected = (session.selected || []).find((s) => s.original_path === path);
    if (fromSelected) return fromSelected;
    const fromExcluded = (session.excluded_series || []).find(
      (e) => e.original_path === path);
    return fromExcluded || null;
  };
```

Render `session.required` — never a local list:

```jsx
        <div>
          <Text strong>Отобранные модальности</Text>
          <List
            size="small"
            dataSource={session.required || []}
            renderItem={(modality) => {
              const path = draft[modality];
              const info = path ? describe(path) : null;
              return (
                <List.Item>
                  <Space>
                    <Checkbox
                      checked={!!path}
                      disabled={isReadOnly || !path}
                      onChange={() => setDraft((prev) => {
                        const next = { ...prev };
                        delete next[modality];
                        return next;
                      })}
                    />
                    <Tag color={path ? 'green' : 'default'}>{modality}</Tag>
                    <Text type={path ? undefined : 'secondary'}>
                      {info
                        ? `${info.series_description} (${info.slice_count} срезов)`
                        : 'не назначена'}
                    </Text>
                  </Space>
                </List.Item>
              );
            }}
          />
        </div>
```

Below it, the unselected pool — everything not currently in the draft, split
into recognized and unrecognized:

```javascript
  const taken = new Set(Object.values(draft));
  const pool = (session.excluded_series || [])
    .concat((session.selected || []).map((s) => ({
      original_path: s.original_path,
      series_description: s.series_description,
      slice_count: s.slice_count,
      detected_modality: s.modality,
      reason: 'currently_selected',
    })))
    .filter((e) => !taken.has(e.original_path));
  const recognized = pool.filter((e) => e.detected_modality);
  const unrecognized = pool.filter((e) => !e.detected_modality);
```

Each pool row offers the free slots — `session.required.filter((m) => !draft[m])`
— plus an explicit "заменить" option for occupied ones, and assigning only
mutates `draft`.

Render `unrecognized` under its own heading, with the same controls. Their
separation is presentational: it tells "the algorithm considered this and
rejected it" apart from "the algorithm had no opinion", and both are assignable.

Finally the footer:

```jsx
        {!isReadOnly && (
          <Space>
            <Button type="primary" disabled={!isDirty()} loading={saving}
                    onClick={handleSave}>
              Сохранить
            </Button>
            <Button disabled={!isDirty()} onClick={resetDraft}>Отменить</Button>
          </Space>
        )}
```

```javascript
  const handleSave = async () => {
    setSaving(true);
    try {
      const result = await saveAssignment(
        runId, session.patient_id, session.session_id, draft);
      message.success(
        result.needs_reprocess
          ? 'Сохранено. Пациент будет переобработан при следующем запуске.'
          : 'Сохранено.',
      );
      if (result.kappa_warning) {
        message.warning(result.kappa_warning, 8);
      }
      onActionComplete();
    } catch (e) {
      message.error(e?.response?.data?.detail || 'Не удалось сохранить набор');
    } finally {
      setSaving(false);
    }
  };
```

- [ ] **Step 3: Lint**

Run: `cd frontend && npm run lint`
Expected: no new errors. The repo baseline is 11 errors and 7 warnings, all
pre-existing in other files; `IncompletePatientDetail.jsx` must not appear.

- [ ] **Step 4: Build**

Run: `cd frontend && npm run build`
Expected: builds.

- [ ] **Step 5: Verify no modality name survives in the component**

```bash
grep -nE "'t1'|'t1c'|'t2'|'t2fl'|T1c|FLAIR" frontend/src/components/IncompletePatientDetail.jsx \
  && echo "ОСТАЛИСЬ зашитые модальности" || echo "чисто"
```

Expected: `чисто`. This is the check that makes the behaviour identical across
lesion types rather than identical-looking.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/services/api.js \
        frontend/src/components/IncompletePatientDetail.jsx
git commit -m "feat(review): show what was selected, and let the doctor change it

The dialog showed only what was missing, so a modality the detector picked
wrongly could not be argued with — only an empty slot was actionable. Both
lists are now on screen and a series moves between them, with unrecognized
ones under their own heading rather than looking unusable.

Nothing reaches the disk until Сохранить, so a change can be reconsidered.
The component no longer contains a single modality name: it renders the set
the API gives it, which is what makes multiple sclerosis behave like
glioblastoma instead of being offered a t1c it has no concept of.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end verification

**Files:** none — this task changes no code.

- [ ] **Step 1: Run everything**

```bash
cd /home/ubuntu/mri_ai_service && source venv/bin/activate
python -m pytest backend/ -q
FILES=$(ls test_*.py | grep -v '^test_stage05_fixes.py$' | tr '\n' ' ')
python -m pytest $FILES -q
cd frontend && npm run lint
```

Expected: backend green; the root suite shows the same pre-existing failures as
`main` (10 failed, 1 error — `test_stage01_stage03_fixes`, `test_stage04_fixes`,
`test_real_config`) and no others; lint unchanged at 11 errors / 7 warnings.

- [ ] **Step 2: Rebuild the image**

`backend/` and `frontend/` are baked into the image, not mounted.

```bash
docker compose --profile full up --build -d
```

- [ ] **Step 3: Check a multiple-sclerosis run**

Open a review dialog on an MS run.

Expected: three slots — `t1`, `t2`, `t2fl`. **No `t1c` anywhere**, including the
dropdowns on unselected series.

- [ ] **Step 4: Reassign, save, reprocess**

On an already-processed patient: clear one modality, assign a different series
to it — pick an **unrecognized** one, to check that path — and save.

Expected: the dialog reports that the patient will be reprocessed. If that
patient is already in Kappa, a second warning says the Kappa copy stays as it
is.

Then requeue the run and watch:

```bash
watch -n 5 'ls /путь/к/output/preprocessed'
```

Expected: that session's folder disappears when the run starts and is rebuilt;
no other patient's folder is touched at any point.

- [ ] **Step 5: Report**

Report what you saw at each step, including anything that did not match. Do not
mark this task complete on a partial verification.
