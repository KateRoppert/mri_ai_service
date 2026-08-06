# Piece B: Per-Session Completeness Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop sessions missing required modalities from reaching stage_03+ by physically routing them into a side directory during stage 01, while capturing enough data (unrecognized series, completeness status) for a future doctor-facing UI to review and fix them.

**Architecture:** All changes live in `scripts/01_reorganize_folders.py`. Per-session completeness is already knowable right after `SeriesDeduplicator.deduplicate_session()` (before any file is copied), so the gate is a routing decision made at that point, not a new validation pass. Unrecognized series (currently only debug-logged, never retained) get collected the same way recognized series are, riding through the existing date-grouping machinery, and are persisted into `dataset_mapping.json` per session alongside a new `status` field. Physical separation uses a `_incomplete/` subdirectory nested inside the existing output root, invisible to every downstream stage's `glob("sub-*")` scan without touching stages 03-08 at all.

**Tech Stack:** Python 3.12, pytest, existing `SeriesInfo`/`SessionInfo` dataclasses.

## Global Constraints

- Gating is per-session, not per-patient (see design spec `docs/superpowers/specs/2026-08-06-incomplete-patient-workflow-design.md`).
- Default behavior changes (gate is ON by default); `--include-incomplete` restores the old behavior (everything in the main tree, unfiltered).
- No DB/API/frontend work in this plan — out of scope, deferred to piece C's plan.
- No changes to scripts/03-08 — the whole point of the design is that `_incomplete/` is structurally invisible to their `glob("sub-*")` scans.
- Follow existing code conventions in `scripts/01_reorganize_folders.py` exactly (dataclasses, per-process component construction in `process_single_patient`, `action='store_true'` for boolean CLI flags).

---

### Task 1: `SessionInfo` gains `unrecognized_series`; grouping and dedup carry it through

**Files:**
- Modify: `scripts/01_reorganize_folders.py:102-106` (`SessionInfo` dataclass)
- Modify: `scripts/01_reorganize_folders.py:802-836` (`SessionGrouper.group_by_date`)
- Modify: `scripts/01_reorganize_folders.py:924-949` (`SeriesDeduplicator.deduplicate_session`)
- Test: `tests/stage01/test_session_grouper_unrecognized.py` (new)

**Interfaces:**
- Consumes: existing `SeriesInfo` dataclass (`scripts/01_reorganize_folders.py:88-99`) — a `SeriesInfo` with `modality=None` now represents "found but not recognized".
- Produces: `SessionInfo.unrecognized_series: List[SeriesInfo]` — consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/stage01/test_session_grouper_unrecognized.py
import logging
import sys
import importlib.util
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_unrecognized")
SeriesInfo = reorganize_mod.SeriesInfo
SessionGrouper = reorganize_mod.SessionGrouper
SeriesDeduplicator = reorganize_mod.SeriesDeduplicator

LOGGER = logging.getLogger("test_unrecognized")


def _series(desc, modality, date="20230101"):
    return SeriesInfo(
        original_path=Path(f"/fake/{desc}"),
        patient_id="P1", date=date,
        modality=modality, series_description=desc,
    )


class TestGroupByDateUnrecognized:
    def test_unrecognized_series_routed_to_session_unrecognized_list(self):
        recognized = _series("CE_T1-TFE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([recognized, unrecognized])
        assert len(sessions) == 1
        assert "t1c" in sessions[0].series
        assert len(sessions[0].unrecognized_series) == 1
        assert sessions[0].unrecognized_series[0].series_description == "some_weird_protocol_47"

    def test_unrecognized_series_grouped_by_own_date(self):
        unrecognized_day1 = _series("weird_a", None, date="20230101")
        unrecognized_day2 = _series("weird_b", None, date="20230115")
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([unrecognized_day1, unrecognized_day2])
        assert len(sessions) == 2
        by_date = {s.date: s for s in sessions}
        assert by_date["20230101"].unrecognized_series[0].series_description == "weird_a"
        assert by_date["20230115"].unrecognized_series[0].series_description == "weird_b"


class TestDeduplicateSessionCarriesUnrecognized:
    def test_deduplicate_session_preserves_unrecognized_series(self):
        recognized = _series("CE_T1-TFE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([recognized, unrecognized])[0]

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert len(result.unrecognized_series) == 1
        assert result.unrecognized_series[0].series_description == "some_weird_protocol_47"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/stage01/test_session_grouper_unrecognized.py -v`
Expected: FAIL — `AttributeError: 'SessionInfo' object has no attribute 'unrecognized_series'` (first test), collection may fail entirely since the attribute doesn't exist yet.

- [ ] **Step 3: Add the field and wire it through**

In `scripts/01_reorganize_folders.py`, `SessionInfo` (currently lines 102-106):

```python
@dataclass
class SessionInfo:
    """Information about a session (grouped by date)."""
    date: str  # YYYYMMDD format
    series: Dict[str, SeriesInfo] = field(default_factory=dict)  # modality -> SeriesInfo
    unrecognized_series: List[SeriesInfo] = field(default_factory=list)  # modality=None, found but not classified
```

In `SessionGrouper.group_by_date` (currently lines 802-836), the loop body (lines 826-834) currently reads:

```python
            for series in date_groups[date]:
                if series.modality:
                    if series.modality not in session.series:
                        session.series[series.modality] = []

                    if not isinstance(session.series[series.modality], list):
                        session.series[series.modality] = [session.series[series.modality]]

                    session.series[series.modality].append(series)
```

Change to:

```python
            for series in date_groups[date]:
                if series.modality:
                    if series.modality not in session.series:
                        session.series[series.modality] = []

                    if not isinstance(session.series[series.modality], list):
                        session.series[series.modality] = [session.series[series.modality]]

                    session.series[series.modality].append(series)
                else:
                    session.unrecognized_series.append(series)
```

In `SeriesDeduplicator.deduplicate_session` (currently lines 924-949), it builds a fresh `SessionInfo` (line 934: `deduplicated = SessionInfo(date=session.date)`) and never copies `unrecognized_series`. Add right after that line:

```python
        deduplicated = SessionInfo(date=session.date)
        deduplicated.unrecognized_series = session.unrecognized_series
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/stage01/test_session_grouper_unrecognized.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Run full stage01 suite to check no regression**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: all existing tests still pass (baseline was 107 passed before this task).

- [ ] **Step 6: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_session_grouper_unrecognized.py
git commit -m "feat(stage01): carry unrecognized (modality=None) series through session grouping"
```

---

### Task 2: Capture unrecognized series + per-session `status` in `_process_one_patient_core`

**Files:**
- Modify: `scripts/01_reorganize_folders.py:1416-1574` (`_process_one_patient_core`)
- Modify: `scripts/01_reorganize_folders.py:1577-1683` (`process_single_patient`)
- Modify: `scripts/01_reorganize_folders.py:1799` area (`run_sequential`) — pass `completeness_checker` into the `_process_one_patient_core` call
- Modify: `scripts/01_reorganize_folders.py:1955` area (`run_parallel`) — pass `lesion_type` into `process_func` partial (already does; no change needed there since `process_single_patient` builds its own checker)
- Test: `tests/stage01/test_unrecognized_series_in_patient_data.py` (new)

**Interfaces:**
- Consumes: `Task 1`'s `SessionInfo.unrecognized_series`; existing `CompletenessChecker.check_session(session) -> Tuple[bool, Set[str]]` (`scripts/01_reorganize_folders.py:1009-1020`, already implemented, currently unused for gating — this task is its first real caller).
- Produces: `patient_data['sessions'][ses_id]['status']` (`"complete"` / `"incomplete"`) and `patient_data['sessions'][ses_id]['unrecognized_series']` (list of dicts) — consumed by Task 3 (routing decision) and later by piece C (API/UI).

- [ ] **Step 1: Write the failing test**

```python
# tests/stage01/test_unrecognized_series_in_patient_data.py
import logging
import sys
import importlib.util
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_patientdata")


class TestUnrecognizedSeriesInPatientData:
    def test_incomplete_session_has_status_and_unrecognized_series(
        self, make_dicom_series, tmp_path
    ):
        patient_dir = tmp_path / "PAT001"
        # Recognized: t1, t2, t2fl (missing t1c) — glioblastoma requires all 4
        make_dicom_series(
            patient_dir / "2023-01-01" / "t1_series",
            protocol_name="t1_mprage_sag", series_description="t1_mprage_sag",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "t2_series",
            protocol_name="t2_tse_tra", series_description="t2_tse_tra",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "flair_series",
            protocol_name="t2_flair", series_description="t2_flair",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        # Unrecognized: nonsense protocol name, no modality keyword matches
        make_dicom_series(
            patient_dir / "2023-01-01" / "weird_series",
            protocol_name="xyz_47_unknown", series_description="xyz_47_unknown",
            image_type=['ORIGINAL', 'PRIMARY'],
        )

        logger = logging.getLogger("test_patientdata")
        modality_detector = reorganize_mod.ModalityDetector(logger)
        scanner = reorganize_mod.DatasetScanner(logger)
        grouper = reorganize_mod.SessionGrouper(logger)
        deduplicator = reorganize_mod.SeriesDeduplicator(logger)
        completeness_checker = reorganize_mod.CompletenessChecker(logger, lesion_type="glioblastoma")
        file_organizer = reorganize_mod.FileOrganizer(tmp_path / "output", logger)

        patient_data = reorganize_mod._process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id="sub-001",
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
            lesion_type="glioblastoma",
            completeness_checker=completeness_checker,
        )

        session = patient_data['sessions']['ses-001']
        assert session['status'] == 'incomplete'
        assert len(session['unrecognized_series']) == 1
        assert session['unrecognized_series'][0]['series_description'] == "xyz_47_unknown | xyz_47_unknown"
        assert 'original_path' in session['unrecognized_series'][0]
        assert 'slice_count' in session['unrecognized_series'][0]

    def test_complete_session_has_status_complete(self, make_dicom_series, tmp_path):
        patient_dir = tmp_path / "PAT002"
        for mod_kw, name in [
            ("t1_mprage_sag", "t1"), ("t1_mprage_sag_KM", "t1c"),
            ("t2_tse_tra", "t2"), ("t2_flair", "t2fl"),
        ]:
            make_dicom_series(
                patient_dir / "2023-01-01" / name,
                protocol_name=mod_kw, series_description=mod_kw,
                image_type=['ORIGINAL', 'PRIMARY'],
            )

        logger = logging.getLogger("test_patientdata_complete")
        modality_detector = reorganize_mod.ModalityDetector(logger)
        scanner = reorganize_mod.DatasetScanner(logger)
        grouper = reorganize_mod.SessionGrouper(logger)
        deduplicator = reorganize_mod.SeriesDeduplicator(logger)
        completeness_checker = reorganize_mod.CompletenessChecker(logger, lesion_type="glioblastoma")
        file_organizer = reorganize_mod.FileOrganizer(tmp_path / "output2", logger)

        patient_data = reorganize_mod._process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id="sub-002",
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
            lesion_type="glioblastoma",
            completeness_checker=completeness_checker,
        )

        session = patient_data['sessions']['ses-001']
        assert session['status'] == 'complete'
        assert session['unrecognized_series'] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/stage01/test_unrecognized_series_in_patient_data.py -v`
Expected: FAIL — `TypeError: _process_one_patient_core() got an unexpected keyword argument 'completeness_checker'`

- [ ] **Step 3: Implement**

In `_process_one_patient_core` (currently lines 1416-1426), add the parameter:

```python
def _process_one_patient_core(
    patient_dir: Path,
    new_patient_id: str,
    modality_detector: 'ModalityDetector',
    scanner: 'DatasetScanner',
    grouper: 'SessionGrouper',
    deduplicator: 'SeriesDeduplicator',
    file_organizer: 'FileOrganizer',
    logger: logging.Logger,
    completeness_checker: 'CompletenessChecker',
    lesion_type: str = 'glioblastoma',
) -> Optional[Dict]:
```

In the per-series loop (currently lines 1480-1502), the final `else` branch currently only logs. Replace:

```python
        else:
            logger.debug(f"  Series {series_dir}: unknown modality (filtered out)")
```

with:

```python
        else:
            unrecognized_info = SeriesInfo(
                original_path=series_dir,
                patient_id=original_patient_id,
                date=date,
                modality=None,
                series_description=series_description,
            )
            unrecognized_info.slice_count = len(find_dicom_files(series_dir))
            series_list.append(unrecognized_info)
            logger.debug(f"  Series {series_dir}: unknown modality (retained as unrecognized)")
```

In the per-session loop (currently lines 1528-1566), right after `session_data: Dict = {...}` is created (line 1532-1535), compute status and unrecognized list before the modality copy loop:

```python
    for session_idx, session in enumerate(sessions, 1):
        new_session_id = f"ses-{session_idx:03d}"
        logger.debug(f"  Session {new_session_id} (date: {session.date})")

        is_complete, missing_modalities = completeness_checker.check_session(session)

        session_data: Dict = {
            'original_date': session.date,
            'status': 'complete' if is_complete else 'incomplete',
            'series': {},
            'unrecognized_series': [
                {
                    'original_path': str(u.original_path),
                    'series_description': u.series_description,
                    'slice_count': u.slice_count,
                }
                for u in session.unrecognized_series
            ],
        }
```

(The existing `for modality, series in session.series.items():` loop right after stays unchanged for this task.)

In `process_single_patient` (currently lines 1577-1654), add `completeness_checker` construction alongside the other per-process components (currently lines 1627-1640):

```python
        scoring_config = load_series_scoring_config()
        modality_detector = ModalityDetector(logger, scoring_config=scoring_config)
        scanner = DatasetScanner(logger)
        grouper = SessionGrouper(logger)
        deduplicator = SeriesDeduplicator(logger, scoring_config=scoring_config)
        completeness_checker = CompletenessChecker(logger, lesion_type=lesion_type)
        _metadata_extractor = None
```

and pass it into the `_process_one_patient_core(...)` call (currently lines 1644-1654):

```python
        patient_data = _process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id=new_patient_id,
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
            completeness_checker=completeness_checker,
            lesion_type=lesion_type,
        )
```

In `run_sequential`, the existing `completeness_checker = CompletenessChecker(logger, lesion_type=lesion_type)` (already present near line 1830) is currently only used later for the end-of-run report. Pass it into the `_process_one_patient_core` call in that same function (the one immediately following `check_patient_exists`):

```python
        patient_data = _process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id=new_patient_id,
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
            completeness_checker=completeness_checker,
            lesion_type=lesion_type,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/stage01/test_unrecognized_series_in_patient_data.py -v`
Expected: PASS (2/2)

- [ ] **Step 5: Run full stage01 suite**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: all pass, no regression.

- [ ] **Step 6: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_unrecognized_series_in_patient_data.py
git commit -m "feat(stage01): persist per-session status and unrecognized series in dataset_mapping.json"
```

---

### Task 3: `--include-incomplete` flag + physical routing into `_incomplete/`

**Files:**
- Modify: `scripts/01_reorganize_folders.py:1069-1107` (`FileOrganizer.create_bids_structure`)
- Modify: `scripts/01_reorganize_folders.py` per-session loop in `_process_one_patient_core` (from Task 2)
- Modify: `scripts/01_reorganize_folders.py:1577-1683` (`process_single_patient`), `run_sequential`, `run_parallel` — thread `include_incomplete: bool` through
- Modify: `scripts/01_reorganize_folders.py` argparse section (~line 2155-2170) — new `--include-incomplete` flag
- Modify: `scripts/01_reorganize_folders.py` ~line 2291-2334 — pass `args.include_incomplete` into `run_sequential`/`run_parallel` calls
- Test: `tests/stage01/test_incomplete_session_routing.py` (new)

**Interfaces:**
- Consumes: `session_data['status']` from Task 2.
- Produces: sessions physically land under `output_dir/_incomplete/sub-XXX/ses-YYY/anat/<modality>/` instead of `output_dir/sub-XXX/ses-YYY/anat/<modality>/` when incomplete and `include_incomplete=False` (default).

- [ ] **Step 1: Write the failing test**

```python
# tests/stage01/test_incomplete_session_routing.py
import logging
import sys
import importlib.util
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_routing")


def _run_one_patient(tmp_path, make_dicom_series, series_specs, include_incomplete=False):
    patient_dir = tmp_path / "PAT001"
    for name, protocol in series_specs:
        make_dicom_series(
            patient_dir / "2023-01-01" / name,
            protocol_name=protocol, series_description=protocol,
            image_type=['ORIGINAL', 'PRIMARY'],
        )
    output_dir = tmp_path / "output"
    logger = logging.getLogger(f"test_routing_{include_incomplete}")
    modality_detector = reorganize_mod.ModalityDetector(logger)
    scanner = reorganize_mod.DatasetScanner(logger)
    grouper = reorganize_mod.SessionGrouper(logger)
    deduplicator = reorganize_mod.SeriesDeduplicator(logger)
    completeness_checker = reorganize_mod.CompletenessChecker(logger, lesion_type="glioblastoma")
    file_organizer = reorganize_mod.FileOrganizer(output_dir, logger)

    reorganize_mod._process_one_patient_core(
        patient_dir=patient_dir,
        new_patient_id="sub-001",
        modality_detector=modality_detector,
        scanner=scanner,
        grouper=grouper,
        deduplicator=deduplicator,
        file_organizer=file_organizer,
        logger=logger,
        completeness_checker=completeness_checker,
        lesion_type="glioblastoma",
        include_incomplete=include_incomplete,
    )
    return output_dir


class TestIncompleteSessionRouting:
    def test_incomplete_session_goes_to_incomplete_subdir_by_default(self, make_dicom_series, tmp_path):
        output_dir = _run_one_patient(
            tmp_path, make_dicom_series,
            [("t1", "t1_mprage_sag"), ("t2", "t2_tse_tra")],  # missing t1c, t2fl
        )
        assert not (output_dir / "sub-001" / "ses-001").exists()
        assert (output_dir / "_incomplete" / "sub-001" / "ses-001" / "anat" / "t1").exists()

    def test_complete_session_goes_to_main_tree(self, make_dicom_series, tmp_path):
        output_dir = _run_one_patient(
            tmp_path, make_dicom_series,
            [
                ("t1", "t1_mprage_sag"), ("t1c", "t1_mprage_sag_KM"),
                ("t2", "t2_tse_tra"), ("t2fl", "t2_flair"),
            ],
        )
        assert (output_dir / "sub-001" / "ses-001" / "anat" / "t1").exists()
        assert not (output_dir / "_incomplete").exists()

    def test_include_incomplete_flag_forces_main_tree(self, make_dicom_series, tmp_path):
        output_dir = _run_one_patient(
            tmp_path, make_dicom_series,
            [("t1", "t1_mprage_sag"), ("t2", "t2_tse_tra")],  # missing t1c, t2fl
            include_incomplete=True,
        )
        assert (output_dir / "sub-001" / "ses-001" / "anat" / "t1").exists()
        assert not (output_dir / "_incomplete").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/stage01/test_incomplete_session_routing.py -v`
Expected: FAIL — `TypeError: _process_one_patient_core() got an unexpected keyword argument 'include_incomplete'`

- [ ] **Step 3: Implement**

`FileOrganizer.create_bids_structure` (currently lines 1085-1107) gains an `incomplete: bool = False` parameter:

```python
    def create_bids_structure(
        self,
        patient_id: str,
        session_id: str,
        modality: str,
        incomplete: bool = False
    ) -> Path:
        """
        Create BIDS directory structure.

        Structure: output/sub-XXX/ses-XXX/anat/MODALITY/
        Structure (incomplete session): output/_incomplete/sub-XXX/ses-XXX/anat/MODALITY/

        Returns:
            Path to modality directory
        """
        root = (self.output_dir / '_incomplete') if incomplete else self.output_dir
        modality_dir = (
            root /
            patient_id /
            session_id /
            'anat' /
            modality
        )

        if not self.dry_run:
            modality_dir.mkdir(parents=True, exist_ok=True)
        return modality_dir
```

`_process_one_patient_core` gains `include_incomplete: bool = False` as a trailing parameter, appended after Task 2's `completeness_checker`/`lesion_type`:

```python
def _process_one_patient_core(
    patient_dir: Path,
    new_patient_id: str,
    modality_detector: 'ModalityDetector',
    scanner: 'DatasetScanner',
    grouper: 'SessionGrouper',
    deduplicator: 'SeriesDeduplicator',
    file_organizer: 'FileOrganizer',
    logger: logging.Logger,
    completeness_checker: 'CompletenessChecker',
    lesion_type: str = 'glioblastoma',
    include_incomplete: bool = False,
) -> Optional[Dict]:
```

In the per-session loop, after `is_complete, missing_modalities = completeness_checker.check_session(session)`, compute the routing decision and pass it to `create_bids_structure`:

```python
        is_complete, missing_modalities = completeness_checker.check_session(session)
        route_to_incomplete = (not is_complete) and (not include_incomplete)

        session_data: Dict = {
            'original_date': session.date,
            'status': 'complete' if is_complete else 'incomplete',
            'series': {},
            'unrecognized_series': [
                {
                    'original_path': str(u.original_path),
                    'series_description': u.series_description,
                    'slice_count': u.slice_count,
                }
                for u in session.unrecognized_series
            ],
        }

        for modality, series in session.series.items():
            target_dir = file_organizer.create_bids_structure(
                new_patient_id, new_session_id, modality, incomplete=route_to_incomplete
            )
```

(Rest of that inner loop — `copy_series`, `validate_copy`, `session_data['series'][modality] = {...}` — is unchanged.)

Thread `include_incomplete` through `process_single_patient` (add parameter, pass to `_process_one_patient_core` call), `run_sequential` (add parameter, default `False`, pass to `_process_one_patient_core` call), `run_parallel` (add parameter, default `False`, add to the `process_func = partial(process_single_patient, ...)` call).

Add the CLI flag in the argparse section (near the existing `--force`/`--dry_run`, ~line 2166-2170):

```python
    parser.add_argument(
        '--include-incomplete',
        action='store_true',
        help='Disable the completeness gate: write incomplete sessions to the '
             'main tree instead of _incomplete/ (old behavior, process everything)'
    )
```

Pass it through at the `run_sequential`/`run_parallel` call sites (~lines 2291-2334):

```python
        ) = run_sequential(
            ...
            lesion_type=args.lesion_type,
            include_incomplete=args.include_incomplete,
        )
```
```python
        ) = run_parallel(
            ...
            lesion_type=args.lesion_type,
            include_incomplete=args.include_incomplete,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/stage01/test_incomplete_session_routing.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Run full stage01 suite**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: all pass, no regression.

- [ ] **Step 6: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_incomplete_session_routing.py
git commit -m "feat(stage01): route incomplete sessions to _incomplete/, add --include-incomplete override"
```

---

### Task 4: Wire the flag into `pipeline_config.yaml`, verify on real BO data

**Files:**
- Modify: `pipeline_config.yaml` (`stages.stage_01_reorganize.args`)
- No changes needed in `orchestrator.py` — `build_command()` already generically passes any boolean `args` entry as a `--flag-name` (see `scripts/pipeline_manager` args loop: `if arg_value is True: cmd.append(f'--{arg_name}')`), so `include-incomplete: false` simply produces no flag (default gate-on behavior), and `true` produces `--include-incomplete`.

**Interfaces:**
- Consumes: Task 3's `--include-incomplete` CLI flag.
- Produces: end-to-end config-driven behavior, verified against real data.

- [ ] **Step 1: Add the config key**

In `pipeline_config.yaml`, under `stages.stage_01_reorganize.args`, add:

```yaml
      include-incomplete: false   # true = old behavior, process every session regardless of completeness
```

- [ ] **Step 2: Verify config wiring with a dry run**

Run (adjust path to match how other stage01 CLI checks were done earlier in this session — via `docker exec mri_ai_service-web-1`):

```bash
docker exec mri_ai_service-web-1 python /app/scripts/01_reorganize_folders.py --help | grep -A2 "include-incomplete"
```

Expected: flag listed with the help text from Task 3.

- [ ] **Step 3: Real-data verification on a known-incomplete BO patient**

Using the same pattern as the bare-c fix verification earlier (isolated single-patient run into a scratch output dir):

```bash
docker exec mri_ai_service-web-1 python /app/scripts/01_reorganize_folders.py \
  "/home/ubuntu/mri_ai_service/data/clinical_dicom/BO/BO-181" \
  "/home/ubuntu/mri_ai_service/demo_workspace/scratch_gate_verify/bo181_out" \
  --lesion-type glioblastoma --mode sequential --force
```

Expected: BO-181 (only ever had a single t1c series, confirmed in the earlier BO investigation) lands under `bo181_out/_incomplete/sub-001/ses-001/`, not `bo181_out/sub-001/`. Inspect `dataset_mapping.json` in that output dir — session should show `"status": "incomplete"`.

- [ ] **Step 4: Real-data verification with `--include-incomplete`**

```bash
docker exec mri_ai_service-web-1 python /app/scripts/01_reorganize_folders.py \
  "/home/ubuntu/mri_ai_service/data/clinical_dicom/BO/BO-181" \
  "/home/ubuntu/mri_ai_service/demo_workspace/scratch_gate_verify/bo181_out_full" \
  --lesion-type glioblastoma --mode sequential --force --include-incomplete
```

Expected: BO-181 lands under `bo181_out_full/sub-001/ses-001/` (main tree), matching pre-gate behavior.

- [ ] **Step 5: Clean up scratch verification output**

```bash
rm -rf /home/ubuntu/mri_ai_service/demo_workspace/scratch_gate_verify
```

- [ ] **Step 6: Run the full test suite one more time**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pipeline_config.yaml
git commit -m "chore(pipeline-config): wire include-incomplete flag for stage_01_reorganize"
```
