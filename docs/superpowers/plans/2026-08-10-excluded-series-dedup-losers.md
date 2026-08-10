# Excluded Series: Capture Dedup Losers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `unrecognized_series` → `excluded_series`: doctors must see every series that didn't make it into the final output, not just ones the classifier never recognized — series that lost deduplication (classified, but a better candidate for the same modality won) currently vanish with no trace.

**Architecture:** `SessionInfo.unrecognized_series` (a `List[SeriesInfo]`) is renamed to `excluded_series`. `SeriesDeduplicator.deduplicate_session()` currently computes `best_series` per modality and discards every other candidate outright (`removed_count = len(series_list) - 1` is counted but the actual `SeriesInfo` objects are thrown away) — it now appends the losers to `deduplicated.excluded_series` instead. Since every `SeriesInfo` already carries its own `.modality` field (set correctly at grouping time — everything in one `series_list` bucket shares the same modality key), no new field is needed on the dataclass to know what a dedup-loser "was detected as." The JSON serialization step in `_process_one_patient_core` computes `detected_modality`/`reason` per entry: `reason="unrecognized"` + `detected_modality=null` when `.modality is None`, else `reason="lost_deduplication"` + `detected_modality=<that modality>`.

**Tech Stack:** Python 3.12, pytest, existing `SeriesInfo`/`SessionInfo` dataclasses.

## Global Constraints

- Branch off `main` directly (this patches piece B, already merged) — not the open `feat/incomplete-patient-backend-api` branch, which gets updated separately afterward to consume the new field (see design spec's "Backend API" section update, commit `bb9ff12`).
- No backward-compat reader for old `dataset_mapping.json` files carrying `unrecognized_series` — old runs need `--force` reprocessing (spec's explicit decision, see `docs/superpowers/specs/2026-08-06-incomplete-patient-workflow-design.md`).
- Doctor is not restricted to a series' `detected_modality` when relabeling — that's a piece C-backend concern (separate plan), not touched here. This plan only changes what stage 01 captures and persists.
- Follow existing code conventions in `scripts/01_reorganize_folders.py` exactly.

---

### Task 1: Rename to `excluded_series`; `SeriesDeduplicator` retains dedup losers

**Files:**
- Modify: `scripts/01_reorganize_folders.py:107` (`SessionInfo.excluded_series` field)
- Modify: `scripts/01_reorganize_folders.py:836-837` (`SessionGrouper.group_by_date`'s `else` branch)
- Modify: `scripts/01_reorganize_folders.py:937-954` (`SeriesDeduplicator.deduplicate_session`)
- Modify (rename `unrecognized_series` → `excluded_series` throughout, update assertions): `tests/stage01/test_session_grouper_unrecognized.py`
- Test: add new dedup-loser-retention test cases to the same file

**Interfaces:**
- Produces: `SessionInfo.excluded_series: List[SeriesInfo]` now contains BOTH modality=None entries (as before, just renamed) AND dedup-losing `SeriesInfo` objects (each still carrying its correct `.modality`) — consumed by Task 2.

- [ ] **Step 1: Update the existing test file for the rename + add dedup-loser coverage**

Rewrite `tests/stage01/test_session_grouper_unrecognized.py` in full:

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_excluded")
SeriesInfo = reorganize_mod.SeriesInfo
SessionGrouper = reorganize_mod.SessionGrouper
SeriesDeduplicator = reorganize_mod.SeriesDeduplicator

LOGGER = logging.getLogger("test_excluded")


def _series(desc, modality, date="20230101"):
    return SeriesInfo(
        original_path=Path(f"/fake/{desc}"),
        patient_id="P1", date=date,
        modality=modality, series_description=desc,
    )


class TestGroupByDateExcluded:
    def test_unrecognized_series_routed_to_session_excluded_list(self):
        recognized = _series("CE_T1-TFE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([recognized, unrecognized])
        assert len(sessions) == 1
        assert "t1c" in sessions[0].series
        assert len(sessions[0].excluded_series) == 1
        assert sessions[0].excluded_series[0].series_description == "some_weird_protocol_47"

    def test_unrecognized_series_grouped_by_own_date(self):
        unrecognized_day1 = _series("weird_a", None, date="20230101")
        unrecognized_day2 = _series("weird_b", None, date="20230115")
        grouper = SessionGrouper(LOGGER)
        sessions = grouper.group_by_date([unrecognized_day1, unrecognized_day2])
        assert len(sessions) == 2
        by_date = {s.date: s for s in sessions}
        assert by_date["20230101"].excluded_series[0].series_description == "weird_a"
        assert by_date["20230115"].excluded_series[0].series_description == "weird_b"


class TestDeduplicateSessionRetainsLosers:
    def test_dedup_losers_added_to_excluded_series(self):
        winner = _series("CE_T1-TFE (3D brain)", "t1c")
        loser = _series("CE_T1-TSE (3D brain)", "t1c")
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([winner, loser])[0]
        assert isinstance(session.series["t1c"], list) and len(session.series["t1c"]) == 2

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert result.series["t1c"] in (winner, loser)  # exactly one is kept
        assert len(result.excluded_series) == 1
        loser_kept = result.excluded_series[0]
        assert loser_kept is not result.series["t1c"]
        assert loser_kept.modality == "t1c"  # detected_modality is just .modality — no new field needed

    def test_deduplicate_session_preserves_prior_excluded_series(self):
        """Modality=None entries from group_by_date must survive dedup untouched,
        alongside any new dedup-loser entries added in the same pass."""
        winner = _series("CE_T1-TFE (3D brain)", "t1c")
        loser = _series("CE_T1-TSE (3D brain)", "t1c")
        unrecognized = _series("some_weird_protocol_47", None)
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([winner, loser, unrecognized])[0]

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert len(result.excluded_series) == 2
        descriptions = {s.series_description for s in result.excluded_series}
        assert "some_weird_protocol_47" in descriptions
        assert loser.series_description in descriptions

    def test_no_duplicates_means_no_excluded_series_added(self):
        only = _series("CE_T1-TFE (3D brain)", "t1c")
        grouper = SessionGrouper(LOGGER)
        session = grouper.group_by_date([only])[0]

        dedup = SeriesDeduplicator(LOGGER)
        result = dedup.deduplicate_session(session)

        assert result.excluded_series == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/stage01/test_session_grouper_unrecognized.py -v`
Expected: FAIL — `AttributeError: 'SessionInfo' object has no attribute 'excluded_series'` (the field is still named `unrecognized_series` at this point).

- [ ] **Step 3: Rename the field and wire dedup-loser retention**

In `scripts/01_reorganize_folders.py`, `SessionInfo` (currently line 107):

```python
    excluded_series: List[SeriesInfo] = field(default_factory=list)  # modality=None (unrecognized) or lost deduplication
```

`SessionGrouper.group_by_date`'s `else` branch (currently lines 836-837):

```python
                else:
                    session.excluded_series.append(series)
```

`SeriesDeduplicator.deduplicate_session` (currently lines 927-954) — rename the carry-over line and append losers:

```python
    def deduplicate_session(self, session: SessionInfo) -> SessionInfo:
        """
        Keep only one series per modality (one with most slices/best score).
        Every other candidate for that modality is retained in
        excluded_series (detected_modality = its own .modality field, reason
        computed later at JSON-serialization time in _process_one_patient_core)
        rather than discarded — a doctor may want to pick the loser instead.

        Args:
            session: Session with potentially duplicate modalities

        Returns:
            Session with single series per modality; excluded_series carries
            forward session.excluded_series plus any new dedup losers.
        """
        deduplicated = SessionInfo(date=session.date)
        deduplicated.excluded_series = list(session.excluded_series)

        for modality, series_list in session.series.items():
            if isinstance(series_list, list):
                if len(series_list) > 1:
                    # Multiple series for same modality — _select_best_series logs the INFO line
                    best_series = self._select_best_series(series_list, modality)
                    deduplicated.series[modality] = best_series

                    losers = [s for s in series_list if s is not best_series]
                    deduplicated.excluded_series.extend(losers)

                    removed_count = len(series_list) - 1
                    self.duplicates_removed += removed_count
                else:
                    deduplicated.series[modality] = series_list[0]
            else:
                deduplicated.series[modality] = series_list

        return deduplicated
```

(`list(session.excluded_series)` — a copy, not the same list object, so mutating `deduplicated.excluded_series` afterward can never leak back into `session.excluded_series`. This also resolves the Minor finding recorded in the piece-B ledger about the old carry-over line assigning by reference.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/stage01/test_session_grouper_unrecognized.py -v`
Expected: PASS (7/7)

- [ ] **Step 5: Run full stage01 suite — expect exactly 2 failures, both from `_process_one_patient_core` itself, not yet touched by this task**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: `tests/stage01/test_unrecognized_series_in_patient_data.py` and `tests/stage01/test_incomplete_session_routing.py` now FAIL with `AttributeError: 'SessionInfo' object has no attribute 'unrecognized_series'`. Both call `_process_one_patient_core` (confirmed: `grep -n "_process_one_patient_core" tests/stage01/test_incomplete_session_routing.py` matches), whose JSON-building code (line ~1563, `for u in session.unrecognized_series`) still references the field's old name — Task 2 fixes that line. Every other test still passes. This is expected at this checkpoint, not a regression to chase down in this task; do not attempt to fix those two files here (`test_incomplete_session_routing.py` itself needs no content change at all — see Task 2 Step 1).

- [ ] **Step 6: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_session_grouper_unrecognized.py
git commit -m "feat(stage01): SessionInfo.excluded_series — dedup losers now retained, not discarded"
```

---

### Task 2: Wire `excluded_series` into `dataset_mapping.json` with `detected_modality`/`reason`

**Files:**
- Modify: `scripts/01_reorganize_folders.py:1546-1565` (`_process_one_patient_core`'s per-session JSON block)
- Modify (rename + add `detected_modality`/`reason` assertions): `tests/stage01/test_unrecognized_series_in_patient_data.py` → rename file to `tests/stage01/test_excluded_series_in_patient_data.py`
- No content change (confirmed): `tests/stage01/test_incomplete_session_routing.py` — it will pass again once Task 2 fixes `_process_one_patient_core`, with zero edits of its own
- Test: add a dedup-loser-in-JSON-output test case

**Interfaces:**
- Consumes: Task 1's `SessionInfo.excluded_series: List[SeriesInfo]`.
- Produces: `dataset_mapping.json`'s `session_data['excluded_series']` — list of `{original_path, series_description, slice_count, detected_modality, reason}` — this is the exact shape the (separately updated) `feat/incomplete-patient-backend-api` branch will consume.

- [ ] **Step 1: Confirm no change needed in the routing test file**

`tests/stage01/test_incomplete_session_routing.py` does NOT reference `unrecognized_series` anywhere (confirmed via `grep -n "unrecognized" tests/stage01/test_incomplete_session_routing.py` — zero matches; that file only asserts directory placement, not series-list content). No edit needed there. Skip straight to Step 2.

- [ ] **Step 2: Rewrite the patient-data test file for the rename + new fields**

`git mv tests/stage01/test_unrecognized_series_in_patient_data.py tests/stage01/test_excluded_series_in_patient_data.py`, then rewrite its contents:

```python
# tests/stage01/test_excluded_series_in_patient_data.py
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_excluded_patientdata")


class TestExcludedSeriesInPatientData:
    def test_incomplete_session_has_status_and_unrecognized_excluded_series(
        self, make_dicom_series, tmp_path
    ):
        patient_dir = tmp_path / "PAT001"
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
        make_dicom_series(
            patient_dir / "2023-01-01" / "weird_series",
            protocol_name="xyz_47_unknown", series_description="xyz_47_unknown",
            image_type=['ORIGINAL', 'PRIMARY'],
        )

        logger = logging.getLogger("test_excluded_patientdata")
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
        assert len(session['excluded_series']) == 1
        entry = session['excluded_series'][0]
        assert entry['series_description'] == "xyz_47_unknown | xyz_47_unknown"
        assert entry['detected_modality'] is None
        assert entry['reason'] == 'unrecognized'
        assert 'original_path' in entry
        assert 'slice_count' in entry

    def test_dedup_loser_appears_in_excluded_series_with_modality_and_reason(
        self, make_dicom_series, tmp_path
    ):
        patient_dir = tmp_path / "PAT002"
        # Two plain T1 candidates for the same session -> one wins dedup, one is retained as excluded
        make_dicom_series(
            patient_dir / "2023-01-01" / "t1_a",
            protocol_name="t1_mprage_sag", series_description="t1_mprage_sag",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "t1_b",
            protocol_name="t1_mprage_sag_repeat", series_description="t1_mprage_sag_repeat",
            image_type=['ORIGINAL', 'PRIMARY'],
        )

        logger = logging.getLogger("test_excluded_dedup")
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
        assert len(session['excluded_series']) == 1
        entry = session['excluded_series'][0]
        assert entry['detected_modality'] == 't1'
        assert entry['reason'] == 'lost_deduplication'
        # the winning t1 is still the one actually present in series
        assert 't1' in session['series']
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/stage01/test_excluded_series_in_patient_data.py -v`
Expected: FAIL — `KeyError: 'excluded_series'` (the JSON block still writes `unrecognized_series` and has no `detected_modality`/`reason` keys).

- [ ] **Step 4: Update `_process_one_patient_core`'s JSON block**

Replace the `session_data` construction (currently lines 1553-1565):

```python
        session_data: Dict = {
            'original_date': session.date,
            'status': 'complete' if is_complete else 'incomplete',
            'series': {},
            'excluded_series': [
                {
                    'original_path': str(u.original_path),
                    'series_description': u.series_description,
                    'slice_count': u.slice_count,
                    'detected_modality': u.modality,
                    'reason': 'unrecognized' if u.modality is None else 'lost_deduplication',
                }
                for u in session.excluded_series
            ],
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/stage01/test_excluded_series_in_patient_data.py -v`
Expected: PASS (2/2)

- [ ] **Step 6: Run the full stage01 suite**

Run: `python3 -m pytest tests/stage01/ -q`
Expected: all pass, no regressions (baseline before this plan was 111; this plan adds tests in both tasks, expect 111 + new counts from Task 1 Step 1 and Task 2 Step 2, no failures).

- [ ] **Step 7: Real-data verification on the exact case that triggered this plan**

```bash
source venv/bin/activate
rm -rf /tmp/claude-1000/-home-ubuntu-mri-ai-service/1b1b558d-6c44-45b8-9a9e-f919ec038ba6/scratchpad/excluded_series_verify
PYTHONPATH=. python3 scripts/01_reorganize_folders.py \
  "data/clinical_dicom/BO/test/BO-68" \
  "/tmp/claude-1000/-home-ubuntu-mri-ai-service/1b1b558d-6c44-45b8-9a9e-f919ec038ba6/scratchpad/excluded_series_verify/bids_organized" \
  --lesion-type glioblastoma --mode sequential --force
python3 -c "
import json
mapping = json.load(open('/tmp/claude-1000/-home-ubuntu-mri-ai-service/1b1b558d-6c44-45b8-9a9e-f919ec038ba6/scratchpad/excluded_series_verify/bids_organized/dataset_mapping.json'))
for pid, pdata in mapping['patients'].items():
    for sid, sdata in pdata['sessions'].items():
        print(pid, sdata['status'], 'excluded_series:', sdata['excluded_series'])
"
rm -rf /tmp/claude-1000/-home-ubuntu-mri-ai-service/1b1b558d-6c44-45b8-9a9e-f919ec038ba6/scratchpad/excluded_series_verify
```

Expected: BO-68's session shows `excluded_series` with exactly one entry — the losing `sT1W_3D_TFE` duplicate (`detected_modality: "t1"`, `reason: "lost_deduplication"`) — confirming the exact real-world case that motivated this plan (see design spec's "Обновлено после реального прогона" note) now surfaces correctly instead of vanishing.

- [ ] **Step 8: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_excluded_series_in_patient_data.py
git commit -m "feat(stage01): persist excluded_series (detected_modality + reason) in dataset_mapping.json"
```
