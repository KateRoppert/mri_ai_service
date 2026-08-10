# C-backend v2: consume excluded_series, support replace-with-swap-back — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `backend/pipeline_manager.py`/`backend/models.py` up to date with stage 01's renamed `excluded_series` field (was `unrecognized_series`), and let `relabel_series` REPLACE an already-filled modality slot (not just fill an empty one) — the currently-selected series is bumped back into `excluded_series` instead of being discarded, per the design spec's "Обновлено после реального прогона" note.

**Architecture:** No new endpoints. `get_incomplete_patients` reads `excluded_series` (already the exact shape stage 01 now writes: `original_path`, `series_description`, `slice_count`, `detected_modality`, `reason`) instead of the old `unrecognized_series`. `relabel_series` searches `excluded_series` for the source (was `unrecognized_series`), and — before overwriting `session_data['series'][modality]` — captures whatever was there, appends it back into `excluded_series` tagged `detected_modality: modality`, `reason: "replaced_by_manual_relabel"`. Doctor is not restricted to a series' `detected_modality` when picking a target modality (full freedom, per spec).

**Tech Stack:** Python 3.12, FastAPI, Pydantic, pytest.

## Global Constraints

- No DB changes, no new endpoints — this plan only updates existing `PipelineManager`/`models.py`/`app.py` code already built by the earlier `feat/incomplete-patient-backend-api` plan.
- No backward-compat reader for `unrecognized_series` — `main`'s stage 01 no longer produces that key at all (confirmed: `grep -n "unrecognized_series" scripts/01_reorganize_folders.py` returns zero matches on `main`), so no dataset_mapping.json on a freshly (re)processed run will ever have it.
- Doctor is never restricted to a series' `detected_modality` — any excluded series can be assigned to any of the 4 modalities (already true of the existing `modality` parameter validation; this plan doesn't add a restriction).
- Follow existing code conventions in `backend/pipeline_manager.py`/`backend/models.py` exactly.

---

### Task 1: Rename to `excluded_series` in models + `get_incomplete_patients`

**Files:**
- Modify: `backend/models.py:131-148` (`UnrecognizedSeriesInfo` → `ExcludedSeriesInfo`, `IncompletePatientSession.unrecognized_series` → `excluded_series`, `RelabelSeriesRequest` docstring)
- Modify: `backend/pipeline_manager.py:578` (`get_incomplete_patients`'s dict key)
- Modify: `backend/test_incomplete_patients_api.py` (`_write_mapping`'s test fixtures + `TestGetIncompletePatients` assertions)

**Interfaces:**
- Produces: `IncompletePatientsResponse.sessions[].excluded_series: List[ExcludedSeriesInfo]`, each with `original_path`, `series_description`, `slice_count`, `detected_modality: Optional[str]`, `reason: str` — consumed by the (separate, future) frontend plan.

- [ ] **Step 1: Update the failing test's fixtures and assertions**

In `backend/test_incomplete_patients_api.py`, `TestGetIncompletePatients::test_returns_only_incomplete_non_discarded_sessions` currently builds `"unrecognized_series": [{"original_path": "/raw/weird", "series_description": "xyz", "slice_count": 20}]` for `sub-001`'s `ses-002`. Change the key name and add the two new fields:

```python
                        "excluded_series": [
                            {
                                "original_path": "/raw/weird", "series_description": "xyz",
                                "slice_count": 20, "detected_modality": None, "reason": "unrecognized",
                            }
                        ],
```

Then update the assertion at the end of that test:

```python
        assert len(result[0]["excluded_series"]) == 1
        assert result[0]["excluded_series"][0]["detected_modality"] is None
        assert result[0]["excluded_series"][0]["reason"] == "unrecognized"
```

(Every other `"unrecognized_series": []` occurrence in this file's other fixtures — e.g. `sub-001`'s `ses-001`, `sub-002`'s `ses-001` — rename the key to `"excluded_series": []`; run `grep -n "unrecognized_series" backend/test_incomplete_patients_api.py` first to find every occurrence before editing, there are more than the ones shown above.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k TestGetIncompletePatients`
Expected: FAIL — `KeyError: 'excluded_series'` (production code still reads/writes `unrecognized_series`).

- [ ] **Step 3: Rename in `models.py`**

Replace (currently lines 131-135):

```python
class ExcludedSeriesInfo(BaseModel):
    """Серия, не попавшая в финальный набор — не распознана алгоритмом,
    или распознана, но проиграла дедупликацию другому кандидату"""
    original_path: str = Field(..., description="Путь к исходной DICOM-серии")
    series_description: str = Field(..., description="ProtocolName | SeriesDescription")
    slice_count: int = Field(..., description="Число DICOM-файлов в серии")
    detected_modality: Optional[str] = Field(None, description="Модальность по мнению алгоритма (null — вообще не распознана)")
    reason: str = Field(..., description="unrecognized | lost_deduplication | replaced_by_manual_relabel")
```

Replace (currently lines 138-148, `IncompletePatientSession`):

```python
    excluded_series: List[ExcludedSeriesInfo] = Field(
        default_factory=list, description="Серии вне финального набора — кандидаты на ручную переразметку"
    )
```

(Only the field name and its type/description change — every other field on `IncompletePatientSession` stays as-is.)

In `RelabelSeriesRequest` (currently line 159), update the docstring reference only:

```python
    original_path: str = Field(..., description="Путь к исходной DICOM-серии (из excluded_series)")
```

Confirm `Optional` is already imported in `models.py` (it's used elsewhere in the file for other fields — check `grep -n "^from typing import" backend/models.py` before assuming).

- [ ] **Step 4: Rename in `pipeline_manager.py`**

Line 578, inside `get_incomplete_patients`:

```python
                    "excluded_series": session_data.get('excluded_series', []),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k TestGetIncompletePatients`
Expected: PASS (2/2)

- [ ] **Step 6: Run full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: some tests in `TestRelabelSeries*`/`TestDiscardSession*` will now fail (they still write/read `unrecognized_series` in their own fixtures — Task 2 fixes those) — confirm the failures are confined to tests that reference `unrecognized_series` (`grep -n "unrecognized_series" backend/test_incomplete_patients_api.py` after this task's edits should show zero remaining occurrences in `TestGetIncompletePatients`, but still show occurrences in the relabel/discard test classes — those are Task 2's scope, not a regression here).

- [ ] **Step 7: Commit**

```bash
git add backend/models.py backend/pipeline_manager.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): consume excluded_series (was unrecognized_series) in list endpoint"
```

---

### Task 2: `relabel_series` reads `excluded_series`, supports replace-with-swap-back

**Files:**
- Modify: `backend/pipeline_manager.py:604-712` (`relabel_series` — full rewrite of the body from the `unrecognized`/`matches` lookup through the `session_data['series'][modality]` assignment)
- Modify: `backend/test_incomplete_patients_api.py` (`TestRelabelSeries*`/`TestRelabelSeriesInputValidation`/`TestDiscardSession*` fixtures — rename `unrecognized_series` → `excluded_series` in every `_write_mapping` call across the file)
- Test: add a new replace-with-swap-back test case

**Interfaces:**
- Consumes: Task 1's renamed field.
- Produces: `relabel_series`'s behavior when `session_data['series'][modality]` is already occupied — the prior occupant is preserved (moved into `excluded_series`), not discarded.

- [ ] **Step 1: Rename remaining `unrecognized_series` references across the whole test file, then add the replace test**

Run `grep -n "unrecognized_series" backend/test_incomplete_patients_api.py` and rename every remaining occurrence (in `TestRelabelSeries`, `TestRelabelSeriesInputValidation`, `TestDiscardSession*`, and the copy-failure regression test added in the C-backend plan's post-review fix) to `excluded_series`, adding `"detected_modality": None, "reason": "unrecognized"` to each fixture entry that represents a never-classified series (matching Task 1 Step 1's pattern) — these tests only care that the source lookup finds the entry by `original_path`, so the exact `detected_modality`/`reason` values on PRE-EXISTING fixtures don't need to vary, just be present and internally consistent (`None`/`"unrecognized"` is fine for all of them unless a test specifically needs a `lost_deduplication` fixture, per Step 2 below).

Then append a new test class:

```python
class TestRelabelSeriesReplace:
    def _make_dicom_series(self, series_dir: Path, n_files=2):
        series_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake dicom bytes")
        return series_dir

    def test_replacing_an_already_filled_modality_bumps_old_series_into_excluded(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-001" / "ses-001"
        new_series = self._make_dicom_series(tmp_path / "raw" / "better_t1")

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {
                            "t1": {
                                "original_path": "/raw/old_t1", "slice_count": 150,
                                "series_description": "old_t1_series",
                            },
                            "t2": {}, "t2fl": {},
                        },
                        "excluded_series": [
                            {
                                "original_path": str(new_series), "series_description": "better_t1_series",
                                "slice_count": 2, "detected_modality": "t1", "reason": "lost_deduplication",
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
            original_path=str(new_series),
            modality="t1",
        )

        assert result["status"] == "complete"  # t1/t2/t2fl all present now

        mapping = json.loads((bids_dir / "dataset_mapping.json").read_text())
        session = mapping["patients"]["sub-001"]["sessions"]["ses-001"]

        # new series is now the active t1
        assert session["series"]["t1"]["original_path"] == str(new_series)

        # old t1 was NOT discarded — it's back in excluded_series
        assert len(session["excluded_series"]) == 1
        bumped = session["excluded_series"][0]
        assert bumped["original_path"] == "/raw/old_t1"
        assert bumped["detected_modality"] == "t1"
        assert bumped["reason"] == "replaced_by_manual_relabel"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k TestRelabelSeriesReplace`
Expected: FAIL — either a `KeyError`/`ValueError` from the old `unrecognized_series`-based lookup not finding the source (since the fixture now writes `excluded_series`), or (once Step 1's renames are in place) the old series simply being overwritten with no trace in `excluded_series` — either way, `assert len(session["excluded_series"]) == 1` fails against the pre-fix code's behavior (it would be `0`, since nothing bumps the old occupant back).

- [ ] **Step 3: Rewrite `relabel_series`'s body**

Replace lines 645-691 (from `unrecognized = session_data.get('unrecognized_series', [])` through the `session_data['series'][modality] = {...}` assignment) with:

```python
        excluded = session_data.get('excluded_series', [])
        matches = [u for u in excluded if u['original_path'] == original_path]
        if not matches:
            raise ValueError(
                f"No excluded series with original_path={original_path!r} "
                f"in {patient_id}/{session_id}"
            )
        series_entry = matches[0]

        bids_dir = Path(output_path) / "bids_organized"
        was_incomplete = session_data.get('status') == 'incomplete'
        current_root = (bids_dir / "_incomplete") if was_incomplete else bids_dir
        target_dir = current_root / patient_id / session_id / "anat" / modality
        target_dir.mkdir(parents=True, exist_ok=True)

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
            # Partial/failed copy — do NOT touch session_data, recompute status,
            # move the session directory, or write the mapping file back. Silently
            # accepting this would let a session with fewer files than expected
            # flow downstream to segmentation as "complete" (the exact failure
            # mode this incomplete-patient feature exists to prevent).
            raise ValueError(
                f"Copy failed: only {copied}/{len(source_files)} files copied for "
                f"{patient_id}/{session_id}/{modality} (source: {original_path})"
            )

        # Remove the newly-chosen series from the excluded pool.
        remaining_excluded = [u for u in excluded if u['original_path'] != original_path]

        # If modality was already occupied, its previous occupant is not
        # discarded — bumped back into excluded_series so the doctor can
        # switch back later. This is what makes relabel a genuine replace,
        # not just a fill-the-empty-slot operation.
        previous = session_data['series'].get(modality)
        if previous is not None:
            remaining_excluded.append({
                'original_path': previous['original_path'],
                'series_description': previous['series_description'],
                'slice_count': previous['slice_count'],
                'detected_modality': modality,
                'reason': 'replaced_by_manual_relabel',
            })

        session_data['excluded_series'] = remaining_excluded
        session_data['series'][modality] = {
            'original_path': original_path,
            'slice_count': len(source_files),
            'series_description': series_entry['series_description'],
        }
```

Also update the method's docstring (currently lines 613-618) to say "piece B's excluded_series" instead of "unrecognized_series", and to mention replace: add one sentence noting that assigning to an already-filled modality replaces it, with the previous occupant preserved in `excluded_series` rather than discarded.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k TestRelabelSeriesReplace`
Expected: PASS (1/1)

- [ ] **Step 5: Run full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: all pass except the one pre-existing unrelated failure (`test_preprocessing_version.py::test_dataset_mapping`) — same baseline as every prior task in this project's history.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): relabel_series replaces an occupied modality slot, bumping the previous occupant back into excluded_series"
```

---

### Task 3: Real end-to-end verification against real BO data

**Files:** none modified — verification only.

- [ ] **Step 1: Reproduce the exact motivating case end-to-end through the backend, not just stage 01**

Using the real `data/clinical_dicom/BO/test/BO-68` case (two identical T1 duplicates, one wins dedup, one lands in `excluded_series` with `detected_modality: "t1"`, `reason: "lost_deduplication"` — already confirmed at the stage-01 level by the prior plan's Task 2):

1. Run stage 01 on `data/clinical_dicom/BO/test/BO-68` into a scratch output dir (same pattern as the prior plan's verification — `PYTHONPATH=.` venv invocation, nested `bids_organized` output path, `--force`).
2. Call `PipelineManager.get_incomplete_patients(scratch_output_dir)` directly (no DB/HTTP needed, same pattern as the original C-backend plan's Task 5) — confirm the session appears with `excluded_series` containing the losing T1 duplicate, `detected_modality: "t1"`, `reason: "lost_deduplication"`.
3. Call `PipelineManager.relabel_series(...)` passing that duplicate's `original_path` and `modality="t1"` (i.e., explicitly replacing the already-selected winner with the loser, the exact "врач может захотеть заменить t1 на другой t1" scenario from the user's original feedback) — confirm: the call succeeds, the previously-selected T1 is now in `excluded_series` with `reason: "replaced_by_manual_relabel"`, and the new T1 is the one physically present in `series['t1']`'s target directory on disk.
4. Clean up the scratch directory.

- [ ] **Step 2: Final full-suite run**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && python3 -m pytest tests/stage01/ -q && cd backend && python3 -m pytest -q`
Expected: `tests/stage01/` all pass; `backend/` all pass except the one pre-existing unrelated failure.
