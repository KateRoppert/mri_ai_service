# Manual Session Merge ("Piece A") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a doctor manually pull one session's series into another session of the same patient as review candidates, and see a patient's sessions grouped together in the review list.

**Architecture:** No new file-copying/anonymization code — every series (assigned or excluded) already carries its `original_path` back to the raw DICOM source, so merging is just moving JSON entries from the donor session's `series`/`excluded_series` into the primary session's `excluded_series` pool (tagged `reason: "from_other_session"`). The doctor then assigns them through the already-built `relabel_series` flow. The donor becomes a new permanent status, `merged`, alongside the existing `discarded`.

**Tech Stack:** Python/FastAPI backend (`backend/pipeline_manager.py`, `backend/app.py`, `backend/models.py`), React 19 + Ant Design v6 frontend (`frontend/src/components/`, `frontend/src/services/api.js`), pytest.

**Branch:** This plan builds on `feat/requeue-progress-linking` (not plain `main`) — that branch already contains the one-shot `manually_reviewed` fix (`get_incomplete_patients`'s `needs_review` logic and the exact line numbers below assume that fix is present). Create `feat/manual-session-merge` from `feat/requeue-progress-linking`.

## Global Constraints

- No physical file moves/deletes for the donor session — only `dataset_mapping.json` entries change. The donor's `original_path` pointers (now living in the primary's `excluded_series`) remain valid for `relabel_series` to re-copy+anonymize from later.
- `merged` is a **permanent** audit-trail status (like `discarded`), not a one-shot confirmation (unlike the `manually_reviewed` "became complete" case) — it must not disappear from `get_incomplete_patients()` on a later fetch.
- Merging is idempotent: calling it twice with the same primary/donor must not duplicate candidates already present in the primary's `excluded_series` (dedup by `original_path`).
- No "undo merge" action in this pass.
- No heuristic/auto-detection of merge candidates — the doctor always picks both sessions explicitly.
- No new frontend data fetch for "other sessions of this patient" — filtered client-side from the already-loaded `sessions` array.

---

### Task 1: Backend — `merge_sessions()` method

**Files:**
- Modify: `backend/pipeline_manager.py` (add method after `discard_session`, currently ending around line 647, before `relabel_series` at line 649)
- Test: `backend/test_incomplete_patients_api.py`

**Interfaces:**
- Produces: `PipelineManager.merge_sessions(output_path: str, patient_id: str, primary_session_id: str, donor_session_id: str) -> Dict[str, Any]` returning `{"status": "merged", "primary_session_id": str, "donor_session_id": str, "pulled_series": int}`. Raises `ValueError` for malformed IDs or `primary_session_id == donor_session_id`; raises `KeyError` (uncaught, same convention as `relabel_series`/`discard_session`) if the patient or either session isn't found.

- [ ] **Step 1: Write the failing tests**

Add to `backend/test_incomplete_patients_api.py`, as a new class after `TestDiscardSessionInputValidation`:

```python
class TestMergeSessions:
    def test_merge_pulls_assigned_and_excluded_series_from_donor(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {"t1": {}, "t2": {}},
                        "excluded_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230201", "status": "incomplete",
                        "series": {
                            "t1c": {"original_path": "/raw/donor_t1c", "series_description": "t1c series", "slice_count": 30},
                        },
                        "excluded_series": [
                            {
                                "original_path": "/raw/donor_extra", "series_description": "extra t2fl",
                                "slice_count": 20, "detected_modality": "t2fl", "reason": "unrecognized",
                            },
                        ],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        assert result == {
            "status": "merged",
            "primary_session_id": "ses-001",
            "donor_session_id": "ses-002",
            "pulled_series": 2,
        }

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        primary = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        excluded_paths = {u["original_path"]: u for u in primary["excluded_series"]}
        assert "/raw/donor_t1c" in excluded_paths
        assert excluded_paths["/raw/donor_t1c"]["detected_modality"] == "t1c"
        assert excluded_paths["/raw/donor_t1c"]["reason"] == "from_other_session"
        assert "/raw/donor_extra" in excluded_paths
        assert excluded_paths["/raw/donor_extra"]["detected_modality"] == "t2fl"
        assert excluded_paths["/raw/donor_extra"]["reason"] == "from_other_session"

    def test_merge_conflicting_modality_offers_both_versions(self, tmp_path):
        """Primary already has a t1; donor also has a t1 — both must be
        offered as alternatives, not silently preferred either way."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {"t1": {"original_path": "/raw/primary_t1", "series_description": "primary t1", "slice_count": 10}},
                        "excluded_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230201", "status": "incomplete",
                        "series": {"t1": {"original_path": "/raw/donor_t1", "series_description": "donor t1", "slice_count": 12}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        primary = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        # primary's own t1 assignment is untouched
        assert primary["series"]["t1"]["original_path"] == "/raw/primary_t1"
        # donor's t1 is offered as an alternative, not silently dropped
        assert len(primary["excluded_series"]) == 1
        assert primary["excluded_series"][0]["original_path"] == "/raw/donor_t1"
        assert primary["excluded_series"][0]["detected_modality"] == "t1"

    def test_merge_marks_donor_session_merged_with_pointer(self, tmp_path):
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
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        donor = mapping["patients"]["sub-001"]["sessions"]["ses-002"]
        assert donor["status"] == "merged"
        assert donor["merged_into_session_id"] == "ses-001"

    def test_merge_is_idempotent_no_duplicate_candidates(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {"original_date": "20230101", "status": "incomplete", "series": {}, "excluded_series": []},
                    "ses-002": {
                        "original_date": "20230201", "status": "incomplete",
                        "series": {"t1c": {"original_path": "/raw/donor_t1c", "series_description": "t1c", "slice_count": 30}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")
        result_second_call = pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        assert result_second_call["pulled_series"] == 0
        mapping = json.loads((tmp_path / "bids_organized" / "dataset_mapping.json").read_text())
        primary = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        assert len(primary["excluded_series"]) == 1


class TestMergeSessionsInputValidation:
    def test_rejects_path_traversal_patient_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid patient_id"):
            pm.merge_sessions(str(tmp_path), "..", "ses-001", "ses-002")

    def test_rejects_path_traversal_primary_session_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid primary_session_id"):
            pm.merge_sessions(str(tmp_path), "sub-001", "..", "ses-002")

    def test_rejects_path_traversal_donor_session_id(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="Invalid donor_session_id"):
            pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "..")

    def test_rejects_same_session_as_primary_and_donor(self, tmp_path):
        _write_mapping(tmp_path, {})
        pm = PipelineManager()
        with pytest.raises(ValueError, match="must be different"):
            pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-001")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k "TestMergeSessions or TestMergeSessionsInputValidation"`
Expected: all FAIL with `AttributeError: 'PipelineManager' object has no attribute 'merge_sessions'`.

- [ ] **Step 3: Implement**

In `backend/pipeline_manager.py`, add this method after `discard_session` (currently ends at line 647) and before `relabel_series` (currently starts at line 649):

```python
    def merge_sessions(
        self, output_path: str, patient_id: str,
        primary_session_id: str, donor_session_id: str,
    ) -> Dict[str, Any]:
        """
        Pull a donor session's series (both currently-assigned and its own
        leftover excluded_series) into the primary session's excluded_series
        pool as candidates, tagged reason="from_other_session". The doctor
        then assigns them through the existing relabel_series flow — this
        method does no file copying or anonymization itself, since every
        series already carries its original_path back to the raw DICOM
        source, which relabel_series already knows how to re-copy+anonymize.

        The donor session becomes a permanent 'merged' audit-trail entry
        (like 'discarded', but semantically distinct — merged means
        consolidated into another session, not skipped) pointing at
        primary_session_id via merged_into_session_id. No physical files
        are touched or deleted.

        Returns:
            Dict with status, primary_session_id, donor_session_id, and
            pulled_series (how many new candidates were added — 0 if
            merging again after everything was already pulled).

        Raises:
            ValueError if patient_id/primary_session_id/donor_session_id
            are malformed, or primary and donor are the same session.
        """
        if not _BIDS_PATIENT_ID_PATTERN.match(patient_id):
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(primary_session_id):
            raise ValueError(f"Invalid primary_session_id: {primary_session_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(donor_session_id):
            raise ValueError(f"Invalid donor_session_id: {donor_session_id!r}")
        if primary_session_id == donor_session_id:
            raise ValueError("primary_session_id and donor_session_id must be different")

        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        sessions = mapping_data['patients'][patient_id]['sessions']
        primary_session_data = sessions[primary_session_id]
        donor_session_data = sessions[donor_session_id]

        primary_session_data.setdefault('excluded_series', [])
        existing_paths = {u['original_path'] for u in primary_session_data['excluded_series']}

        pulled = 0
        for modality, series_info in donor_session_data.get('series', {}).items():
            if series_info['original_path'] in existing_paths:
                continue
            primary_session_data['excluded_series'].append({
                'original_path': series_info['original_path'],
                'series_description': series_info['series_description'],
                'slice_count': series_info['slice_count'],
                'detected_modality': modality,
                'reason': 'from_other_session',
            })
            existing_paths.add(series_info['original_path'])
            pulled += 1

        for entry in donor_session_data.get('excluded_series', []):
            if entry['original_path'] in existing_paths:
                continue
            primary_session_data['excluded_series'].append({
                'original_path': entry['original_path'],
                'series_description': entry['series_description'],
                'slice_count': entry['slice_count'],
                'detected_modality': entry.get('detected_modality'),
                'reason': 'from_other_session',
            })
            existing_paths.add(entry['original_path'])
            pulled += 1

        donor_session_data['status'] = 'merged'
        donor_session_data['merged_into_session_id'] = primary_session_id

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        return {
            'status': 'merged',
            'primary_session_id': primary_session_id,
            'donor_session_id': donor_session_id,
            'pulled_series': pulled,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k "TestMergeSessions or TestMergeSessionsInputValidation"`
Expected: PASS (8/8 — 4 in `TestMergeSessions`, 4 in `TestMergeSessionsInputValidation`).

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: same baseline as before this task (94 passed, 1 pre-existing unrelated failure in `test_preprocessing_version.py::test_dataset_mapping`) plus this task's 8 new tests = 102 passed, 1 pre-existing failure. Run `git checkout -- configs/preprocessing_versions.json` afterward (known harmless test-pollution side effect) before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): merge_sessions pulls a donor session's series into a primary session's excluded_series"
```

---

### Task 2: Backend — endpoint, models, and permanent `merged` status in the review queue

**Files:**
- Modify: `backend/models.py` (`ExcludedSeriesInfo` at line 131, `IncompletePatientSession` at line 141, add `MergeSessionsRequest`/`MergeSessionsResponse` near `DiscardSessionResponse` at line 173)
- Modify: `backend/app.py` (add endpoint after `discard_session` route, currently ending at line 981)
- Modify: `backend/pipeline_manager.py` (`get_incomplete_patients`, currently lines 553-622)
- Test: `backend/test_incomplete_patients_api.py`

**Interfaces:**
- Consumes: `PipelineManager.merge_sessions(...)` from Task 1.
- Produces: `POST /api/incomplete-patients/{run_id}/{patient_id}/merge` (body `{primary_session_id, donor_session_id}`, response `MergeSessionsResponse`). `IncompletePatientSession.merged_into_session_id: Optional[str]` — consumed by Task 3's frontend status tag.

- [ ] **Step 1: Write the failing test**

Add to `backend/test_incomplete_patients_api.py`'s `TestDiscardSession` class area — add as a new test method in a new class right after it:

```python
class TestGetIncompletePatientsMerged:
    def test_merged_session_appears_permanently_in_review_queue(self, tmp_path):
        """Like discarded, merged is a permanent audit-trail entry — must
        NOT disappear after being read once (unlike the one-shot
        manually_reviewed 'became complete' confirmation)."""
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
        pm.merge_sessions(str(tmp_path), "sub-001", "ses-001", "ses-002")

        first = pm.get_incomplete_patients(str(tmp_path))
        second = pm.get_incomplete_patients(str(tmp_path))

        by_key_first = {(r["patient_id"], r["session_id"]): r for r in first}
        by_key_second = {(r["patient_id"], r["session_id"]): r for r in second}
        assert by_key_first[("sub-001", "ses-002")]["status"] == "merged"
        assert by_key_first[("sub-001", "ses-002")]["merged_into_session_id"] == "ses-001"
        assert by_key_second[("sub-001", "ses-002")]["status"] == "merged"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k test_merged_session_appears_permanently_in_review_queue`
Expected: FAIL — `by_key_first` is missing the `("sub-001", "ses-002")` key entirely, since `get_incomplete_patients` doesn't yet include `status == 'merged'` sessions (KeyError on the dict lookup).

- [ ] **Step 3: Implement**

In `backend/pipeline_manager.py`'s `get_incomplete_patients` (currently lines 553-622), add the `merged` clause to `needs_review` (currently lines 588-593):

```python
                needs_review = (
                    status == 'incomplete'
                    or (status == 'complete' and has_alternatives)
                    or status == 'discarded'
                    or status == 'merged'
                    or manually_reviewed
                )
```

And add `merged_into_session_id` to the result dict (currently lines 607-616):

```python
                results.append({
                    "patient_id": patient_id,
                    "original_id": patient_data.get('original_id', ''),
                    "session_id": session_id,
                    "date": session_data.get('original_date', ''),
                    "status": status,
                    "available": available,
                    "missing": sorted(required - set(available)),
                    "excluded_series": session_data.get('excluded_series', []),
                    "merged_into_session_id": session_data.get('merged_into_session_id'),
                })
```

In `backend/models.py`, update `ExcludedSeriesInfo.reason`'s docstring (currently line 138) to include the new reason value:

```python
    reason: str = Field(..., description="unrecognized | lost_deduplication | replaced_by_manual_relabel | from_other_session")
```

Add `merged_into_session_id` to `IncompletePatientSession` (currently ends around line 152, right after the `excluded_series` field):

```python
    merged_into_session_id: Optional[str] = Field(None, description="Если статус 'merged' — ID сессии, в которую объединили")
```

Add two new models right after `DiscardSessionResponse` (currently lines 173-175):

```python
class MergeSessionsRequest(BaseModel):
    """Запрос на объединение сессии-донора с основной сессией"""
    primary_session_id: str = Field(..., description="ID основной (целевой) сессии — BIDS ses-XXX")
    donor_session_id: str = Field(..., description="ID сессии-донора, чьи серии переносятся — BIDS ses-XXX")


class MergeSessionsResponse(BaseModel):
    """Результат объединения сессий"""
    status: str = Field(..., description="Статус после объединения: merged")
    primary_session_id: str = Field(..., description="ID основной (целевой) сессии")
    donor_session_id: str = Field(..., description="ID сессии-донора")
    pulled_series: int = Field(..., description="Сколько новых серий добавлено в excluded_series основной сессии")
```

In `backend/app.py`, add the endpoint right after the `discard_session` route (currently ends at line 981):

```python
@app.post(
    "/api/incomplete-patients/{run_id}/{patient_id}/merge",
    response_model=MergeSessionsResponse,
)
async def merge_sessions(
    run_id: str,
    patient_id: str,
    request: MergeSessionsRequest,
    db: Session = Depends(get_db)
):
    """Объединить сессию-донора с основной сессией пациента"""
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    try:
        result = pipeline_manager.merge_sessions(
            output_path=run.output_path,
            patient_id=patient_id,
            primary_session_id=request.primary_session_id,
            donor_session_id=request.donor_session_id,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(
        f"Объединение {patient_id}: донор {request.donor_session_id} -> "
        f"основная {request.primary_session_id} (run_id={run_id})"
    )
    return MergeSessionsResponse(**result)
```

Confirm `MergeSessionsRequest`/`MergeSessionsResponse` are importable where `app.py` imports its other models from (check the existing `from models import (...)` block and add the two new names there — same place `RelabelSeriesRequest`/`DiscardSessionResponse` etc. are already imported from).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k test_merged_session_appears_permanently_in_review_queue`
Expected: PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: baseline from Task 1 (94 + 8 = 102 passed, 1 pre-existing failure) plus this task's 1 new test = 103 passed, 1 pre-existing failure. Run `git checkout -- configs/preprocessing_versions.json` afterward before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/models.py backend/app.py backend/pipeline_manager.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): POST .../merge endpoint, merged is a permanent review-queue status"
```

---

### Task 3: Frontend — grouping, merge UI, and the `merged` status tag

**Files:**
- Modify: `frontend/src/services/api.js` (add `mergeSessions`, after `discardSession` — currently ends around line 278)
- Modify: `frontend/src/components/IncompletePatients.jsx` (sorting/grouping, status tag, pass `sessions` prop to detail modal)
- Modify: `frontend/src/components/IncompletePatientDetail.jsx` (new merge section, `isReadOnly` extension, `REASON_LABELS` entry)

**Interfaces:**
- Consumes: `mergeSessions(runId, patientId, primarySessionId, donorSessionId)`, `merged_into_session_id` field on each session object (Task 2).
- Produces: none consumed by later tasks — this is the last code task.

- [ ] **Step 1: `api.js` — add `mergeSessions`**

Read the file first to confirm current line numbers. Add after `discardSession` (currently ends around line 278):

```js
/**
 * Объединить сессию-донора с основной (целевой) сессией пациента —
 * переносит серии донора в excluded_series основной как кандидатов
 */
export const mergeSessions = async (runId, patientId, primarySessionId, donorSessionId) => {
  const response = await apiClient.post(
    `/incomplete-patients/${runId}/${patientId}/merge`,
    { primary_session_id: primarySessionId, donor_session_id: donorSessionId }
  );
  return response.data;
};
```

Add `mergeSessions` to the default-export object at the bottom of the file, alongside `discardSession`.

- [ ] **Step 2: `IncompletePatients.jsx` — grouping and the `merged` tag**

Read the file first to confirm current line numbers. Replace the status column's `render` function (currently lines 80-91) to add the `merged` case:

```jsx
      render: (status, record) => {
        if (status === 'incomplete') {
          return <Tag color="orange">Неполная</Tag>;
        }
        if (status === 'discarded') {
          return <Tag color="default">Отброшена</Tag>;
        }
        if (status === 'merged') {
          return <Tag color="default">Объединена → {record.merged_into_session_id}</Tag>;
        }
        const hasAlternatives = (record.excluded_series || []).length > 0;
        return hasAlternatives
          ? <Tag color="blue">Есть альтернативы</Tag>
          : <Tag color="green">Полная</Tag>;
      },
```

Add grouping. Right before the `columns` array declaration, compute a sorted copy of `sessions` and a rowSpan map for the "Пациент" column:

```jsx
  const sortedSessions = [...sessions].sort((a, b) => {
    if (a.original_id !== b.original_id) return a.original_id.localeCompare(b.original_id);
    return a.session_id.localeCompare(b.session_id);
  });

  const patientRowSpans = {};
  sortedSessions.forEach((s, idx) => {
    if (idx === 0 || sortedSessions[idx - 1].original_id !== s.original_id) {
      let count = 1;
      while (sortedSessions[idx + count] && sortedSessions[idx + count].original_id === s.original_id) {
        count++;
      }
      patientRowSpans[idx] = count;
    } else {
      patientRowSpans[idx] = 0;
    }
  });
```

Change the "Пациент" column (currently `{ title: 'Пациент', dataIndex: 'original_id', key: 'original_id' }`) to use the rowSpan map:

```jsx
    {
      title: 'Пациент',
      dataIndex: 'original_id',
      key: 'original_id',
      render: (value, record, index) => ({
        children: value,
        props: { rowSpan: patientRowSpans[index] },
      }),
    },
```

Change the `<Table>` element's `dataSource` from `sessions` to `sortedSessions`, and **remove pagination** (currently `pagination={{ pageSize: 10 }}`, change to `pagination={false}`) — antd's per-row `rowSpan` grouping only works within a single rendered page; with pagination active, a patient's sessions could get split across pages and the rowSpan math would be wrong. Review queues are small (tens of sessions per run, not hundreds), so an unpaginated table is fine here:

```jsx
        <Table
          columns={columns}
          dataSource={sortedSessions}
          rowKey={(r) => `${r.patient_id}_${r.session_id}`}
          pagination={false}
          locale={{ emptyText: 'Нет сессий, требующих внимания' }}
        />
```

Pass the full `sessions` array down to the detail modal so it can compute "other sessions of this patient" without a new fetch (currently the `<IncompletePatientDetail>` mount at the bottom of the file):

```jsx
      <IncompletePatientDetail
        runId={runId}
        session={selectedSession}
        sessions={sessions}
        visible={!!selectedSession}
        onClose={() => setSelectedSession(null)}
        onActionComplete={fetchSessions}
      />
```

- [ ] **Step 3: `IncompletePatientDetail.jsx` — merge section**

Read the file first to confirm current line numbers. Add `mergeSessions` to the import (currently line 9):

```jsx
import { relabelSeries, discardSession, mergeSessions } from '../services/api';
```

Add `from_other_session` to `REASON_LABELS` (currently lines 20-24):

```jsx
const REASON_LABELS = {
  unrecognized: 'алгоритм не распознал',
  lost_deduplication: 'алгоритм распознал, но выбрал другую копию',
  replaced_by_manual_relabel: 'заменена вручную ранее',
  from_other_session: 'перенесена из другой сессии пациента',
};
```

Add `sessions` to the component's props (currently line 26) and extend `isReadOnly` to also cover `merged`:

```jsx
const IncompletePatientDetail = ({ runId, session, sessions = [], visible, onClose, onActionComplete }) => {
  const [selectedModality, setSelectedModality] = useState({});
  const [loadingPath, setLoadingPath] = useState(null);
  const [discarding, setDiscarding] = useState(false);
  const [donorSessionId, setDonorSessionId] = useState(undefined);
  const [merging, setMerging] = useState(false);

  if (!session) return null;

  const isReadOnly = session.status === 'discarded' || session.status === 'merged';
```

Add `handleMerge` and the `otherSessions` filter, right after `handleDiscard` (currently ends at line 73):

```jsx
  const handleMerge = async () => {
    if (!donorSessionId) {
      message.error('Выберите сессию для объединения');
      return;
    }
    setMerging(true);
    try {
      const result = await mergeSessions(runId, session.patient_id, session.session_id, donorSessionId);
      message.success(`Серии из ${donorSessionId} добавлены как альтернативы (${result.pulled_series})`);
      setDonorSessionId(undefined);
      onActionComplete();
    } catch (err) {
      console.error('Ошибка объединения:', err);
      message.error(err.response?.data?.detail || 'Не удалось объединить сессии');
    } finally {
      setMerging(false);
    }
  };

  const otherSessions = sessions.filter(
    (s) => s.patient_id === session.patient_id
      && s.session_id !== session.session_id
      && s.status !== 'merged'
      && s.status !== 'discarded'
  );
```

Add the merge section's JSX, right before the final discard `Popconfirm` block (currently starts at line 171, `{!isReadOnly && (`):

```jsx
        {!isReadOnly && otherSessions.length > 0 && (
          <>
            <Divider style={{ margin: '8px 0' }} />
            <div>
              <Text strong>Объединить с другой сессией пациента:</Text>
              <div style={{ marginTop: 8 }}>
                <Space>
                  <Select
                    size="small"
                    style={{ width: 220 }}
                    placeholder="Выберите сессию"
                    value={donorSessionId}
                    onChange={setDonorSessionId}
                    options={otherSessions.map((s) => ({
                      label: `${s.session_id} (${s.date})`,
                      value: s.session_id,
                    }))}
                  />
                  <Popconfirm
                    title="Перенести серии выбранной сессии сюда как альтернативы?"
                    onConfirm={handleMerge}
                    okText="Да"
                    cancelText="Нет"
                  >
                    <Button size="small" loading={merging} disabled={!donorSessionId}>
                      Объединить
                    </Button>
                  </Popconfirm>
                </Space>
              </div>
            </div>
          </>
        )}
```

- [ ] **Step 4: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors from the three modified files (baseline at time of writing: 21 pre-existing problems in unrelated files).

- [ ] **Step 5: Build check**

Run: `cd frontend && npm run build`
Expected: builds successfully (no new warnings/errors beyond the pre-existing chunk-size notice).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/services/api.js frontend/src/components/IncompletePatients.jsx frontend/src/components/IncompletePatientDetail.jsx
git commit -m "feat(frontend): group a patient's sessions in the review list, manual merge action in the detail modal"
```

---

### Task 4: Manual verification

**Files:** none modified — verification only.

- [ ] **Step 1: Rebuild and restart the stack**

```bash
docker compose --profile full up --build
```

- [ ] **Step 2: Verify grouping**

Open "Пациенты, требующие внимания" for a run where at least one patient has 2+ sessions in the queue (e.g. BO-69 or BO-55 from earlier testing on real BO data). Confirm their rows are adjacent and the "Пациент" cell only shows the name once, spanning both rows.

- [ ] **Step 3: Verify a complementary merge (the split-visit case)**

Open the detail view for one of that patient's incomplete sessions. Confirm the new "Объединить с другой сессией пациента" section lists the patient's other session(s). Pick one, click "Объединить", confirm. Verify: success message with a pulled-series count, the "Неотобранные серии" list in the SAME modal now includes the donor's series (reason text "перенесена из другой сессии пациента"), and — after closing and reopening the list — the donor session's row now shows "Объединена → ses-XXX".

- [ ] **Step 4: Assign a pulled candidate**

From the newly-pulled candidates, pick one and click "Назначить" (the existing relabel flow) — confirm it works exactly as it does for any other excluded-series candidate (fills the slot, or offers the replace confirmation if that modality is already occupied).

- [ ] **Step 5: Verify the merged session is read-only and permanent**

Open the merged (donor) session's own detail view. Confirm it renders read-only (no assign/discard/merge controls, matching the existing discarded-session read-only treatment). Click "Обновить" in the list (or close/reopen) — confirm the merged row is still there, not just shown once.

- [ ] **Step 6: Report findings**

Note any issues found — this is real, unscripted verification.
