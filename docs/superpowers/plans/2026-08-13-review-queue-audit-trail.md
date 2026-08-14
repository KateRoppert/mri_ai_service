# Review Queue Audit Trail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the incomplete-patients review queue a durable audit trail — a session the doctor has ever acted on (relabeled to completion, or discarded) keeps showing up in the report, instead of silently vanishing the moment it's resolved.

**Architecture:** Add one backend-only bookkeeping field, `manually_reviewed: bool`, to each session's dict in `dataset_mapping.json`, set by `relabel_series()` and `discard_session()`. `get_incomplete_patients()`'s inclusion filter gains one more `OR` clause for it, and stops excluding `status == 'discarded'` sessions outright. The frontend then just needs two more status-tag cases (green "Полная", grey "Отброшена") and a read-only rendering mode in the detail modal for discarded sessions — no new frontend state-tracking, since the backend now naturally keeps returning these rows on every existing `fetchSessions()` refresh.

**Tech Stack:** Python/FastAPI backend (`backend/pipeline_manager.py`), React 19 + Ant Design v6 frontend (`frontend/src/components/`), pytest for backend TDD, no automated frontend tests (manual browser verification only, matching the rest of this branch).

## Global Constraints

- No new endpoints, no schema migration — `manually_reviewed` defaults to `False` via `.get('manually_reviewed', False)` for any session dict written before this change.
- `manually_reviewed` is backend-internal bookkeeping — it does NOT get added to `IncompletePatientSession` (`backend/models.py`) or round-trip to the frontend. The frontend derives all four visual states from `status` + `excluded_series` presence only, exactly as it already does for the first two states.
- No "undo discard" action in this pass (explicitly deferred in the spec).
- No backend-side guard rejecting `relabel_series`/`discard_session` against an already-discarded session — the frontend simply won't expose the controls for a discarded row (explicitly deferred in the spec).
- Discarded sessions render read-only in `IncompletePatientDetail.jsx`: modality tags and excluded-series list still show, but the assign (`Select` + "Назначить") and "Отбросить сессию" controls are hidden.

---

### Task 1: Backend — `manually_reviewed` marker and filter update

**Files:**
- Modify: `backend/pipeline_manager.py` (`discard_session` at line 596, `relabel_series` at line 618, `get_incomplete_patients` at line 553)
- Test: `backend/test_incomplete_patients_api.py`

**Interfaces:**
- Produces: no new public method signatures — `get_incomplete_patients()`, `relabel_series()`, `discard_session()` keep their existing signatures and return shapes. Only their persisted-data side effects and filtering logic change.

- [ ] **Step 1: Update two existing tests that assert the OLD (soon-to-be-wrong) behavior**

Two existing tests currently assert that a discarded session is excluded from `get_incomplete_patients()`. Under the new spec, discarded sessions are included (with `status: "discarded"`). Update both now, before touching production code, so the RED step in Step 2 is meaningful (these two updated assertions should fail against today's code, then pass after Step 4's implementation).

In `backend/test_incomplete_patients_api.py`, replace `test_returns_only_incomplete_non_discarded_sessions` (currently lines 21-67) with:

```python
    def test_returns_incomplete_and_discarded_but_not_untouched_complete_sessions(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "complete",
                        "series": {"t1": {}, "t1c": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230115",
                        "status": "incomplete",
                        "series": {"t1c": {}},
                        "excluded_series": [
                            {
                                "original_path": "/raw/weird", "series_description": "xyz",
                                "slice_count": 20, "detected_modality": None, "reason": "unrecognized",
                            }
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
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))

        # sub-001/ses-001 is untouched, always-complete, no alternatives — excluded.
        # sub-001/ses-002 is incomplete — included.
        # sub-002/ses-001 is discarded — now included (this is the behavior change).
        assert len(result) == 2
        by_key = {(r["patient_id"], r["session_id"]): r for r in result}
        assert ("sub-001", "ses-002") in by_key
        assert by_key[("sub-001", "ses-002")]["status"] == "incomplete"
        assert ("sub-002", "ses-001") in by_key
        assert by_key[("sub-002", "ses-001")]["status"] == "discarded"
        assert ("sub-001", "ses-001") not in by_key
```

In the same file, replace `test_discarded_session_excluded_from_incomplete_list` (currently lines 363-377, inside `class TestDiscardSession`) with:

```python
    def test_discarded_session_still_appears_in_review_queue(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "incomplete",
                        "series": {}, "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.discard_session(str(tmp_path), "sub-001", "ses-001")

        result = pm.get_incomplete_patients(str(tmp_path))
        assert len(result) == 1
        assert result[0]["status"] == "discarded"
```

- [ ] **Step 2: Run the updated and new tests to confirm they fail against current code**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k "test_returns_incomplete_and_discarded_but_not_untouched_complete_sessions or test_discarded_session_still_appears_in_review_queue"`
Expected: Both FAIL. `test_returns_incomplete_and_discarded_but_not_untouched_complete_sessions` fails on `assert len(result) == 2` (currently returns 1, since discarded is excluded). `test_discarded_session_still_appears_in_review_queue` fails on `assert len(result) == 1` (currently returns 0).

- [ ] **Step 3: Add two new tests for the `manually_reviewed` marker itself**

Add to `class TestRelabelSeries` in `backend/test_incomplete_patients_api.py` (find the class, add as a new method):

```python
    def test_relabel_that_completes_session_still_appears_in_review_queue(self, tmp_path):
        """Filling the last missing modality makes excluded_series empty and
        status complete — under the OLD filter this would silently drop out
        of the review queue with no confirmation the action worked. The new
        manually_reviewed marker keeps it visible."""
        source_dir = tmp_path / "raw_series"
        source_dir.mkdir()
        (source_dir / "IM001.dcm").write_bytes(b"\x00" * 200)

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {"original_path": "x", "slice_count": 1, "series_description": "t1"},
                                   "t2": {"original_path": "x", "slice_count": 1, "series_description": "t2"},
                                   "t2fl": {"original_path": "x", "slice_count": 1, "series_description": "t2fl"}},
                        "excluded_series": [
                            {"original_path": str(source_dir), "series_description": "candidate t1c",
                             "slice_count": 1, "detected_modality": "t1c", "reason": "unrecognized"},
                        ],
                    },
                },
            },
        })
        pm = PipelineManager()
        pm.relabel_series(str(tmp_path), "sub-001", "ses-001", str(source_dir), "t1c")

        result = pm.get_incomplete_patients(str(tmp_path))
        assert len(result) == 1
        assert result[0]["status"] == "complete"
        assert result[0]["excluded_series"] == []

    def test_untouched_complete_session_has_manually_reviewed_false_by_default(self, tmp_path):
        """Regression guard: manually_reviewed must default False for session
        dicts that predate this feature (no key present at all), not True."""
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101", "status": "complete",
                        "series": {"t1": {}, "t1c": {}, "t2": {}, "t2fl": {}},
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        assert pm.get_incomplete_patients(str(tmp_path)) == []
```

Note: `test_relabel_that_completes_session_still_appears_in_review_queue` relies on `relabel_series`'s real DICOM copy path (`find_dicom_files`/`copy_and_anonymize_series`/`_build_metadata_extractor`) — this mirrors the exact fixture pattern already used by other tests in `class TestRelabelSeries` in this same file (a temp `source_dir` with a single fake `.dcm` file). Read one of the existing passing tests in that class first (e.g. around line 150) to match the exact fixture shape if the DICOM file needs specific bytes/tags to pass anonymization — copy that pattern exactly rather than inventing a new one.

- [ ] **Step 4: Run the new tests to confirm they fail correctly**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k "test_relabel_that_completes_session_still_appears_in_review_queue or test_untouched_complete_session_has_manually_reviewed_false_by_default"`
Expected: `test_relabel_that_completes_session_still_appears_in_review_queue` FAILS on `assert len(result) == 1` (currently 0, since a complete session with empty excluded_series is excluded). `test_untouched_complete_session_has_manually_reviewed_false_by_default` PASSES already (it's a regression guard for behavior that shouldn't change) — that's expected, not a problem.

- [ ] **Step 5: Implement the production code changes**

In `backend/pipeline_manager.py`, `discard_session` (currently lines 596-616), add the marker right before writing the mapping file back:

```python
        session_data = mapping_data['patients'][patient_id]['sessions'][session_id]
        session_data['status'] = 'discarded'
        session_data['manually_reviewed'] = True

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
```

In `relabel_series` (currently lines 618-761), add the marker alongside the other `session_data` mutations, right after `session_data['status'] = 'complete' if is_complete else 'incomplete'` (currently line 746):

```python
        is_complete = required.issubset(session_data['series'].keys())
        session_data['status'] = 'complete' if is_complete else 'incomplete'
        session_data['manually_reviewed'] = True
```

In `get_incomplete_patients` (currently lines 553-594), update the docstring and the inclusion rule (currently lines 578-582):

```python
        """
        Read dataset_mapping.json and return the doctor-review queue for a
        run: every incomplete session, every complete session that still has
        excluded_series (e.g. a dedup loser that lost to the winner but is
        still a plausible alternative), every discarded session, and any
        session the doctor has ever manually touched (relabeled or
        discarded) — even if that action fully resolved it. This makes the
        report a durable audit trail: once you've acted on a session, it
        keeps showing up (with its current status), so a resolved-complete
        or discarded row confirms the action worked instead of silently
        vanishing. A session nobody has ever needed to look at (always
        complete, no alternatives, never touched) is excluded.
        """
```

```python
                status = session_data.get('status')
                has_alternatives = bool(session_data.get('excluded_series'))
                needs_review = (
                    status == 'incomplete'
                    or (status == 'complete' and has_alternatives)
                    or status == 'discarded'
                    or session_data.get('manually_reviewed', False)
                )
```

- [ ] **Step 6: Run all four tests to confirm they pass**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k "test_returns_incomplete_and_discarded_but_not_untouched_complete_sessions or test_discarded_session_still_appears_in_review_queue or test_relabel_that_completes_session_still_appears_in_review_queue or test_untouched_complete_session_has_manually_reviewed_false_by_default"`
Expected: PASS (4/4).

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: same baseline as before this task (84 passed, 1 pre-existing unrelated failure in `test_preprocessing_version.py::test_dataset_mapping`) plus the net-new tests from this task. Run `git checkout -- configs/preprocessing_versions.json` afterward (known harmless test-pollution side effect from an unrelated test) before committing.

- [ ] **Step 8: Commit**

```bash
git add backend/pipeline_manager.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): manually_reviewed marker makes review queue a durable audit trail"
```

---

### Task 2: Frontend — four-state status display and read-only discarded view

**Files:**
- Modify: `frontend/src/components/IncompletePatients.jsx` (status column `render`, currently lines 73-82)
- Modify: `frontend/src/components/IncompletePatientDetail.jsx` (currently 177 lines)

**Interfaces:**
- Consumes: `status` (`'incomplete' | 'complete' | 'discarded'`) and `excluded_series` (array) from each session object returned by `getIncompletePatients()` — both fields already exist in the API response (Task 1 changes which sessions are included, not the shape of each session).

- [ ] **Step 1: Update the status column in `IncompletePatients.jsx`**

Replace the `render` function of the "Статус" column (currently lines 73-82):

```jsx
    {
      title: 'Статус',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => {
        if (status === 'incomplete') {
          return <Tag color="orange">Неполная</Tag>;
        }
        if (status === 'discarded') {
          return <Tag color="default">Отброшена</Tag>;
        }
        const hasAlternatives = (record.excluded_series || []).length > 0;
        return hasAlternatives
          ? <Tag color="blue">Есть альтернативы</Tag>
          : <Tag color="green">Полная</Tag>;
      },
    },
```

No import changes needed — `Tag` is already imported in this file.

- [ ] **Step 2: Add the read-only guard to `IncompletePatientDetail.jsx`**

Right after the existing early return (currently `if (!session) return null;` at line 31), add:

```jsx
  const isReadOnly = session.status === 'discarded';
```

- [ ] **Step 3: Show a read-only notice and hide the assign controls**

In the outer `<Space direction="vertical" ...>` (currently starting at line 83), add a notice as the first child, right after the opening tag:

```jsx
        {isReadOnly && (
          <Text type="secondary" style={{ fontStyle: 'italic' }}>
            Сессия отброшена — показано только для справки, действия недоступны.
          </Text>
        )}
```

In the `excluded_series` `List`'s `renderItem` (currently lines 105-155), wrap the `<Space>` containing the `Select` and relabel button (currently lines 128-151) in a read-only guard:

```jsx
                      {!isReadOnly && (
                        <Space>
                          <Select
                            size="small"
                            style={{ width: 160 }}
                            placeholder="Модальность"
                            options={MODALITY_OPTIONS}
                            value={modality}
                            onChange={(value) =>
                              setSelectedModality((prev) => ({ ...prev, [entry.original_path]: value }))
                            }
                          />
                          {willReplace ? (
                            <Popconfirm
                              title="Эта модальность уже заполнена другой серией — заменить?"
                              onConfirm={() => handleRelabel(entry)}
                              okText="Да"
                              cancelText="Нет"
                            >
                              {relabelButton}
                            </Popconfirm>
                          ) : (
                            relabelButton
                          )}
                        </Space>
                      )}
```

(This is the existing block, unchanged internally — only the wrapping `{!isReadOnly && (...)}` is new.)

- [ ] **Step 4: Hide the discard button in read-only mode**

Wrap the final `Popconfirm` + "Отбросить сессию" button (currently lines 162-171) the same way:

```jsx
        {!isReadOnly && (
          <Popconfirm
            title="Отбросить сессию? Данные не удаляются, но она уйдёт из очереди review."
            onConfirm={handleDiscard}
            okText="Да"
            cancelText="Нет"
          >
            <Button danger icon={<DeleteOutlined />} loading={discarding}>
              Отбросить сессию
            </Button>
          </Popconfirm>
        )}
```

- [ ] **Step 5: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors from either file (baseline at time of writing: 21 pre-existing problems in unrelated files — confirm the count/file-list is unchanged by this task's two files).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/IncompletePatients.jsx frontend/src/components/IncompletePatientDetail.jsx
git commit -m "feat(frontend): four-state status tags and read-only view for discarded sessions"
```

---

### Task 3: Manual verification

**Files:** none modified — verification only, folded into the same manual pass already in progress on this branch for Task 5 of the prior plan.

- [ ] **Step 1: Rebuild and restart the stack**

```bash
docker compose --profile full up --build
```

- [ ] **Step 2: Verify the "resolved, stays visible" case**

Open a run's "Пациенты, требующие внимания" list, pick an incomplete session with exactly one missing modality, assign its last missing modality via an excluded-series entry (or via a fresh manual DICOM path if no excluded entry exists for it). Confirm: after the action succeeds and you close the detail modal, the row is STILL in the list, now tagged green "Полная" with all required modalities shown — it does not vanish.

- [ ] **Step 3: Verify the "discarded, stays visible, read-only" case**

Pick a different incomplete session, click "Отбросить сессию", confirm. Confirm: the row stays in the list, tagged grey "Отброшена". Click "Подробнее" on it — confirm the detail modal shows the read-only notice, modality tags, and excluded-series list (if any), but NO "Назначить" controls and NO "Отбросить сессию" button.

- [ ] **Step 4: Verify an untouched, always-complete session still doesn't appear**

Confirm a patient session that was complete from the very first run, with no `excluded_series` and never manually touched, does NOT appear in the list at all — the queue isn't cluttered with sessions nobody needed to look at.

- [ ] **Step 5: Report findings**

Note any issues found; this folds into the same manual-verification pass already underway for this branch before merge.
