# Piece C-frontend: Incomplete-Patient Review UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the doctor a real UI, reachable from both an in-progress run (as soon as stage 1 finishes) and run history, to review sessions needing attention, relabel/replace series, discard sessions, and requeue the pipeline — wired to the already-merged backend API.

**Architecture:** Follows the codebase's existing modal-per-concern idiom exactly (`QualityReport.jsx`/`ClinicalReport.jsx`/`NIfTIViewer.jsx`) rather than inventing a new navigation pattern — no router exists in this app. Two new components: `IncompletePatientDetail.jsx` (one session's detail — modality tags, `excluded_series` list with relabel actions, discard) and `IncompletePatients.jsx` (the list — a table of sessions needing review, requeue button, opens `IncompletePatientDetail` when a row is clicked). Entry points duplicate the codebase's existing dual-path pattern: `ProgressMonitor.jsx` owns its own modal-visibility state for an active run (gated on stage 1 completion, not the whole run), while `PipelineHistory.jsx`/`App.jsx` follow the lifted-state pattern for historical runs. A small backend addition computes `missing` server-side (lesion-type-aware, matching `relabel_series`'s existing pattern) so the frontend never duplicates which modalities are required per lesion type.

**Tech Stack:** React 19, Ant Design v6, axios (`frontend/`); FastAPI/Pydantic (`backend/`, one small addition). No frontend test framework exists in this project (`frontend/package.json` has no `test` script) — verification for frontend tasks is a real dev-server + browser check, not automated tests.

## Global Constraints

- No routing library, no state management library — follow existing `useState`/props patterns exactly, do not introduce either.
- No new CSS files — inline `style={{}}` + antd component props only, matching every existing component.
- Piece A (session-merge candidate confirm/reject UI) is explicitly out of scope — not implemented on the backend yet.
- The list endpoint's scope (complete-with-`excluded_series` sessions included, not just incomplete ones) is already correct in the merged backend — the frontend must not re-filter or hide those sessions itself.
- Follow `frontend/src/services/api.js`'s established function-per-endpoint convention exactly: one named async arrow function per call, added to both the named exports and the default-export object at the bottom.

---

### Task 1: Backend — `missing` field, lesion-type-aware

**Files:**
- Modify: `backend/pipeline_manager.py` (`get_incomplete_patients`, currently lines 553-580)
- Modify: `backend/models.py` (`IncompletePatientSession`, currently lines 138-149)
- Modify: `backend/app.py` (`get_incomplete_patients` endpoint, currently lines 886-906)
- Test: `backend/test_incomplete_patients_api.py` (`TestGetIncompletePatients`)

**Interfaces:**
- Produces: `IncompletePatientSession.missing: List[str]` — consumed by `IncompletePatients.jsx`/`IncompletePatientDetail.jsx` (Task 3) to render "not present" modality tags without duplicating the lesion-type → required-modalities mapping in JS.

- [ ] **Step 1: Write the failing test**

Add to `backend/test_incomplete_patients_api.py`'s `TestGetIncompletePatients` class:

```python
    def test_missing_field_is_lesion_type_aware(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},  # MS-complete, glio-incomplete (no t1c)
                        "excluded_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()

        glio_result = pm.get_incomplete_patients(str(tmp_path), lesion_type="glioblastoma")
        assert glio_result[0]["missing"] == ["t1c"]

        ms_result = pm.get_incomplete_patients(str(tmp_path), lesion_type="multiple_sclerosis")
        assert ms_result[0]["missing"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k test_missing_field_is_lesion_type_aware`
Expected: FAIL — `TypeError: get_incomplete_patients() got an unexpected keyword argument 'lesion_type'`

- [ ] **Step 3: Implement**

`backend/pipeline_manager.py` — add `lesion_type` parameter and compute `missing` (mirroring `relabel_series`'s existing `load_lesion_type_config` fallback pattern, currently at lines 712-716):

```python
    def get_incomplete_patients(self, output_path: str, lesion_type: str = 'glioblastoma') -> List[Dict[str, Any]]:
        """
        Read dataset_mapping.json and return the doctor-review queue for a
        run: every incomplete session, PLUS every complete session that
        still has excluded_series (e.g. a dedup loser that lost to the
        winner but is still a plausible alternative — a doctor may want to
        reconsider which series won, not just fill a gap). A complete
        session with nothing left to reconsider is excluded, as is any
        discarded session.
        """
        mapping_file = self._dataset_mapping_path(output_path)
        if not mapping_file.exists():
            return []

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        try:
            required = set(load_lesion_type_config(lesion_type)['required_modalities'])
        except KeyError:
            required = {'t1', 't1c', 't2', 't2fl'}

        results: List[Dict[str, Any]] = []
        for patient_id, patient_data in mapping_data.get('patients', {}).items():
            for session_id, session_data in patient_data.get('sessions', {}).items():
                status = session_data.get('status')
                has_alternatives = bool(session_data.get('excluded_series'))
                needs_review = status == 'incomplete' or (status == 'complete' and has_alternatives)
                if not needs_review:
                    continue
                available = sorted(session_data.get('series', {}).keys())
                results.append({
                    "patient_id": patient_id,
                    "original_id": patient_data.get('original_id', ''),
                    "session_id": session_id,
                    "date": session_data.get('original_date', ''),
                    "status": status,
                    "available": available,
                    "missing": sorted(required - set(available)),
                    "excluded_series": session_data.get('excluded_series', []),
                })
        return results
```

`backend/models.py` — add to `IncompletePatientSession` (currently lines 138-149), right after `available`:

```python
    missing: List[str] = Field(..., description="Модальности, которых не хватает (lesion-type-aware)")
```

`backend/app.py` — pass `run.lesion_type` through (currently line 901):

```python
    sessions = pipeline_manager.get_incomplete_patients(run.output_path, lesion_type=run.lesion_type or 'glioblastoma')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python3 -m pytest test_incomplete_patients_api.py -v -k TestGetIncompletePatients`
Expected: PASS (4/4 — the 3 pre-existing tests in this class plus the new one)

- [ ] **Step 5: Run full backend suite**

Run: `cd backend && python3 -m pytest -q`
Expected: same pass count as the current baseline +1, same single pre-existing unrelated failure (`test_preprocessing_version.py::test_dataset_mapping`) as every prior task in this project's history — not a new regression.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline_manager.py backend/models.py backend/app.py backend/test_incomplete_patients_api.py
git commit -m "feat(backend): get_incomplete_patients computes missing modalities, lesion-type-aware"
```

---

### Task 2: Frontend — `api.js` client functions

**Files:**
- Modify: `frontend/src/services/api.js`

**Interfaces:**
- Produces: `getIncompletePatients(runId)`, `relabelSeries(runId, patientId, sessionId, originalPath, modality)`, `discardSession(runId, patientId, sessionId)`, `requeuePipelineRun(runId)` — consumed by Task 3's components.

- [ ] **Step 1: Add the four functions**

Add to `frontend/src/services/api.js`, after `getEntityRunInfo` (currently ends at line 248) and before the "Скачать zip-пакет" comment:

```js
/**
 * Список сессий текущего запуска, требующих внимания врача
 * (неполные + полные с непустым excluded_series)
 */
export const getIncompletePatients = async (runId) => {
  const response = await apiClient.get(`/incomplete-patients/${runId}`);
  return response.data;
};

/**
 * Вручную назначить (или заменить) модальность для серии из excluded_series
 */
export const relabelSeries = async (runId, patientId, sessionId, originalPath, modality) => {
  const response = await apiClient.post(
    `/incomplete-patients/${runId}/${patientId}/${sessionId}/relabel`,
    { original_path: originalPath, modality }
  );
  return response.data;
};

/**
 * Отбросить сессию — пометить как намеренно исключённую из очереди review
 */
export const discardSession = async (runId, patientId, sessionId) => {
  const response = await apiClient.post(
    `/incomplete-patients/${runId}/${patientId}/${sessionId}/discard`
  );
  return response.data;
};

/**
 * Перезапустить pipeline на тех же путях (skip_existing обработает только новое)
 */
export const requeuePipelineRun = async (runId) => {
  const response = await apiClient.post(`/pipeline-runs/${runId}/requeue`);
  return response.data;
};
```

- [ ] **Step 2: Add to the default-export object**

At the bottom of the file (currently lines 328-355), add all four names to the exported object, e.g. right after `getEntityRunInfo,`:

```js
  getIncompletePatients,
  relabelSeries,
  discardSession,
  requeuePipelineRun,
```

- [ ] **Step 3: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors introduced by this file (pre-existing errors/warnings elsewhere in the codebase, if any, are not this task's concern — compare the error count/list before and after this change).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.js
git commit -m "feat(frontend): api client functions for incomplete-patients review"
```

---

### Task 3: Frontend — `IncompletePatientDetail.jsx` + `IncompletePatients.jsx`

**Files:**
- Create: `frontend/src/components/IncompletePatientDetail.jsx`
- Create: `frontend/src/components/IncompletePatients.jsx`

**Interfaces:**
- Consumes: Task 2's four api.js functions.
- Produces: `<IncompletePatients runId={string} visible={bool} onClose={fn} />` — the top-level modal, self-contained (fetches its own data on open, following `QualityReport.jsx`'s `useEffect(() => { if (visible && runId) fetch... }, [visible, runId])` pattern exactly) — consumed by Task 4's wiring into `ProgressMonitor.jsx` and `App.jsx`/`PipelineHistory.jsx`.

- [ ] **Step 1: Create `IncompletePatientDetail.jsx`**

```jsx
/**
 * Модальное окно с деталями одной сессии, требующей внимания врача:
 * какие модальности есть/не хватает, список исключённых серий с
 * возможностью назначить их на модальность, кнопка отбросить сессию.
 */
import { useState } from 'react';
import { Modal, Tag, Space, List, Select, Button, Popconfirm, message, Divider, Typography } from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import { relabelSeries, discardSession } from '../services/api';

const { Text } = Typography;

const MODALITY_OPTIONS = [
  { label: 'T1', value: 't1' },
  { label: 'T1c', value: 't1c' },
  { label: 'T2', value: 't2' },
  { label: 'FLAIR (T2fl)', value: 't2fl' },
];

const REASON_LABELS = {
  unrecognized: 'алгоритм не распознал',
  lost_deduplication: 'алгоритм распознал, но выбрал другую копию',
  replaced_by_manual_relabel: 'заменена вручную ранее',
};

const IncompletePatientDetail = ({ runId, session, visible, onClose, onActionComplete }) => {
  const [selectedModality, setSelectedModality] = useState({});
  const [loadingPath, setLoadingPath] = useState(null);
  const [discarding, setDiscarding] = useState(false);

  if (!session) return null;

  const handleRelabel = async (excludedEntry) => {
    const modality = selectedModality[excludedEntry.original_path] || excludedEntry.detected_modality;
    if (!modality) {
      message.error('Выберите модальность');
      return;
    }
    setLoadingPath(excludedEntry.original_path);
    try {
      const result = await relabelSeries(
        runId, session.patient_id, session.session_id, excludedEntry.original_path, modality
      );
      message.success(
        result.status === 'complete'
          ? 'Серия назначена, сессия теперь полная'
          : 'Серия назначена'
      );
      onActionComplete();
    } catch (err) {
      console.error('Ошибка переразметки:', err);
      message.error(err.response?.data?.detail || 'Не удалось назначить серию');
    } finally {
      setLoadingPath(null);
    }
  };

  const handleDiscard = async () => {
    setDiscarding(true);
    try {
      await discardSession(runId, session.patient_id, session.session_id);
      message.success('Сессия отброшена');
      onClose();
      onActionComplete();
    } catch (err) {
      console.error('Ошибка:', err);
      message.error('Не удалось отбросить сессию');
    } finally {
      setDiscarding(false);
    }
  };

  const isAlreadyFilled = (modality) => session.available.includes(modality);

  return (
    <Modal
      title={`${session.original_id} — ${session.session_id}`}
      open={visible}
      onCancel={onClose}
      width={700}
      footer={null}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <div>
          <Text strong>Модальности: </Text>
          <Space wrap>
            {session.available.map((m) => (
              <Tag color="green" key={m}>{m}</Tag>
            ))}
            {session.missing.map((m) => (
              <Tag color="default" key={m}>{m} — нет</Tag>
            ))}
          </Space>
        </div>

        <Divider style={{ margin: '8px 0' }} />

        <div>
          <Text strong>Неотобранные серии:</Text>
          {session.excluded_series.length === 0 ? (
            <p style={{ color: '#999' }}>Нет неотобранных серий</p>
          ) : (
            <List
              dataSource={session.excluded_series}
              renderItem={(entry) => {
                const modality = selectedModality[entry.original_path] || entry.detected_modality || undefined;
                const willReplace = modality && isAlreadyFilled(modality);
                const relabelButton = (
                  <Button
                    type="primary"
                    size="small"
                    loading={loadingPath === entry.original_path}
                    disabled={!modality}
                    onClick={willReplace ? undefined : () => handleRelabel(entry)}
                  >
                    Назначить
                  </Button>
                );
                return (
                  <List.Item>
                    <Space direction="vertical" size={2} style={{ width: '100%' }}>
                      <Text>{entry.series_description} ({entry.slice_count} срезов)</Text>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {entry.detected_modality
                          ? `Похоже на: ${entry.detected_modality} — ${REASON_LABELS[entry.reason] || entry.reason}`
                          : REASON_LABELS[entry.reason] || entry.reason}
                      </Text>
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
                    </Space>
                  </List.Item>
                );
              }}
            />
          )}
        </div>

        <Divider style={{ margin: '8px 0' }} />

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
      </Space>
    </Modal>
  );
};

export default IncompletePatientDetail;
```

- [ ] **Step 2: Create `IncompletePatients.jsx`**

```jsx
/**
 * Модальное окно со списком сессий текущего запуска, требующих
 * внимания врача — неполные, и полные, где есть альтернативные
 * (исключённые) серии-кандидаты.
 */
import { useState, useEffect } from 'react';
import { Modal, Table, Tag, Space, Button, Alert, Spin, message, Popconfirm } from 'antd';
import { ReloadOutlined, SyncOutlined } from '@ant-design/icons';
import { getIncompletePatients, requeuePipelineRun } from '../services/api';
import IncompletePatientDetail from './IncompletePatientDetail';

const IncompletePatients = ({ runId, visible, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [error, setError] = useState(null);
  const [selectedSession, setSelectedSession] = useState(null);
  const [requeuing, setRequeuing] = useState(false);

  useEffect(() => {
    if (visible && runId) {
      fetchSessions();
    }
  }, [visible, runId]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getIncompletePatients(runId);
      setSessions(data.sessions || []);
    } catch (err) {
      console.error('Ошибка загрузки списка неполных пациентов:', err);
      setError('Не удалось загрузить список');
    } finally {
      setLoading(false);
    }
  };

  const handleRequeue = async () => {
    setRequeuing(true);
    try {
      const result = await requeuePipelineRun(runId);
      message.success(`Обработка перезапущена (новый run_id: ${result.run_id.substring(0, 8)}...)`);
    } catch (err) {
      console.error('Ошибка перезапуска:', err);
      message.error('Не удалось перезапустить обработку');
    } finally {
      setRequeuing(false);
    }
  };

  const columns = [
    { title: 'Пациент', dataIndex: 'original_id', key: 'original_id' },
    { title: 'Дата', dataIndex: 'date', key: 'date' },
    {
      title: 'Статус',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'incomplete' ? 'orange' : 'blue'}>
          {status === 'incomplete' ? 'Неполная' : 'Есть альтернативы'}
        </Tag>
      ),
    },
    {
      title: 'Модальности',
      dataIndex: 'available',
      key: 'available',
      render: (available) => (
        <Space wrap>
          {available.map((m) => <Tag color="green" key={m}>{m}</Tag>)}
        </Space>
      ),
    },
    {
      title: '',
      key: 'actions',
      render: (_, record) => (
        <Button size="small" onClick={() => setSelectedSession(record)}>
          Подробнее
        </Button>
      ),
    },
  ];

  return (
    <Modal
      title="Пациенты, требующие внимания"
      open={visible}
      onCancel={onClose}
      width={900}
      footer={null}
    >
      <Space style={{ marginBottom: 16 }}>
        <Popconfirm
          title="Перезапустить обработку? Уже обработанные пациенты будут пропущены."
          onConfirm={handleRequeue}
          okText="Да"
          cancelText="Нет"
        >
          <Button type="primary" icon={<SyncOutlined />} loading={requeuing}>
            Запустить обработку
          </Button>
        </Popconfirm>
        <Button icon={<ReloadOutlined />} onClick={fetchSessions}>
          Обновить
        </Button>
      </Space>

      {error && <Alert type="error" description={error} showIcon style={{ marginBottom: 16 }} />}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin size="large" />
        </div>
      ) : (
        <Table
          columns={columns}
          dataSource={sessions}
          rowKey={(r) => `${r.patient_id}_${r.session_id}`}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: 'Нет сессий, требующих внимания' }}
        />
      )}

      <IncompletePatientDetail
        runId={runId}
        session={selectedSession}
        visible={!!selectedSession}
        onClose={() => setSelectedSession(null)}
        onActionComplete={fetchSessions}
      />
    </Modal>
  );
};

export default IncompletePatients;
```

- [ ] **Step 3: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors from these two files.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/IncompletePatientDetail.jsx frontend/src/components/IncompletePatients.jsx
git commit -m "feat(frontend): IncompletePatients list modal + IncompletePatientDetail modal"
```

---

### Task 4: Wire into both entry points

**Files:**
- Modify: `frontend/src/components/StageProgress.jsx` (new button, gated on stage 1)
- Modify: `frontend/src/components/ProgressMonitor.jsx` (self-contained modal state, active-run path)
- Modify: `frontend/src/components/PipelineHistory.jsx` (new action-column button, history path)
- Modify: `frontend/src/App.jsx` (lifted modal state, history path)

**Interfaces:**
- Consumes: Task 3's `<IncompletePatients>`.

- [ ] **Step 1: `StageProgress.jsx` — new button gated on stage 1**

Add `TeamOutlined` (or similar — any antd icon distinct from the existing four) to the icon imports (currently lines 5-13), add `onShowIncompletePatients` to the component's props (currently line 15), and add a new gate constant alongside the existing three (currently lines 55-60):

```js
  const showIncompletePatientsButton = stageNumber === 1 && status === 'completed' && onShowIncompletePatients;
```

Add the button inside the existing `<Space>` of action buttons (currently lines 73-108), anywhere among the other three:

```jsx
          {showIncompletePatientsButton && (
            <Button
              type="link"
              size="small"
              icon={<TeamOutlined />}
              onClick={onShowIncompletePatients}
            >
              Неполные пациенты
            </Button>
          )}
```

- [ ] **Step 2: `ProgressMonitor.jsx` — self-contained modal state**

Add import: `import IncompletePatients from './IncompletePatients';`

Add state (alongside the existing three `showX` states, currently lines 26-28):

```js
  const [showIncompletePatients, setShowIncompletePatients] = useState(false);
```

Add handler (alongside `handleShowQualityReport`, currently lines 145-147):

```js
  const handleShowIncompletePatients = () => {
    setShowIncompletePatients(true);
  };
```

Pass the callback into `StageProgress` (currently lines 231-240, add alongside the other three `onShowX` props):

```jsx
            onShowIncompletePatients={stageData.stage_number === 1 ? handleShowIncompletePatients : null}
```

Mount the modal (alongside `<QualityReport>`/`<NIfTIViewer>`/`<ClinicalReport>`, currently lines 260-280):

```jsx
      <IncompletePatients
        runId={runId}
        visible={showIncompletePatients}
        onClose={() => setShowIncompletePatients(false)}
      />
```

- [ ] **Step 3: `PipelineHistory.jsx` — new action-column button**

Add `onShowIncompletePatients` to the component's destructured props (currently line 16).

Add a button inside the "Действия" column's `render` function (currently lines 215-247), alongside the other three, gated only on `record.status === 'completed'` (stage 1 always precedes any completed run, no `current_stage` check needed):

```jsx
          {record.status === 'completed' && (
            <Button
              type="link"
              size="small"
              onClick={() => onShowIncompletePatients(record.run_id)}
            >
              Неполные пациенты
            </Button>
          )}
```

- [ ] **Step 4: `App.jsx` — lifted modal state**

Add import: `import IncompletePatients from './components/IncompletePatients';`

Add state (alongside the existing `historyX`/`showHistoryX` states, currently lines 25-32):

```js
  const [historyIncompletePatientsRunId, setHistoryIncompletePatientsRunId] = useState(null);
  const [showHistoryIncompletePatients, setShowHistoryIncompletePatients] = useState(false);
```

Add handler (alongside `handleShowHistoryQualityReport`, currently lines 58-61):

```js
  const handleShowHistoryIncompletePatients = (runId) => {
    setHistoryIncompletePatientsRunId(runId);
    setShowHistoryIncompletePatients(true);
  };
```

Pass the callback into `<PipelineHistory>` (currently lines 205-209):

```jsx
                      onShowIncompletePatients={handleShowHistoryIncompletePatients}
```

Mount the modal (alongside the other three history modals, currently lines 225-251):

```jsx
            {showHistoryIncompletePatients && (
              <IncompletePatients
                runId={historyIncompletePatientsRunId}
                visible={showHistoryIncompletePatients}
                onClose={() => setShowHistoryIncompletePatients(false)}
              />
            )}
```

- [ ] **Step 5: Lint check**

Run: `cd frontend && npm run lint`
Expected: no new errors from the four modified files.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/StageProgress.jsx frontend/src/components/ProgressMonitor.jsx frontend/src/components/PipelineHistory.jsx frontend/src/App.jsx
git commit -m "feat(frontend): wire IncompletePatients into ProgressMonitor (stage 1) and PipelineHistory"
```

---

### Task 5: Manual end-to-end browser verification

**Files:** none modified — verification only.

- [ ] **Step 1: Start the stack**

```bash
docker compose --profile full up --build
```

(A rebuild is required — `backend/` and `frontend/` are baked into the image, not bind-mounted, per `web.Dockerfile`.)

- [ ] **Step 2: Trigger a real run against BO data with a genuinely incomplete/reconsiderable case**

Use `data/clinical_dicom/BO/test` (5 real patients, includes `BO-68`'s tied-T1-duplicate case confirmed multiple times earlier in this project's history) as the input path via the "Запуск обработки" tab.

- [ ] **Step 3: Verify the active-run entry point**

As soon as stage 1 completes (should be within seconds — this is a 5-patient input), confirm the "Неполные пациенты" button appears next to stage 1 in `ProgressMonitor`, without waiting for later stages. Click it — confirm the list modal opens and shows real sessions (expect `BO-68`'s session, `status: "complete"`, with one `excluded_series` entry for the losing T1 duplicate).

- [ ] **Step 4: Verify the detail modal and relabel (replace) flow**

Click "Подробнее" on `BO-68`'s row. Confirm: available modalities shown as green tags, `missing` empty (it's complete), the excluded T1 duplicate listed with its `detected_modality`/`reason` hint. Select `t1` in the dropdown (should already default to it) and click "Назначить" — confirm the `Popconfirm` fires (since `t1` is already filled) asking to confirm the replace, confirm it, and verify: success message, the list re-fetches, and (optionally) re-run stage 01's dataset_mapping.json check via the same pattern used in earlier verification (`python3 -c "import json; ..."` on the real output path) to confirm the swap landed on disk correctly — matching Task 3 of the original C-backend plan's verification depth, not just trusting the UI's success toast.

- [ ] **Step 5: Verify discard**

Pick a different session (or the same one after undoing, if easy), click "Отбросить сессию" in the detail modal, confirm the `Popconfirm`, confirm it disappears from the list on next fetch.

- [ ] **Step 6: Verify requeue**

Click "Запустить обработку" in the list modal, confirm the `Popconfirm`, confirm success message with a new `run_id`. Check `История запусков` — a new run should appear.

- [ ] **Step 7: Verify the history entry point independently**

Navigate to "История запусков", find the just-completed run, click its own "Неполные пациенты" button (separate from the `ProgressMonitor` one, per the dual-path architecture) — confirm it opens the same modal with current data.

- [ ] **Step 8: Report findings**

Note any visual/UX issues found (this is real, unscripted verification — expect to find at least minor polish issues on a first pass) separately from functional correctness. Functional bugs block completion; polish issues can be logged as follow-ups if minor.
