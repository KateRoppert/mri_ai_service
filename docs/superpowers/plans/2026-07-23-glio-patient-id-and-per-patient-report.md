# Glio Original-ID Parity + Per-Patient Clinical Report Scoping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Show the original patient ID next to the BIDS `sub-XXX` id for
glioblastoma clinical reports, matching existing MS behavior, everywhere
except validation. (2) Scope the clinical report embedded in the 3D
visualization modal to the patient currently selected in the series dropdown,
instead of rendering every patient in the run as one continuous sheet.

**Architecture:** Two isolated, additive changes to two existing React
components (`ClinicalReportContent.jsx`, `NIfTIViewer.jsx`). No backend
changes, no new dependencies, no new components.

**Tech Stack:** React (Vite), Ant Design (`Tag`, `Alert`).

## Global Constraints

- No backend changes. No new dependencies. No new components.
- Validation (`ValidationPanel.jsx` → `NIfTIViewer` with `kappaReport` set)
  must render identically to today after both features ship.
- `ClinicalReport.jsx` (standalone modal, no series dropdown) must keep
  showing every patient in the run — it never passes `selectedPatientId`.
- **No frontend test framework exists in this project** (`frontend/package.json`
  has no vitest/jest/testing-library — confirmed before writing this plan).
  Introducing one is a separate decision outside this feature's scope
  (YAGNI — two isolated, small, purely-presentational changes don't justify
  standing up new test infrastructure). Verification for each task is
  therefore: `npm run lint`, `npm run build` (both must pass — this is the
  project's actual stated bar: "code must at least compile; check syntax and
  imports"), plus a **manual browser check that the implementing agent cannot
  perform itself** (no browser/screenshot tool available) — call this out
  explicitly to the human partner at the end of each task rather than
  claiming the UI behavior was verified.
- Code comments in English. Conventional-commit messages. One commit per task.

---

### Task 1: Show original patient ID in the GLIO clinical report branch

**Files:**
- Modify: `frontend/src/components/ClinicalReportContent.jsx` (GLIO render
  path, currently lines 673-676)

**Interfaces:**
- Consumes: existing `patientMap` state (line 152, already populated for
  both lesion types by `fetchAllData()`), existing `report.patient_id` loop
  variable inside the GLIO branch's `volumeReports.map(...)`.
- Produces: no new interface — presentational change only.

- [ ] **Step 1: Locate the current GLIO patient/session tag block**

Run: `grep -n "report.patient_id" frontend/src/components/ClinicalReportContent.jsx`
Expected: a match inside the GLIO render path showing:
```jsx
            <div style={{ marginBottom: 16 }}>
              <Tag>{report.patient_id}</Tag>
              <Tag>{report.session_id}</Tag>
            </div>
```

- [ ] **Step 2: Mirror the MS branch's original-ID Tag into the GLIO branch**

Replace:
```jsx
            <div style={{ marginBottom: 16 }}>
              <Tag>{report.patient_id}</Tag>
              <Tag>{report.session_id}</Tag>
            </div>
```
with:
```jsx
            <div style={{ marginBottom: 16 }}>
              {patientMap[report.patient_id] && (
                <Tag color="blue">{patientMap[report.patient_id]}</Tag>
              )}
              <Tag>{report.patient_id}</Tag>
              <Tag>{report.session_id}</Tag>
            </div>
```

This is the exact same pattern already used in the MS branch (lines 544-548:
`{patientMap[patientId] && (<Tag color="blue">{patientMap[patientId]}</Tag>)}`),
applied to the GLIO branch's loop variable (`report.patient_id` instead of
`patientId`).

- [ ] **Step 3: Lint and build**

Run: `cd frontend && npm run lint && npm run build`
Expected: both exit 0, no new errors or warnings introduced.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ClinicalReportContent.jsx
git commit -m "feat(clinical-report): show original patient id for glioblastoma

Mirrors the existing MS behavior (patientMap[...] tag) into the GLIO
render branch. patientMap is already fetched for both lesion types and
stays empty on the Kappa/validation path, so validation is unaffected
by construction — no lesion-type check needed."
```

- [ ] **Step 5: Flag manual verification (cannot be performed by the implementing agent)**

State explicitly to the human partner: this change has not been visually
verified in a browser (no browser tool available to the implementing agent).
Before considering Task 1 done, manually check:
1. Open a completed glioblastoma run's clinical report (active run view or
   history → 3D visualization). Confirm a blue `Tag` with the original
   patient id now appears next to each `sub-XXX` tag.
2. Open the validation view (Каппа-датасет verification). Confirm **no**
   original-id tag appears there — only the anonymous BIDS-derived name, same
   as before this change.

---

### Task 2: Scope the 3D-visualization clinical report to the selected patient

**Files:**
- Modify: `frontend/src/components/NIfTIViewer.jsx` (`ClinicalReportContent`
  embed, currently lines 779-784)
- Modify: `frontend/src/components/ClinicalReportContent.jsx` (prop
  destructuring at line 142; render guards/branches from line 530 onward)

**Interfaces:**
- Consumes: `NIfTIViewer`'s existing `selectedFile` state (line 82, already
  holds `{ patient_id, session_id, modality, ... }` once the file list loads)
  and existing `kappaReport` prop (line 69).
- Produces: `ClinicalReportContent` gains a new optional prop
  `selectedPatientId` (default `null`) — backward compatible; any existing
  caller that doesn't pass it (`ClinicalReport.jsx`) behaves exactly as
  before.

- [ ] **Step 1: Thread `selectedPatientId` from `NIfTIViewer` into `ClinicalReportContent`**

In `frontend/src/components/NIfTIViewer.jsx`, locate the embed (around line
779):
```jsx
              <ClinicalReportContent
                runId={resolvedRunId}
                autoLoad={true}
                lesionType={lesionType}
                kappaEntityInfo={kappaReport}
              />
```
Replace with:
```jsx
              <ClinicalReportContent
                runId={resolvedRunId}
                autoLoad={true}
                lesionType={lesionType}
                kappaEntityInfo={kappaReport}
                selectedPatientId={kappaReport ? undefined : selectedFile?.patient_id}
              />
```

- [ ] **Step 2: Accept the new prop in `ClinicalReportContent`**

In `frontend/src/components/ClinicalReportContent.jsx`, locate line 142:
```jsx
const ClinicalReportContent = ({ runId, autoLoad = false, lesionType = 'glioblastoma', kappaEntityInfo = null }) => {
```
Replace with:
```jsx
const ClinicalReportContent = ({ runId, autoLoad = false, lesionType = 'glioblastoma', kappaEntityInfo = null, selectedPatientId = null }) => {
```

- [ ] **Step 3: Compute scoped report lists after the existing top-level guards**

Locate (around line 530):
```jsx
  if (!loaded || (lesionType !== 'multiple_sclerosis' && volumeReports.length === 0)) {
    return null;
  }

  // ===== MS RENDER PATH =====
```
Insert the scoped-list computation between the guard and the MS render path
comment:
```jsx
  if (!loaded || (lesionType !== 'multiple_sclerosis' && volumeReports.length === 0)) {
    return null;
  }

  // Scope to the patient selected in NIfTIViewer's series dropdown, when
  // provided. Falsy selectedPatientId (e.g. the standalone ClinicalReport
  // modal, which has no dropdown) leaves the full run unfiltered — same
  // behavior as before this feature.
  const scopedVolumeReports = selectedPatientId
    ? volumeReports.filter((r) => r.patient_id === selectedPatientId)
    : volumeReports;
  const scopedLesionStatsReports = selectedPatientId
    ? lesionStatsReports.filter((r) => r.patient_id === selectedPatientId)
    : lesionStatsReports;

  // ===== MS RENDER PATH =====
```

- [ ] **Step 4: Use the scoped list and add the empty-state Alert in the MS branch**

Locate (around line 536-539):
```jsx
  if (lesionType === 'multiple_sclerosis') {
    if (!loaded || lesionStatsReports.length === 0) return null;
    return (
      <>
        {groupByPatient(lesionStatsReports).map(([patientId, sessions]) => (
```
Replace with:
```jsx
  if (lesionType === 'multiple_sclerosis') {
    if (!loaded || lesionStatsReports.length === 0) return null;
    if (selectedPatientId && scopedLesionStatsReports.length === 0) {
      return (
        <Alert
          type="info"
          message="Отчёт для выбранного пациента пока недоступен"
          showIcon
        />
      );
    }
    return (
      <>
        {groupByPatient(scopedLesionStatsReports).map(([patientId, sessions]) => (
```

- [ ] **Step 5: Use the scoped list and add the empty-state Alert in the GLIO branch**

Locate (around line 660-663):
```jsx
  // ===== GLIO RENDER PATH =====
  return (
    <>
      {volumeReports.map((report, idx) => {
```
Replace with:
```jsx
  // ===== GLIO RENDER PATH =====
  if (selectedPatientId && scopedVolumeReports.length === 0) {
    return (
      <Alert
        type="info"
        message="Отчёт для выбранного пациента пока недоступен"
        showIcon
      />
    );
  }
  return (
    <>
      {scopedVolumeReports.map((report, idx) => {
```

- [ ] **Step 6: Verify `Alert` is already imported**

Run: `grep -n "^import.*Alert" frontend/src/components/ClinicalReportContent.jsx`
Expected: `Alert` is already in the `antd` import on line 10 (it's used
today for the error state) — no import change needed. If this check fails,
add `Alert` to the existing `import { ... } from 'antd';` line.

- [ ] **Step 7: Lint and build**

Run: `cd frontend && npm run lint && npm run build`
Expected: both exit 0, no new errors or warnings introduced.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/ClinicalReportContent.jsx frontend/src/components/NIfTIViewer.jsx
git commit -m "feat(clinical-report): scope embedded report to the selected patient

NIfTIViewer's series dropdown already resolves to a file object with a
patient_id field (selectedFile.patient_id). Thread it into
ClinicalReportContent as selectedPatientId and filter volumeReports /
lesionStatsReports by it before rendering, so the 3D-visualization
modal shows one patient's report at a time instead of every patient in
the run concatenated. Falsy selectedPatientId (the standalone
ClinicalReport modal, which has no dropdown) is unaffected. Explicitly
undefined on the validation/Kappa path (kappaReport set), which already
renders a single pre-normalized entity regardless."
```

- [ ] **Step 9: Flag manual verification (cannot be performed by the implementing agent)**

State explicitly to the human partner: this change has not been visually
verified in a browser. Before considering Task 2 done, manually check, using
a run with 2+ patients that have clinical reports:
1. Open the 3D visualization modal for that run. Confirm the report shown
   matches whichever patient is selected in the top-left dropdown on open.
2. Switch the dropdown to a different modality/session of the **same**
   patient (e.g. `sub-001` t1 → `sub-001` t2). Confirm the report does **not**
   change.
3. Switch the dropdown to a **different** patient (e.g. `sub-001` → `sub-002`,
   any modality). Confirm the report changes to that patient's data.
4. Open the standalone clinical-report modal (not the 3D viewer) for the same
   run, if reachable from the UI. Confirm it still shows every patient in the
   run, unfiltered (regression check — this modal has no dropdown and must be
   unaffected).
5. Open the validation view. Confirm behavior is unchanged from before this
   feature (single anonymous entity, no filtering artifacts).

---

## Verification After Both Tasks

- [ ] `cd frontend && npm run lint && npm run build` — final combined check,
  both must pass.
- [ ] All 5 manual checks from Task 2 Step 9, plus both checks from Task 1
  Step 5 — performed by the human partner (or delegate to a browser-capable
  agent if available), not claimed as done by the implementing agent.
