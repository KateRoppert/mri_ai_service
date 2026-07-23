# Glio original-patient-ID parity + per-patient clinical report scoping

**Date:** 2026-07-23
**Status:** design — approved, awaiting SPEC self-review sign-off

## Goal

Two small, related frontend fixes to the clinical report views:

1. **Feature 1:** show the original (pre-BIDS) patient ID alongside the BIDS
   `sub-XXX` id for glioblastoma (GBM), matching what already exists for
   multiple sclerosis (MS) — everywhere except the validation view, which must
   stay anonymous.
2. **Feature 2:** in the 3D-visualization modal (`NIfTIViewer`), scope the
   embedded clinical report to the patient currently selected in the
   series dropdown, instead of rendering every patient in the run as one
   continuous sheet. Switching modality/session for the same patient keeps
   the same report visible; switching patient switches the report.

## Current state (verified against source, not paraphrased)

- The BIDS↔original-id mapping is **already lesion-type-agnostic** on the
  backend: `scripts/01_reorganize_folders.py` writes
  `bids_organized/dataset_mapping.json` for every run regardless of
  `lesion_type`; `backend/pipeline_manager.py:564-593`
  `get_patient_map(output_path)` reads it; `GET
  /api/run/{run_id}/patient-map` (`backend/app.py:859-870`) exposes it;
  frontend `getPatientMap` (`frontend/src/services/api.js:91-93`) fetches it.
  No backend change needed for Feature 1.
- The gap is entirely in `frontend/src/components/ClinicalReportContent.jsx`:
  - `patientMap` state (line 152) is fetched once in `fetchAllData()`
    (line 204) whenever the **local/clinical path** runs (`!kappaEntityInfo`).
  - MS render path (lines 544-548) already shows it:
    ```jsx
    {patientMap[patientId] && (
      <Tag color="blue">{patientMap[patientId]}</Tag>
    )}
    <Tag>{patientId}</Tag>
    ```
  - GLIO render path (lines 673-676) does not:
    ```jsx
    <div style={{ marginBottom: 16 }}>
      <Tag>{report.patient_id}</Tag>
      <Tag>{report.session_id}</Tag>
    </div>
    ```
- Validation stays anonymous **by construction, not by a lesion-type check**:
  `ValidationPanel.jsx` opens `NIfTIViewer` with `kappaReport={viewerEntityInfo}`;
  inside `ClinicalReportContent`, the `kappaEntityInfo` branch
  (`normalizeKappaEntity`, lines 79-140) never populates `patientMap` — the
  reset effect (lines 178-188) only runs `setPatientMap({})` and is guarded by
  `!kappaEntityInfo`. This mechanism is identical for MS and GLIO, so mirroring
  the Tag into the GLIO branch automatically preserves the validation
  exclusion — no extra code required.
- **`NIfTIViewer.jsx`** — the "3D visualization modal" — holds a `Select`
  dropdown (lines 564-576) whose `value` is the raw preprocessed filename, but
  `handleFileChange` (lines 336-355) already resolves it to a full file object
  (`NIfTIFile` from `backend/models.py:27-41`) with **separate** fields
  `patient_id`, `session_id`, `modality` — stored in `selectedFile` state
  (line 82). After the file list loads, `setSelectedFile(filesData[0])` (line
  214) fires immediately, so `selectedFile` is never null once loaded — no
  "nothing selected" state to design around.
- All four report types returned by the per-run endpoints
  (`GET /api/volume-reports/{run_id}`, `/lobar-reports/`, `/lesion-stats/`,
  `/mcdonald-reports/`, `backend/app.py:737,770,803,836`) use a consistently
  named and consistently formatted `patient_id: str` field (e.g. `"sub-001"`,
  confirmed against `backend/models.py` and the `patient_id = parts[0]  #
  sub-001` assignment at `backend/app.py:663`) — identical format to
  `selectedFile.patient_id`. No format translation needed to filter by it.
- `ClinicalReportContent` currently receives `runId`, `autoLoad`, `lesionType`,
  `kappaEntityInfo` (`NIfTIViewer.jsx:779-784`) — no patient-scoping prop
  exists today. It renders **all** patients/sessions in the run:
  GLIO via `volumeReports.map(...)` (line 663), MS via
  `groupByPatient(lesionStatsReports).map(...)` (line 539).
- `frontend/src/components/ClinicalReport.jsx` is a **separate**, simpler
  standalone modal (`<ClinicalReportContent runId={runId} autoLoad={visible}
  lesionType={lesionType} />`) with **no dropdown/series selector at all** —
  it is not "the 3D visualization modal" the user described and is
  intentionally out of scope for Feature 2 (it has nothing to scope by).
  It is in scope for Feature 1 (inherits the GLIO-branch fix automatically,
  since it renders the same shared component).
- "История запусков" (`PipelineHistory.jsx`) never displays any patient-level
  ID (original or BIDS) — it is a table of runs, not patients. Not affected by
  either feature; mentioned only to confirm it needs no change.

## Design

### Feature 1 — mirror the original-ID Tag into the GLIO branch

In `ClinicalReportContent.jsx`, change the GLIO render path
(currently lines 673-676) to match the MS branch's pattern:

```jsx
<div style={{ marginBottom: 16 }}>
  {patientMap[report.patient_id] && (
    <Tag color="blue">{patientMap[report.patient_id]}</Tag>
  )}
  <Tag>{report.patient_id}</Tag>
  <Tag>{report.session_id}</Tag>
</div>
```

`patientMap` is already fetched unconditionally by lesion type in
`fetchAllData()` — no fetch-path change needed. Validation stays anonymous
because `patientMap` is never populated on the Kappa path, regardless of
lesion type (see Current State above).

### Feature 2 — scope the embedded report to the selected patient

**`NIfTIViewer.jsx`** (embed at lines 779-784): add a new prop, computed from
existing state — no new state needed:

```jsx
<ClinicalReportContent
  runId={resolvedRunId}
  autoLoad={true}
  lesionType={lesionType}
  kappaEntityInfo={kappaReport}
  selectedPatientId={kappaReport ? undefined : selectedFile?.patient_id}
/>
```

`undefined` whenever `kappaReport` is set (validation path) — an explicit,
redundant-but-cheap guarantee that Feature 2 never touches validation, on top
of the fact that the Kappa path already renders a single pre-normalized
entity regardless.

**`ClinicalReportContent.jsx`**:
1. Accept a new prop: `selectedPatientId = null`.
2. Immediately before the two render branches (after the existing
   `loading`/`error`/`!loaded` guards, which must keep checking the
   **unfiltered** arrays — an empty run has nothing to scope, that is a
   different state than "this run has data, just not for this patient"),
   compute scoped copies:
   ```js
   const scopedVolumeReports = selectedPatientId
     ? volumeReports.filter(r => r.patient_id === selectedPatientId)
     : volumeReports;
   const scopedLesionStatsReports = selectedPatientId
     ? lesionStatsReports.filter(r => r.patient_id === selectedPatientId)
     : lesionStatsReports;
   ```
3. GLIO branch: replace `volumeReports.map(...)` with
   `scopedVolumeReports.map(...)`.
4. MS branch: replace `groupByPatient(lesionStatsReports).map(...)` with
   `groupByPatient(scopedLesionStatsReports).map(...)`.
5. `lobarReports` and `mcdonaldReports` are **not** filtered — both are only
   ever consulted via `.find(x => x.patient_id === report.patient_id &&
   x.session_id === report.session_id)` against an already-scoped outer
   report, so they resolve correctly unfiltered (and filtering them would be
   redundant work with no behavioral effect).
6. When `selectedPatientId` is falsy (the `ClinicalReport.jsx` standalone
   modal never passes it), `scopedX === X` — behavior is byte-for-byte
   unchanged from today.
7. **Empty-for-this-patient state:** if `selectedPatientId` is set, the
   unfiltered array is non-empty (run has data), but the scoped array is
   empty (no report for this specific patient yet — e.g. a later pipeline
   stage hasn't produced this patient's report), render an
   `<Alert type="info" message="Отчёт для выбранного пациента пока
   недоступен" showIcon />` instead of silently rendering nothing. Applies
   independently to the GLIO branch (checking `scopedVolumeReports`) and the
   MS branch (checking `scopedLesionStatsReports`).

### Why client-side filtering, not new backend endpoints

The report arrays are already fetched in full for the run (one `Promise.all`
per modal open); the data volume per run is small (a handful of patients).
Adding `?patient_id=` query params to four endpoints would touch backend
routes, `pipeline_manager.py` report readers, and four Pydantic response
paths for a payload-size saving that doesn't matter at this scale. Filtering
an already-in-memory array client-side is the minimal correct change.

## Files touched

- `frontend/src/components/ClinicalReportContent.jsx` — both features.
- `frontend/src/components/NIfTIViewer.jsx` — Feature 2 prop threading only.

No backend changes. No new dependencies. No database/schema changes.

## Global constraints

- Validation (`ValidationPanel.jsx` → `NIfTIViewer` with `kappaReport` set)
  must render identically to today after both features ship — verified by
  the `kappaReport ? undefined : ...` guard (Feature 2) and by `patientMap`
  never populating on the Kappa path (Feature 1, pre-existing mechanism).
- `ClinicalReport.jsx` (standalone modal, no dropdown) keeps showing every
  patient in the run — Feature 2 must not change its behavior (it simply
  never passes `selectedPatientId`).
- `PipelineHistory.jsx` is unaffected by both features (no patient-level ID
  display there today; out of scope).
- No change to report data shape/fields — filtering only, no new backend
  response fields.

## Out of scope

- Session-level scoping within a patient (the user's spec explicitly groups
  by patient across all sessions/modalities — MS already does this via
  `groupByPatient`; GLIO's per-report loop is already one row per
  patient+session, and scoping to the selected patient naturally shows all of
  that patient's sessions, matching the requirement).
- Any change to how `patientMap`/original IDs are populated or stored —
  purely a display-layer fix.
- Any change to `ClinicalReport.jsx`'s lack of a patient selector — the user
  only described the 3D-visualization modal.
