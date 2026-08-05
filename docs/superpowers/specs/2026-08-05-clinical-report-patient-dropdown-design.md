# Patient dropdown in the standalone clinical report modal

**Date:** 2026-08-05
**Status:** approved

## Goal

`ClinicalReport.jsx` (the standalone clinical-report modal — distinct from the
report embedded under the 3D-visualization modal, `NIfTIViewer.jsx`) currently
renders every patient in a run as one continuous sheet, with no way to jump to
a specific patient. Add a patient-selector dropdown, identical in both places
this modal is used: the active-run tab (`ProgressMonitor.jsx`) and history
(`App.jsx`).

## Current state

- `ClinicalReport.jsx` wraps `ClinicalReportContent` in a `Modal`, passing only
  `runId`, `autoLoad`, `lesionType`. No patient list, no selection state.
- `ClinicalReportContent.jsx` already supports patient-scoped rendering via a
  `selectedPatientId` prop (built earlier for `NIfTIViewer.jsx`'s embedded
  report, same feature request, different location — see
  `docs/superpowers/specs/2026-07-23-glio-patient-id-and-per-patient-report-design.md`).
  It fetches `volumeReports` (GBM) / `lesionStatsReports` (MS) and
  `patientMap` (BIDS id → original id) internally; nothing outside the
  component currently sees that data.
- Both call sites (`ProgressMonitor.jsx:275-280`, `App.jsx:245-250`) invoke the
  same `ClinicalReport` component with the same props — a fix inside it
  applies identically in both places with no separate wiring.

## Design

**`ClinicalReportContent.jsx`**: add an optional `onPatientsChange` callback
prop. A `useEffect` (dependencies: `volumeReports`, `lesionStatsReports`,
`patientMap`, `lesionType`) computes the de-duplicated, ordered list of
patients from whichever report array is active for the lesion type, and calls
`onPatientsChange?.(patients)` with
`[{ patient_id, original_id: patientMap[patient_id] ?? null }, ...]`. Purely
additive — `NIfTIViewer.jsx` doesn't pass this prop and is unaffected.

**`ClinicalReport.jsx`**: becomes stateful.
- `patients` (from `onPatientsChange`) and `selectedPatientId` state, reset
  when `runId` changes.
- On receiving a new patient list, auto-select the first patient (matching
  `NIfTIViewer`'s existing dropdown behavior — the same default already
  shipped for the embedded report).
- Render a `Select` above the report body, options labelled
  `patient_id (original_id)` when an original id is known, else just
  `patient_id`. **Hidden when there are 0–1 patients** — no selector needed
  for a single-patient run.
- Pass `selectedPatientId` through to `ClinicalReportContent` (same prop it
  already consumes).

## Global constraints

- No backend changes — the patient list is derived from data
  `ClinicalReportContent` already fetches.
- No change to `NIfTIViewer.jsx`'s own dropdown/report scoping — this is a
  separate modal or reachable from a different UI, not affected by this work.
- Identical behavior at both call sites by construction (same shared
  component, same props) — no per-caller logic to keep in sync.

## Out of scope

- An "all patients" fallback view — not requested; can be added later if
  needed.
- Any change to the embedded report inside `NIfTIViewer.jsx`, which already
  has its own selection mechanism (the series dropdown).
