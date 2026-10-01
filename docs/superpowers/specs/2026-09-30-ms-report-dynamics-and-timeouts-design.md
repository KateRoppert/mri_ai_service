# MS Report Dynamics, Mask Colour, Stall Timeout — Design

**Date:** 2026-09-30
**Branch:** `fix/ms-report-dynamics-and-timeouts` (from `main`)
**Type:** Bugfixes (UI + backend) + one KNOWN_ISSUES entry

Four items found while running the SibBMS MS patient `P000010` (2 sessions) on
the demob laptop, run `e8d73f9c` in `demo_workspace/input/30_09_1616`.

## 1. "Dynamics between sessions" missing when the report is opened from the run tab

### Symptom

For an MS run, the report under the visualization opened from the main run tab
(`ProgressMonitor`) has no "📈 Динамика между сессиями" block. The same run
opened from the history tab shows it.

### Cause

The dynamics block (`LongitudinalTimeline`) is fed by
`GET /api/longitudinal/{patient}`, which builds points from
`patient_registry` rows. Those rows are written **only by the Kappa uploader**
(`backend/kappa_uploader.py:567`), which runs **after** the run completes.
For this run: completed `09:45:44`, registry rows written `09:45:54` and
`09:46:05`. Until then the endpoint answers 404 ("need >= 2") and the timeline
renders nothing.

That alone would be a short window. It becomes permanent in the run tab
because of component lifetime:

| Where | How the viewer / report is mounted | Effect |
|---|---|---|
| History tab (`App.jsx:546`, `:558`) | `{show && <NIfTIViewer/>}` — created on each open | Fresh fetch every time → works |
| Run tab (`ProgressMonitor.jsx:529`, `:537`) | Always mounted, only `visible` toggles; the Modal keeps its children | Report (`loaded` flag) and timeline (`useEffect` on `[patientId, lesionType, runId]`) fetch **once** and never again |

The visualization button appears as soon as UI stage 6 (inverse transform)
completes — before stage 08 and before the Kappa upload. Any open before the
registry is written freezes an empty timeline for the life of the tab.

Verified: the endpoint returns both sessions for every run id of this output
folder now, so the backend data is correct; only the run-tab view is stale.

### Fix

Mount `NIfTIViewer` and `ClinicalReport` in `ProgressMonitor` only while open
(`{showVisualization && …}`, `{showClinicalReport && …}`), the same as `App.jsx`
already does for history. Each open then fetches current data. `validationRef`
lives in `ProgressMonitor` state, so it survives remounts.

### Out of scope (recorded, not fixed)

Dynamics depend on a successful Kappa upload: a run with no Kappa session or
a failed upload never gets registry rows, so it never shows dynamics anywhere.
Decoupling the registry from the upload is a separate change → new KI entry.

## 2. MS mask colour: green → red

MS lesions are small; the current green (`rgb(82,196,26)`) is hard to see on
grey MRI. Change the MS colormap (`createMsColormap`, `NIfTIViewer.jsx:40`) and
the MS legend entry (`NIfTIViewer.jsx:766`) to pure red `rgb(255,0,0)`.
GBM colours are untouched (red there means enhancing tumour, but the two
lesion types never share a view). Slicer segment colours are not changed.

## 3. KNOWN_ISSUES: memory planning on small hosts (new KI-058)

Stage 07 on the laptop (15 GiB RAM) caused a **host-wide** OOM on 2026-09-30:

- `utils/resource_planner.py` sizes workers from the container cgroup limit
  only. The compose cap is `20g`, above the laptop's physical RAM, so the
  planner saw an 18.3 GB budget and started 2 workers.
- The stage 07 cost estimate (3.85 GB/worker) is below the measured peak on
  SibBMS: ~5.5 GB per worker while warping the mask onto native `t2fl`.

Record both, plus the laptop workaround (`mem_limit: 9g` in the local
`docker-compose.override.yml`). Proper fix candidates: planner also caps by
host `MemAvailable`; re-measure `k_stage` for 07. Doc-only in this branch.

Also add **KI-059**: longitudinal dynamics require a completed Kappa upload
(from item 1).

## 4. Pipeline timeout does not fit multi-session patients

### Problem

`estimate_pipeline_timeout` = `1200 s + 90 s × (top-level subdirectories of the
input)`. It counts folders, not work:

- a longitudinal MS patient is one folder but N sessions;
- one real session of SibBMS takes ~5 min in stage 05 alone, 7–8 min end-to-end
  (GBM 1–2 min), so 90 s is already short for one session;
- the folder count depends on how the input is laid out (a patient folder with
  study subfolders counts as several "patients").

On 2026-09-28 a 3-patient / 11-session run was SIGKILLed at exactly
1200 + 3×90 = 1470 s, mid-stage 05, with nothing in any log.

Estimating better needs a DICOM scan before the run and a per-lesion-type
cost table — and still breaks on the next dataset with a different resolution
(see KI-042).

### Design: stall timeout instead of a total-duration timeout

The timeout exists to catch **hung** runs (KI-052), not long ones. A long,
healthy run keeps producing files; a hung one does not. So:

- Kill the run only when nothing under its `output_path` (logs and stage
  outputs) has been modified for `pipeline_stall_timeout_seconds`
  (default **3600**, env `PIPELINE_STALL_TIMEOUT_SECONDS`).
- Watch the whole output tree, not just logs: in parallel mode worker logs do
  not reach the stage log file (KI-032), but every stage writes its outputs
  per session.
- The backend waits in a loop: `process.communicate(timeout=60)`; on each
  `TimeoutExpired` check the newest mtime under `output_path`. Retrying
  `communicate` after `TimeoutExpired` loses no output (documented behaviour),
  so the stdout/stderr pipes keep being drained.
- On a stall: write a line to `{output}/logs/pipeline_master.log`
  ("остановлен: нет активности N мин") **before** the group kill, and store
  the same text as `error_message`. Today the kill leaves no trace in any log.
- Remove `estimate_pipeline_timeout` and the two per-patient settings; replace
  their test with tests for the activity check and the stall decision.
- No absolute cap: the Stop button covers "I want it to end".

Cost of the check: one `os.scandir` walk of the output tree per minute —
seconds even for ~50k files on the 175-patient BO dataset.

## Testing

- Backend (pytest, next to the code): newest-mtime helper (empty dir, nested
  files, missing dir); stall loop with a fake process — finishes normally,
  stalls → killed + log line + `error_message`, active output → not killed.
- Frontend: `npm run lint`; manual check on run `30_09_1616`: open the viewer
  from the run tab → dynamics table with 2 sessions; MS mask is red; legend red.
- Existing backend suite stays green.
