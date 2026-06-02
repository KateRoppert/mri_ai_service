# Design: Lesion-Type-Aware Pipeline

**Branch:** `feat/lesion-type-aware-pipeline`
**Date:** 2026-06-02
**Author:** Kate Roppert
**Closes:** KI-004, KI-005, KI-016, KI-019

---

## 1. Context

The MAS infrastructure (Stages 1–3) and post-MAS cleanup (`chore/post-mas-cleanup`) are
complete and merged into `main`. The MS pipeline runs end-to-end without hard blockers,
but several stages are either unaware of `lesion_type` or contain glio-only assumptions
that cause suboptimal behavior for MS (wasted processing time, incorrect fallbacks,
glio-only clinical reports).

This branch makes MS a true first-class citizen across all pipeline stages and the
frontend, and closes the partially-open KI-016 (lesion_types.yaml not yet wired to code).

**Out of scope (deferred):**
- KI-001 / KI-027 — ModalityDetector confidence scoring and series selection heuristics
- KI-010 — GPU sharing coordination
- Everything else tagged `feat/prod-readiness` or `feat/mas-coordinator` in KNOWN_ISSUES.md

---

## 2. Approach

Work stage-by-stage in pipeline order (01 → 03 → 04 → 05 → 06–08 → backend → frontend).
Each stage is independently testable by disabling others in the orchestrator config.
After each stage fix, run a smoke test on the MS case.

---

## 3. Section 1 — Config migration (KI-016)

### Problem

`lesion_types.yaml` was created in `chore/post-mas-cleanup` but is not consumed by any
pipeline stage. Each stage has its own hardcoded constants:
- `CompletenessChecker.LESION_TYPE_MODALITIES` in Stage 01
- `modalities: [t1c, t1, t2, t2fl]` default in Stage 05
- No lesion_type awareness at all in Stage 04

### Solution

Add `load_lesion_type_config(lesion_type: str) -> dict` to `utils/config_loader.py`.
Returns the full entry from `lesion_types.yaml` for the given lesion type.

```python
def load_lesion_type_config(lesion_type: str) -> dict:
    """
    Load per-lesion-type configuration from configs/lesion_types.yaml.
    Returns required_modalities, reference_modality, reports.
    """
```

All pipeline stages replace their hardcoded modality constants with a call to this
function. Single source of truth — adding a new lesion type requires only a YAML edit.

---

## 4. Section 2 — Pipeline stage adaptation

### Stage 01 (`01_reorganize_folders.py`) — KI-016 wiring only

Already adapted for MS in `chore/post-mas-cleanup`. Only change: replace the hardcoded
`LESION_TYPE_MODALITIES` dict in `CompletenessChecker` with `load_lesion_type_config()`.
No behavioral change, pure DRY refactor.

### Stage 03 (`03_convert_to_nifti.py`) — no changes needed

Converts whatever DICOM files Stage 01 organized into BIDS. T1c simply will not be
present in the BIDS folder for MS patients, so Stage 03 processes only the correct
modalities without modification.

### Stage 04 (`04_assess_quality.py`) — KI-004

**Problem:** No `--lesion-type` argument. Stage 04 processes whatever NIfTI files exist,
so it does not crash for MS, but it has no explicit knowledge of which modalities are
expected for which disease.

**Changes:**
1. Add `--lesion-type` to argparse (same pattern as Stage 01, 06, 07, 08).
2. Read `required_modalities` via `load_lesion_type_config()`.
3. Use the modality list when building quality assessment scope and completeness report.
4. Add `--lesion-type` to orchestrator's Stage 04 command builder.

### Stage 05 (`05_preprocessing.py`) — KI-005

**Problem:** Two glio-only assumptions:

1. `modalities` defaults to `['t1c', 't1', 't2', 't2fl']` (line 735). For MS, T1c does
   not exist. The current code gracefully skips missing files, but T1c remains in the
   processing list and the code iterates over it unnecessarily. Fixing this explicitly
   saves ~25% of Stage 05 runtime for MS (T1c N4 bias correction alone takes ~280s).

2. The reference modality fallback hardcodes `'t1c'` in two places (lines 303 and 307).
   This contradicts `preprocessing_config.yaml` which already sets `reference_modality: "t1"`.
   The fallback fires only when the registration step is absent from config or has no
   `reference_modality` key — but if it fires for an MS case without T1c files, Stage 05
   would skip the subject with `missing_reference_modality_t1c`.

**Changes:**
1. Add `--lesion-type` to argparse.
2. Read `required_modalities` and `reference_modality` via `load_lesion_type_config()`.
3. Override `modalities` list (from config) with the lesion-type-specific list. For MS:
   `[t1, t2, t2fl]`; for glio: `[t1c, t1, t2, t2fl]`.
4. Replace both `'t1c'` fallbacks with `lt_config['reference_modality']`
   (= `'t1'` for both glio and MS per `lesion_types.yaml`).
5. Add `--lesion-type` to orchestrator Stage 05 command builder.

### Stages 06 / 07 / 08 — verify only

Already parameterized with `--lesion-type` from `feat/mas-refactor`. Run a smoke MS
pass to confirm no regressions. Fix only if a specific issue surfaces.

---

## 5. Section 3 — Backend for MS report

### Lesion count computation

**Where:** Stage 08 (`08_lobar_localization.py`).

Stage 08 already reads the segmentation mask for lobar atlas mapping. Add a branch:
if `lesion_type == 'multiple_sclerosis'`, run `scipy.ndimage.label()` on the binary
mask and compute connected component statistics.

Output file: `lesion_stats_report.json` (written alongside existing `lobar_report.json`):

```json
{
  "patient_id": "sub-P000915",
  "session_id": "ses-002",
  "lesion_count": 14,
  "total_volume_cm3": 3.21,
  "mean_lesion_volume_cm3": 0.23,
  "lesion_volumes_cm3": [0.12, 0.45, 0.08, ...]
}
```

### New backend endpoints

`GET /api/reports/lesion-stats/{run_id}`
Returns lesion_stats_report.json contents for each patient/session in the run.
Pattern follows existing `get_volume_reports` / `get_lobar_reports`.

`GET /api/reports/longitudinal/{patient_id}?lesion_type=multiple_sclerosis`
Queries `patient_registry` (SQLite) for all sessions matching `patient_id` + `lesion_type`,
sorted by `scan_date`. For each session reads `lesion_stats_report.json` (MS) or
volume_report.txt (glio) to get total volume. Returns:

```json
[
  {"session_id": "ses-001", "scan_date": "2022-01-18", "total_volume_cm3": 2.14, "lesion_count": 11},
  {"session_id": "ses-002", "scan_date": "2023-03-25", "total_volume_cm3": 3.21, "lesion_count": 14}
]
```

Appears in the UI only when 2+ sessions are returned.

---

## 6. Section 4 — Frontend

### `NIfTIViewer.jsx` — dynamic colormap

**Problem:** Colormap hardcodes 4 glio classes (NCR/ED/NET/ET).

**Solution:** Accept `lesionType` prop (already available in `ValidationPanel` from the
run record). Build colormap from lesion type:
- `multiple_sclerosis` → binary colormap: 0=background, 1=lesion (`#52c41a`)
- `glioblastoma` → existing 4-class palette (NCR/ED/NET/ET), unchanged

This is simpler than threading `outputClasses` from the inference result (which is not
persisted in the database after pipeline completion) and correctly captures the fixed
class structure of each model.

### `ClinicalReportContent.jsx` — lesion_type routing

**Problem:** Monolithic component, fully glio-specific (NCR/ED/NET/ET volumes, CE+/CE−).

**Solution:** Add `lesionType` prop. Split render path:

```jsx
if (lesionType === 'multiple_sclerosis') {
  return <MsReportSection data={lesionStats} lobarData={lobarReports} />;
}
return <GbmReportSection volumeReports={volumeReports} lobarReports={lobarReports} />;
```

`GbmReportSection` — current code extracted into a sub-component, no behavioral change.
`MsReportSection` — new sub-component (same file) showing:
- Total lesion volume (Statistic)
- Lesion count (Statistic)
- Lobar distribution table (existing lobar_report, already works for MS)
- `LongitudinalTimeline` (see below, only if 2+ sessions found)

### `LongitudinalTimeline` — new sub-component

Located alongside `ClinicalReport.jsx`. Fetches
`GET /api/reports/longitudinal/{patient_id}?lesion_type=...` on mount. If API returns
fewer than 2 rows, renders nothing. Otherwise renders a simple Ant Design table:

| Дата | Сессия | Объём (см³) | Кол-во очагов |
|------|--------|------------|---------------|
| 2022-01-18 | ses-001 | 2.14 | 11 |
| 2023-03-25 | ses-002 | 3.21 | 14 |

Δ-column (volume change vs. previous session) added if 3+ sessions.

---

## 7. Work order summary

| # | Area | Files | Closes |
|---|------|-------|--------|
| 1 | Config utility | `utils/config_loader.py` | KI-016 |
| 2 | Stage 01 wiring | `scripts/01_reorganize_folders.py` | KI-016 |
| 3 | Stage 04 | `scripts/04_assess_quality.py`, `orchestrator.py` | KI-004 |
| 4 | Stage 05 | `scripts/05_preprocessing.py`, `orchestrator.py` | KI-005 |
| 5 | Stages 06–08 smoke | verify only | — |
| 6 | Stage 08 lesion count | `scripts/08_lobar_localization.py` | KI-019 (partial) |
| 7 | Backend endpoints | `backend/app.py` | KI-019 |
| 8 | NIfTIViewer colormap | `frontend/src/components/NIfTIViewer.jsx` | SPEC §5 |
| 9 | ClinicalReportContent routing | `frontend/src/components/ClinicalReportContent.jsx` | KI-019 |
| 10 | LongitudinalTimeline | `frontend/src/components/LongitudinalTimeline.jsx`, `backend/app.py` | KI-019 |

---

## 8. Testing

Each pipeline stage is tested independently (disable others in orchestrator config).
Smoke test sequence per stage:
1. `python3 -m py_compile <script>` — syntax check
2. Run stage on MS case (P000915), verify log output
3. Run stage on glio case (UPENN-GBM), verify no regression

Frontend: manual test in browser after dev server start.

---

## 9. Commit conventions

`feat(stage04): ...`, `fix(stage05): ...`, `refactor(config): ...` etc.
Each commit covers one completed sub-task with a passing smoke test.

---

*Document status: approved. Next step — implementation plan.*
