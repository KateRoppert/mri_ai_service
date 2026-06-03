# Design: MS Clinical Report Redesign (variant A)

**Branch:** `feat/lesion-type-aware-pipeline`
**Date:** 2026-06-03
**Author:** Kate Roppert
**Relates to:** KI-019 (clinical report for MS)

---

## 1. Context

The MS pipeline produces lesion statistics (`compute_lesion_stats` in Stage 08)
and a Kappa-sourced clinical report renders in the UI. But the MS render path in
`ClinicalReportContent` currently **reuses the glioblastoma layout**: CE+/CE−
volumes, NCR/ED/NET/ET class detail, and cortical-lobe localization with
glio-specific columns. None of these are clinically meaningful for multiple
sclerosis.

This redesign rebuilds the MS report around what MS clinicians actually need and
do manually, so the tool saves them time rather than producing decorative charts.

### Clinical grounding

What MS radiologists/neurologists do by hand that we can automate:
- **Count lesions** (T2/FLAIR) — tedious and error-prone for 20–50 small lesions.
- **Estimate total lesion volume (burden)** — correlates with disability; manual
  volumetry is too slow for routine use, so it is usually eyeballed. A computed
  number is something they normally cannot get.
- **Measure a specific lesion** — manual ROI measurement when they want to know
  "how big is *this* one." Hover-to-read replaces that, tied to the actual lesion.

Out of scope here (deferred — see §6): McDonald-criteria localization
(periventricular / juxtacortical / infratentorial) and new/enlarging-lesion
detection between sessions. Both require pipeline computation and go to a separate
branch `feat/ms-clinical-metrics` (variant C).

---

## 2. Scope

This branch (variant A): a frontend report redesign **plus a small Stage 08
addition** to support per-lesion hover. No McDonald localization, no longitudinal
lesion-level diffing.

---

## 3. MS report composition

The MS render path in `ClinicalReportContent` is rebuilt with these sections, top
to bottom. The glioblastoma render path is unchanged.

### 3.1 Очаговая нагрузка (hero metrics)
Three `Statistic` widgets from `lesion_stats`:
- **Количество очагов** — `lesion_count`
- **Суммарный объём поражения** — `total_volume_cm3` (green)
- **Средний объём очага** — `mean_lesion_volume_cm3`

### 3.2 Характер поражения (burden character)
Three size-category chips computed from `lesion_volumes_cm3`:
- **крупные** ≥ 0.1 см³
- **средние** 0.01–0.1 см³
- **точечные** < 0.01 см³

Each chip shows the count of lesions in that band. Thresholds are pragmatic
(a ~3 mm punctate lesion ≈ 0.014 см³); define them as named constants so they are
easy to tune.

### 3.3 Объём очага по наведению (hover)
In the viewer, hovering over a lesion shows a tooltip with that lesion's volume.
Replaces manual ROI measurement and ties the number to the lesion being examined.
Requires the labeled lesion mask + label→volume map (see §4).

### 3.4 Объёмы всех очагов (per-lesion table, collapsible)
A collapsed-by-default, scrollable table (`№`, `объём см³`) listing every lesion,
sorted descending. For protocol/documentation export. Built from
`lesion_volumes_cm3` (already sorted descending).

### 3.5 Динамика между сессиями (longitudinal)
Keep the existing `LongitudinalTimeline` (count + total volume + Δ across
sessions). Unchanged. Full new/enlarging detection is variant C.

### Removed from the MS path
- CE+/CE− clinical volumes (glio Section 1)
- Class detail table (NCR/ED/NET/ET)
- Cortical-lobe localization (Harvard-Oxford) — until variant C provides proper
  MS localization

---

## 4. Stage 08 changes (hover support)

`compute_lesion_stats` already runs `scipy.ndimage.label` to count connected
components. Two additions:

1. **Save the labeled mask** as `*_lesion_labels.nii.gz` alongside the binary
   segmask — each lesion carries its integer label (1..N).
2. **Extend `lesion_stats_report.json`** with a label→volume mapping
   (`lesion_volumes_by_label`: `{label: volume_cm3}`), so the viewer can map a
   hovered voxel's label to its volume. Keep `lesion_volumes_cm3` (sorted desc)
   for the table and burden categories.

The labeled array already exists inside `compute_lesion_stats`; this is mostly
saving it and keying volumes by label instead of only sorting them.

---

## 5. Data-source paths for hover

The report and the labeled mask must be reachable in both contexts:

- **Run / History (local source):** the viewer resolves `runId` and loads the
  labeled mask via the existing local NIfTI serving endpoint; the label→volume map
  comes from `lesion_stats_report.json` (local).
- **Validation (Kappa-only):** the labeled mask must be uploaded to Kappa as a
  file, and the label→volume map must live in `dsEntityInfo`. So `kappa_uploader`
  uploads `*_lesion_labels.nii.gz` and embeds `lesion_volumes_by_label` in
  `dsEntityInfo`. This keeps validation self-contained from Kappa (consistent with
  the earlier Kappa-only decision).

The hero metrics, burden categories, per-lesion table, and longitudinal section
already flow through the existing local-vs-Kappa normalization in
`ClinicalReportContent` (`normalizeKappaEntity`); they need no new transport.

---

## 6. Frontend hover implementation

In `NIfTIViewer`, for MS:
- Load the **labeled** mask as the lesion overlay (so the cursor can read a
  per-lesion label, not just 0/1). All labels 1..N must still display as the
  single MS green — achieved by clamping (`cal_min: 0.5`, `cal_max: 1` so any
  value ≥1 maps to the top colormap entry) or an equivalent flat colormap. Color
  conveys "lesion"; the label value is used only for volume lookup.
- On cursor move, read the overlay voxel value (label) at the crosshair via the
  niivue location/voxel API, look up the volume from the label→volume map, and
  show a small tooltip near the cursor. Hide it when the label is 0 (background).

**Technical risk:** reading the overlay voxel value at the cursor depends on the
niivue API surface. If direct per-voxel read of the overlay is not exposed,
fall back to reading the label from the loaded volume's data array at the voxel
index reported by the location-change callback. This is the one unknown to
validate early.

---

## 7. Units of change

| Unit | File | Responsibility |
|------|------|----------------|
| Lesion labels + map | `scripts/08_lobar_localization.py` | save labeled mask, add `lesion_volumes_by_label` |
| Kappa transport | `backend/kappa_uploader.py` | upload labeled mask, embed label→volume map |
| MS report render | `frontend/src/components/ClinicalReportContent.jsx` | rebuild MS path: hero, burden categories, per-lesion table; drop glio sections from MS |
| Burden categories | same | compute size bands from `lesion_volumes_cm3` |
| Hover | `frontend/src/components/NIfTIViewer.jsx` | load labeled mask, read label at cursor, tooltip with volume |
| Normalizer | `ClinicalReportContent.jsx` (`normalizeKappaEntity`) | surface `lesion_volumes_by_label` for hover lookup in validation |

---

## 8. Testing

- Stage 08: run on the MS case; verify `*_lesion_labels.nii.gz` written and
  `lesion_volumes_by_label` present with N entries matching `lesion_count`.
- Frontend: build passes; MS report shows the five sections, glio sections absent;
  glio report unchanged.
- Hover: verify tooltip shows the correct volume for the largest lesion and hides
  on background. Verify in both run/history (local) and validation (Kappa) after a
  fresh run.

---

## 9. Out of scope (future — variant C, `feat/ms-clinical-metrics`)

- McDonald-criteria localization: periventricular / juxtacortical / infratentorial
  classification (distance-to-ventricles / distance-to-cortex computation).
- New / enlarging lesion detection between sessions (inter-session registration +
  mask diff) — the clinically highest-value metric for disease monitoring.
- Gadolinium-enhancing lesion detection (requires MS contrast series, not in the
  current MS protocol).

---

*Document status: approved layout. Next — implementation plan.*
