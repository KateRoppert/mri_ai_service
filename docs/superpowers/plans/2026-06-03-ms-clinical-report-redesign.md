# MS Clinical Report Redesign (variant A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the MS clinical report around clinical utility (lesion burden, burden character, per-lesion hover volume, protocol table, longitudinal trend) and add the Stage 08 + Kappa support for per-lesion hover.

**Architecture:** Stage 08 already runs connected-component labeling inside `compute_lesion_stats`; we persist the labeled mask and a label→volume map. The Kappa uploader carries both so validation stays Kappa-only. The frontend MS render path in `ClinicalReportContent` is rebuilt; `NIfTIViewer` loads the labeled mask and shows a volume tooltip on hover.

**Tech Stack:** Python (nibabel, numpy, scipy.ndimage), FastAPI, React + Ant Design, niivue 0.67.

---

## File Structure

- `scripts/08_lobar_localization.py` — `compute_lesion_stats` returns label→volume map + labeled array; `process_one_mask` saves the labeled mask NIfTI.
- `tests/test_lesion_stats.py` — NEW unit tests for `compute_lesion_stats`.
- `backend/kappa_uploader.py` — discover + upload labeled mask; embed `lesion_volumes_by_label` in `dsEntityInfo`.
- `backend/app.py` — `nifti-files` response surfaces the labeled-mask URL when present.
- `backend/models.py` — add `mask_labels_url` to `NIfTIFile`.
- `frontend/src/components/ClinicalReportContent.jsx` — rebuild MS render path; surface `lesion_volumes_by_label` in `normalizeKappaEntity`.
- `frontend/src/components/ValidationPanel.jsx` — surface the labeled-mask file in `buildCustomFiles`.
- `frontend/src/components/NIfTIViewer.jsx` — load labeled mask for MS, hover tooltip with volume.

---

## Task 1: Stage 08 — label→volume map + saved labeled mask

**Files:**
- Modify: `scripts/08_lobar_localization.py` (`compute_lesion_stats` ~line 33, `process_one_mask` ~line 211)
- Test: `tests/test_lesion_stats.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lesion_stats.py
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import importlib.util

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "stage08", str(Path(__file__).parent.parent / "scripts" / "08_lobar_localization.py")
)
stage08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage08)


def _make_mask(tmp_path):
    # 1mm isotropic, two separate blobs: 8 voxels and 1 voxel
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1   # 8 voxels -> 0.008 cm3
    data[7, 7, 7] = 1         # 1 voxel  -> 0.001 cm3
    p = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    return p


def test_compute_lesion_stats_returns_label_map(tmp_path):
    mask = _make_mask(tmp_path)
    stats, labeled, affine = stage08.compute_lesion_stats(mask)

    assert stats["lesion_count"] == 2
    # label->volume map has one entry per lesion, keyed by string label
    by_label = stats["lesion_volumes_by_label"]
    assert len(by_label) == 2
    assert set(round(v, 3) for v in by_label.values()) == {0.008, 0.001}
    # labeled array carries integer labels matching the map keys
    assert set(int(x) for x in np.unique(labeled) if x != 0) == {int(k) for k in by_label}
    # display list stays sorted descending
    assert stats["lesion_volumes_cm3"] == sorted(stats["lesion_volumes_cm3"], reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/test_lesion_stats.py -v`
Expected: FAIL — `compute_lesion_stats` returns a dict (not a 3-tuple) and has no `lesion_volumes_by_label`.

- [ ] **Step 3: Rewrite `compute_lesion_stats`**

Replace the function body (lines ~33–62) with:

```python
def compute_lesion_stats(mask_path: Path):
    """
    Count connected components (individual lesions) in a binary mask.
    Used for MS where each component = one lesion.

    Returns (stats_dict, labeled_array, affine):
      stats_dict: lesion_count, total_volume_cm3, mean_lesion_volume_cm3,
                  lesion_volumes_cm3 (sorted desc, for display/table),
                  lesion_volumes_by_label ({str(label): volume_cm3}, for hover).
      labeled_array: int array, each lesion its own integer label 1..N.
      affine: source affine (to save the labeled mask).
    """
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)
    voxel_vol_mm3 = float(np.prod(np.abs(np.diag(img.affine[:3, :3]))))
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    binary = (data > 0).astype(np.uint8)
    labeled, n_components = ndimage_label(binary)

    volumes_by_label = {}
    for i in range(1, n_components + 1):
        voxel_count = int(np.sum(labeled == i))
        volumes_by_label[str(i)] = round(voxel_count * voxel_vol_cm3, 4)

    volumes = list(volumes_by_label.values())
    total = round(sum(volumes), 4)
    mean = round(total / n_components, 4) if n_components > 0 else 0.0

    stats = {
        "lesion_count": n_components,
        "total_volume_cm3": total,
        "mean_lesion_volume_cm3": mean,
        "lesion_volumes_cm3": sorted(volumes, reverse=True),
        "lesion_volumes_by_label": volumes_by_label,
    }
    return stats, labeled.astype(np.int16), img.affine
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/test_lesion_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Update the caller in `process_one_mask`**

Replace the MS block (lines ~211–224) with:

```python
        # For MS: compute per-lesion statistics + save labeled mask for hover
        if lesion_type == 'multiple_sclerosis':
            stats, labeled, affine = compute_lesion_stats(mask_path)
            stats["patient_id"] = subject_id
            stats["session_id"] = session_id
            stats_path = report_path.parent / report_path.name.replace(
                "_lobar_report.json", "_lesion_stats_report.json"
            )
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            # Save labeled mask alongside the binary segmask so the viewer can
            # read a per-lesion label under the cursor and show its volume.
            labels_path = mask_path.parent / mask_path.name.replace(
                "_segmask.nii.gz", "_segmask_labels.nii.gz"
            )
            nib.save(nib.Nifti1Image(labeled, affine), str(labels_path))

            logger.info(
                f"Lesion stats: {stats['lesion_count']} lesions, "
                f"{stats['total_volume_cm3']:.3f} cm³ → {stats_path.name}; "
                f"labels → {labels_path.name}"
            )
```

- [ ] **Step 6: Syntax check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/08_lobar_localization.py`
Expected: no output (success).

- [ ] **Step 7: Commit**

```bash
git add scripts/08_lobar_localization.py tests/test_lesion_stats.py
git commit -m "feat(stage08): save labeled lesion mask and label-to-volume map for hover"
```

---

## Task 2: Kappa transport — labeled mask + label→volume map

**Files:**
- Modify: `backend/kappa_uploader.py` (session discovery ~line 225, `_build_entity_info` lesion_stats block)

- [ ] **Step 1: Discover the labeled mask in session_data**

In `_discover_sessions` (the `sessions.setdefault` dict, ~line 225) add a key:

```python
            sessions.setdefault(session_key, {
                "preprocessed": [],
                "masks": [],
                "quality_reports": [],
                "volume_report": None,
                "lobar_report": None,
                "lesion_stats_report": None,
                "lesion_labels_mask": None,
            })
```

After the lesion_stats discovery loop (the `*_lesion_stats_report.json` rglob block), add:

```python
        # Find labeled lesion masks (МС — для hover объёма очага)
        for lbl in sorted(segmentation_dir.rglob("*_segmask_labels.nii.gz")):
            session_key = self._extract_session_key(lbl)
            if session_key and session_key in sessions:
                sessions[session_key]["lesion_labels_mask"] = lbl
```

- [ ] **Step 2: Upload the labeled mask file**

In `_upload_session` where `file_paths` is assembled (it lists preprocessed + main masks), append the labeled mask if present. Find the `file_paths = list(session_data["preprocessed"])` line and after the masks are added, add:

```python
        # Include the labeled lesion mask so validation hover works Kappa-only
        if session_data.get("lesion_labels_mask"):
            file_paths.append(session_data["lesion_labels_mask"])
```

- [ ] **Step 3: Embed label→volume map in dsEntityInfo**

In `_build_entity_info`, in the lesion_stats block, add `lesion_volumes_by_label`:

```python
        # Lesion stats report (МС — количество и объёмы очагов)
        if session_data.get("lesion_stats_report"):
            try:
                with open(session_data["lesion_stats_report"], "r") as f:
                    ls = json.load(f)
                info["lesion_stats"] = {
                    "lesion_count": ls.get("lesion_count"),
                    "total_volume_cm3": ls.get("total_volume_cm3"),
                    "mean_lesion_volume_cm3": ls.get("mean_lesion_volume_cm3"),
                    "lesion_volumes_cm3": ls.get("lesion_volumes_cm3", []),
                    "lesion_volumes_by_label": ls.get("lesion_volumes_by_label", {}),
                }
                if session_data.get("lesion_labels_mask"):
                    info["lesion_labels_file"] = session_data["lesion_labels_mask"].name
            except Exception as e:
                logger.warning("Failed to read lesion stats report: %s", e)
```

- [ ] **Step 4: Syntax check**

Run: `cd /home/ubuntu/mri_ai_service/backend && python -m py_compile kappa_uploader.py`
Expected: no output (success).

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_uploader.py
git commit -m "feat(kappa): upload labeled lesion mask and embed label-to-volume map"
```

---

## Task 3: Frontend — rebuild MS report render

**Files:**
- Modify: `frontend/src/components/ClinicalReportContent.jsx` (MS render path ~lines 410–490; `normalizeKappaEntity`)

No frontend unit-test framework exists in this project; verification is `npm run build` + manual browser check.

- [ ] **Step 1: Surface `lesion_volumes_by_label` in the Kappa normalizer**

In `normalizeKappaEntity`, in the `lesionStatsReports` object, add the two hover fields:

```javascript
  const lesionStatsReports = info.lesion_stats
    ? [{
        patient_id,
        session_id,
        lesion_count: info.lesion_stats.lesion_count,
        total_volume_cm3: info.lesion_stats.total_volume_cm3,
        mean_lesion_volume_cm3: info.lesion_stats.mean_lesion_volume_cm3,
        lesion_volumes_cm3: info.lesion_stats.lesion_volumes_cm3 || [],
        lesion_volumes_by_label: info.lesion_stats.lesion_volumes_by_label || {},
      }]
    : [];
```

- [ ] **Step 2: Add a burden-category helper above the component**

After `sortByPatientSession` (module scope), add:

```javascript
// MS lesion size bands (pragmatic; a ~3 mm punctate lesion ≈ 0.014 см³).
const LESION_SIZE_BANDS = [
  { key: 'large',  label: 'крупные ≥0.1 см³',  bg: '#f6ffed', border: '#b7eb8f', color: '#389e0d', test: (v) => v >= 0.1 },
  { key: 'medium', label: 'средние 0.01–0.1',  bg: '#fcffe6', border: '#eaff8f', color: '#7cb305', test: (v) => v >= 0.01 && v < 0.1 },
  { key: 'small',  label: 'точечные <0.01',    bg: '#fafafa', border: '#e8e8e8', color: '#8c8c8c', test: (v) => v < 0.01 },
];

const countLesionBands = (volumes) =>
  LESION_SIZE_BANDS.map((b) => ({ ...b, count: (volumes || []).filter(b.test).length }));
```

- [ ] **Step 3: Replace the MS render path**

Replace the entire MS render block (from `// ===== MS RENDER PATH =====` through its closing — the `if (lesionType === 'multiple_sclerosis') { ... }` block, ~lines 410–490) with:

```javascript
  // ===== MS RENDER PATH =====
  if (lesionType === 'multiple_sclerosis') {
    if (!loaded || lesionStatsReports.length === 0) return null;
    return (
      <>
        {lesionStatsReports.map((stats, idx) => {
          const bands = countLesionBands(stats.lesion_volumes_cm3);
          const perLesionRows = (stats.lesion_volumes_cm3 || []).map((v, i) => ({
            key: i, n: i + 1, cm3: v,
          }));
          return (
            <div key={idx} style={{ marginBottom: 32 }}>
              <div style={{ marginBottom: 16 }}>
                <Tag>{stats.patient_id}</Tag>
                <Tag>{stats.session_id}</Tag>
              </div>

              {/* Очаговая нагрузка */}
              <Divider orientation="left" style={{ fontSize: 14 }}>
                <Space><MedicineBoxOutlined /> Очаговая нагрузка</Space>
              </Divider>
              <Row gutter={32} style={{ marginBottom: 16 }}>
                <Col>
                  <Statistic title="Количество очагов" value={stats.lesion_count}
                    valueStyle={{ color: '#1890ff' }} />
                </Col>
                <Col>
                  <Statistic title="Суммарный объём поражения" value={stats.total_volume_cm3}
                    precision={3} suffix="см³" valueStyle={{ color: '#52c41a' }} />
                </Col>
                <Col>
                  <Statistic title="Средний объём очага" value={stats.mean_lesion_volume_cm3}
                    precision={3} suffix="см³" />
                </Col>
              </Row>

              {/* Характер поражения */}
              <Divider orientation="left" style={{ fontSize: 14 }}>
                <Space><ExperimentOutlined /> Характер поражения</Space>
              </Divider>
              <Row gutter={12} style={{ marginBottom: 16, maxWidth: 460 }}>
                {bands.map((b) => (
                  <Col span={8} key={b.key}>
                    <div style={{
                      textAlign: 'center', background: b.bg, border: `1px solid ${b.border}`,
                      borderRadius: 6, padding: '10px 4px',
                    }}>
                      <div style={{ fontSize: 22, fontWeight: 700, color: b.color }}>{b.count}</div>
                      <div style={{ fontSize: 11, color: '#888' }}>{b.label}</div>
                    </div>
                  </Col>
                ))}
              </Row>

              {/* Объёмы всех очагов — свёрнуто, для протокола */}
              <Collapse ghost style={{ marginBottom: 8 }} items={[{
                key: 'lesions',
                label: `Объёмы всех очагов (${perLesionRows.length}) — для протокола`,
                children: (
                  <Table
                    columns={[
                      { title: '№', dataIndex: 'n', key: 'n', width: 60 },
                      { title: 'Объём (см³)', dataIndex: 'cm3', key: 'cm3', align: 'right',
                        render: (v) => v.toFixed(4) },
                    ]}
                    dataSource={perLesionRows}
                    pagination={false}
                    size="small"
                    bordered
                    scroll={{ y: 220 }}
                    style={{ maxWidth: 300 }}
                  />
                ),
              }]} />

              {/* Динамика между сессиями */}
              <Divider orientation="left" style={{ fontSize: 14 }}>
                <Space>📈 Динамика между сессиями</Space>
              </Divider>
              <LongitudinalTimeline patientId={stats.patient_id} lesionType="multiple_sclerosis" />
            </div>
          );
        })}
      </>
    );
  }
```

- [ ] **Step 4: Add the `Collapse` import**

In the antd import line (line ~10), add `Collapse`:

```javascript
import { Table, Space, Spin, Alert, Tag, Tooltip, Row, Col, Statistic, Divider, Collapse } from 'antd';
```

- [ ] **Step 5: Build to verify it compiles**

Run: `cd /home/ubuntu/mri_ai_service/frontend && npm run build 2>&1 | tail -3`
Expected: `✓ built in ...` with no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ClinicalReportContent.jsx
git commit -m "feat(frontend): rebuild MS report (burden, size bands, per-lesion table, trend)"
```

---

## Task 4: Hover spike — verify niivue location data

**Files:**
- Modify (temporary): `frontend/src/components/NIfTIViewer.jsx`

Goal: confirm how niivue 0.67 reports the voxel value of the overlay under the cursor before building the tooltip. This de-risks the one unknown.

- [ ] **Step 1: Add a temporary location-change logger after niivue init**

In the niivue init effect (after `nvRef.current = nv;` ~line 109), add:

```javascript
          nv.onLocationChange = (loc) => {
            // SPIKE: inspect shape — loc.values is array per loaded volume.
            console.log('niivue loc', loc?.values, loc?.vox);
          };
```

- [ ] **Step 2: Build and inspect manually**

Run: `cd /home/ubuntu/mri_ai_service/frontend && npm run build 2>&1 | tail -3`
Then (after redeploying the web container) open a case, move the cursor over a lesion, and read the browser console.

Confirm: `loc.values` contains an entry per volume (gray image, mask) and the mask entry equals the voxel label, and `loc.vox` gives integer voxel coords. Note the exact field names observed.

- [ ] **Step 3: Remove the spike logger**

Delete the temporary `nv.onLocationChange` logger added in Step 1.

- [ ] **Step 4: Commit the spike removal (no functional change)**

No commit needed if nothing else changed. If `loc` field names differ from the assumptions in Task 5, update Task 5's handler to match the observed shape before implementing.

---

## Task 5: Hover-to-volume in the viewer

**Files:**
- Modify: `backend/models.py` (`NIfTIFile` — add `mask_labels_url`)
- Modify: `backend/app.py` (`nifti-files` — set `mask_labels_url` when the labeled mask exists)
- Modify: `frontend/src/components/ValidationPanel.jsx` (`buildCustomFiles` — surface labeled mask + map)
- Modify: `frontend/src/components/NIfTIViewer.jsx` (load labeled mask for MS, tooltip)

- [ ] **Step 1: Add hover fields to the NIfTIFile model**

In `backend/models.py`, find the `NIfTIFile` model and add (the map is per-session but attached to each modality file so the viewer can read it from `selectedFile` regardless of source):

```python
    mask_labels_url: Optional[str] = None
    lesion_volumes_by_label: Optional[Dict[str, float]] = None
```

If `Dict` is not imported in `models.py`, add it to the typing import line (`from typing import ..., Dict`).

- [ ] **Step 2: Populate them in the nifti-files endpoint**

In `backend/app.py` `get_nifti_files_list`, where each `NIfTIFile(...)` is built (~line 648), before constructing it, resolve the labeled mask URL and read the session's label→volume map (so local run/history hover has the same data validation gets from Kappa):

```python
                    labels_name = mask_filename.replace("_segmask.nii.gz", "_segmask_labels.nii.gz")
                    mask_labels_url = None
                    for labels_path in segmentation_dir.rglob(labels_name):
                        mask_labels_url = f"/api/nifti/{run_id}/segmentation/{labels_name}"
                        break

                    # base_name was derived above as mask_filename without _segmask.nii.gz
                    volumes_by_label = None
                    stats_name = f"{base_name}_lesion_stats_report.json"
                    for stats_p in segmentation_dir.rglob(stats_name):
                        try:
                            with open(stats_p, "r", encoding="utf-8") as sf:
                                volumes_by_label = json.load(sf).get("lesion_volumes_by_label")
                        except Exception:
                            volumes_by_label = None
                        break
```

Then add to the `NIfTIFile(...)` constructor call:

```python
                        mask_labels_url=mask_labels_url,
                        lesion_volumes_by_label=volumes_by_label,
```

(`json` is already imported in `app.py`.)

- [ ] **Step 3: Surface labeled mask + map in validation buildCustomFiles**

In `frontend/src/components/ValidationPanel.jsx` `buildCustomFiles`, the entity's `dsEntityInfo` has `lesion_labels_file` and `lesion_stats.lesion_volumes_by_label`. Resolve the labeled-mask file URL and attach both to each returned file object. After computing `maskUrl`, add:

```javascript
  const labelsFileName = info.lesion_labels_file;
  const labelsFileId = labelsFileName ? fileIdByName[labelsFileName] : null;
  const maskLabelsUrl = labelsFileId ? getValidationFileUrl(datasetId, labelsFileId) : null;
  const volumesByLabel = info.lesion_stats?.lesion_volumes_by_label || null;
```

Then in the returned file object (inside the `.map`), add:

```javascript
        mask_labels_url: maskLabelsUrl,
        lesion_volumes_by_label: volumesByLabel,
```

- [ ] **Step 4: Thread hover data into NIfTIViewer files**

In `NIfTIViewer`, the per-file object already flows through `files`/`selectedFile`. For run/history, the backend `mask_labels_url` is on each file. For validation, `mask_labels_url` + `lesion_volumes_by_label` are on each custom file. No new prop needed — read from `selectedFile`.

- [ ] **Step 5: Load the labeled mask for MS and add tooltip state**

In `NIfTIViewer`, add state near the other `useState` hooks:

```javascript
  const [hoverVolume, setHoverVolume] = useState(null); // {x, y, cm3} | null
```

In `loadNIfTI`, when `lesionType === 'multiple_sclerosis'` and `file.mask_labels_url` exists, load the labeled mask as the overlay instead of the binary mask (same green via clamp). Change the second volume in `nv.loadVolumes([...])`:

```javascript
        {
          url: (lesionType === 'multiple_sclerosis' && file.mask_labels_url)
            ? file.mask_labels_url
            : maskUrl,
          name: file.mask_filename,
          colormap: 'seg_custom',
          opacity: maskOpacity,
          cal_min: 0,
          cal_max: lesionType === 'multiple_sclerosis' ? 1 : 4,
        }
```

(`cal_max: 1` clamps all labels ≥1 to the single green entry — display unchanged.)

- [ ] **Step 6: Wire the hover handler**

In the niivue init effect (after `nvRef.current = nv;`), add (adjust field names to match the Task 4 spike if needed):

```javascript
          nv.onLocationChange = (loc) => {
            const file = selectedFileRef.current;
            const byLabel = file?.lesion_volumes_by_label;
            // mask overlay is the last loaded volume; its value is the label
            const values = loc?.values || [];
            const labelVal = values.length ? Math.round(values[values.length - 1].value ?? values[values.length - 1]) : 0;
            if (byLabel && labelVal > 0 && byLabel[String(labelVal)] != null) {
              setHoverVolume({ cm3: byLabel[String(labelVal)] });
            } else {
              setHoverVolume(null);
            }
          };
```

Add a ref to read the latest selected file inside the niivue callback (the callback closes over the init render). Near the other refs:

```javascript
  const selectedFileRef = useRef(null);
  useEffect(() => { selectedFileRef.current = selectedFile; }, [selectedFile]);
```

- [ ] **Step 7: Render the tooltip**

Just inside the canvas container (near the `<canvas>`), add an overlay badge shown when `hoverVolume` is set:

```javascript
          {hoverVolume && (
            <div style={{
              position: 'absolute', top: 8, right: 8, zIndex: 5,
              background: 'rgba(0,0,0,0.75)', color: '#fff',
              padding: '4px 10px', borderRadius: 4, fontSize: 13, pointerEvents: 'none',
            }}>
              Очаг: {hoverVolume.cm3.toFixed(3)} см³
            </div>
          )}
```

Ensure the canvas wrapper has `position: relative`.

- [ ] **Step 8: Syntax + build check**

Run:
```bash
cd /home/ubuntu/mri_ai_service/backend && python -m py_compile models.py app.py
cd /home/ubuntu/mri_ai_service/frontend && npm run build 2>&1 | tail -3
```
Expected: no Python errors; `✓ built in ...`.

- [ ] **Step 9: Commit**

```bash
git add backend/models.py backend/app.py frontend/src/components/ValidationPanel.jsx frontend/src/components/NIfTIViewer.jsx
git commit -m "feat(viewer): per-lesion volume tooltip on hover (local + Kappa sources)"
```

---

## Task 6: End-to-end verification

- [ ] **Step 1: Rebuild and restart the web container**

```bash
cd /home/ubuntu/mri_ai_service && docker compose build --no-cache web && docker compose up -d web
```

- [ ] **Step 2: Fresh MS run**

Delete the P000915 entities from Kappa, then run the MS case (P000915) through the web UI so Stage 08 writes the labeled mask + map and the uploader carries them.

- [ ] **Step 3: Verify report**

In Запуск, История, and Валидация for the MS case, confirm: Очаговая нагрузка (count + total + mean), Характер поражения (3 size bands), collapsible per-lesion table, Динамика. Confirm CE+/CE−, NCR/ED/NET/ET, and cortical-lobe sections are absent for MS.

- [ ] **Step 4: Verify hover**

Hover over a lesion in the viewer (run/history and validation) — the volume badge appears with the lesion's volume and disappears over background.

- [ ] **Step 5: Verify glio unchanged**

Open a glioblastoma case report — confirm the glio layout (CE+/CE−, classes, lobar) is unchanged.

---

## Notes for the implementer

- The frontend has no unit-test framework; `npm run build` + manual browser verification is the contract here.
- niivue's `onLocationChange` payload shape is the one real unknown — Task 4 validates it before Task 5 depends on it. If the observed field names differ, adjust the handler in Task 5 Step 6 accordingly.
- Stage 08 only produces the labeled mask for `lesion_type == 'multiple_sclerosis'`; glio is untouched.
