# Slicer from validation: Kappa → host cache — Design

**Branch:** `fix/slicer-mask-not-loading`  
**Date:** 2026-08-14  
**Scope of this spec:** expert/validation tab only. Clinical «Запуск» Slicer path is unchanged.

## Context

Expert mode is a **separate deployment**. The expert logs into Kappa, opens the validation tab, and never ran the pipeline for these patients. The in-app viewer already works: `ValidationPanel.buildCustomFiles` builds HTTP URLs from Kappa `fileId`.

Slicer does not. `ValidationActions` refuses to open without `runId`. `NIfTIViewer` fills `runId` via `getEntityRunInfo` → local `patient_registry.pipeline_run_id`. Then `POST /api/slicer/open/{run_id}` reads volumes from that run’s `output_path` and the mask from `mask_versions.file_path` on **this** machine. On the expert host those rows/files are missing or point at another machine’s disk. Slicer starts with volumes and no mask, or errors «Нет данных пациента».

Slicer Agent (`localhost:8001`) can only load host filesystem paths. Kappa files must be materialized to a directory that is the **same absolute path** inside the web container and on the host.

## Goal

From the validation tab, «Редактировать» downloads the entity’s preprocessed volumes + current mask from Kappa into a host-visible cache, launches Slicer on those paths, and on «Сохранить и отправить» writes the new mask into the same cache and `replace_entity_file` back to Kappa — with **no** local `pipeline_runs` / `output_path` required.

## Out of scope (deferred)

- Content-aware Kappa update on repeat pipeline runs (KI-035). Separate spec.
- Changing clinical-tab `POST /api/slicer/open/{run_id}` (still local disk).
- Rebuilding the mask-version picker from Kappa file lists (expert machine has empty `mask_versions`). This pass opens the **current** Kappa mask only.
- Native-space files, `*_segmask_labels.nii.gz`, JSON reports — not sent to Slicer.
- Multi-center `kappa_datasets.yaml` key `(lesion_type × center)`.
- Cache GC / TTL. Cache may grow; operator can delete `slicer_cache/`.

## Design

### Source of truth

| Surface | Source |
|---|---|
| Validation list + NiiVue | Kappa HTTP (unchanged) |
| Slicer open from validation | Kappa files, materialized to cache |
| Slicer open from clinical run | Local `output_path` (unchanged) |
| After expert save | New file in Kappa via existing `replace_entity_file`; local cache copy |

### Cache location (path identity)

Slicer Agent checks `Path(p).exists()` on the **host**. Paths in the `/open` payload must be host paths.

Web container already bind-mounts `/home:/home`. Cache directory must be an **absolute host path** that is also valid inside the container (same string). On this machine that is `{repo}/backend/data/slicer_cache`.

- Docker: `SLICER_CACHE_DIR` on `web` = host absolute path of `backend/data/slicer_cache` (writable via `/home:/home`). Do not use `/app/backend/data/...` in the agent payload.
- Local backend: default `Path(__file__).resolve().parent / "data" / "slicer_cache"` (absolute). Agent runs on the same machine, so the path matches.
- Layout: `{SLICER_CACHE_DIR}/{entity_id}/{original_filename.nii.gz}`.
- Gitignore `backend/data/slicer_cache/`. Never commit NIfTI.

Do **not** write cache as `/app/backend/data/...` in the agent payload: that path does not exist on the host.

### Materialize

New module `backend/slicer_workspace.py`:

```
materialize_from_kappa(session, dataset_id, entity_id) -> SlicerWorkspace
```

1. `get_entity_details` (already in `kappa_client.py`).
2. Classify files from `dsEntityInfo` + `files[]`:
   - **Volumes:** names in `data_files` (preprocessed MRI).
   - **Mask (one):** among entity files whose name matches `*_segmask*.nii.gz` and does **not** contain `_native_` or `_labels`:
     - if any `*_segmask_v{N}.nii.gz`, pick the highest `N`;
     - else the `prediction_files` entry that is a non-native segmask;
     - else 404 «в сущности нет маски».
3. For each chosen file: if cache file missing or size ≠ `fileSize` from Kappa (when provided), `download_entity_file` and write under `{cache}/{entity_id}/{fileName}`.
4. Return host paths: `image_paths`, `mask_path`, `cache_dir`, `patient_id`/`session_id` parsed from `dsEntityInfo.bids_id` the same way `ValidationPanel` does, `lesion_type` from `dsEntityInfo` (default `glioblastoma`).

Always use Kappa as the file list; ignore local `mask_versions` for this path.

### Open API

New endpoint, not an overload of `open/{run_id}`:

```
POST /api/slicer/open-from-kappa/{entity_id}?session_id=&dataset_id=
```

- 401 if Kappa session missing.
- 404 if entity or mask missing in Kappa.
- Calls `materialize_from_kappa`, then the existing Slicer Agent `POST {SLICER_AGENT_URL}/open` with:
  - `image_paths` / `mask_path` from cache (host paths)
  - `entity_id`, `dataset_id`, `kappa_session_id`
  - `run_id`: empty string
  - `segmentation_dir`: the entity cache dir (writable host path)
  - `lesion_type`, `patient_id`, `session_id`

Clinical `POST /api/slicer/open/{run_id}` is not modified.

### Frontend

If `entityId` and `datasetId` are set (validation tab always has them), call `openInSlicerFromKappa(entityId, datasetId)` — do not require `runId`, do not send `selected_mask_version` from local DB.

Clinical viewer still passes `runId` and keeps using `openInSlicer(runId, …)`.

### Save round-trip

`/api/validation/upload-mask` save fallback order:

1. If `{SLICER_CACHE_DIR}/{entity_id}` exists → save the new `*_segmask_vN.nii.gz` there.
2. Else existing `get_ai_mask_dir` / run `output_path` logic (clinical machine).

Then existing `replace_entity_file` + `register_expert_mask`.

Slicer Agent: relax upload-button guard to `entity_id` only (`run_id` not required when `segmentation_dir` + `entity_id` + `dataset_id` are set). Backend must not 404 on missing run when cache dir exists.

### Errors

- Kappa download failure: 502/504 with the file name, no Slicer launch.
- Agent down: existing 503.
- Missing mask in entity: 404, frontend `message.error`.
- Log at INFO: entity_id, cache dir, number of volumes, mask filename, whether downloaded or cache-hit.

### Testing

Pytest, mocks, no live Kappa.

- `materialize_from_kappa`: writes volumes + highest `vN` mask; skips `_native_` / `_labels`; 404 when no mask.
- Cache hit: matching size → no second download.
- `open-from-kappa` builds agent payload with cache paths; `mask_path` non-empty.
- `upload-mask` with empty `run_id` and populated cache dir saves under cache and still calls `replace_entity_file` (mocked).

Frontend: `npm run lint`. Manual: validation tab on a machine without that run’s `output_path` → Slicer shows mask → save → entity in Kappa gains `*_segmask_vN.nii.gz`.

## Success criteria

1. Expert deployment, no local run for the patient: validation → Редактировать → Slicer has MRI + current Kappa mask.
2. Save from Slicer updates Kappa; another browser session’s validation viewer shows the new mask (HTTP), without that expert host’s cache.
3. Clinical «открыть Slicer» from a just-finished run still uses local `output_path` (no mandatory Kappa round-trip).
