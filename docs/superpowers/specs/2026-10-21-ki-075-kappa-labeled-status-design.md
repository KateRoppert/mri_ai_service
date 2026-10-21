# KI-075: Kappa Statuses 3 and 5 Need `labels` and an `output` Mask — Design

**Date:** 2026-10-21
**Branch:** `fix/ki-075-kappa-labeled-status` (from `main` 67de8f7)
**Type:** Fix (also closes KI-036)

## What Kappa wants now (probed live, throwaway dataset 357, 2026-10-21)

| attempt | result |
|---|---|
| v1 `PUT` status 3 with `empty.json` (today's `update_entity_status`) | 422 `missing required fields: labels; missing required files: output_image` |
| v2 `PUT` status 3, no files | same 422 |
| v2 `mark-labeled` | 422 `missing required fields: labels` |
| `dsEntityInfo.labels = ["lesion"]` only | 422 `missing required files: output_image` |
| main mask re-uploaded with `file_category=output` only | 422 `missing required fields: labels` |
| **both** | **200, status 3** |
| **both, set at creation** (`filesCategory: {mask: "output"}` in the v1 create JSON + `labels` in info) | **200, status 3** — no extra upload |
| `filesCategory` in a `PUT` without files | ignored (categories of existing files unchanged) |
| status 2→4, 3→4 | 200 (v1 and v2) |
| status 4→5 without labels/output | **422** `Cannot mark as verified; missing required fields: labels; missing required files: output_image` |
| status 4/5 via v1 | works, but **adds an `empty.json` each time** (KI-036) |
| status via v2 `PUT` with no files | works, no stub |

What the real datasets hold: every file is category `Input`, no entity has
`labels`. Status 3 was still set on 2026-10-06 (dataset 158); entities from
2026-10-09 on (337: 18, 353: all 13) are stuck at 2 — the rule appeared
between those dates.

Consequences today:

- No new entity reaches Labeled (warning in the log only).
- **The validation tab cannot verify anything**: approving sets status 5,
  which Kappa refuses for entities without `labels` / an `output` mask —
  i.e. all of them. The endpoint ignores `update_entity_status`'s result,
  so the doctor sees the vote accepted while Kappa keeps the old status.

## Change

1. **Upload** (`kappa_client.upload_entity`, used by `_upload_session`): the
   create JSON gets `filesCategory` — the main AI mask (`*_segmask.nii.gz`,
   not native, not expert, not `_labels`) → `output`, everything else →
   `input`; `dsEntityInfo.labels` from the lesion type (below).
2. **Status changes** (`update_entity_status`): v2 `PUT
   /datasets/datasetEntities/{ds}/{id}` with `{"dsEntityStatus", "remark"}`
   and no files. No more `empty.json` (KI-036). Returns `(ok, kappa_message)`
   so callers can say why.
3. **Replace / KI-035 completion** (`kappa_entity_files`): `patch_file` and
   `add_files` pass `file_category=output` for the main mask (input for the
   rest); the info update carries `labels`.
4. **Validation endpoint**: if Kappa refuses the status, the response says
   so (`kappa_status: {ok, message}`) and the validation panel shows a
   warning instead of a silent success. (Frontend change → rebuild.)
5. **Existing entities** (status stuck at 2, verification impossible):
   a script `scripts/kappa_backfill_labels.py --dataset N [--apply]`, dry-run
   by default: per entity, PUT `labels` into the info and re-upload the main
   mask with `file_category=output` (bytes downloaded from Kappa itself, so
   it works for entities uploaded from another machine), optionally drop
   `empty.json` stubs, then status 3 if it was 2. Run per dataset only on
   Kate's go-ahead.

## Decisions for Kate

- **What goes into `labels`.** Kappa accepts any non-empty list; dataset
  label classes are not defined in our datasets (`GET /datasets/labels` →
  404). Proposed: the mask's classes from the segmentation service manifest
  (`output.classes`, background left out) — GBM `["ed", "net", "et", "ncr"]`
  (`services/gbm-seg/manifest.yaml`), MS `["lesion"]`
  (`services/ms-seg/manifest.yaml`). One source of truth, no new config key;
  fallback `[lesion_type]` if a manifest has no classes.
- **Backfill scope.** Old datasets (249, 337, 353, 158) are to be replaced
  by new ones for the new preprocessing order. Backfill only datasets that
  will keep being validated — possibly none.

## Verification

- Unit tests: create JSON carries `filesCategory` and `labels`; status
  update is a v2 PUT with no files; replace passes `file_category`;
  validation response reports a refused status.
- Live, throwaway dataset: upload via the uploader → status 3 without manual
  steps; validation-style 3→4→5 → 5 reached, no `empty.json`; KI-035
  completion on an incomplete entity → mask `Output`, status 3.
- Backfill dry-run on one real dataset (read-only), `--apply` only on a test
  copy unless Kate says otherwise.

## Results (2026-10-21)

Deviation from the plan: the web image has no `services/`, so the mask
classes are copied into `configs/lesion_types.yaml` (`kappa_labels`) and a
test (`backend/test_kappa_labeled_status.py`) keeps them equal to
`services/*/manifest.yaml`.

Live, dataset 357, branch code: uploader → new entities status 3, mask
`Output`; KI-035 completion → status 3, mask `Output`; validation-style
3→4→5 via `update_entity_status` → 5, no `empty.json`; backfill on an
old-style entity (no labels, `Input` mask, one `empty.json`) → labels,
`Output`, stub removed, status 3, second dry run "0 need work". Test entities
deleted.

Backfill dry run on the real datasets: 353 13/13, 337 22/22 (sub-015: 9
stubs), 249 84/84, 158 13/13.

Tests: 15 in `test_kappa_labeled_status.py`, 6 in
`scripts/test_kappa_backfill_labels.py`; full `scripts/` + `backend/` 633
passed, the 3 known environment failures unchanged. ESLint on
`ValidationActions.jsx`: the same 5 pre-existing findings as on main, none
new.
