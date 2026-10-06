# Kappa Upload: Complete Sessions Only — Plan

Spec: `docs/superpowers/specs/2026-10-01-kappa-upload-complete-sessions-only-design.md`
Branch: `fix/kappa-upload-complete-sessions-only`

Tests run in a throwaway `brain-lesion-web:demo` container (the laptop venv
is incomplete). Commit per step only when Kate asks.

## Step 1 — Readiness and universe in the uploader (TDD)

1. Tests in `backend/test_kappa_uploader_readiness.py` (cases from the spec,
   built on a tmp output tree: `bids_organized/`, `preprocessed/`,
   `segmentation/`).
2. `backend/kappa_uploader.py`:
   - `_session_universe()` → session keys from `bids_organized/sub-*/ses-*`
     (skip `_incomplete`), fallback to the keys `_discover_sessions` finds;
   - `_readiness(session_key, session_data)` → `(ready, reason)` using
     `load_lesion_type_config(self.lesion_type)['required_modalities']` and
     the presence of a non-native `*_segmask.nii.gz`;
   - in `upload_results`: not-ready → `error="not_processed"` result, no
     upload; `total = len(universe)`.
- Check: new tests + `backend/test_kappa_uploader*.py` green.

## Step 2 — Delivery classification (TDD)

1. Tests in `backend/test_kappa_delivery_not_processed.py` (cases from spec).
2. `backend/kappa_delivery.py`: collect `not_processed`; `done` when all ready
   sessions delivered (`delivered + len(not_processed) >= total`); store the
   list in `detail.not_processed` (also in `_detail` defaults).
- Check: new tests + existing delivery tests green; full `backend/` suite.

## Step 3 — History column

- `frontend/src/components/PipelineHistory.jsx`: `deliveryLabel` for `done`
  with `delivered < total` → `Kappa 4/6 · N не обработаны`; `deliveryHint`
  lists `detail.not_processed` lines + pointer to the lost-patients report.
- Check: lint in the `web` image — no new findings vs `main`.

## Step 4 — Verify on the real run

- Rebuild `web`. Delete the incomplete `ses-005` entity in Kappa (Kate, by
  hand). Re-run `P000067` into a fresh folder with the laptop's GPU config.
- Expected: all 6 sessions processed → `Kappa 6/6`. To see the new path, the
  old folder's state can be re-classified offline in a test against a copy of
  `30_09_1752` (no upload): 4 ready, 2 not processed, `total=6`.
