# KI-060: Automatic BIDS Allocation Migration — Plan

Spec: `docs/superpowers/specs/2026-10-05-ki-060-auto-bids-migration-design.md`
Branch: `fix/ki-060-auto-bids-migration`

Tests in a throwaway `brain-lesion-web:demo` container. Commit per step only
when Kate asks. The first commit on this branch is the KI housekeeping
(statuses of KI-004/005/042/043/052/053/054/057/058).

## Step 1 — Tests first

`utils/test_bids_allocation_auto_migration.py` with an old-format DB fixture
(old `bids_patient_allocation` + `patient_registry`), cases from the spec.
Run → fail (module/behaviour missing).

## Step 2 — Move the migration into utils

- `utils/bids_allocation_migration.py`: `is_legacy(conn)`, `find_conflicts`,
  `migrate(db_path, dry_run)` (from the script), plus backup, registry-absent
  handling, re-check under `BEGIN IMMEDIATE`, `RuntimeError` on conflicts,
  WARNING summary.
- `scripts/migrate_bids_allocation_scopes.py`: CLI over it; prints the
  conflict list and exits 1 on conflicts (CLI behaviour unchanged).
- Check: `test_migrate_bids_allocation_scopes.py` green.

## Step 3 — Call it from `_connect`

- `bids_allocator._connect`: `PRAGMA table_info` → legacy → migrate before
  `CREATE TABLE IF NOT EXISTS`.
- Check: new tests + full `backend/` suite + allocator tests green.

## Step 4 — Real data

- Run against a copy of `backend/data/brain_lesion.db.bak-20260930-before-scope-migration`
  in the container: counts 69 / 252 / 6 floors, as the manual run.
- Update KI-060 (closed, how), and the CLAUDE.md gotcha if one mentions the
  manual step (check).
