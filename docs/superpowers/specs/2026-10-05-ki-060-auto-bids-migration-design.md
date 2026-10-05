# KI-060: Migrate the BIDS Allocation Table Automatically — Design

**Date:** 2026-10-05
**Branch:** `fix/ki-060-auto-bids-migration` (from `main`)
**Type:** Bugfix (first run on an upgraded machine crashes)

## Problem

`feat/bids-per-dataset` changed `bids_patient_allocation` from per-lesion-type
numbering (`lesion_type` column) to per-dataset numbering (`scope` column).
`utils/bids_allocator.py::_connect` only runs `CREATE TABLE IF NOT EXISTS`, so
an existing old-format table stays as it is. Converting it is a one-off script,
`scripts/migrate_bids_allocation_scopes.py`, that nothing calls.

On the demob laptop (2026-09-30) the first run after `git pull` died in stage
01: `sqlite3.OperationalError: no such column: scope`. The backend startup
check (`reconcile_registry_bids_ids` in `lifespan`) hits the same table and
fails too, but only logs an error. The migration had to be run by hand.

## Design

### Where: `_connect`, the one door to the table

Every user of the table opens it through `bids_allocator._connect`: the
backend (startup reconcile, `numbering.py`, uploader), stage 01 in its own
process, and CLI runs with no backend at all. Migrating there covers all of
them; migrating in the backend's `lifespan` alone would leave CLI runs broken.

On every connect (cheap: one `PRAGMA table_info`):

- table absent → create the new schema, as today;
- table has `scope` → nothing to do, as today;
- table has `lesion_type` and no `scope` → **migrate**, then continue.

### How: the existing, tested migration — moved, not rewritten

The logic in `scripts/migrate_bids_allocation_scopes.py` (re-file rows into
`ds:<dataset>` from `patient_registry`, else `local:<lesion>`; floors per
dataset; old table kept as `bids_patient_allocation_legacy`; abort on any
in-dataset conflict) moves to `utils/bids_allocation_migration.py`. The script
stays as a thin CLI over it (dry run by default), so the manual path and its
tests keep working.

Changes to that logic:

1. **Backup first.** Before writing, copy the DB with SQLite's online backup
   API to `brain_lesion.db.bak-<YYYYmmdd-HHMMSS>-before-scope-migration` next
   to it (safe while the backend holds the file open).
2. **Race-safe.** Backend and stage 01 can connect at the same moment. The
   schema is re-checked after `BEGIN IMMEDIATE`; whoever gets the lock second
   sees `scope` and does nothing.
3. **No `patient_registry` table** (a CLI-only DB) → nothing was uploaded,
   everything goes to `local:<lesion>` instead of crashing.
4. **Conflicts stop the run with a readable error**, not `SystemExit`:
   `RuntimeError` listing the conflicts, saying nothing was changed and
   naming the script for a manual dry run. Same safety as before — never
   guess how to split a conflict.
5. **Logged** at WARNING: rows moved to datasets / to local, floors, backup
   path. Lands in the backend log or the stage 01 log, whoever migrated.

### Out of scope

- Other future schema changes — a general migration framework is not worth
  it for one table; this is the pattern to copy if another one comes.
- The `_legacy` table and backup files are never deleted automatically.

## Testing (pytest, next to the code)

`utils/test_bids_allocation_auto_migration.py`, on a temp DB built in the old
format:

- `get_allocations()` on an old DB returns the migrated map (no exception);
  rows land in `ds:`/`local:` scopes per `patient_registry`; numbers unchanged;
  `_legacy` table present; floors set.
- A backup file is created and is a readable SQLite DB with the old schema.
- Second connect: no migration, no second backup.
- Fresh DB: new schema, no backup, nothing logged.
- Old DB without `patient_registry`: everything to `local:`.
- In-dataset conflict: `RuntimeError` naming the conflict and the script; DB
  unchanged (still old schema, no `_legacy`).
- Two threads connecting at once to an old DB: both succeed, one migration,
  one backup.
- Existing `test_migrate_bids_allocation_scopes.py` stays green (script API).
- Real: copy of the laptop's pre-migration backup
  (`brain_lesion.db.bak-20260930-before-scope-migration`) → first connect
  migrates it with the same counts as the manual run (69 / 252 / 6 floors).
