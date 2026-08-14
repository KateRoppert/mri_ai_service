# Requeue Progress Linking — Design

## Context

The doctor-review workflow (incomplete-patients queue, merged in `feat/incomplete-patients-frontend`) lets a doctor manually fix incomplete sessions and click "Запустить обработку" (requeue) to reprocess them. Today the requeued run only becomes visible in the "История запусков" tab — the active-run screen (`ProgressMonitor`, on the "Запуск обработки" tab) keeps showing whatever run was active before, with no link to the new one. The doctor has no way to watch the requeued run's live progress without manually switching tabs and finding it in the history table, and once found, nothing on screen connects it back to the original run it continues.

## Design

### Data model: `parent_run_id`

Add a nullable `parent_run_id` column to `pipeline_runs`, following the exact precedent already used for `lesion_type` (`backend/database.py:253-263`): a small `_migrate_add_parent_run_id()` function, called from `init_db()` alongside the existing migration, checking `PRAGMA table_info(pipeline_runs)` before running `ALTER TABLE pipeline_runs ADD COLUMN parent_run_id VARCHAR`. This is additive and safe against existing databases (matches the project's established migration convention — see `CLAUDE.md`'s note that `backend/data` is a volume, so the DB is created fresh on a new machine but must also tolerate being an existing file with real data on a developer's machine).

`requeue_pipeline_run` (`backend/app.py`, the `POST /api/pipeline-runs/{run_id}/requeue` handler) sets the new run's `parent_run_id` to the `run_id` from the URL path — the run being requeued. An ordinary `/api/pipeline/start` run always has `parent_run_id = NULL`. This naturally supports chains (a requeue of a requeue points at its immediate parent, not the original ancestor) — no special-casing needed, and no need to resolve the whole chain for this feature.

### Backend: exposing it

Add `parent_run_id: Optional[str] = None` to `PipelineStatusResponse` (`backend/models.py:234-244`), the response model for `GET /api/pipeline/status/{run_id}` — the endpoint `ProgressMonitor` already calls once via `fetchInitialStatus()` on mount (`frontend/src/components/ProgressMonitor.jsx:65-68`). Since a run's `parent_run_id` never changes during its lifetime, it doesn't need to travel through the WebSocket live-update messages (`updateStatus`) — the one-time REST fetch is sufficient. The handler for this endpoint reads the value off the `PipelineRun` DB row it already loads and includes it in the response, no new query.

### Frontend: auto-switch + banner

- `IncompletePatients.jsx` gains an optional `onRequeued(newRunId)` prop, called right after `onClose()` on a successful requeue (in `handleRequeue`).
- `App.jsx` wires this callback from BOTH mount points of `IncompletePatients`:
  - The `ProgressMonitor`-owned instance (active-run path): the callback calls the same `setActiveRun({...})` shape already used by `handlePipelineStarted` (`App.jsx:111-119`), pointing at the new run. This means requeuing — whether triggered from the active screen or from history — always makes the requeued run the one shown on the "Запуск обработки" tab, matching how starting any new run already behaves today.
  - The history-triggered instance: same callback, same effect — requeuing from history also activates the new run on the pipeline tab, since it's now live and worth watching.
- `ProgressMonitor` reads `parent_run_id` from its `fetchInitialStatus()` response into local state. If present, it renders a small banner above the stage list: "Это повторный запуск после ручной правки (исходный запуск: `<parent_run_id first 8 chars>`...). [Перейти к истории]". Clicking the link switches the active tab to "История запусков" — the original run's row (with all its report buttons) is already there, nothing needs to be duplicated or specially fetched.
- To let a child component (the banner's link) switch tabs, `App.jsx`'s `<Tabs>` (currently uncontrolled, `defaultActiveKey="pipeline"`) becomes controlled: a new `activeTabKey` state, `activeKey={activeTabKey}` + `onChange={setActiveTabKey}` on the `Tabs` component, and a `switchToHistoryTab` handler passed down to `ProgressMonitor` for the banner's link to call.

Reports (quality, volumes, masks, NIfTI viewer, etc.) are unaffected by any of this — they're already read from disk keyed by `output_path`, not by `run_id`, and the requeued run shares the same `output_path` as its parent. Switching which run is "active" never hides or duplicates this data.

### Testing

Backend (pytest, following the existing test structure in `backend/test_app_requeue_endpoint.py`):
- `requeue_pipeline_run` sets `parent_run_id` on the newly created run to the original `run_id`.
- `GET /api/pipeline/status/{run_id}` includes `parent_run_id` in its response (both the `None` case for an ordinary run and the populated case for a requeued one).
- Migration test: `_migrate_add_parent_run_id()` is idempotent (running it twice doesn't error) and adds the column to a table that predates it — same shape as whatever test coverage (if any) exists for `_migrate_add_lesion_type`; if none exists for that one, a lightweight one for the new migration is still worth adding since this is new code, not a modification of already-covered code.

Frontend: no automated test framework exists in this project (confirmed fact carried over from the prior branch's work) — manual browser verification is the only functional check, same as everything else touching these components.

## Out of Scope

- Merging or visually combining the two runs' stage timelines into one view (rejected during brainstorming in favor of the simpler switch + banner-link approach).
- Resolving/displaying the full ancestor chain for a multiply-requeued run — only the immediate `parent_run_id` is shown.
- Any change to report-fetching endpoints — they are already `output_path`-keyed and need no changes.
