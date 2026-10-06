# MS Report Dynamics, Mask Colour, Stall Timeout — Plan

Spec: `docs/superpowers/specs/2026-09-30-ms-report-dynamics-and-timeouts-design.md`
Branch: `fix/ms-report-dynamics-and-timeouts`

Each step ends with its check; commit per step only when Kate asks.

## Step 1 — Run-tab report refetches on every open

- `frontend/src/components/ProgressMonitor.jsx`: wrap `<NIfTIViewer …>` in
  `{showVisualization && (…)}` and `<ClinicalReport …>` in
  `{showClinicalReport && (…)}`.
- Check: `npm run lint`; open run `30_09_1616` from the run tab → "Динамика
  между сессиями" shows 2 sessions; close/reopen still works; validation
  panel state kept.

## Step 2 — MS mask red

- `frontend/src/components/NIfTIViewer.jsx`: `createMsColormap` →
  `R:[0,255] G:[0,0] B:[0,0]`; comments "green" → "red" (lines ~40, ~289,
  ~300, ~495); MS legend colour → `rgb(255, 0, 0)`.
- Check: lint; viewer shows red lesions in atlas and native modes and after
  switching mask versions.

## Step 3 — Stall timeout (TDD)

1. Tests first, `backend/test_pipeline_stall_timeout.py`:
   - `newest_mtime(path)`: missing dir → None; nested file newest wins.
   - `wait_for_pipeline(process, output_path, stall_seconds, poll_seconds)`
     with a fake process: normal exit returns `(rc, out, err)`; no activity
     past the threshold → returns stalled result; touching a file inside the
     tree resets the timer.
2. Implement in `backend/pipeline_manager.py` (`newest_mtime`,
   `wait_for_pipeline`) and use it in `backend/app.py:run_pipeline_background`
   instead of `estimate_pipeline_timeout` + `communicate(timeout=…)`.
   On stall: append the reason to `{output}/logs/pipeline_master.log`, then
   `_kill_process_tree`, then `error_message` = same text.
3. `backend/config.py`: replace the two `pipeline_timeout_*` settings with
   `pipeline_stall_timeout_seconds = 3600`; add to `.env.example` (commented).
4. Delete `backend/test_pipeline_manager_timeout.py` and
   `estimate_pipeline_timeout`.
- Check: new tests pass; `python -m pytest backend/ -q` green.

## Step 4 — KNOWN_ISSUES

- KI-052: add a note that the total-duration timeout was replaced by the stall
  timeout (this branch), with the 2026-09-28 11-session case.
- KI-058: memory planner vs host RAM + stage 07 cost underestimate.
- KI-059: longitudinal dynamics require a completed Kappa upload.
- Check: entries readable, numbering unique.

## Step 5 — Verify end to end

- Rebuild `web`: the frontend is built into the image
  (`web.Dockerfile:91-94`) and `backend/` is copied, not mounted
  (`web.Dockerfile:97`) — `docker compose --profile full up -d --build web`.
- Re-open run `30_09_1616` from the run tab (items 1–2).
- Optional: short run with `PIPELINE_STALL_TIMEOUT_SECONDS=120` and a stage
  artificially paused to see the stall kill + log line.
