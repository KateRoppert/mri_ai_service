# KI-058: Memory-Aware Planning — Plan

Spec: `docs/superpowers/specs/2026-10-02-ki-058-memory-aware-planning-design.md`
Branch: `fix/ki-058-memory-aware-planning`

Tests in a throwaway `brain-lesion-web:demo` container (the laptop's copied
`venv/` has no interpreter). Commit per step only when Kate asks.

## Step 1 — Budget from host MemAvailable (TDD)

- Tests: `utils/test_resource_planner_host_budget.py` (next to code; check
  how existing planner tests are laid out first and follow them).
- `utils/resource_planner.py`: `host_available_bytes(path="/proc/meminfo")`;
  `plan_workers(..., host_bytes=None)` — when budget not injected, raw budget
  = min of available sources; reason shows them. `plan_stage_workers` passes
  through; tests inject both values.
- Check: new + existing planner tests green.

## Step 2 — OVER BUDGET warning (TDD)

- `plan_workers` marks the reason; `plan_stage_workers` logs WARNING with
  per-worker estimate, usable budget, max input voxels.
- Stages 04/05/07 already log the reason — no stage change needed beyond
  the logger the planner uses being visible in stage logs (verify).

## Step 3 — Per-task peak meter (TDD)

- `task_peak_meter()` in `utils/resource_planner.py` (reset `clear_refs`,
  read `VmHWM`); tests as in spec.
- Wire into the per-session task function of stages 04, 05, 07 (one `with`
  + one log line each), with that task's max input voxels for the k figure.
- Check: tests; stage scripts still import under `python -c`.

## Step 4 — Measure and recalibrate

- Kate's laptop: run P000067 into a fresh folder (web rebuilt not required —
  `scripts/`, `utils/`, `configs/` are mounted).
- Read per-session peaks from 04/05/07 logs; update k in
  `configs/resource_config.yaml` with the measurement in the comment.
- Update KI-058 (what was fixed, measured k, what remains).
