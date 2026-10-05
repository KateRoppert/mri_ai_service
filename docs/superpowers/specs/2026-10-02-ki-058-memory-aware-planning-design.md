# KI-058: Plan Workers Against Real Free Memory, Measure What Workers Use — Design

**Date:** 2026-10-02
**Branch:** `fix/ki-058-memory-aware-planning` (from `main`)
**Type:** Bugfix + observability (follow-up to
`2026-07-17-memory-aware-worker-sizing-design.md`)

## Problem

Memory-aware worker sizing (July) caps each stage's workers at
`(budget - reserve) / (max_voxels × k_stage)`. On the 15 GiB demob laptop it
failed three ways (KI-058):

1. **Budget ignores the host.** `budget = cgroup memory.max × 0.85`. The
   compose cap for `web` is `20g`, above the laptop's physical RAM, so stage 07
   planned 2 workers against an 18.3 GB budget while ~2 GB was held by
   `ms-seg`'s loaded model and ~3 GB by the desktop → **host-wide** OOM
   (2026-09-30). Workaround in place: local override `mem_limit: 9g`.
2. **k is wrong and nobody can tell.** Stage 05 `k = 17.7` B/voxel predicts
   4.1 GB for the 231 Mvox SibBMS session; the worker was killed above 9.2 GB
   (before the registration fix). Stage 07 predicted 3.85 GB, workers were
   killed at 5.1–5.5 GB (not even their peak). No stage logs what a worker
   actually used, so k cannot be checked or recalibrated.
3. **Silent when even one worker does not fit.** The July spec promised a
   WARNING; `plan_workers` only returns `min_workers=1`. The operator learns
   about it from the OOM.

Facts checked on 2026-10-02: inside the container `/proc/meminfo` reports the
host (`MemTotal` 15.2 GiB, `MemAvailable` ~11.5 GiB) — so a stage can see
real free memory without new mounts.

## Design

### A. Budget = min(container limit, host memory available now)

```
raw_budget = min(cgroup memory.max, host MemAvailable)   # either may be absent
budget     = raw_budget × safety_factor                   # unchanged, 0.85
usable     = budget − reserve_bytes                       # unchanged
```

`MemAvailable` is read when the stage starts planning: it already excludes
`ms-seg`'s model, the desktop, other containers. If neither source exists
(not Linux / not in a container) → unbounded, as today. Logged reason shows
both inputs, e.g. `budget=min(cgroup 9.0, host-avail 11.4)GB×0.85`.

Not using `MemTotal`: it ignores what other processes hold, which is exactly
what broke on 2026-09-30.

### B. Say it when one worker does not fit

When `per_worker_bytes > usable`, `plan_workers` still returns 1 but the reason
carries `OVER BUDGET`, and the stage logs a WARNING:
`одна задача по оценке ~X ГБ, доступно ~Y ГБ — возможен OOM (sub-…/ses-… <voxels> Мвокс)`.
Purely informational — the run still tries.

### C. Measure per-task peak memory (stages 04, 05, 07)

Each worker task, at start, resets the process' high-water mark
(`echo 5 > /proc/self/clear_refs`), and at the end reads `VmHWM` from
`/proc/self/status`. Logged per session:

`peak RSS 6.2 GB · max input 231.4 Mvox · k=26.8 B/voxel`

Works in sequential and parallel mode (each pool worker is its own process;
the reset makes the reading per task, not cumulative). Fail-safe: if
`/proc` is unavailable, log nothing. Lives in `utils/resource_planner.py`
(`task_peak_meter()` context manager) so stages add two lines.

This turns every run into calibration data and makes the next OOM explainable
from the log alone.

### D. Recalibrate k for stages 05 and 07

After C is in, run SibBMS `P000067` (2D sessions 1–9 Mvox and 3D sessions
120–231 Mvox) and set `k` from the measured worst case (+ the usual
`safety_factor` margin), in `configs/resource_config.yaml`, with the
measurement recorded in the comment. Stage 04 too if its measured k differs.

### Out of scope

- Scheduling by per-session cost (big sessions alone, small ones in parallel):
  today the max voxels of the whole run sets one worker count for all
  sessions. Worth doing later (KI-043 admission control).
- Making the compose `mem_limit` follow the machine (env var): backend
  `Settings` reads the same `.env`, need to confirm unknown keys are ignored
  before putting compose vars there. The laptop already has its override.
- VRAM planning for stage 06.

## Testing

- `host_available_bytes(meminfo_path)`: parses `MemAvailable`; missing file /
  field → None.
- `plan_workers`: budget is the min of cgroup and host; either None; both None
  → unbounded; reason string shows both.
- Over-budget: `per_worker_bytes > usable` → 1 worker, reason contains
  `OVER BUDGET`; stage wrapper logs a WARNING (caplog).
- `task_peak_meter`: allocate ~200 MB inside → reported peak ≥ that; a second
  task after a large first one reports its own (small) peak — proves the reset;
  unreadable `/proc` → yields None, no exception.
- Existing planner tests updated where they assumed cgroup-only budget
  (inject host value explicitly; no test reads the real `/proc/meminfo`).
- Real: P000067 run → per-session peak lines in 04/05/07 logs; k updated.
