# KI-032: Worker Logs Reach the Stage Log File — Design

**Date:** 2026-10-23
**Branch:** `fix/ki-032-worker-logs` (from `main` 682f6c0)
**Type:** Fix (debuggability, before the resource-allocation MAS research)

## Why

What a worker logs must end up in `{output}/logs/stages/stage_XX.log`;
today, for three stages, it does not (checked on run `14_10_1651` and in the
code):

| stage | file handler on | worker logs via | in the file? |
|---|---|---|---|
| 01 | logger `reorganize_folders` | `patient_<id>` with its own StreamHandler | **no** — stderr, in parallel *and* sequential mode |
| 03 | logger `nifti_converter` | `converter_worker_<id>_<mod>`, no handler | **no** — WARNING+ to stderr via logging's last resort |
| 04 | logger `quality_assessor` | the stage logger (inherited on fork) | yes; but helper modules' `getLogger(__name__)` do not |
| 05, 07, 08 | root logger | module loggers | yes |

All of this rests on `fork` copying the parent's handlers into workers. The
image runs Python 3.12 (`fork`); from 3.14 Linux defaults to `forkserver`,
and no stage would keep worker logs.

## Change

1. `utils/stage_logging.py`:
   - `configure_stage_logging(log_file, console_level, file_level)` — console
     and file handlers on the **root** logger (what 05/07/08 already do),
     replacing handlers it added before; file format carries the logger name
     and the process name, so interleaved worker lines can be told apart.
   - `worker_initializer` / `worker_initargs()` — for pools: in a worker
     that did not inherit handlers (spawn/forkserver) re-attach the same file
     handler; a no-op under fork.
2. Stages 01, 03, 04: `setup_logging` uses it and returns the stage logger
   (no handlers of its own, propagates). Worker loggers become children of
   the stage logger without handlers (`reorganize_folders.patient_<id>`,
   `nifti_converter.worker`).
3. Every process pool in stages 01, 03, 04, 05, 07, 08 passes
   `initializer=worker_initializer, initargs=worker_initargs()`.

## Verification

- `utils/test_stage_logging.py`: a warning from an arbitrary logger in a
  worker lands in the file — under `fork` and under `spawn`; no duplicate
  lines; reconfiguring does not stack handlers.
- Stage-level: stage 01 and 03 run in parallel mode on a small input
  (throwaway container) — a per-patient / per-series warning appears in
  `stage_01_reorganize.log` / `stage_03_convert.log`, which it does not on
  `main`.
- All test suites stay green.

## Results (2026-10-23)

- `utils/test_stage_logging.py` (7): worker warnings reach the file under
  fork and spawn, no duplicate lines, stage DEBUG kept and third-party DEBUG
  dropped, a stage that configures root itself keeps its level in workers.
- Live, stage 01 parallel, synthetic patient with dates stripped + KA08,
  `--dry_run`: "Could not parse date" — `main`: stderr only; this branch: in
  the stage log with process and logger name. Sequential mode: in the log on
  both.
- Stage 03 is covered by the same mechanism and the tests; no failing
  conversion was staged live.
- Suites (with `fix/ki-063-stale-root-tests`): utils 31, root 225, backend
  561, scripts 75, tests/ 212 + 6 skipped for a fixture dataset absent from
  the temporary worktree.

### Follow-up (same day): workers say what they do

Kate ran two patients and saw no worker lines: warnings reached the file,
but on a clean run workers of 01/03/04 logged nothing (01 ran them at
WARNING in parallel mode, 03 too, 04 had no INFO on the worker path). Now
one INFO line per task — 01 per patient, 03 per series with outcome and
time, 04 per image with category, time and peak memory — and 05/07/08 carry
the process name. Live, KA08 + KA45, parallel: stage 01 11 lines per worker,
03 one per series (`ForkPoolWorker-1 … 001/ses-001 t2: ok in 0.3s`), 04 one
per image (`ForkProcess-3 … t2: POOR in 0.6s — peak RSS 0.13 GB …`).
