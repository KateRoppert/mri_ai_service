# KI-063: Every Test Suite Green — Design

**Date:** 2026-10-23
**Branch:** `fix/ki-063-stale-root-tests` (from `main` 682f6c0)
**Type:** Test hygiene before two research branches start in parallel

## Why

Always-red tests stop catching anything: a regression of the stage-05 memory
fix (`singleprecision` in `tests/preprocessing_steps`) went unnoticed among
15 known failures. One test overwrote the real `configs/series_scoring.yaml`
(restored in `finally` — a killed run left it empty); two others wrote into
the repo root or depended on other files having been collected first.

## Rules applied

- A test that encodes an outdated behaviour is updated to the decided one
  (MPR: `e4f592d`/`42967c7`; GBM reference `t1c`; dict worker results;
  `ProcessPoolExecutor`; integrity-checked resume).
- Numbers that come from configuration are computed from it in the test
  (planner worker counts), not written as literals.
- Tests write only under `tmp_path`; production code gets an optional path
  argument where a test needs one (`load_series_scoring_config`).
- A print-only script is not a test: `test_real_config.py` became a real
  smoke test of `pipeline_config.yaml`, keeping its printout as a CLI.
- No production behaviour changes, except the optional argument above.

## Result

Root 225, `backend/` 561, `scripts/` 75, tracked `tests/` 218 — all passing,
in the throwaway container with the repository mounted read-only (which is
also how the writes-into-repo cases were found). Commands in CLAUDE.md.
