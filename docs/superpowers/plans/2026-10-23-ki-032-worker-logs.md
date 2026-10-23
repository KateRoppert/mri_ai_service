# KI-032 — Plan

Spec: `docs/superpowers/specs/2026-10-23-ki-032-worker-logs-design.md`

1. Test first: `utils/test_stage_logging.py` (fork + spawn, no duplicates,
   idempotent setup).
2. `utils/stage_logging.py`.
3. Stages 01/03/04 `setup_logging` + worker loggers; pool initializers in
   01/03/04/05/07/08.
4. Live: stages 01 and 03 in parallel mode on a few patients; compare with
   `main`.
5. KNOWN_ISSUES KI-032 closed; all suites.
