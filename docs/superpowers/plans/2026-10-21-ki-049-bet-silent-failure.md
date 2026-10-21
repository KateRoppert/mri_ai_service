# KI-049 — Plan

Spec: `docs/superpowers/specs/2026-10-21-ki-049-bet-silent-failure-design.md`
Branch: `fix/ki-049-bet-silent-failure`

1. Tests first: `scripts/preprocessing_steps/skull_stripping/tests/test_bet_paths.py`
   (real `bet` on a small phantom/real t1 copy; spaced dir; invalid input;
   rc 0 without mask via a fake `bet` on PATH) and a stage-05 test for a
   failing stripper (`scripts/preprocessing_steps/test_stage05_skull_strip_failure.py`).
2. `bet.py::run_bet`: private temp dir + symlinked input, move results,
   stdout+stderr in errors, existence checks.
3. `05_preprocessing.py` step 4: raise on failure (reference or any modality),
   handle the `{"success", "error"}` shape.
4. Live check in a throwaway container: stage 05 on one SibBMS session into
   `…/Рабочий стол/…`, BET and HD-BET, both positions.
5. KNOWN_ISSUES: KI-049 closed; full test suites.

No commit / merge / push without Kate's request.
