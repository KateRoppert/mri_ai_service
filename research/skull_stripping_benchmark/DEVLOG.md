# Skull Stripping Benchmark — Development Log

Updated at the end of each working session: date · done · blockers · next step.
Tracks actual vs estimated timeline (estimate: 6–8 weeks part-time).

## 2026-09-04 (evening)
- **Done:** Task A — cascade + two-tier mask validation. `validation.py` (`mask_metrics`, catastrophe vs review flags, speckle drop `< 1 ml`). `build_cascade_order` / `try_stripper`; `process_subject_skull_stripping` retries only on hard fail; GPU slot released before CPU BET. Production config: `validation: {}`, cascade omitted (= hdbet → bet). Paper §3.6 EN+RU filled with the gate design (no numbers yet). Tests: 32 passed (`test_validation`, `test_cascade`, GPU pool/dispatch).
- **Not done:** none for Task A.
- **Blockers:** none.
- **Next step:** Task B SynthStrip.

## 2026-09-04
- **Done:** Reconnected onto `origin/main` as `feat/skull-stripping-research-v2` (docs cherry-picked; June BET-only plugin dropped). Executable plan rewritten: [2026-09-04-skull-stripping-research.md](../../docs/superpowers/plans/2026-09-04-skull-stripping-research.md). June plan frozen.
- **On main already:** SkullStripperBase, BET, HD-BET 2.x, dispatcher fallback, GPU pool, production `method: hdbet`.
- **Still to do:** cascade + mask validation, SynthStrip / BrainMaGe / MNI / SAM / DeepBET, benchmark harness, paper fill.
- **Blockers:** none for Task A (cascade). MosMed + clinical sets needed later for the full CSV.
- **Next step:** Task A — cascade + `validation.py` without breaking the GPU pool.

## 2026-06-16
- **Done:** Scaffolding created (research dir, requirements, paper skeleton dirs).
- **Blockers:** none.
- **Next step:** implement plugin architecture (SkullStripperBase + BET refactor).
