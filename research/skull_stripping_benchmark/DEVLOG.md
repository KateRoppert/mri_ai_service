# Skull Stripping Benchmark — Development Log

Updated at the end of each working session: date · done · blockers · next step.
Tracks actual vs estimated timeline (estimate: 6–8 weeks part-time).

## 2026-09-04 (Task B)
- **Done:** SynthStrip plugin: `synthstrip.py`, `STRIPPERS["synthstrip"]`, manifest `services/skull-stripping/synthstrip/manifest.yaml`. CLI: `mri_synthstrip -i -o -m`, optional `-b` from `tool_params.border`. `uses_gpu = False` (does not take the HD-BET pool). Official CLI also has `-g/--gpu` — not wired. Production config unchanged (hdbet → bet). Tests: `test_synthstrip.py` + cascade/GPU still green.
- **Install smoke:** On the **host venv** `pip install surfa` still fails without `sudo python3-dev`. Stage 05 does **not** use that venv: the web image already ships FreeSurfer `mri_synthstrip` (`FREESURFER_HOME=/opt/freesurfer`, weights in `models/synthstrip.1.pt`). `shutil.which` inside the container returns `/opt/freesurfer/bin/mri_synthstrip`. `--help` matches our flags (`-i/-o/-m/-b/-g/--no-csf/--model`). A synthetic 64³ cube crashed inside SynthStrip's CC step (not a brain); real Stage 05 T1 is the smoke. Host venv `is_available()` remains False.
- **Blockers:** none for Stage 05 (FreeSurfer CLI in the web image). Host venv still has no `mri_synthstrip`.
- **Next step:** Task D (MNI baseline, no extra install).

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
