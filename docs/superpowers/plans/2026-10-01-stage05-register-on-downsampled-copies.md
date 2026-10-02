# Stage 05: Register on ≤1 mm Copies — Plan

Spec: `docs/superpowers/specs/2026-10-01-stage05-register-on-downsampled-copies-design.md`
Branch: `fix/stage05-register-on-downsampled-copies`

Tests run in a throwaway `brain-lesion-web:demo` container (has ANTs).
Commit per step only when Kate asks.

## Step 1 — Tests first

`scripts/preprocessing_steps/test_registration_downsample.py`:
- `registration_spacing((0.347, 0.347, 0.55), 1.0) == (1.0, 1.0, 1.0)`;
  `(0.47, 0.47, 5.0) → (1.0, 1.0, 5.0)`; `(1.2, 1.2, 6.0)` unchanged.
- Synthetic: a 0.5 mm blob volume and a copy shifted by 3 mm; spy on
  `ants.registration` → receives images with spacing ≥1 mm; recovered
  translation within 0.5 mm of 3 mm; transform file written.
- A ≥1 mm pair is passed to `ants.registration` unresampled (same object
  shape/spacing) — sessions 001–004 behave exactly as before.

## Step 2 — Implement

- `registration.py`: `registration_spacing(spacing, max_mm)`;
  `_downsample_for_registration(img, max_mm)` (returns `img` itself when no
  axis is finer); `register_modalities(..., max_registration_spacing_mm=1.0)`
  uses it for fixed and moving, logs original → registration shapes.
- `process_subject_registration`: pass
  `params.get("max_registration_spacing_mm", 1.0)`.
- `configs/preprocessing_config.yaml`: add the key with a comment (why, and
  that it only affects images finer than the value).
- Check: new tests + `scripts/preprocessing_steps/skull_stripping/tests`
  + `python -m pytest backend/ -q` green.

## Step 3 — Real run

- No rebuild or restart: `scripts/` and `configs/` are bind-mounted into
  `web`, and each stage starts a fresh Python process.
- Re-run `P000067` into a fresh folder. Expect 6/6 through 05–08 and
  `Kappa 6/6`; watch stage 07 peak memory on ses-005/006 (spec risk).
- Update KI-058 with the corrected cause (and stage 07 result).
