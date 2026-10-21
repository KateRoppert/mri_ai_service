# KI-075 — Plan

Spec: `docs/superpowers/specs/2026-10-21-ki-075-kappa-labeled-status-design.md`
Branch: `fix/ki-075-kappa-labeled-status`

1. Labels from the segmentation service manifest of the lesion type
   (`output.classes` minus background), fallback `[lesion_type]`.
2. Tests first (backend): `upload_entity` create JSON (`filesCategory`,
   `labels`); `update_entity_status` → v2 PUT, no files, `(ok, message)`;
   `patch_file` / `add_files` category; validation endpoint surfaces a refusal.
3. Implement 1–4 of the spec; frontend warning in `ValidationPanel.jsx`;
   `npm run lint`.
4. `scripts/kappa_backfill_labels.py` (dry-run default) + test with fakes.
5. Live check on dataset 357 (uploader, validation transitions, KI-035
   completion); backfill dry-run on a real dataset.
6. KNOWN_ISSUES: KI-075 and KI-036 closed; full suites.
7. Kate: rebuild `web`; decide backfill.

No commit / merge / push without Kate's request.
