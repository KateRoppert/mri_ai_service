# KI-035 — Plan

Spec: `docs/superpowers/specs/2026-10-19-ki-035-complete-incomplete-entity-design.md`
Branch: `fix/ki-035-kappa-completes-incomplete-study`

1. Tests first: `backend/test_kappa_uploader_completion.py` (cases from the
   spec's Verification), with `kappa_client.get_dataset_entities` and
   `kappa_entity_files.replace_entity_contents` faked; POST asserted never
   called for a known hash.
2. `kappa_uploader.py`:
   - `_entity_by_hash` from the per-attempt listing (KI-069 cache);
   - `_entity_completeness(entity, session_key)` → complete / incomplete /
     unknown (no `files` key), plus has-expert-masks;
   - duplicate branch: complete → as today; incomplete + no expert →
     `_complete_entity` (replace, status, info update, registration);
     incomplete + expert → `supersedes`.
3. `kappa_entity_files.py`: `update_entity_info(...)` behind the verification
   in step 5; until verified it is not called (flag in code, not config).
4. `kappa_run_log.py`: "дополнено (entity …)" for `completed`.
5. Live check of `PUT` with metadata only, on a throwaway entity in a test
   dataset — only with Kate's go-ahead (writes to Kappa).
6. KNOWN_ISSUES: KI-035 closed with the audit table; full `backend/` suite.
7. Kate: rebuild `web`, re-run one audited patient.

No commit / merge / push without Kate's request.
