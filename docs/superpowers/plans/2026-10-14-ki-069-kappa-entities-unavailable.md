# KI-069 — Plan

Spec: `docs/superpowers/specs/2026-10-14-ki-069-kappa-entities-unavailable-design.md`
Branch: `fix/ki-069-kappa-entities-unavailable` (worktree `../mri_ai_service-mem`)

1. Tests first (`backend/test_kappa_uploader_entities_unavailable.py`),
   with `get_dataset_entities` faked and POST asserted not called:
   - listing `None` → result `error="entities_unavailable"`, no POST, no
     name clash; `classify` gives `pending` with reason `network`;
   - listing fetched once per `upload_results`;
   - listing with the session's name and hash → duplicate path;
   - listing `[]` → upload attempted.
2. `kappa_uploader.py`: `KappaUnavailable`; `_fetch_dataset_entities`
   (per-attempt cache, reset at the start of `upload_results`); the two
   helpers read it; `upload_results` catches `KappaUnavailable` → result.
3. KNOWN_ISSUES: KI-067 (max_subjects), KI-068 (start guard), KI-069 (this,
   closed by the branch) — renumbered after main's KI-066; the local
   diffusion entry is dropped (fixed by `fix/stage01-exclude-diffusion`).
4. Full `backend/` suite in a throwaway container.
5. Kate: rebuild `web`, "retry upload" on `14_10_1651`.

No commit / merge / push without Kate's request.
