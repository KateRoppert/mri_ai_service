# KI-069: An Unreadable Dataset Listing Is "Kappa Unavailable", Not "Empty" — Design

**Date:** 2026-10-14
**Branch:** `fix/ki-069-kappa-entities-unavailable` (from `main` 03a38e1)
**Type:** Fix

## Why

Two symptoms on SibBMS dataset 353, one cause:

1. **False name clash.** Run `14_10_1651` reprocessed the same three
   patients as `12_10_1901` and ended "требует внимания — 0 из 11 (номер
   занят другими данными)". There is no clash: `PatientID:StudyInstanceUID`
   hashes of all 11 sessions are identical in both runs. Status
   `needs_attention` is never retried automatically.
2. **Duplicate entities.** Dataset 353 holds **29 entities for 13 names**
   (sub-001_ses-001 ×4, sub-002_ses-003 ×4, sub-003_ses-003 ×4, six more ×2–3).
   Run `12_10_1901` needed 4 delivery attempts; every attempt re-POSTed
   sessions that an earlier attempt had already delivered (sub-001_ses-001
   got a new entity id in each of three attempts).

Cause (`backend/kappa_uploader.py`, `upload_results` step 3): the dataset
listing is fetched **twice** per attempt — `_get_existing_study_hashes`, then
`_get_existing_entity_names`. `get_dataset_entities` returns `None` on any
transport or HTTP failure, and both helpers turn `None` into an empty set:

- hashes failed, names succeeded → every delivered session is a name clash
  (symptom 1, 14 Oct: one `ConnectTimeout`);
- both failed → nothing looks delivered, everything is POSTed again
  (symptom 2, 12 Oct: dozens of "Error getting dataset entities" during the
  attempts).

The retry added in e26a5d4 (connect 8 s, up to 3 attempts) makes a failed
listing rarer but not impossible — its own commit message records listings
that failed all three attempts.

## Change

1. **One listing per attempt.** `upload_results` fetches the dataset entities
   once (`_fetch_dataset_entities`) and both helpers read from it. Half the
   calls, and the two views can no longer disagree.
2. **No listing → no decisions.** If the listing is `None`, the attempt
   uploads nothing and returns
   `{"dataset_id": …, "error": "entities_unavailable", "total": …}`.
   `kappa_delivery.classify` already maps any top-level `error` to a
   transient `network` state with backoff — no change there. The run log
   shows "Kappa недоступна" as for any network failure.
3. The helper signatures stay `(dataset_id) -> set` (tests monkeypatch
   them); they read the per-attempt cache and raise `KappaUnavailable` if the
   listing failed, which `upload_results` turns into the result above.

An empty dataset (listing `[]`) is unchanged: nothing exists, everything
uploads.

## Out of scope (decisions for Kate)

- **The 16 surplus entities in dataset 353.** Deleting in Kappa is
  irreversible and the copy to keep is the one `patient_registry` /
  `mask_versions` point to. A read-only report script (which entity ids are
  referenced locally, which are orphans) can come first; deletion only on
  explicit request.
- `_reconcile_session` after a POST without entity id also reads the
  listing; with this fix a failed reconcile still ends as "upload failed",
  and the next attempt finds the entity by hash instead of POSTing again.

## Rollout

Backend is baked into the `web` image: rebuild, then the manual
"retry upload" for run `14_10_1651` turns its 11 sessions into duplicates
already in the dataset → done.

## Verification

- Unit tests (`backend/test_kappa_uploader_entities_unavailable.py`):
  listing `None` → no POST, `error == "entities_unavailable"`, `classify`
  → `pending`/`network`; listing fetched exactly once per attempt; hashes
  present → duplicate path, not name clash; `[]` → uploads.
- Existing uploader / delivery tests stay green.
