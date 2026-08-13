# Review Queue Audit Trail — Design

## Context

The incomplete-patients review UI (built in `feat/incomplete-patients-frontend`) currently treats the review queue as an *ephemeral filter*: `get_incomplete_patients()` returns only sessions currently needing attention (incomplete, or complete-with-unused-alternatives). The moment a session is resolved — either because a manual relabel completed it, or because the doctor discarded it — it drops out of the report entirely on the next fetch.

Manual testing against real data (2026-08-13, run `13_08_1806`, 5 real BO patients) surfaced this as a real gap, not just a cosmetic one:

- Assigning the last missing modality to a session makes it vanish from the list with no visible confirmation that the action succeeded.
- Discarding a session makes it vanish too, with no record that the doctor already made that call — a later reviewer (or the same doctor, later) has no way to tell "nobody's looked at this" from "I already decided to skip this."

This spec covers only that gap. Three related findings from the same testing session were explicitly scoped OUT and deferred to their own specs:
- Showing a patient's *other* visits for cross-session context (this is "Piece A" / session-merge, already deferred before this branch existed).
- Surfacing a requeued run's progress from the original active-run view (`ProgressMonitor`) instead of only in run history — no `parent_run_id` concept exists in the data model today; a separate, small design question.
- An end-to-end "did this patient reach the output" report spanning all 8 pipeline stages — a materially larger feature than the other three combined.

## Design

### Data model & backend filter

Add one field to each session's dict inside `bids_organized/dataset_mapping.json`: `manually_reviewed: bool`. Absent on any session dict written before this change; `.get('manually_reviewed', False)` treats that as `False`, so no migration is needed.

Both `relabel_series()` and `discard_session()` (in `backend/pipeline_manager.py`) set `manually_reviewed = True` on the session dict they touch, in addition to whatever else they already do (updating `series`/`excluded_series`/`status`).

`get_incomplete_patients()`'s inclusion rule gains one more `OR` clause:

```python
needs_review = (
    status == 'incomplete'
    or (status == 'complete' and has_alternatives)
    or status == 'discarded'
    or session_data.get('manually_reviewed', False)
)
```

The `status == 'discarded'` sessions were previously excluded outright — that exclusion is removed; they're now explicitly included via this same rule.

Because `dataset_mapping.json` lives at the run's `output_path`, and requeues reuse that same `output_path` (that's how `skip_existing` already works), this marker persists across page reloads and across a requeue automatically — no separate tracking table, no new endpoint, no frontend-side merge logic.

### API response shape

`IncompletePatientSession` (backend/models.py) gains no new field — `status` already distinguishes `incomplete` / `complete` / `discarded`, and the frontend derives its four visual states from `status` + `excluded_series` presence, exactly as it already derives "Неполная" vs "Есть альтернативы" today. `manually_reviewed` itself is a backend-internal bookkeeping field and does not need to round-trip to the frontend.

### Frontend display

`IncompletePatients.jsx`'s status column gains two more cases on top of the existing two:

| Condition | Tag | Color |
|---|---|---|
| `status === 'incomplete'` | Неполная | orange (existing) |
| `status === 'complete'` and `excluded_series.length > 0` | Есть альтернативы | blue (existing) |
| `status === 'complete'` and `excluded_series.length === 0` | Полная | green (new) |
| `status === 'discarded'` | Отброшена | grey (new) |

The third row only appears at all because of the backend's `manually_reviewed` clause — an untouched, always-complete session never reaches the frontend, so there's no risk of the list filling up with sessions nobody ever needed to look at.

`IncompletePatientDetail.jsx` renders read-only when `session.status === 'discarded'`: the modality tags and excluded-series list still show (for reference — what did this session look like when it was discarded), but the `Select` + "Назначить" controls and the "Отбросить сессию" button are hidden. No "undo discard" action in this pass (explicitly deferred).

No new frontend state-tracking mechanism is needed for "row updates instead of vanishing": since the backend now keeps returning a manually-touched session, the existing `fetchSessions()` → `setSessions()` flow (already triggered by `onActionComplete` after every successful action) naturally re-renders the row with its new status. This was the whole point of making the backend the source of truth rather than patching frontend state.

### Testing

Backend (TDD, pytest, matching the existing `TestRelabelSeries*`/`TestDiscardSession*` structure in `backend/test_incomplete_patients_api.py`):
- Relabeling the last missing modality → session becomes `complete`, `excluded_series` empties out → `get_incomplete_patients()` still returns it (`manually_reviewed` clause).
- Discarding a session → `get_incomplete_patients()` still returns it with `status: "discarded"`.
- An untouched, always-complete session with empty `excluded_series` → still correctly excluded (regression guard — `manually_reviewed` must default `False`, not `True`).
- A `discarded` session's `relabel_series`/`discard_session` are not called again if the frontend prevents it — not a backend concern; the backend doesn't need to reject a relabel attempt against a discarded session since the frontend won't offer the controls (out of scope to add a backend-side guard for this in the current pass).

Frontend: no automated test framework exists in this project — manual verification (dev server + browser) is the only check, same as the rest of this branch, plus the existing `npm run lint` check.

## Out of Scope (explicitly deferred)

- Un-discarding a session (bringing it back to `incomplete`/`complete` for re-review).
- Cross-session visibility of a patient's other visits (Piece A / session-merge).
- Surfacing a requeued run's progress in the original `ProgressMonitor` view.
- End-to-end, all-stages patient-funnel/audit report.
- A backend-side guard rejecting `relabel_series`/`discard_session` calls against an already-discarded session (the frontend simply won't expose the controls for a discarded row in this pass).
