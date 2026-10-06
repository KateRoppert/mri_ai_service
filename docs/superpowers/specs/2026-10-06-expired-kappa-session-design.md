# Treat an Expired Kappa Session as No Session — Design

**Date:** 2026-10-06
**Branch:** `fix/slicer-ms-red-and-agent-deps` (with the Slicer fixes found in
the same check)
**Type:** Bugfix (silent empty lists after the token expires)

## Problem

On 2026-10-06 the validation tab showed no MS sessions for dataset 353. The
Kappa session in the DB had `token_expiry = 2026-10-05T11:15:31Z`. Kappa
answered `401` to every request; `get_dataset_entities` returns `None` on any
failure and `/api/kappa/entities` turned that into `200 {"entities": []}`.

The UI already knows how to recover: on page load it calls `/api/kappa/me`,
and on `401` it forgets `kappa_session_id` and shows the login form
(`App.jsx`, restore effect). It never gets the chance, because
`kappa_auth.get_session()` does not look at `token_expiry` — an expired session
is returned as live, `/me` answers `200`, and every Kappa call goes out with a
dead token.

`find_live_session_for_user()` (deferred upload) already skips expired rows;
`get_session()` was simply never given the same rule.

## Design

1. `get_session(session_id, include_expired=False)`: returns `None` when the
   token has expired. Same rule as `find_live_session_for_user`: a NULL or
   unparseable `token_expiry` counts as live ("cannot tell" ≠ "expired");
   expiry compared in UTC via the existing `_parse_expiry`.
2. `/api/kappa/logout` calls `get_session(..., include_expired=True)`: logging
   out of an expired session must still forget the pending login and clear
   the outcome, as it does today.
3. Everything else keeps calling `get_session(sid)` and gets `None` for an
   expired session — which each caller already handles as "no session":
   - `/api/kappa/me` → `401` → UI shows the login form on load;
   - Kappa data endpoints (`/kappa/entities`, `/validation/*`, Slicer
     open-from-kappa) → `401` instead of empty data;
   - run start (`numbering.scope_for_run`) → offline numbering from
     `kappa_datasets.yaml`, instead of calling Kappa with a dead token;
   - end-of-run upload (`pipeline_monitor`) → no uploader; the deferred
     delivery worker picks up the run with the user's live session
     (`find_live_session_for_user`) once they log in again.
4. `/api/kappa/entities` detail becomes «Сессия Kappa не найдена или истекла —
   войдите заново»; `ValidationPanel` shows the backend's detail on `401`
   instead of the generic «Не удалось загрузить список сессий». Covers a
   session that expires while the page is open.

### Out of scope

- Distinguishing Kappa's own `401` from network errors inside
  `kappa_client` (all failures return `None`). With expiry checked up front,
  the remaining case — Kappa revoking a token before its expiry — is rare.
- A global axios interceptor that sends every `401` to the login form.

## Testing

- `backend/test_kappa_session_expiry.py` (temp DB rows via `SessionLocal`):
  live → returned; expired → `None`; expired with `include_expired=True` →
  returned; NULL / unparseable expiry → returned.
- `/api/kappa/me` with an expired session → `401`.
- `/api/kappa/logout` with an expired session → `200`, pending login forgotten.
- `/api/kappa/entities` with an expired session → `401`, no Kappa call.
- Full `backend/` suite green; frontend lint: no new findings.
- Real: the laptop's current session (expired 2026-10-05) → reload the page →
  login form; after login the MS sessions are listed.
