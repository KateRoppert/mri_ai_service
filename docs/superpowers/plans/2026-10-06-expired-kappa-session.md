# Expired Kappa Session — Plan

Spec: `docs/superpowers/specs/2026-10-06-expired-kappa-session-design.md`
Branch: `fix/slicer-ms-red-and-agent-deps`

## Step 1 — Tests first
`backend/test_kappa_session_expiry.py`: cases from the spec. Run → fail.

## Step 2 — Backend
- `kappa_auth.get_session(session_id, include_expired=False)` using
  `_parse_expiry`.
- `app.kappa_logout`: `include_expired=True`.
- `app.get_kappa_entities`: detail text.
- Check: new tests + full `backend/` suite (stubs of `get_session` in other
  tests take one positional arg — keep the new parameter keyword-only with a
  default so they still fit).

## Step 3 — Frontend
- `ValidationPanel.loadEntities`: on `err.response?.status === 401` show
  `err.response.data.detail`.
- Check: lint in the `web` image, no new findings vs `main`.

## Step 4 — Real check
Rebuild `web` (backend and frontend are baked into the image). Reload the
page with the expired session → login form; log in → MS sessions listed.

## Also on this branch (done before this plan)
MS segment red in Slicer; `slicer/requirements.txt` (httpx); CLAUDE.md agent
command; KI-050 / KI-056 closed; user systemd unit for the agent.
