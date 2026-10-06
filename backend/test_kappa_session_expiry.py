"""
An expired Kappa session is no session.

2026-10-06: the validation tab listed no MS sessions. The session's token had
expired the day before; get_session() returned it as live anyway, so /me said
"logged in", every Kappa call went out with a dead token, Kappa answered 401
and /api/kappa/entities turned that into an empty list. The UI already shows
the login form when /me answers 401 — it just never got one.
"""
import pytest
from fastapi import HTTPException

from database import SessionLocal
from kappa_auth import get_session
from registry_models import KappaSession

PAST = "2000-01-01T00:00:00.000000Z"
FUTURE = "2099-01-01T00:00:00.000000Z"


@pytest.fixture
def sessions():
    """Insert sessions by id -> expiry; clean them up afterwards."""
    db = SessionLocal()
    added = []

    def add(session_id, expiry):
        db.add(KappaSession(session_id=session_id, kappa_token=f"token-{session_id}",
                            user_id=990061, user_type_id=3, user_name="test-user",
                            token_expiry=expiry))
        db.commit()
        added.append(session_id)
        return session_id

    yield add

    for sid in added:
        row = db.query(KappaSession).filter(KappaSession.session_id == sid).first()
        if row:
            db.delete(row)
    db.commit()
    db.close()


def test_live_session_is_returned(sessions):
    sid = sessions("test_exp_live", FUTURE)
    assert get_session(sid)["kappa_token"] == "token-test_exp_live"


def test_expired_session_is_none(sessions):
    sid = sessions("test_exp_dead", PAST)
    assert get_session(sid) is None


def test_expired_session_on_request(sessions):
    sid = sessions("test_exp_dead_incl", PAST)
    assert get_session(sid, include_expired=True)["kappa_token"] == "token-test_exp_dead_incl"


@pytest.mark.parametrize("expiry", [None, "", "not a date"])
def test_unknown_expiry_counts_as_live(sessions, expiry):
    """Same rule as find_live_session_for_user: "cannot tell" is not "expired"."""
    sid = sessions(f"test_exp_unknown_{expiry!r}", expiry)
    assert get_session(sid) is not None


@pytest.mark.asyncio
async def test_me_with_expired_session_is_401(sessions):
    import app
    sid = sessions("test_exp_me", PAST)
    with pytest.raises(HTTPException) as err:
        await app.kappa_me(session_id=sid)
    assert err.value.status_code == 401


@pytest.mark.asyncio
async def test_logout_works_on_an_expired_session(sessions, monkeypatch):
    """Logging out must still forget the pending login, expired or not."""
    import app
    import kappa_pending_login
    forgotten = []
    monkeypatch.setattr(kappa_pending_login, "forget", lambda reason: forgotten.append(reason))
    sid = sessions("test_exp_logout", PAST)

    await app.kappa_logout(session_id=sid)

    assert forgotten


@pytest.mark.asyncio
async def test_entities_with_expired_session_is_401_without_calling_kappa(sessions, monkeypatch):
    import app
    import kappa_client

    async def must_not_be_called(**kwargs):
        raise AssertionError("asked Kappa with a dead token")

    monkeypatch.setattr(kappa_client, "get_dataset_entities", must_not_be_called)
    sid = sessions("test_exp_entities", PAST)

    with pytest.raises(HTTPException) as err:
        await app.get_kappa_entities(dataset_id=353, session_id=sid)

    assert err.value.status_code == 401
    assert "истекла" in err.value.detail
