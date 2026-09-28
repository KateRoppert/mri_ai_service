"""Credentials held across a Kappa outage — and dropped the moment they
should be."""
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, "backend")

import kappa_pending_login as pending  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    pending.forget("тест")
    yield
    pending.forget("тест")


def test_remembers_the_login_id_but_never_exposes_the_password():
    pending.remember("e.roppert", "secret")
    assert pending.held_login() == "e.roppert"
    # Nothing public may hand the password back out.
    public = {n: getattr(pending, n) for n in dir(pending) if not n.startswith("_")}
    assert "secret" not in repr(public)


@pytest.mark.asyncio
async def test_successful_login_drops_the_credentials(monkeypatch):
    """Once a session exists the password is redundant, so it must not linger."""
    async def _ok(login_id, passwd):
        return {"session_id": "s1", "user_id": 26}

    monkeypatch.setattr("kappa_auth.kappa_login", _ok)
    pending.remember("e.roppert", "secret")

    result = await pending.try_login_now()

    assert result["session_id"] == "s1"
    assert pending.held_login() is None


@pytest.mark.asyncio
async def test_rejected_credentials_are_dropped_not_retried(monkeypatch):
    """Retrying a wrong password is how an account gets locked out."""
    calls = []

    async def _rejected(login_id, passwd):
        calls.append(login_id)
        return None

    monkeypatch.setattr("kappa_auth.kappa_login", _rejected)
    pending.remember("e.roppert", "wrong")

    assert await pending.try_login_now() is None
    assert pending.held_login() is None

    # A second tick must not try again — there is nothing held any more.
    assert await pending.try_login_now() is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unreachable_kappa_keeps_the_credentials(monkeypatch):
    """Still down is not a rejection: keep waiting."""
    from kappa_auth import KappaUnreachable

    async def _down(login_id, passwd):
        raise KappaUnreachable("connect timed out")

    monkeypatch.setattr("kappa_auth.kappa_login", _down)
    pending.remember("e.roppert", "secret")

    assert await pending.try_login_now() is None
    assert pending.held_login() == "e.roppert"


@pytest.mark.asyncio
async def test_an_unexpected_error_does_not_escape(monkeypatch):
    """This runs inside the delivery loop; an exception would stop every
    deferred upload in the system."""
    async def _boom(login_id, passwd):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("kappa_auth.kappa_login", _boom)
    pending.remember("e.roppert", "secret")

    assert await pending.try_login_now() is None
    assert pending.held_login() == "e.roppert"


def test_credentials_expire_after_the_ttl():
    pending.remember("e.roppert", "secret")
    pending._held["stored_at"] = (
        datetime.now(timezone.utc)
        - timedelta(hours=pending.CREDENTIAL_TTL_HOURS + 1)
    )
    assert pending.held_login() is None


def test_nothing_is_held_without_a_password():
    pending.remember("e.roppert", "")
    assert pending.held_login() is None
