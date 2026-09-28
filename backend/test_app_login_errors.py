"""Login must fail in words the operator can act on.

An unhandled httpx error made FastAPI answer 500, whose body is the plain
string "Internal Server Error" — not JSON. The login form parsed it and
showed "JSON.parse: unexpected character at line 1 column 1", which tells
the operator nothing about Kappa being down.
"""
import sys

import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, "backend")


@pytest.mark.asyncio
async def test_unreachable_kappa_answers_503_not_500(monkeypatch):
    import app
    from kappa_auth import KappaUnreachable
    from app import KappaLoginRequest

    async def _unreachable(login_id, passwd):
        raise KappaUnreachable("connect timed out")

    monkeypatch.setattr(app, "kappa_login", _unreachable)

    with pytest.raises(HTTPException) as caught:
        await app.kappa_login_endpoint(
            KappaLoginRequest(login_id="x", passwd="y")
        )

    assert caught.value.status_code == 503
    assert "Kappa" in caught.value.detail
    # Says what the operator can still do, not just that something broke.
    assert "без входа" in caught.value.detail


@pytest.mark.asyncio
async def test_bad_credentials_still_answer_401(monkeypatch):
    """The two failures must stay distinguishable."""
    import app
    from app import KappaLoginRequest

    async def _rejected(login_id, passwd):
        return None

    monkeypatch.setattr(app, "kappa_login", _rejected)

    with pytest.raises(HTTPException) as caught:
        await app.kappa_login_endpoint(
            KappaLoginRequest(login_id="x", passwd="y")
        )
    assert caught.value.status_code == 401


def test_kappa_login_raises_a_named_error_on_network_failure(monkeypatch):
    """kappa_login must convert transport errors, not let them escape."""
    import asyncio

    import kappa_auth

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(kappa_auth.httpx, "AsyncClient", lambda **kw: _Client())

    with pytest.raises(kappa_auth.KappaUnreachable):
        asyncio.run(kappa_auth.kappa_login("user", "pass"))
