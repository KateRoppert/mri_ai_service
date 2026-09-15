import asyncio

import app as backend_app


def test_endpoint_resolves_user_from_session(monkeypatch):
    calls = {}
    monkeypatch.setattr(backend_app, "get_session",
                        lambda sid: {"user_id": 52, "user_type_id": 4, "kappa_token": "t"}
                        if sid == "sess-52" else None)
    monkeypatch.setattr(backend_app, "get_lesion_types",
                        lambda user_id: calls.setdefault("uid", user_id) or [])

    asyncio.run(backend_app.get_lesion_types_endpoint(kappa_session_id="sess-52"))
    assert calls["uid"] == 52


def test_endpoint_without_session_uses_none(monkeypatch):
    calls = {}
    monkeypatch.setattr(backend_app, "get_session", lambda sid: None)
    monkeypatch.setattr(backend_app, "get_lesion_types",
                        lambda user_id: calls.setdefault("uid", user_id) or [])

    asyncio.run(backend_app.get_lesion_types_endpoint(kappa_session_id=None))
    assert calls["uid"] is None
