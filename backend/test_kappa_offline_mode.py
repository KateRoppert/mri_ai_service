"""Working with Kappa down.

The whole deferred-upload feature is pointless if the operator cannot start
a run while Kappa is unreachable — which is exactly when they need it.
"""
import sys

import pytest

sys.path.insert(0, "backend")

import database as db_mod  # noqa: E402
import numbering  # noqa: E402
from database import SessionLocal, create_pipeline_run  # noqa: E402


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_run_intending_upload_gets_a_pending_scope_when_kappa_is_down():
    """Numbers issued offline must be rebindable to whatever dataset the run
    eventually lands in. A `local:` scope cannot be rebound, so its numbers
    would collide with the dataset's own and surface as name_clash.
    """
    scope, dataset_id, warning = await numbering.scope_for_run(
        "test_offline_run", "glioblastoma", None, intends_upload=True,
    )
    assert scope == numbering.pending_scope("test_offline_run")
    assert dataset_id is None
    assert warning is not None


@pytest.mark.asyncio
async def test_cli_run_keeps_the_local_scope():
    """Runs that never intended to upload must not change behaviour."""
    scope, dataset_id, warning = await numbering.scope_for_run(
        "test_cli_run", "glioblastoma", None,
    )
    assert scope == numbering.local_scope("glioblastoma")
    assert dataset_id is None


def test_adopt_orphan_runs_assigns_the_logging_in_user():
    """A run started while Kappa was down has no owner, so the worker can
    never find a token for it. Logging in is what tells us whose it is."""
    from kappa_delivery_worker import adopt_orphan_runs

    db = SessionLocal()
    ids = ["test_adopt_orphan", "test_adopt_owned", "test_adopt_done"]
    try:
        for rid, user, upload in [
            ("test_adopt_orphan", None, "pending"),
            ("test_adopt_owned", 52, "pending"),
            ("test_adopt_done", None, "done"),
        ]:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out", run_id=rid,
                kappa_upload_status=upload, kappa_user_id=user,
            )
            run.status = "completed"
            db.commit()

        adopted = adopt_orphan_runs(26)
        assert adopted == 1

        db.expire_all()
        assert db_mod.get_pipeline_run(db, "test_adopt_orphan").kappa_user_id == 26
        # Someone else's run is not stolen, and a finished one is not revived.
        assert db_mod.get_pipeline_run(db, "test_adopt_owned").kappa_user_id == 52
        assert db_mod.get_pipeline_run(db, "test_adopt_done").kappa_user_id is None
    finally:
        _cleanup(db, *ids)
        db.close()


def test_adoption_clears_the_backoff_so_delivery_resumes_at_once():
    """Logging in is a strong signal that Kappa is reachable again. Making
    the operator wait out a 60-minute backoff after that is indefensible."""
    from datetime import datetime, timedelta, timezone

    from database import set_kappa_delivery
    from kappa_delivery_worker import adopt_orphan_runs

    db = SessionLocal()
    run_id = "test_adopt_resets_backoff"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="pending", kappa_user_id=None,
        )
        run.status = "completed"
        db.commit()
        set_kappa_delivery(
            db, run_id, "pending",
            datetime.now(timezone.utc) + timedelta(hours=1), {"attempts": 6},
        )

        adopt_orphan_runs(26)

        db.expire_all()
        assert db_mod.get_pipeline_run(db, run_id).kappa_upload_next_attempt is None
    finally:
        _cleanup(db, run_id)
        db.close()


def test_reachability_transition_resumes_waiting_runs(monkeypatch):
    """Kappa coming back must pull in the waiting runs by itself.

    Backoff grows to an hour, so after an outage ends a run can sit that long
    doing nothing — which reads as broken. Only the down->up TRANSITION
    resumes, so a genuinely failing run still backs off instead of hammering
    Kappa every minute.
    """
    import kappa_delivery_worker as worker

    worker._last_reachable = None
    calls = []
    monkeypatch.setattr(worker, "resume_all_pending",
                        lambda reason: calls.append(reason) or 1)

    monkeypatch.setattr(worker, "kappa_reachable", lambda: False)
    worker.note_kappa_reachability()
    assert calls == []                      # still down, nothing to do

    monkeypatch.setattr(worker, "kappa_reachable", lambda: True)
    worker.note_kappa_reachability()
    assert len(calls) == 1                  # came back -> resume

    worker.note_kappa_reachability()
    assert len(calls) == 1                  # still up -> backoff respected


def test_kappa_login_returns_what_adoption_needs(monkeypatch):
    """Contract between kappa_login and adopt_orphan_runs.

    adopt_orphan_runs(result.get("user_id")) silently did nothing for weeks
    because kappa_login never returned that key: it exits early on a None
    user id. Logging in appeared to work while no run was ever adopted, so
    offline runs waited on "no session" forever. Testing the two pieces
    separately is exactly what let this through.
    """
    import asyncio
    import json as _json

    import kappa_auth

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "token": "t", "userId": 26, "userTypeId": 1,
                "userName": "e.roppert", "firstName": "Kate", "lastName": "R",
                "tokenExpiryDate": "2099-01-01T00:00:00.000000Z",
                "orgDetails": None,
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            return _Response()

    monkeypatch.setattr(kappa_auth.httpx, "AsyncClient", lambda **kw: _Client())

    result = asyncio.run(kappa_auth.kappa_login("e.roppert", "secret"))

    assert result["user_id"] == 26, "adopt_orphan_runs cannot work without this"
    assert result["user_type_id"] == 1
    _json.dumps(result)          # ответ уходит в HTTP — должен сериализоваться

    # Чистим созданную сессию, чтобы не мешала другим тестам.
    from database import SessionLocal
    from registry_models import KappaSession
    db = SessionLocal()
    try:
        row = db.query(KappaSession).filter(
            KappaSession.session_id == result["session_id"]
        ).first()
        if row:
            db.delete(row)
            db.commit()
    finally:
        db.close()


def test_offline_numbering_uses_the_configured_dataset(monkeypatch, tmp_path):
    """Offline runs must continue the dataset's numbering, not restart at 1.

    pending:<run_id> always starts at sub-001, so binding it into a dataset
    that already holds sub-001 collided on UNIQUE(scope, bids_id) and took the
    whole upload down. The dataset is knowable offline — it is in a YAML file.
    """
    import asyncio

    import kappa_auth
    import kappa_dataset_mapping as kdm
    import numbering

    mapping = tmp_path / "kappa_datasets.yaml"
    mapping.write_text(
        "datasets:\n  26:glioblastoma:current: 351\n", encoding="utf-8",
    )
    monkeypatch.setattr(kdm, "MAPPING_FILE", mapping)
    monkeypatch.setattr(kappa_auth, "sole_known_user_id", lambda: 26)

    scope, dataset_id, warning = asyncio.run(numbering.scope_for_run(
        "test_offline_ds", "glioblastoma", None, intends_upload=True,
    ))

    assert scope == numbering.dataset_scope(351)
    assert dataset_id == 351
    assert "351" in warning


def test_offline_numbering_refuses_to_guess_between_accounts(monkeypatch):
    """Two accounts and no queued login: numbering into the wrong dataset is
    worse than postponing, so fall back to the pending scope."""
    import asyncio

    import kappa_auth
    import kappa_pending_login
    import numbering

    monkeypatch.setattr(kappa_auth, "sole_known_user_id", lambda: None)
    kappa_pending_login.forget("подготовка теста")

    scope, dataset_id, _ = asyncio.run(numbering.scope_for_run(
        "test_offline_ambiguous", "glioblastoma", None, intends_upload=True,
    ))

    assert scope == numbering.pending_scope("test_offline_ambiguous")
    assert dataset_id is None
