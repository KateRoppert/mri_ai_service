"""The background worker: who it picks up, and what it does with no token."""
from datetime import datetime, timedelta, timezone

import pytest

import database as db_mod
import kappa_delivery_worker as worker
from database import (
    SessionLocal, create_pipeline_run, get_kappa_delivery, set_kappa_delivery,
)

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _make_run(db, run_id, **kwargs):
    run = create_pipeline_run(
        db, input_path="/in", output_path="/out", run_id=run_id,
        kappa_upload_status=kwargs.pop("upload_status", "pending"),
        kappa_user_id=kwargs.pop("user_id", 26),
        kappa_dataset_id=kwargs.pop("dataset_id", 350),
    )
    run.status = "completed"
    run.completed_at = NOW - timedelta(hours=1)
    db.commit()
    return run


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_no_live_session_defers_without_counting_an_attempt(monkeypatch):
    db = SessionLocal()
    run_id = "test_worker_no_session"
    try:
        _make_run(db, run_id)
        set_kappa_delivery(db, run_id, "pending", None, {"attempts": 3})
        monkeypatch.setattr(worker, "find_live_session_for_user",
                            lambda user_id, now=None: None)

        await worker.deliver_one(run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        detail = get_kappa_delivery(run)
        assert detail["reason"] == "no_session"
        assert detail["attempts"] == 3
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_successful_attempt_marks_the_run_done(monkeypatch):
    db = SessionLocal()
    run_id = "test_worker_success"
    try:
        _make_run(db, run_id)
        monkeypatch.setattr(
            worker, "find_live_session_for_user",
            lambda user_id, now=None: {
                "kappa_token": "fresh-token", "user_id": 26, "user_type_id": 1,
            },
        )

        class _Uploader:
            async def upload_results(self):
                return {"dataset_id": 350, "uploaded": 1, "total": 1,
                        "sessions": [{"session": "sub-001_ses-001",
                                      "success": True, "entity_id": "e1"}]}

        monkeypatch.setattr(worker, "build_uploader",
                            lambda run, session: _Uploader())

        await worker.deliver_one(run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "done"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_uploader_is_built_with_the_current_token(monkeypatch):
    """Regression guard: the worker must not reuse a token snapshotted when
    the run started — that token is what expired in the first place."""
    db = SessionLocal()
    run_id = "test_worker_fresh_token"
    seen = {}
    try:
        _make_run(db, run_id)
        monkeypatch.setattr(
            worker, "find_live_session_for_user",
            lambda user_id, now=None: {
                "kappa_token": "fresh-token", "user_id": 26, "user_type_id": 1,
            },
        )

        class _Uploader:
            async def upload_results(self):
                return {"error": "Failed to resolve dataset_id"}

        def _build(run, session):
            seen["token"] = session["kappa_token"]
            return _Uploader()

        monkeypatch.setattr(worker, "build_uploader", _build)

        await worker.deliver_one(run_id)
        assert seen["token"] == "fresh-token"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_tick_respects_the_batch_size(monkeypatch):
    db = SessionLocal()
    ids = [f"test_worker_batch_{i}" for i in range(7)]
    attempted = []
    try:
        for rid in ids:
            _make_run(db, rid)

        async def _fake_deliver(run_id):
            attempted.append(run_id)
            return None

        monkeypatch.setattr(worker, "deliver_one", _fake_deliver)
        touched = await worker.tick(now=NOW)

        assert touched == worker.BATCH_SIZE
        assert len(attempted) == worker.BATCH_SIZE
    finally:
        _cleanup(db, *ids)
        db.close()


@pytest.mark.asyncio
async def test_tick_skips_runs_that_are_not_due(monkeypatch):
    db = SessionLocal()
    run_id = "test_worker_not_due"
    attempted = []
    try:
        _make_run(db, run_id)
        set_kappa_delivery(db, run_id, "pending", NOW + timedelta(hours=1), {})

        async def _fake_deliver(rid):
            attempted.append(rid)
            return None

        monkeypatch.setattr(worker, "deliver_one", _fake_deliver)
        await worker.tick(now=NOW)

        assert run_id not in attempted
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_the_verdict_is_stamped_when_the_upload_finished(monkeypatch):
    """`now` was captured before the upload and reused for the verdict, so
    last_attempt_at reported the attempt's START and the backoff window was
    short by however long the upload took.

    Since set_kappa_delivery refuses a verdict older than the stored one,
    this also meant a slow attempt's result could be thrown away: its own
    in-flight seed, written with the same stamp, is not older — but any
    verdict written meanwhile is newer, and the real outcome lost to it.
    """
    import json
    from datetime import datetime, timezone

    db = SessionLocal()
    run_id = "test_worker_fresh_stamp"
    try:
        _make_run(db, run_id)
        monkeypatch.setattr(
            worker, "find_live_session_for_user",
            lambda user_id, now=None: {
                "kappa_token": "t", "user_id": 26, "user_type_id": 1,
            },
        )

        started = datetime.now(timezone.utc)

        class _SlowUploader:
            async def upload_results(self):
                # Stands in for a multi-second upload without spending them.
                import asyncio
                await asyncio.sleep(0.05)
                return {"dataset_id": 350, "uploaded": 1, "total": 1,
                        "sessions": [{"session": "sub-001_ses-001",
                                      "success": True, "entity_id": "e1"}]}

        monkeypatch.setattr(worker, "build_uploader",
                            lambda run, session: _SlowUploader())

        await worker.deliver_one(run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        stamp = datetime.fromisoformat(
            json.loads(run.kappa_upload_detail)["last_attempt_at"]
        )
        assert stamp >= started, "вердикт помечен временем до начала попытки"
        assert (stamp - started).total_seconds() >= 0.05, (
            "метка взята до выгрузки, а не после — "
            f"прошло {(stamp - started).total_seconds():.3f}с"
        )
    finally:
        _cleanup(db, run_id)
        db.close()
