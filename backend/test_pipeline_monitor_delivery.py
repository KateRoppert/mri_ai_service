"""The post-run upload must record what happened — including the failures
that arrive as a return value rather than an exception."""
import pytest

import database as db_mod
import pipeline_monitor as pm
from database import SessionLocal, create_pipeline_run, get_kappa_delivery


class _Uploader:
    """Stands in for KappaUploader: returns or raises whatever it was given."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    async def upload_results(self):
        if self.exc:
            raise self.exc
        return self.result


def _make_run(db, run_id):
    run = create_pipeline_run(
        db, input_path="/in", output_path="/out",
        run_id=run_id, kappa_upload_status="pending", kappa_user_id=26,
    )
    run.status = "completed"
    db.commit()
    return run


def _cleanup(db, run_id):
    run = db_mod.get_pipeline_run(db, run_id)
    if run:
        db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_returned_error_is_recorded_as_pending():
    db = SessionLocal()
    run_id = "test_monitor_returned_error"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(result={"error": "Failed to resolve dataset_id"})
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        assert get_kappa_delivery(run)["reason"] == "network"
        assert run.kappa_upload_next_attempt is not None
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_exception_is_recorded_as_pending():
    db = SessionLocal()
    run_id = "test_monitor_exception"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(exc=RuntimeError("connection reset"))
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        assert "connection reset" in get_kappa_delivery(run)["last_error"]
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_full_success_is_recorded_as_done():
    db = SessionLocal()
    run_id = "test_monitor_success"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(result={
            "dataset_id": 350, "uploaded": 1, "total": 1,
            "sessions": [{"session": "sub-001_ses-001", "success": True,
                          "entity_id": "e1"}],
        })
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "done"
        assert run.kappa_upload_next_attempt is None
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_name_clash_is_recorded_as_needs_attention():
    db = SessionLocal()
    run_id = "test_monitor_clash"
    try:
        _make_run(db, run_id)
        uploader = _Uploader(result={
            "dataset_id": 350, "uploaded": 0, "total": 1,
            "sessions": [{"session": "sub-003_ses-001", "success": False,
                          "error": "name_clash", "message": "уже есть"}],
        })
        await pm.pipeline_monitor._kappa_upload_safe(uploader, run_id)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "needs_attention"
        assert get_kappa_delivery(run)["blocked"][0]["reason"] == "name_clash"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_missing_uploader_is_recorded_as_waiting_for_a_session():
    """No usable Kappa session when the run finished means no attempt was
    ever made. The run must stay pending, and say WHY, so the worker picks it
    up once someone logs in — and so the UI does not call it a network
    problem."""
    from kappa_delivery import NO_SESSION

    db = SessionLocal()
    run_id = "test_monitor_no_uploader"
    try:
        _make_run(db, run_id)
        pm.pipeline_monitor._record_delivery(run_id, NO_SESSION, None)

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
        detail = get_kappa_delivery(run)
        assert detail["reason"] == "no_session"
        assert detail["attempts"] == 0
    finally:
        _cleanup(db, run_id)
        db.close()


def test_run_without_upload_intent_is_left_alone():
    """A CLI run (NULL status) must never acquire delivery state, or the
    worker would start chasing runs nobody meant to upload."""
    db = SessionLocal()
    run_id = "test_monitor_no_intent"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
        )
        run.status = "completed"
        db.commit()

        verdict = pm.pipeline_monitor._record_delivery(
            run_id, {"error": "Failed to resolve dataset_id"}, None
        )
        assert verdict is None

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status is None
    finally:
        _cleanup(db, run_id)
        db.close()
