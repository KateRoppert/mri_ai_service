"""A failed run must not sit in the delivery queue forever.

The worker only ever picks up runs with status='completed'. If a failed run
keeps kappa_upload_status='pending', nothing will ever resolve it, but the
summary banner keeps counting it — a phantom the operator cannot clear.
"""
import sys

import pytest

sys.path.insert(0, "backend")

import database as db_mod  # noqa: E402
import pipeline_monitor as pm  # noqa: E402
from database import SessionLocal, create_pipeline_run  # noqa: E402


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


def test_failed_run_drops_out_of_the_delivery_queue():
    db = SessionLocal()
    run_id = "test_failed_drops_out"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="pending", kappa_user_id=26,
        )
        run.status = "failed"
        db.commit()

        pm.pipeline_monitor._abandon_delivery(run_id, "прогон завершился ошибкой")

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status is None
        assert run.kappa_upload_next_attempt is None
    finally:
        _cleanup(db, run_id)
        db.close()


def test_abandon_leaves_a_completed_run_alone():
    """Guard rail: the helper must never disarm a run that can still deliver."""
    db = SessionLocal()
    run_id = "test_abandon_skips_completed"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="pending", kappa_user_id=26,
        )
        run.status = "completed"
        db.commit()

        pm.pipeline_monitor._abandon_delivery(run_id, "не должно сработать")

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "pending"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_summary_ignores_runs_the_worker_cannot_touch():
    """The banner must count exactly what the worker can act on. Any other
    predicate produces a number the operator has no way to make go away."""
    import app

    db = SessionLocal()
    ids = ["test_sum_failed_pending", "test_sum_running_pending",
           "test_sum_real_pending"]
    try:
        for rid, status in [
            ("test_sum_failed_pending", "failed"),
            ("test_sum_running_pending", "running"),
            ("test_sum_real_pending", "completed"),
        ]:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out", run_id=rid,
                kappa_upload_status="pending",
            )
            run.status = status
            db.commit()

        summary = await app.kappa_delivery_summary(db=db)

        worker_sees = db_mod.runs_due_for_delivery(
            db, __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ), limit=100,
        )
        ours = [r.run_id for r in worker_sees if r.run_id in ids]
        assert ours == ["test_sum_real_pending"]
        assert summary["pending"] == len(ours)
    finally:
        _cleanup(db, *ids)
        db.close()
