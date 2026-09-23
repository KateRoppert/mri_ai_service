"""Delivery-state columns on pipeline_runs and their accessors."""
from datetime import datetime, timedelta, timezone

import database as db_mod
from database import (
    SessionLocal,
    create_pipeline_run,
    get_kappa_delivery,
    runs_due_for_delivery,
    set_kappa_delivery,
)


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


def test_delivery_state_roundtrips_through_the_db():
    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out",
            run_id="test_delivery_roundtrip",
            kappa_upload_status="pending", kappa_user_id=26,
        )
        assert run.kappa_upload_status == "pending"
        assert run.kappa_user_id == 26
        assert get_kappa_delivery(run) == {}

        nxt = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
        set_kappa_delivery(
            db, "test_delivery_roundtrip", "pending", nxt,
            {"total": 5, "delivered": 3, "attempts": 2},
        )
        db.expire_all()
        run = db_mod.get_pipeline_run(db, "test_delivery_roundtrip")
        assert get_kappa_delivery(run)["delivered"] == 3
        # stored naive, returned aware
        assert run.kappa_upload_next_attempt.replace(tzinfo=timezone.utc) == nxt
    finally:
        _cleanup(db, "test_delivery_roundtrip")
        db.close()


def test_runs_due_for_delivery_selects_only_completed_pending_and_due():
    db = SessionLocal()
    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    ids = [
        "test_due_ready", "test_due_future", "test_due_running",
        "test_due_done", "test_due_null",
    ]
    try:
        for rid, status, upload, nxt in [
            ("test_due_ready", "completed", "pending", now - timedelta(minutes=1)),
            ("test_due_future", "completed", "pending", now + timedelta(minutes=30)),
            ("test_due_running", "running", "pending", None),
            ("test_due_done", "completed", "done", None),
            ("test_due_null", "completed", None, None),
        ]:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out",
                run_id=rid, kappa_upload_status=upload,
            )
            run.status = status
            db.commit()
            if nxt is not None:
                set_kappa_delivery(db, rid, upload, nxt, {})

        due = [r.run_id for r in runs_due_for_delivery(db, now)]
        assert "test_due_ready" in due
        assert "test_due_future" not in due
        assert "test_due_running" not in due
        assert "test_due_done" not in due
        assert "test_due_null" not in due
    finally:
        _cleanup(db, *ids)
        db.close()


def test_runs_due_for_delivery_honours_the_limit():
    db = SessionLocal()
    ids = [f"test_due_many_{i}" for i in range(7)]
    now = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
    try:
        for rid in ids:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out",
                run_id=rid, kappa_upload_status="pending",
            )
            run.status = "completed"
            db.commit()
        assert len(runs_due_for_delivery(db, now, limit=5)) == 5
    finally:
        _cleanup(db, *ids)
        db.close()
