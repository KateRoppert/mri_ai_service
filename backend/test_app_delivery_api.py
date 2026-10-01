"""Delivery state reaching the frontend."""
import sys

import pytest

sys.path.insert(0, "backend")

import database as db_mod  # noqa: E402
from database import SessionLocal, create_pipeline_run, set_kappa_delivery  # noqa: E402


def _cleanup(db, *run_ids):
    for rid in run_ids:
        run = db_mod.get_pipeline_run(db, rid)
        if run:
            db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_history_carries_delivery_state():
    import app

    db = SessionLocal()
    run_id = "test_api_history_delivery"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="needs_attention",
        )
        run.status = "completed"
        db.commit()
        set_kappa_delivery(db, run_id, "needs_attention", None, {
            "total": 2, "delivered": 1, "reason": "name_clash",
            "blocked": [{"session": "sub-002_ses-001",
                         "reason": "name_clash", "message": "уже есть"}],
        })

        response = await app.get_history(limit=100, offset=0, db=db)
        item = next(r for r in response.runs if r.run_id == run_id)

        assert item.kappa_upload.status == "needs_attention"
        assert item.kappa_upload.delivered == 1
        assert item.kappa_upload.total == 2
        assert item.kappa_upload.blocked[0].reason == "name_clash"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_history_carries_sessions_lost_in_processing():
    """Run 30_09_1752: 4 of 6 delivered, the other two never processed. The
    history column must be able to say which two and why."""
    import app

    db = SessionLocal()
    run_id = "test_api_history_not_processed"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="done",
        )
        run.status = "completed"
        db.commit()
        set_kappa_delivery(db, run_id, "done", None, {
            "total": 6, "delivered": 4,
            "not_processed": [
                {"session": "sub-003_ses-005", "message": "нет t2fl и маски сегментации"},
                {"session": "sub-003_ses-006", "message": "нет данных после предобработки"},
            ],
        })

        response = await app.get_history(limit=100, offset=0, db=db)
        item = next(r for r in response.runs if r.run_id == run_id)

        assert (item.kappa_upload.delivered, item.kappa_upload.total) == (4, 6)
        assert [s.session for s in item.kappa_upload.not_processed] == [
            "sub-003_ses-005", "sub-003_ses-006",
        ]
        assert item.kappa_upload.not_processed[0].message == "нет t2fl и маски сегментации"
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_history_item_without_delivery_has_none():
    import app

    db = SessionLocal()
    run_id = "test_api_history_no_delivery"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
        )
        run.status = "completed"
        db.commit()

        response = await app.get_history(limit=100, offset=0, db=db)
        item = next(r for r in response.runs if r.run_id == run_id)
        assert item.kappa_upload is None
    finally:
        _cleanup(db, run_id)
        db.close()


@pytest.mark.asyncio
async def test_summary_counts_by_status():
    import app

    db = SessionLocal()
    ids = ["test_api_sum_p1", "test_api_sum_p2", "test_api_sum_na"]
    try:
        for rid, status in [
            ("test_api_sum_p1", "pending"),
            ("test_api_sum_p2", "pending"),
            ("test_api_sum_na", "needs_attention"),
        ]:
            run = create_pipeline_run(
                db, input_path="/in", output_path="/out", run_id=rid,
                kappa_upload_status=status,
            )
            run.status = "completed"
            db.commit()

        summary = await app.kappa_delivery_summary(db=db)
        assert summary["pending"] >= 2
        assert summary["needs_attention"] >= 1
    finally:
        _cleanup(db, *ids)
        db.close()
