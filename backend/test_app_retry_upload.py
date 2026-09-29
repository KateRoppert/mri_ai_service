"""The manual retry button must leave the same trace as every other path."""
import sys

import pytest

sys.path.insert(0, "backend")

import database as db_mod  # noqa: E402
from database import SessionLocal, create_pipeline_run  # noqa: E402


def _cleanup(db, run_id):
    run = db_mod.get_pipeline_run(db, run_id)
    if run:
        db.delete(run)
    db.commit()


@pytest.mark.asyncio
async def test_manual_retry_records_the_verdict(monkeypatch):
    import app
    import pipeline_monitor as pm

    db = SessionLocal()
    run_id = "test_retry_records"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_upload_status="pending", kappa_user_id=26,
            kappa_dataset_id=350,
        )
        run.status = "completed"
        db.commit()

        class _Uploader:
            async def upload_results(self):
                return {"dataset_id": 350, "uploaded": 1, "total": 1,
                        "sessions": [{"session": "sub-001_ses-001",
                                      "success": True, "entity_id": "e1"}]}

        monkeypatch.setattr(
            pm.pipeline_monitor, "_create_kappa_uploader",
            lambda *a, **k: _Uploader(),
        )

        response = await app.retry_kappa_upload(run_id, session_id="sid")
        assert response["delivery"]["status"] == "done"

        db.expire_all()
        run = db_mod.get_pipeline_run(db, run_id)
        assert run.kappa_upload_status == "done"
    finally:
        _cleanup(db, run_id)
        db.close()
