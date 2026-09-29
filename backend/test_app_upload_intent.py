"""A run started with a Kappa session is born owing delivery."""
import database as db_mod
from database import SessionLocal, create_pipeline_run


def _cleanup(db, run_id):
    run = db_mod.get_pipeline_run(db, run_id)
    if run:
        db.delete(run)
    db.commit()


def test_run_with_a_session_is_born_pending():
    db = SessionLocal()
    run_id = "test_intent_with_session"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
            kappa_dataset_id=350, kappa_upload_status="pending",
            kappa_user_id=26,
        )
        assert run.kappa_upload_status == "pending"
        assert run.kappa_user_id == 26
    finally:
        _cleanup(db, run_id)
        db.close()


def test_run_without_a_session_has_no_upload_status():
    db = SessionLocal()
    run_id = "test_intent_no_session"
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path="/out", run_id=run_id,
        )
        assert run.kappa_upload_status is None
        assert run.kappa_user_id is None
    finally:
        _cleanup(db, run_id)
        db.close()
