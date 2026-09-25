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
