"""A verdict must not be overwritten by an older one.

Seen in production twice. Run 2af5d6ed (2026-10-05) ended up with the
status from a correct verdict written at 09:40:59 and the detail from one
stamped 09:40:54 — five seconds earlier, reporting "0 of 2, Kappa
unreachable" for a run that had actually delivered 1 of 2 and was waiting
for a human. Run e3783cbb (2026-09-28) shows the same shape: status=done
with detail=network/attempts=2.

The writing mechanism was never reproduced from the logs. This guard makes
the state unreachable regardless of the mechanism, and logs any attempt so
a recurrence leaves evidence instead of a contradiction.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from database import (
    SessionLocal, create_pipeline_run, get_pipeline_run, set_kappa_delivery,
)

NOW = datetime(2026, 10, 5, 9, 40, 59, tzinfo=timezone.utc)
EARLIER = NOW - timedelta(seconds=5)


def _detail(last_attempt, **over):
    d = {"total": 2, "delivered": 1, "blocked": [], "reason": None,
         "last_error": None, "attempts": 0, "first_failure_at": None,
         "last_attempt_at": last_attempt.isoformat()}
    d.update(over)
    return d


def _run(db):
    return create_pipeline_run(
        db, input_path="/in", output_path="/out",
        kappa_upload_status="pending",
    )


def test_an_older_verdict_cannot_overwrite_a_newer_one():
    db = SessionLocal()
    try:
        run = _run(db)
        set_kappa_delivery(db, run.run_id, "needs_attention", None,
                           _detail(NOW, reason="supersedes_kappa",
                                   blocked=[{"session": "s",
                                             "reason": "supersedes_kappa",
                                             "message": ""}]))

        # The shape observed in production: an attempt that started earlier
        # finishing later and reporting a worse, staler picture.
        set_kappa_delivery(db, run.run_id, "pending",
                           EARLIER + timedelta(minutes=1),
                           _detail(EARLIER, reason="network", delivered=0,
                                   attempts=1))

        fresh = get_pipeline_run(db, run.run_id)
        detail = json.loads(fresh.kappa_upload_detail)
        assert fresh.kappa_upload_status == "needs_attention"
        assert detail["reason"] == "supersedes_kappa"
        assert detail["delivered"] == 1
        assert fresh.kappa_upload_next_attempt is None
    finally:
        db.close()


def test_a_newer_verdict_is_written_normally():
    db = SessionLocal()
    try:
        run = _run(db)
        set_kappa_delivery(db, run.run_id, "pending", None,
                           _detail(EARLIER, reason="network", delivered=0))
        set_kappa_delivery(db, run.run_id, "done", None,
                           _detail(NOW, delivered=2))

        fresh = get_pipeline_run(db, run.run_id)
        assert fresh.kappa_upload_status == "done"
        assert json.loads(fresh.kappa_upload_detail)["delivered"] == 2
    finally:
        db.close()


def test_the_same_timestamp_is_allowed_through():
    """Equal stamps mean the same attempt writing twice — a seed followed by
    its own verdict. Refusing those would strand runs mid-upload."""
    db = SessionLocal()
    try:
        run = _run(db)
        set_kappa_delivery(db, run.run_id, "pending", None, _detail(NOW))
        set_kappa_delivery(db, run.run_id, "done", None,
                           _detail(NOW, delivered=2))

        fresh = get_pipeline_run(db, run.run_id)
        assert fresh.kappa_upload_status == "done"
    finally:
        db.close()


def test_a_first_write_always_lands():
    db = SessionLocal()
    try:
        run = _run(db)
        set_kappa_delivery(db, run.run_id, "done", None, _detail(NOW))

        fresh = get_pipeline_run(db, run.run_id)
        assert fresh.kappa_upload_status == "done"
    finally:
        db.close()


def test_an_unparseable_existing_detail_does_not_block_the_write():
    """A corrupt blob must not freeze a run's state forever."""
    db = SessionLocal()
    try:
        run = _run(db)
        run.kappa_upload_detail = "not json"
        db.commit()

        set_kappa_delivery(db, run.run_id, "done", None, _detail(NOW))

        fresh = get_pipeline_run(db, run.run_id)
        assert fresh.kappa_upload_status == "done"
    finally:
        db.close()


@pytest.mark.parametrize("bad", [None, "", "nonsense-timestamp"])
def test_a_missing_or_bad_stamp_is_not_treated_as_newer(bad):
    db = SessionLocal()
    try:
        run = _run(db)
        set_kappa_delivery(db, run.run_id, "pending", None,
                           _detail(NOW, last_attempt_at=bad))
        set_kappa_delivery(db, run.run_id, "done", None, _detail(NOW))

        fresh = get_pipeline_run(db, run.run_id)
        assert fresh.kappa_upload_status == "done"
    finally:
        db.close()
