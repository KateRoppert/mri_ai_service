"""The queued-requeue flag: one per output_path, carried by the active run.

The flag lives on the run that is currently occupying the path, not on the
run the doctor clicked from — they are often different. The doctor looks at
a completed run X while its requeue child Y is still working, and the
queued request has to fire when Y finishes.

It is deliberately NOT a second `pending` PipelineRun row:
get_active_run_by_output_path treats pending as active, so a queue stored
that way would count as the very run it is waiting for.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from database import (
    SessionLocal, clear_all_queued_requeues, clear_queued_requeue,
    create_pipeline_run, get_active_run_by_output_path, get_pipeline_run,
    queue_requeue,
)


def _run(db, output_path, status="running"):
    run = create_pipeline_run(db, input_path="/in", output_path=output_path)
    run.status = status
    db.commit()
    return run


def test_queueing_marks_the_run_and_records_when():
    db = SessionLocal()
    try:
        run = _run(db, "/out/queue-mark")
        assert queue_requeue(db, run.run_id) is True

        fresh = get_pipeline_run(db, run.run_id)
        assert fresh.queued_requeue_at is not None
    finally:
        db.close()


def test_queueing_twice_keeps_the_first_request():
    """One queue per path. A second click confirms what is already there —
    two consecutive runs would do the same work anyway, since skip_existing
    reads the disk at start."""
    db = SessionLocal()
    try:
        run = _run(db, "/out/queue-twice")
        queue_requeue(db, run.run_id)
        first = get_pipeline_run(db, run.run_id).queued_requeue_at

        queue_requeue(db, run.run_id)

        assert get_pipeline_run(db, run.run_id).queued_requeue_at == first
    finally:
        db.close()


def test_a_queued_run_is_still_the_active_run_on_its_path():
    """The flag must not change what counts as active — otherwise the queue
    would either block itself or let a second orchestrator start."""
    db = SessionLocal()
    try:
        run = _run(db, "/out/queue-active")
        queue_requeue(db, run.run_id)

        active = get_active_run_by_output_path(db, "/out/queue-active")
        assert active is not None and active.run_id == run.run_id
    finally:
        db.close()


def test_clearing_removes_the_flag():
    db = SessionLocal()
    try:
        run = _run(db, "/out/queue-clear")
        queue_requeue(db, run.run_id)

        assert clear_queued_requeue(db, run.run_id) is True
        assert get_pipeline_run(db, run.run_id).queued_requeue_at is None
    finally:
        db.close()


def test_clearing_what_was_never_queued_is_false():
    """So the cancel endpoint can tell 'nothing to cancel' from 'cancelled'."""
    db = SessionLocal()
    try:
        run = _run(db, "/out/queue-noop")
        assert clear_queued_requeue(db, run.run_id) is False
    finally:
        db.close()


def test_queueing_a_missing_run_is_false():
    db = SessionLocal()
    try:
        assert queue_requeue(db, "no-such-run") is False
    finally:
        db.close()


def test_startup_clears_every_queued_request():
    """A pipeline run is a subprocess of this backend, so a restart means the
    run it was waiting for is gone. Starting it automatically after a crash
    is unsafe — the orchestrator may have died midway through writing
    dataset_mapping.json — so the intent is dropped and the doctor re-clicks.
    """
    db = SessionLocal()
    try:
        # conftest gives the whole session one temp DB, so earlier tests have
        # left flags behind. Clear the slate first — asserting on an exact
        # count only means something against a known starting point.
        clear_all_queued_requeues(db)

        a = _run(db, "/out/queue-boot-a")
        b = _run(db, "/out/queue-boot-b")
        c = _run(db, "/out/queue-boot-c")
        queue_requeue(db, a.run_id)
        queue_requeue(db, b.run_id)

        assert clear_all_queued_requeues(db) == 2

        for r in (a, b, c):
            assert get_pipeline_run(db, r.run_id).queued_requeue_at is None
    finally:
        db.close()
