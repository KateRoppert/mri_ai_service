"""The queue fires when the path frees up.

The monitor already reacts to a run finishing — that is where the Kappa
upload is dispatched — so it is also where a queued request starts.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import database as db_mod
import pipeline_monitor as pm
from database import SessionLocal, create_pipeline_run, queue_requeue


def _finished(db, status, output_path, queued=True):
    run = create_pipeline_run(db, input_path="/in", output_path=output_path)
    run.status = status
    db.commit()
    if queued:
        queue_requeue(db, run.run_id)
    return run


@pytest.mark.asyncio
async def test_a_completed_run_starts_the_queued_request():
    db = SessionLocal()
    try:
        run = _finished(db, "completed", "/out/fire")

        with patch("requeue_service.start_requeue",
                   new=AsyncMock()) as start:
            await pm.pipeline_monitor._start_queued_requeue(run.run_id, "sid")

        assert start.await_count == 1
        # The new run must inherit the finished run's paths, so the finished
        # run is what it is started from.
        assert start.await_args.args[1].run_id == run.run_id
        assert start.await_args.args[2] == "sid"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_flag_is_cleared_so_it_cannot_fire_twice():
    db = SessionLocal()
    try:
        run = _finished(db, "completed", "/out/once")

        with patch("requeue_service.start_requeue", new=AsyncMock()):
            await pm.pipeline_monitor._start_queued_requeue(run.run_id, None)

        db.expire_all()
        assert db_mod.get_pipeline_run(db, run.run_id).queued_requeue_at is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_failed_run_drops_the_request_instead_of_starting_it():
    """Auto-starting after a failure would hide the failure, and a pipeline
    broken by its environment would loop. The doctor's correction is on
    disk and the next run picks it up, so nothing is lost by asking for a
    deliberate click."""
    db = SessionLocal()
    try:
        run = _finished(db, "failed", "/out/failed")

        with patch("requeue_service.start_requeue", new=AsyncMock()) as start:
            await pm.pipeline_monitor._start_queued_requeue(run.run_id, None)

        start.assert_not_called()
        db.expire_all()
        assert db_mod.get_pipeline_run(db, run.run_id).queued_requeue_at is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_nothing_queued_does_nothing():
    db = SessionLocal()
    try:
        run = _finished(db, "completed", "/out/noqueue", queued=False)

        with patch("requeue_service.start_requeue", new=AsyncMock()) as start:
            await pm.pipeline_monitor._start_queued_requeue(run.run_id, None)

        start.assert_not_called()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_failure_to_start_does_not_escape_the_monitor():
    """The monitor loop also delivers to Kappa and pushes progress. A queued
    run that cannot start must not take those down with it."""
    db = SessionLocal()
    try:
        run = _finished(db, "completed", "/out/boom")

        with patch("requeue_service.start_requeue",
                   new=AsyncMock(side_effect=RuntimeError("no disk"))):
            await pm.pipeline_monitor._start_queued_requeue(run.run_id, None)

        # And the flag is gone, so it does not retry forever.
        db.expire_all()
        assert db_mod.get_pipeline_run(db, run.run_id).queued_requeue_at is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_monitor_loop_actually_calls_the_hook():
    """Wiring, not just the function. Twice in this project a unit test has
    passed while the caller never invoked what it tested — the adoption bug
    and the superseding-sessions list. The loop is what makes the queue real.
    """
    db = SessionLocal()
    try:
        run = _finished(db, "completed", "/out/wired")

        with patch.object(pm.pipeline_monitor, "_start_queued_requeue",
                          new=AsyncMock()) as hook, \
             patch.object(pm.pipeline_monitor, "_send_update", new=AsyncMock()):
            await pm.pipeline_monitor._monitor_loop(
                run.run_id, "/out/wired", "sid", "glioblastoma")

        assert hook.await_count == 1
        assert hook.await_args.args == (run.run_id, "sid")
    finally:
        db.close()
