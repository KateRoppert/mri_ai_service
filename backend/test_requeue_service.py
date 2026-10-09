"""Starting a requeue — the one path both the endpoint and the queue use.

The point of sharing it is WHEN the work happens. Purging the sessions the
doctor marked, and recording which ones supersede what Kappa holds, must
happen when the run actually starts, not when the button was clicked. A
queued request can wait an hour, and in that hour the doctor may correct
another patient:

  - purging at click time would delete results long before new ones exist;
  - a superseding list snapshotted at click time would miss the patient
    corrected while waiting, and that patient's old version would stay in
    Kappa silently — the exact defect fixed in feat/kappa-replace-on-reprocess.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import requeue_service
from database import SessionLocal, create_pipeline_run, get_pipeline_run


def _original(db, output_path="/out/svc", dataset_id=351):
    run = create_pipeline_run(
        db, input_path="/in/svc", output_path=output_path,
        lesion_type="multiple_sclerosis", kappa_dataset_id=dataset_id,
    )
    run.status = "completed"
    db.commit()
    return run


@pytest.mark.asyncio
async def test_purge_runs_when_the_run_starts_and_its_result_is_recorded():
    db = SessionLocal()
    try:
        original = _original(db)
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={"sub-002/ses-001": [], "sub-003/ses-002": []}), \
             patch("app.run_pipeline_background"), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)

        assert sorted(json.loads(new_run.reprocessed_sessions)) == [
            "sub-002_ses-001", "sub-003_ses-002",
        ]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_session_corrected_while_waiting_is_included():
    """The regression this whole shared path exists to prevent. The purge
    reads dataset_mapping.json at start, so a patient marked after the click
    is in its result — and therefore in reprocessed_sessions too."""
    db = SessionLocal()
    try:
        original = _original(db, output_path="/out/svc-late")

        # What the doctor had marked at click time was only ses-001; by the
        # time the queue fires, ses-002 is marked as well.
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={"sub-002/ses-001": [],
                                 "sub-002/ses-002": []}), \
             patch("app.run_pipeline_background"), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)

        assert "sub-002_ses-002" in json.loads(new_run.reprocessed_sessions)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_nothing_purged_records_nothing():
    db = SessionLocal()
    try:
        original = _original(db, output_path="/out/svc-empty")
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={}), \
             patch("app.run_pipeline_background"), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)

        assert new_run.reprocessed_sessions is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_new_run_inherits_the_path_and_the_dataset():
    db = SessionLocal()
    try:
        original = _original(db, output_path="/out/svc-inherit", dataset_id=158)
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={}), \
             patch("app.run_pipeline_background"), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)

        assert new_run.output_path == "/out/svc-inherit"
        assert new_run.input_path == "/in/svc"
        assert new_run.lesion_type == "multiple_sclerosis"
        assert new_run.kappa_dataset_id == 158
        assert new_run.parent_run_id == original.run_id
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_kappa_session_makes_the_run_owe_delivery():
    db = SessionLocal()
    try:
        original = _original(db, output_path="/out/svc-kappa")
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={}), \
             patch("app.run_pipeline_background"), \
             patch("kappa_auth.get_session",
                   return_value={"user_id": 26, "kappa_token": "t"}), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()) as monitor:
            new_run = await requeue_service.start_requeue(db, original, "sid")

        assert new_run.kappa_upload_status == "pending"
        assert new_run.kappa_user_id == 26
        # call_args, not await_args: monitoring is scheduled with
        # create_task, so the coroutine has been built but not yet run.
        # The session is what matters, and it is already captured.
        assert monitor.call_args.args[2] == "sid"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_without_a_kappa_session_the_run_owes_nothing():
    db = SessionLocal()
    try:
        original = _original(db, output_path="/out/svc-nokappa")
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={}), \
             patch("app.run_pipeline_background"), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)

        assert new_run.kappa_upload_status is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_failed_purge_does_not_stop_the_run():
    """Cleanup is housekeeping. Refusing to start because a stale folder
    could not be removed is a worse outcome than the mess."""
    db = SessionLocal()
    try:
        original = _original(db, output_path="/out/svc-badpurge")
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   side_effect=OSError("disk is read-only")), \
             patch("app.run_pipeline_background") as launch, \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)
            # The launch is dispatched now, so give the loop a turn.
            import asyncio
            await asyncio.sleep(0.1)

        assert new_run is not None
        assert launch.called
        assert get_pipeline_run(db, new_run.run_id) is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_launch_is_dispatched_to_a_thread_not_run_inline():
    """run_pipeline_background waits for the whole pipeline
    (wait_for_pipeline), so calling it inline from an async caller freezes
    the event loop until the run ends — no API responses, no WebSocket
    progress, nothing.

    That shipped: resuming a stopped run froze the backend for minutes and
    read as "resume does not work". The earlier tests here missed it because
    they asserted the launcher was CALLED, never that it was DISPATCHED.

    Asserted by thread identity rather than by a timeout: a blocked event
    loop cannot be detected from inside that same loop, because a sync call
    never yields and asyncio.wait_for has no point at which to cancel. A
    timeout-based version of this test passed while the bug was present.

    FastAPI's BackgroundTasks, which the endpoint used before, ran it in a
    threadpool; this has to keep that property.
    """
    import asyncio
    import threading

    db = SessionLocal()
    launched_on = {}
    entered = threading.Event()

    def _record_thread(*a, **k):
        launched_on["ident"] = threading.get_ident()
        entered.set()

    try:
        original = _original(db, output_path="/out/svc-noblock")
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={}), \
             patch("app.run_pipeline_background", _record_thread), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            new_run = await requeue_service.start_requeue(db, original)

            # In another thread: Event.wait() on the loop's own thread would
            # starve the task we are waiting for — the same class of mistake
            # as the bug under test.
            assert await asyncio.to_thread(entered.wait, 5.0), \
                "запуск не был отправлен вообще"

        assert new_run is not None
        assert launched_on["ident"] != threading.get_ident(), (
            "pipeline запущен в том же потоке, что и event loop — "
            "бэкенд замрёт на всё время прогона"
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_the_launch_gets_its_own_session():
    """The launcher uses its session for the whole pipeline — minutes — to
    update status and check for a stop. It must not borrow the caller's.

    The endpoint's session is closed by FastAPI once the response is sent,
    and the queue's caller (_start_queued_requeue) closes its own in a
    finally the moment start_requeue returns. Either way the borrowed
    session would be closed out from under a thread still using it.
    BackgroundTasks hid this before, because FastAPI tore the dependency
    down after background tasks had run.
    """
    import asyncio
    import threading

    db = SessionLocal()
    seen = {}
    entered = threading.Event()

    def _record_session(run_id, input_path, output_path, session, **k):
        seen["session"] = session
        entered.set()

    try:
        original = _original(db, output_path="/out/svc-session")
        with patch("session_artifacts.purge_sessions_marked_for_reprocess",
                   return_value={}), \
             patch("app.run_pipeline_background", _record_session), \
             patch("pipeline_monitor.pipeline_monitor.start_monitoring",
                   new=AsyncMock()):
            await requeue_service.start_requeue(db, original)
            assert await asyncio.to_thread(entered.wait, 5.0)

        assert seen["session"] is not db, (
            "pipeline получил сессию вызывающего — её закроют у него под руками"
        )
    finally:
        db.close()
