"""Starting another run over an output_path that already has results.

One definition, two callers: the requeue endpoint (the doctor clicked and
nothing is running) and the delivery queue (the doctor clicked while a run
was working, and that run has now finished).

Sharing this is not about saving lines — it is about WHEN the work happens.
Purging the sessions the doctor marked, and recording which ones supersede
what Kappa already holds, must happen as the run starts, not when the
button was pressed. A queued request can wait an hour, and in that hour the
doctor may correct another patient:

  - purging at click time would delete results long before new ones exist;
  - a superseding list snapshotted at click time would miss the patient
    corrected while waiting, and that patient's stale version would sit in
    Kappa with nothing saying so.

Both are avoided by doing it here, at the moment of the actual start.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

import numbering
from database import PipelineRun, SessionLocal, create_pipeline_run

logger = logging.getLogger(__name__)

# A bare create_task is only weakly referenced, so a fire-and-forget launch
# can be garbage-collected mid-flight. Keep a strong reference until it ends.
_background: set = set()


def _spawn(coro, what: str) -> None:
    """Run `coro` detached, and make sure a failure is not swallowed.

    BackgroundTasks logged what went wrong; a bare task does not, and a
    pipeline that failed to launch in silence looks exactly like one that is
    still starting.
    """
    task = asyncio.ensure_future(coro)
    _background.add(task)

    def _done(t: asyncio.Future) -> None:
        _background.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error("Не удалось выполнить %s: %r", what, exc)

    task.add_done_callback(_done)


def _purge_and_record(output_path: str) -> Optional[str]:
    """Delete the outputs of every session flagged for reprocessing, and
    return them as the JSON list the uploader reads.

    Never raises: cleanup is housekeeping, and refusing to start the run
    because a stale folder could not be removed is a worse outcome than the
    mess.
    """
    try:
        from session_artifacts import purge_sessions_marked_for_reprocess
        purged = purge_sessions_marked_for_reprocess(output_path)
    except Exception as exc:  # noqa: BLE001 — запуск важнее уборки
        logger.error("Не удалось очистить помеченные сессии: %s", exc)
        return None

    if not purged:
        return None

    logger.info("Переобработка: очищено сессий — %d", len(purged))
    # These sessions supersede what Kappa holds rather than duplicating it.
    # needs_reprocess is cleared by the purge itself, so without this nothing
    # would remember by the time the upload runs.
    return json.dumps([key.replace("/", "_") for key in purged])


def _upload_intent(kappa_session_id: Optional[str]):
    """Delivery intent recorded at birth, same as the start endpoint: a run
    carrying a Kappa session owes an upload from the moment it exists."""
    if not kappa_session_id:
        return None, None
    from kappa_auth import get_session
    session = get_session(kappa_session_id)
    if not session:
        return None, None
    return "pending", session.get("user_id")


async def start_requeue(
    db: Session,
    original_run: PipelineRun,
    kappa_session_id: Optional[str] = None,
    *,
    snapshot_runtime_config: Optional[Path] = None,
    preprocessing_snapshot: Optional[Path] = None,
) -> PipelineRun:
    """Create and launch another run over `original_run`'s paths.

    The snapshot arguments belong to resuming a stopped run; a queued
    requeue passes neither.
    """
    lesion_type = original_run.lesion_type or "glioblastoma"
    reprocessed = _purge_and_record(original_run.output_path)
    upload_status, upload_user_id = _upload_intent(kappa_session_id)

    run = create_pipeline_run(
        db,
        input_path=original_run.input_path,
        output_path=original_run.output_path,
        lesion_type=lesion_type,
        parent_run_id=original_run.run_id,
        kappa_dataset_id=original_run.kappa_dataset_id,
        kappa_upload_status=upload_status,
        kappa_user_id=upload_user_id,
        reprocessed_sessions=reprocessed,
    )

    # Imported here, not at module scope: app imports pipeline_monitor, which
    # imports this module, so a top-level import of app would close the cycle.
    from app import run_pipeline_background
    from pipeline_monitor import pipeline_monitor

    scope = numbering.scope_from_dataset_id(
        original_run.kappa_dataset_id, lesion_type
    )

    def _launch() -> None:
        # Its own session, not the caller's. The launcher holds one for the
        # whole pipeline — minutes — to publish status and notice a stop,
        # while the caller's is closed as soon as the response is sent (the
        # endpoint) or the moment start_requeue returns (the queue).
        launch_db = SessionLocal()
        try:
            run_pipeline_background(
                run.run_id,
                run.input_path,
                run.output_path,
                launch_db,
                lesion_type=run.lesion_type,
                snapshot_runtime_config=snapshot_runtime_config,
                preprocessing_snapshot=preprocessing_snapshot,
                numbering_scope=scope,
            )
        finally:
            launch_db.close()

    # To a thread, and not awaited. run_pipeline_background waits out the
    # whole pipeline (wait_for_pipeline), so calling it inline from an async
    # caller freezes the event loop until the run ends — no API responses, no
    # WebSocket progress. The endpoint used to hand it to FastAPI's
    # BackgroundTasks, which ran it in a threadpool; this keeps that property
    # and gives the queue the same one.
    _spawn(asyncio.to_thread(_launch), f"launch {run.run_id}")

    # Pass the Kappa session through: without it the monitor never builds an
    # uploader, and the run completes having silently never reached Kappa.
    asyncio.create_task(pipeline_monitor.start_monitoring(
        run.run_id, run.output_path, kappa_session_id, run.lesion_type
    ))

    logger.info(
        "Requeue: новый run_id %s на тех же путях, что и %s",
        run.run_id, original_run.run_id,
    )
    return run
