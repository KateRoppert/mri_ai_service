"""Background delivery of results that did not reach Kappa the first time.

An asyncio task in the web process, started from app's lifespan beside the
orphaned-run reconciliation. No new process, container or scheduler —
pipeline_monitor already works this way.

Every attempt re-runs the existing KappaUploader, which dedups against the
dataset by study_hash. That is what makes retrying safe: an attempt is not
"send it again", it is "reconcile with what is actually there".
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from database import (
    SessionLocal,
    get_kappa_delivery,
    get_pipeline_run,
    runs_due_for_delivery,
    set_kappa_delivery,
)
from kappa_auth import find_live_session_for_user
from kappa_dataset_mapping import owner_of_dataset
from kappa_delivery import (
    NO_SESSION,
    classify,
    count_local_progress,
    mark_in_progress,
    merge_local_counters,
)
from pipeline_monitor import PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)

TICK_SECONDS = 60
BATCH_SIZE = 5


def build_uploader(run, session: Dict[str, Any]):
    """A KappaUploader for this run, using the token we have right now.

    Mirrors PipelineMonitor._create_kappa_uploader, but takes a session dict
    rather than a session id: the worker has already resolved the account by
    user id, because the session id the run started with is long gone.
    """
    from kappa_uploader import KappaUploader

    if not PREPROCESSING_CONFIG.exists():
        logger.error("Preprocessing config not found: %s", PREPROCESSING_CONFIG)
        return None

    return KappaUploader(
        run_id=run.run_id,
        output_path=run.output_path,
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        lesion_type=getattr(run, "lesion_type", None) or "glioblastoma",
        preprocessing_config_path=str(PREPROCESSING_CONFIG),
        dataset_id=run.kappa_dataset_id,
    )


def _owner_of(run) -> Optional[int]:
    """The Kappa account this run belongs to. Runs created before this feature
    have no kappa_user_id, but the migration only backfilled rows that DO have
    a dataset id — so the mapping can answer for them."""
    if run.kappa_user_id is not None:
        return run.kappa_user_id
    if run.kappa_dataset_id is not None:
        return owner_of_dataset(run.kappa_dataset_id)
    return None


# Last observed reachability of Kappa. None = never checked. Only the
# down -> up transition resumes waiting runs; see note_kappa_reachability().
_last_reachable: Optional[bool] = None


def kappa_reachable(timeout: float = 5.0) -> bool:
    """Can we open a TCP connection to Kappa right now?

    Deliberately not an API call: no token needed, nothing to authorise, and
    it answers the only question that matters for scheduling — is the service
    responding at all.
    """
    import socket
    from urllib.parse import urlparse

    from kappa_auth import KAPPA_BASE_URL

    parsed = urlparse(KAPPA_BASE_URL)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def note_kappa_reachability() -> None:
    """Watch Kappa come back, and pull the waiting runs in when it does.

    Backoff reaches an hour, so without this a run can sit idle long after an
    outage ends — which is indistinguishable from broken. Resuming only on the
    down -> up TRANSITION is what keeps that from defeating backoff entirely:
    a run failing for a reason other than the outage still waits its turn.
    """
    global _last_reachable
    reachable = kappa_reachable()
    if reachable and _last_reachable is False:
        resume_all_pending("Kappa снова отвечает")
    _last_reachable = reachable


def adopt_orphan_runs(user_id: int) -> int:
    """Give runs that started while Kappa was down an owner, and retry at once.

    A run started offline has no kappa_user_id and no dataset, so _owner_of()
    cannot name an account and the worker can never find a token for it — it
    would wait on `no_session` forever. Logging in is precisely the event that
    answers "whose is this": the person who just authenticated.

    Only unowned runs are adopted, so a second operator's pending work is
    never taken over. The backoff is cleared too: a fresh login is strong
    evidence Kappa is reachable, and making the operator wait out an hour of
    exponential backoff after that is indefensible.

    Returns how many runs were adopted.
    """
    if user_id is None:
        return 0

    from database import PipelineRun

    db = SessionLocal()
    try:
        orphans = db.query(PipelineRun).filter(
            PipelineRun.status == "completed",
            PipelineRun.kappa_upload_status == "pending",
            PipelineRun.kappa_user_id.is_(None),
        ).all()
        for run in orphans:
            run.kappa_user_id = user_id
            run.kappa_upload_next_attempt = None
        db.commit()
        if orphans:
            logger.info(
                "Kappa login by user %s adopted %d run(s) awaiting delivery",
                user_id, len(orphans),
            )
        return len(orphans)
    finally:
        db.close()


def resume_all_pending(reason: str) -> int:
    """Clear the backoff on every waiting run, so the next tick retries them.

    Used when something tells us Kappa is reachable again — a login, or one
    delivery succeeding while others sit on a long backoff. Without this a
    run can be an hour from its next attempt when the outage ends, which
    reads to the operator as "nothing is happening".
    """
    from database import PipelineRun

    db = SessionLocal()
    try:
        waiting = db.query(PipelineRun).filter(
            PipelineRun.status == "completed",
            PipelineRun.kappa_upload_status == "pending",
            PipelineRun.kappa_upload_next_attempt.isnot(None),
        ).all()
        for run in waiting:
            run.kappa_upload_next_attempt = None
        db.commit()
        if waiting:
            logger.info(
                "Retrying %d waiting run(s) now: %s", len(waiting), reason,
            )
        return len(waiting)
    finally:
        db.close()


async def deliver_one(run_id: str) -> Optional[Dict[str, Any]]:
    """One delivery attempt for one run. Never raises."""
    db = SessionLocal()
    try:
        run = get_pipeline_run(db, run_id)
        if run is None or run.kappa_upload_status != "pending":
            return None

        state = get_kappa_delivery(run)
        now = datetime.now(timezone.utc)
        local = count_local_progress(
            run_id, run.output_path, run.kappa_dataset_id
        )
        state = merge_local_counters(state, local["total"], local["delivered"])

        owner = _owner_of(run)
        session = find_live_session_for_user(owner) if owner else None

        # Bound on every path: both the verdict and the run log read them.
        result, exc = None, None

        if session is None:
            result = NO_SESSION
            verdict = classify(NO_SESSION, None, state, now)
        else:
            seeded = mark_in_progress(
                state, now, state["total"], state["delivered"],
            )
            set_kappa_delivery(
                db, run_id, seeded["status"],
                seeded["next_attempt"], seeded["detail"],
            )
            try:
                uploader = build_uploader(run, session)
                if uploader is None:
                    result = {"error": "preprocessing config not found"}
                else:
                    result = await uploader.upload_results()
            except Exception as e:          # noqa: BLE001 - recorded, not raised
                exc = e
                logger.error("Deferred Kappa upload failed for %s: %s", run_id, e)
            verdict = classify(result, exc, state, now)

        set_kappa_delivery(
            db, run_id, verdict["status"],
            verdict["next_attempt"], verdict["detail"],
        )
        import kappa_run_log
        kappa_run_log.log_attempt(
            run.output_path, "фоновая досылка", result, exc, verdict,
        )
        logger.info(
            "Deferred delivery %s: %s (%s/%s, reason=%s)",
            run_id, verdict["status"],
            verdict["detail"].get("delivered"), verdict["detail"].get("total"),
            verdict["detail"].get("reason"),
        )
        # Same WS types as the post-run upload: the open progress screen
        # already listens for these (Task 9). History polls on pending.
        try:
            from websocket_manager import ws_manager
            if verdict["status"] == "done":
                await ws_manager.broadcast(run_id, {
                    "type": "kappa_upload_complete",
                    "run_id": run_id,
                    "entities": [],
                })
            else:
                await ws_manager.broadcast(run_id, {
                    "type": "kappa_upload_deferred",
                    "run_id": run_id,
                    "status": verdict["status"],
                    "detail": verdict["detail"],
                })
        except Exception as e:  # noqa: BLE001 — UI notify must not fail delivery
            logger.debug("Could not broadcast delivery state for %s: %s", run_id, e)
        return verdict
    finally:
        db.close()


def _waiting_count(db) -> int:
    """Runs that owe delivery but are not due yet."""
    from database import PipelineRun
    return db.query(PipelineRun).filter(
        PipelineRun.status == "completed",
        PipelineRun.kappa_upload_status == "pending",
        PipelineRun.kappa_upload_next_attempt.isnot(None),
    ).count()


async def tick(now: Optional[datetime] = None) -> int:
    """One pass over the due runs. Returns how many were attempted."""
    now = now or datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        due = runs_due_for_delivery(db, now, limit=BATCH_SIZE)
        run_ids = [r.run_id for r in due]
        waiting = _waiting_count(db) if not run_ids else 0
    finally:
        db.close()

    # Only probe when something is actually waiting out a backoff — there is
    # no point poking Kappa on an idle system.
    if waiting:
        await asyncio.to_thread(note_kappa_reachability)

    for run_id in run_ids:
        await deliver_one(run_id)
    return len(run_ids)


async def delivery_loop() -> None:
    """Forever. One tick's failure must never kill the loop — a dead worker
    would silently stop every deferred upload in the system."""
    logger.info("Kappa delivery worker started (tick=%ss)", TICK_SECONDS)
    while True:
        try:
            await tick()
        except asyncio.CancelledError:
            logger.info("Kappa delivery worker cancelled")
            raise
        except Exception as e:              # noqa: BLE001
            logger.error("Kappa delivery tick failed: %s", e)
        await asyncio.sleep(TICK_SECONDS)


def start_delivery_worker() -> asyncio.Task:
    return asyncio.create_task(delivery_loop())
