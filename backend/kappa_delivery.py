"""Retry policy for deferred Kappa uploads.

Pure: no network, no database, no clock. Everything it needs arrives as an
argument and everything it decides comes back as a return value, so the
whole policy is testable without a Kappa instance.

Design note — the vocabulary is narrow on purpose. backend/kappa_client.py
logs HTTP status codes and returns None, so a 403 and a dropped connection
are indistinguishable by the time they reach here. The `stuck` rule below is
what keeps a permanent failure from hiding behind an endless retry.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

BACKOFF_MINUTES = [1, 2, 5, 15, 30, 60]
NO_SESSION_RETRY_MINUTES = 5
STUCK_AFTER_HOURS = 24

# Sentinel the worker passes when no live Kappa session exists for the run's
# user. Routed through classify() like everything else so there is exactly
# one place where delivery state is decided.
NO_SESSION: Dict[str, Any] = {"error": "no_session"}

# Per-session errors that retrying cannot fix.
_BLOCKING = {
    "name_clash": "name_clash",
    "no files": "missing_files",
}


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _detail(state: Dict[str, Any], now: datetime, **over: Any) -> Dict[str, Any]:
    detail = {
        "total": state.get("total", 0),
        "delivered": state.get("delivered", 0),
        "blocked": [],
        "reason": None,
        "last_error": None,
        "attempts": state.get("attempts", 0),
        "first_failure_at": state.get("first_failure_at"),
        "last_attempt_at": now.isoformat(),
    }
    detail.update(over)
    return detail


def _transient(state, now, detail_over) -> Dict[str, Any]:
    """Schedule another attempt, or give up and ask for a human.

    A run only becomes `stuck` when it has delivered nothing at all: partial
    progress means the path works and the rest is worth retrying.
    """
    attempts = state.get("attempts", 0) + 1
    delivered = detail_over.get(
        "delivered", state.get("delivered", 0) or 0
    )

    first_failure = None if delivered else (
        state.get("first_failure_at") or now.isoformat()
    )

    detail = _detail(state, now, attempts=attempts,
                     first_failure_at=first_failure, **detail_over)

    started = _parse(first_failure)
    if (not delivered and started
            and now - started >= timedelta(hours=STUCK_AFTER_HOURS)):
        detail["reason"] = "stuck"
        return {"status": "needs_attention", "next_attempt": None,
                "detail": detail}

    index = min(attempts - 1, len(BACKOFF_MINUTES) - 1)
    return {
        "status": "pending",
        "next_attempt": now + timedelta(minutes=BACKOFF_MINUTES[index]),
        "detail": detail,
    }


def classify(
    result: Optional[Dict[str, Any]],
    exc: Optional[BaseException],
    state: Dict[str, Any],
    now: datetime,
) -> Dict[str, Any]:
    """Turn the outcome of one upload attempt into the run's new state.

    `result` is what KappaUploader.upload_results() returned (or NO_SESSION),
    `exc` is what it raised, `state` is the previous detail blob.
    """
    state = state or {}

    if exc is not None:
        return _transient(state, now,
                          {"reason": "network", "last_error": repr(exc)})

    if result is None:
        return _transient(state, now,
                          {"reason": "network",
                           "last_error": "uploader returned nothing"})

    error = result.get("error")

    if error == "no_session":
        # Waiting for a human to log in is not a failure: it must not burn
        # backoff attempts and must not count toward the stuck window.
        return {
            "status": "pending",
            "next_attempt": now + timedelta(minutes=NO_SESSION_RETRY_MINUTES),
            "detail": _detail(state, now, reason="no_session"),
        }

    if error:
        # Kappa unreachable, dataset gone, or no rights — indistinguishable.
        return _transient(state, now,
                          {"reason": "network", "last_error": str(error)})

    sessions = result.get("sessions") or []
    total = result.get("total", len(sessions))

    delivered = sum(
        1 for s in sessions
        if s.get("success") or s.get("error") == "duplicate"
    )
    blocked = [
        {"session": s.get("session"),
         "reason": _BLOCKING[s.get("error")],
         "message": s.get("message") or ""}
        for s in sessions
        if not s.get("success") and s.get("error") in _BLOCKING
    ]

    counters = {"total": total, "delivered": delivered}

    if not sessions:
        # A completed run that produced nothing to upload. Not transient —
        # retrying an empty directory forever tells the operator nothing.
        return {
            "status": "needs_attention",
            "next_attempt": None,
            "detail": _detail(state, now, reason="missing_files",
                              total=0, delivered=0),
        }

    if blocked:
        return {
            "status": "needs_attention",
            "next_attempt": None,
            "detail": _detail(state, now, reason=blocked[0]["reason"],
                              blocked=blocked, **counters),
        }

    if delivered >= total:
        return {
            "status": "done",
            "next_attempt": None,
            "detail": _detail(state, now, first_failure_at=None, **counters),
        }

    return _transient(state, now, {"reason": "network", **counters})


# How long an in-flight upload holds the worker slot so a second tick
# does not start the same run again.
IN_FLIGHT_HOLD_MINUTES = 15


def mark_in_progress(state: Dict[str, Any], now: datetime,
                     total: int, delivered: int) -> Dict[str, Any]:
    """Persist known counters before upload_results() returns.

    Without this the history column stays at 0/0 for the whole (slow)
    upload, and a Kappa outage on the first attempt has nothing to show.
    """
    state = state or {}
    return {
        "status": "pending",
        "next_attempt": now + timedelta(minutes=IN_FLIGHT_HOLD_MINUTES),
        "detail": _detail(
            state, now,
            total=max(int(state.get("total") or 0), int(total or 0)),
            delivered=max(int(state.get("delivered") or 0), int(delivered or 0)),
            reason=None,
            last_error=None,
        ),
    }


def merge_local_counters(state: Dict[str, Any], total: int,
                         delivered: int) -> Dict[str, Any]:
    """Never let a later snapshot shrink counters we already published."""
    state = dict(state or {})
    state["total"] = max(int(state.get("total") or 0), int(total or 0))
    state["delivered"] = max(int(state.get("delivered") or 0), int(delivered or 0))
    return state


def _session_key_from_path(filepath) -> Optional[str]:
    from pathlib import Path
    filepath = Path(filepath)
    parts = filepath.parts[:-1]
    sub = ses = None
    for part in parts:
        if part.startswith("sub-"):
            sub = part
        elif part.startswith("ses-"):
            ses = part
    if sub and ses:
        return f"{sub}_{ses}"
    name = filepath.stem
    if name.endswith(".nii"):
        name = name[:-4]
    bits = name.split("_")
    for i, part in enumerate(bits):
        if part.startswith("sub-") and i + 1 < len(bits) and bits[i + 1].startswith("ses-"):
            return f"{part}_{bits[i + 1]}"
    return None


def count_local_progress(
    run_id: str, output_path, dataset_id: Optional[int] = None
) -> Dict[str, int]:
    """Sessions on disk vs already registered as uploaded (local registry).

    Does not talk to Kappa. Used to seed the column before an attempt
    finishes, and to keep '3 of 4 already there' when Kappa is down.

    `dataset_id` is the run's own Kappa dataset, and it is what makes the
    answer trustworthy: sub-NNN is unique only WITHIN a dataset, so an
    unscoped registry lookup would count somebody else's sub-001 as this
    run's delivered session. Counters only ever grow (merge_local_counters),
    so such a mistake is permanent — the column would claim data arrived that
    never did. None means the run has no dataset yet (Kappa was unreachable
    at start): nothing can be known to have reached a dataset we cannot name,
    so only this run's own registry rows count.
    """
    from pathlib import Path

    from patient_registry import find_by_bids_id, find_by_run_id

    keys: set = set()
    pre = Path(output_path) / "preprocessed"
    if pre.is_dir():
        for nifti in pre.rglob("*.nii.gz"):
            key = _session_key_from_path(nifti)
            if key:
                keys.add(key)

    this_run = {
        r.get("bids_id") for r in (find_by_run_id(run_id) or [])
        if r.get("kappa_entity_id") and r.get("bids_id")
    }

    def _already_in_kappa(session_key: str) -> bool:
        if session_key in this_run:
            return True
        if dataset_id is None:
            return False
        return any(
            r.get("kappa_entity_id")
            for r in (find_by_bids_id(session_key, {dataset_id}) or [])
        )

    if keys:
        delivered = sum(1 for key in keys if _already_in_kappa(key))
        total = max(len(keys), delivered)
    else:
        delivered = len(this_run)
        total = delivered
    return {"total": total, "delivered": delivered}
