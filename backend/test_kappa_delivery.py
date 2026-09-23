"""The Kappa delivery retry policy. Pure function, no network, no DB."""
from datetime import datetime, timedelta, timezone

import pytest

from kappa_delivery import NO_SESSION, classify

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _ok(session):
    return {"session": session, "success": True, "entity_id": "e1"}


def _fail(session, error):
    return {"session": session, "success": False, "error": error,
            "message": f"{session}: {error}"}


def test_all_sessions_delivered_is_done():
    result = {"dataset_id": 350, "uploaded": 2, "total": 2,
              "sessions": [_ok("sub-001_ses-001"), _ok("sub-002_ses-001")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "done"
    assert out["next_attempt"] is None
    assert out["detail"]["delivered"] == 2


def test_duplicate_counts_as_delivered():
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "duplicate")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "done"
    assert out["detail"]["delivered"] == 2


def test_unreachable_kappa_is_transient():
    out = classify({"error": "Failed to resolve dataset_id"}, None, {}, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "network"
    assert out["detail"]["attempts"] == 1
    assert out["next_attempt"] == NOW + timedelta(minutes=1)


def test_exception_is_transient():
    out = classify(None, RuntimeError("boom"), {}, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "network"
    assert "boom" in out["detail"]["last_error"]


def test_upload_failed_session_is_transient_with_counters():
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "upload failed")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["delivered"] == 1
    assert out["detail"]["total"] == 2


def test_name_clash_needs_attention_and_never_retries():
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "name_clash")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "needs_attention"
    assert out["next_attempt"] is None
    assert out["detail"]["reason"] == "name_clash"
    assert out["detail"]["blocked"][0]["session"] == "sub-002_ses-001"


def test_no_files_needs_attention():
    result = {"dataset_id": 350, "uploaded": 0, "total": 1,
              "sessions": [_fail("sub-001_ses-001", "no files")]}
    out = classify(result, None, {}, NOW)
    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "missing_files"


def test_nothing_on_disk_needs_attention():
    out = classify({"uploaded": 0, "sessions": []}, None, {}, NOW)
    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "missing_files"


def test_no_session_waits_without_counting_an_attempt():
    state = {"attempts": 3, "first_failure_at": "2026-09-23T11:00:00+00:00"}
    out = classify(NO_SESSION, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "no_session"
    assert out["detail"]["attempts"] == 3          # unchanged
    assert out["next_attempt"] == NOW + timedelta(minutes=5)


@pytest.mark.parametrize("prior_attempts,minutes", [
    (0, 1), (1, 2), (2, 5), (3, 15), (4, 30), (5, 60), (9, 60), (99, 60),
])
def test_backoff_schedule_and_cap(prior_attempts, minutes):
    state = {"attempts": prior_attempts,
             "first_failure_at": NOW.isoformat()}
    out = classify({"error": "Failed to resolve dataset_id"}, None, state, NOW)
    assert out["next_attempt"] == NOW + timedelta(minutes=minutes)


def test_stuck_after_24h_with_nothing_delivered():
    state = {"attempts": 40,
             "first_failure_at": (NOW - timedelta(hours=25)).isoformat()}
    out = classify({"error": "Failed to resolve dataset_id"}, None, state, NOW)
    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "stuck"
    assert out["next_attempt"] is None


def test_no_session_waits_do_not_make_a_run_stuck():
    state = {"attempts": 0,
             "first_failure_at": (NOW - timedelta(hours=48)).isoformat()}
    out = classify(NO_SESSION, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "no_session"


def test_progress_resets_the_stuck_window():
    state = {"attempts": 40,
             "first_failure_at": (NOW - timedelta(hours=25)).isoformat()}
    result = {"dataset_id": 350, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "upload failed")]}
    out = classify(result, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["first_failure_at"] is None
    assert out["detail"]["delivered"] == 1
