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


def test_count_local_progress_from_disk(tmp_path, monkeypatch):
    pre1 = tmp_path / "preprocessed" / "sub-001" / "ses-001" / "anat"
    pre1.mkdir(parents=True)
    (pre1 / "sub-001_ses-001_t1.nii.gz").write_bytes(b"x")
    pre2 = tmp_path / "preprocessed" / "sub-002" / "ses-001" / "anat"
    pre2.mkdir(parents=True)
    (pre2 / "sub-002_ses-001_t1.nii.gz").write_bytes(b"x")

    import patient_registry
    from kappa_delivery import count_local_progress

    monkeypatch.setattr(patient_registry, "find_by_run_id", lambda rid: [])
    monkeypatch.setattr(
        patient_registry, "find_by_bids_id",
        lambda bids_id, dataset_ids=None: (
            [{"bids_id": bids_id, "kappa_entity_id": "e1"}]
            if bids_id == "sub-001_ses-001" else []
        ),
    )
    assert count_local_progress("any", tmp_path, dataset_id=350) == {
        "total": 2, "delivered": 1,
    }


def test_mark_in_progress_holds_the_worker_slot():
    from kappa_delivery import mark_in_progress
    out = mark_in_progress({"attempts": 0}, NOW, total=4, delivered=3)
    assert out["status"] == "pending"
    assert out["detail"]["total"] == 4
    assert out["detail"]["delivered"] == 3
    assert out["detail"]["reason"] is None
    assert out["next_attempt"] == NOW + timedelta(minutes=15)


def test_network_error_keeps_known_counters():
    """Kappa down must not wipe '3 of 4 already there' from the column."""
    state = {"total": 4, "delivered": 3, "attempts": 1}
    out = classify({"error": "Failed to resolve dataset_id"}, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "network"
    assert out["detail"]["total"] == 4
    assert out["detail"]["delivered"] == 3


def test_no_session_keeps_known_counters():
    state = {"total": 4, "delivered": 3, "attempts": 2}
    out = classify(NO_SESSION, None, state, NOW)
    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "no_session"
    assert out["detail"]["total"] == 4
    assert out["detail"]["delivered"] == 3


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


def test_count_local_progress_is_scoped_to_this_runs_dataset(tmp_path, monkeypatch):
    """A session key is only "already in Kappa" if it is in THIS run's dataset.

    sub-NNN is unique only WITHIN a dataset, and the live registry really does
    hold e.g. sub-001_ses-001 in two datasets at once. An unscoped lookup
    counts another dataset's patient as delivered, and because counters only
    ever grow, the column then permanently claims data arrived that never did.
    """
    pre = tmp_path / "preprocessed" / "sub-001" / "ses-001" / "anat"
    pre.mkdir(parents=True)
    (pre / "sub-001_ses-001_t1.nii.gz").write_bytes(b"x")

    import patient_registry
    from kappa_delivery import count_local_progress

    seen = {}

    def _find_by_bids_id(bids_id, dataset_ids=None):
        seen["dataset_ids"] = dataset_ids
        # The same BIDS id, uploaded by somebody else into dataset 338.
        rows = [{"bids_id": bids_id, "kappa_entity_id": "other",
                 "kappa_dataset_id": 338}]
        # Filter as the real find_by_bids_id does, so this test proves the
        # scoping works — not merely that an argument was passed along.
        if dataset_ids is not None:
            rows = [r for r in rows if r["kappa_dataset_id"] in dataset_ids]
        return rows

    monkeypatch.setattr(patient_registry, "find_by_run_id", lambda rid: [])
    monkeypatch.setattr(patient_registry, "find_by_bids_id", _find_by_bids_id)

    result = count_local_progress("any", tmp_path, dataset_id=350)

    assert seen["dataset_ids"] == {350}, "lookup must be scoped to the dataset"
    assert result == {"total": 1, "delivered": 0}


def test_count_local_progress_without_a_dataset_trusts_only_this_run(
    tmp_path, monkeypatch
):
    """Kappa was unreachable at start, so the run has no dataset yet. Nothing
    can be known to have reached a dataset we cannot name — only this run's
    own registry rows count."""
    pre = tmp_path / "preprocessed" / "sub-001" / "ses-001" / "anat"
    pre.mkdir(parents=True)
    (pre / "sub-001_ses-001_t1.nii.gz").write_bytes(b"x")

    import patient_registry
    from kappa_delivery import count_local_progress

    def _must_not_be_called(bids_id, dataset_ids=None):
        raise AssertionError("unscoped registry lookup with no dataset context")

    monkeypatch.setattr(patient_registry, "find_by_run_id", lambda rid: [])
    monkeypatch.setattr(patient_registry, "find_by_bids_id", _must_not_be_called)

    assert count_local_progress("any", tmp_path, dataset_id=None) == {
        "total": 1, "delivered": 0,
    }


def test_a_superseding_session_needs_a_human_not_a_retry():
    """Overwriting data in Kappa is not something to retry into."""
    result = {"dataset_id": 351, "uploaded": 0, "total": 1,
              "sessions": [_fail("sub-002_ses-001", "supersedes")]}
    out = classify(result, None, {}, NOW)

    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "supersedes_kappa"
    assert out["next_attempt"] is None
    assert out["detail"]["blocked"][0]["session"] == "sub-002_ses-001"


def test_a_superseding_session_alongside_a_delivered_one():
    """One blocked session must not erase the fact that the other arrived —
    the operator decides about the one patient, not the whole run."""
    result = {"dataset_id": 351, "uploaded": 1, "total": 2,
              "sessions": [_ok("sub-001_ses-001"),
                           _fail("sub-002_ses-001", "supersedes")]}
    out = classify(result, None, {}, NOW)

    assert out["status"] == "needs_attention"
    assert out["detail"]["delivered"] == 1
    assert len(out["detail"]["blocked"]) == 1


# --- Resolving a supersedes blockage by replacing in Kappa ------------------

def test_replacing_a_superseding_session_clears_its_blockage():
    """The operator pressed «Заменить в Kappa» and it worked. If the run
    still reads «требует внимания, 1 из 2», the screen contradicts what just
    happened — which is indistinguishable from Kappa being down."""
    from kappa_delivery import mark_session_delivered

    state = {"total": 2, "delivered": 1, "attempts": 0,
             "blocked": [{"session": "sub-002_ses-001",
                          "reason": "supersedes_kappa", "message": "m"}],
             "reason": "supersedes_kappa"}

    out = mark_session_delivered(state, "sub-002_ses-001", NOW)

    assert out["status"] == "done"
    assert out["detail"]["delivered"] == 2
    assert out["detail"]["blocked"] == []
    assert out["detail"]["reason"] is None
    assert out["next_attempt"] is None


def test_another_blocked_session_keeps_the_run_in_needs_attention():
    from kappa_delivery import mark_session_delivered

    state = {"total": 3, "delivered": 1, "attempts": 0,
             "blocked": [
                 {"session": "a", "reason": "supersedes_kappa", "message": ""},
                 {"session": "b", "reason": "name_clash", "message": ""},
             ],
             "reason": "supersedes_kappa"}

    out = mark_session_delivered(state, "a", NOW)

    assert out["status"] == "needs_attention"
    assert out["detail"]["delivered"] == 2
    assert [b["session"] for b in out["detail"]["blocked"]] == ["b"]
    assert out["detail"]["reason"] == "name_clash"


def test_delivered_never_exceeds_total():
    """A repeated replacement is allowed — it is idempotent in Kappa — so the
    counter must not drift past the number of sessions."""
    from kappa_delivery import mark_session_delivered

    state = {"total": 1, "delivered": 1, "blocked": [], "attempts": 0}
    out = mark_session_delivered(state, "sub-002_ses-001", NOW)

    assert out["detail"]["delivered"] == 1


def test_sessions_still_undelivered_keep_the_run_pending():
    """Nothing is blocked, but not everything arrived: the worker should
    still finish the job rather than the run reading as done."""
    from kappa_delivery import mark_session_delivered

    state = {"total": 3, "delivered": 0, "attempts": 0,
             "blocked": [{"session": "a", "reason": "supersedes_kappa",
                          "message": ""}]}

    out = mark_session_delivered(state, "a", NOW)

    assert out["status"] == "pending"
    assert out["next_attempt"] is not None
    assert out["detail"]["delivered"] == 1


def test_a_replacement_completes_a_run_that_also_lost_a_session():
    """Sessions lost in processing count toward completion (classify's own
    rule), so the replacement of the last blocked one has to finish the run.

    Without this the run stays pending on a total it can never reach, and
    the worker retries it forever. Found by merging this work onto main,
    where not_processed had been added meanwhile.
    """
    from kappa_delivery import mark_session_delivered

    state = {"total": 3, "delivered": 1, "attempts": 0,
             "not_processed": [{"session": "c", "message": "нет данных"}],
             "blocked": [{"session": "b", "reason": "supersedes_kappa",
                          "message": ""}]}

    out = mark_session_delivered(state, "b", NOW)

    assert out["status"] == "done"
    assert out["detail"]["delivered"] == 2
    assert len(out["detail"]["not_processed"]) == 1


def test_a_lost_session_is_not_counted_as_delivered():
    """It is reported separately, and conflating the two would tell the
    operator a patient reached Kappa when nothing was ever uploadable."""
    from kappa_delivery import mark_session_delivered

    state = {"total": 3, "delivered": 0, "attempts": 0,
             "not_processed": [{"session": "c", "message": "нет данных"}],
             "blocked": [{"session": "b", "reason": "supersedes_kappa",
                          "message": ""}]}

    out = mark_session_delivered(state, "b", NOW)

    assert out["detail"]["delivered"] == 1
    assert out["status"] == "pending"      # 1 + 1 < 3, one still owed
