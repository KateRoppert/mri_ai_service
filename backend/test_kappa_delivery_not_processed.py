"""
A session lost in processing is counted, never uploaded, and never retried.

The uploader now reports such sessions as error="not_processed" and counts
them in `total` (run 30_09_1752: 4 delivered of 6). The retry policy treated
any delivered < total as "Kappa still owes us data" and would have retried
the missing two forever — but their data does not exist.
"""
from datetime import datetime, timezone

from kappa_delivery import classify, count_local_progress

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _ok(session):
    return {"session": session, "success": True, "entity_id": "e1"}


def _not_processed(session, why):
    return {"session": session, "success": False, "error": "not_processed",
            "message": why}


def _the_30_09_1752_result():
    return {
        "dataset_id": 353, "uploaded": 4, "total": 6,
        "sessions": [_ok(f"sub-003_ses-00{n}") for n in range(1, 5)] + [
            _not_processed("sub-003_ses-005", "нет t2fl и маски сегментации"),
            _not_processed("sub-003_ses-006", "нет данных после предобработки"),
        ],
    }


def test_everything_ready_delivered_is_done_with_the_true_count():
    out = classify(_the_30_09_1752_result(), None, {}, NOW)

    assert out["status"] == "done"
    assert out["next_attempt"] is None
    assert (out["detail"]["delivered"], out["detail"]["total"]) == (4, 6)


def test_not_processed_sessions_are_listed_with_reasons():
    out = classify(_the_30_09_1752_result(), None, {}, NOW)

    assert out["detail"]["not_processed"] == [
        {"session": "sub-003_ses-005", "message": "нет t2fl и маски сегментации"},
        {"session": "sub-003_ses-006", "message": "нет данных после предобработки"},
    ]
    assert out["detail"]["blocked"] == []


def test_nothing_ready_needs_attention_as_missing_files():
    result = {"dataset_id": 353, "uploaded": 0, "total": 1,
              "sessions": [_not_processed("sub-001_ses-001", "нет данных после предобработки")]}

    out = classify(result, None, {}, NOW)

    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "missing_files"
    assert out["detail"]["total"] == 1
    assert len(out["detail"]["not_processed"]) == 1


def test_a_ready_session_that_failed_to_upload_still_retries():
    result = _the_30_09_1752_result()
    result["sessions"][3] = {"session": "sub-003_ses-004", "success": False,
                             "error": "upload failed", "message": "502"}

    out = classify(result, None, {}, NOW)

    assert out["status"] == "pending"
    assert out["detail"]["reason"] == "network"


def test_name_clash_still_wins_over_not_processed():
    result = _the_30_09_1752_result()
    result["sessions"][0] = {"session": "sub-003_ses-001", "success": False,
                             "error": "name_clash", "message": "clash"}

    out = classify(result, None, {}, NOW)

    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "name_clash"


def test_local_count_includes_sessions_lost_before_preprocessing(tmp_path, monkeypatch):
    """The column is seeded from disk before the upload answers; it must use
    the same denominator, or it shows "of 5" and only grows to 6 later."""
    for n in range(1, 7):
        (tmp_path / "bids_organized" / "sub-003" / f"ses-00{n}" / "anat").mkdir(parents=True)
    for n in range(1, 6):
        anat = tmp_path / "preprocessed" / "sub-003" / f"ses-00{n}" / "anat"
        anat.mkdir(parents=True)
        (anat / f"sub-003_ses-00{n}_t1.nii.gz").write_bytes(b"x")
    (tmp_path / "bids_organized" / "_incomplete" / "sub-009" / "ses-001").mkdir(parents=True)

    import patient_registry
    monkeypatch.setattr(patient_registry, "find_by_run_id", lambda rid: [])
    monkeypatch.setattr(patient_registry, "find_by_bids_id",
                        lambda bids_id, dataset_ids=None: [])

    assert count_local_progress("r1", tmp_path, dataset_id=353) == {
        "total": 6, "delivered": 0,
    }
