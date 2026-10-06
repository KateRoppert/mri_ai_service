"""Replacing an entity's contents, on purpose and only on purpose."""
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, "backend")


class _Run:
    run_id = "run-1"
    output_path = "/tmp/run"
    lesion_type = "glioblastoma"
    kappa_dataset_id = 351
    reprocessed_sessions = json.dumps(["sub-002_ses-001"])
    # The endpoint recomputes delivery state after a successful replacement.
    kappa_upload_status = "needs_attention"
    kappa_upload_next_attempt = None
    kappa_upload_detail = json.dumps({
        "total": 1, "delivered": 0,
        "blocked": [{"session": "sub-002_ses-001",
                     "reason": "supersedes_kappa", "message": "m"}],
        "reason": "supersedes_kappa", "attempts": 0,
    })


class _Uploader:
    token = "t"
    user_id = 26
    user_type_id = 1

    def _discover_sessions(self):
        return {"sub-002_ses-001": {
            "preprocessed": [Path("/tmp/run/t1.nii.gz")],
            "masks": [], "lesion_labels_mask": None,
        }}


@pytest.mark.asyncio
async def test_refuses_a_session_that_supersedes_nothing(monkeypatch):
    """This endpoint overwrites data in someone else's system. It must not
    be a general purpose tool reachable by guessing a URL."""
    import app

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-009", "ses-001", kappa_session_id="sid", db=None)

    assert caught.value.status_code == 400
    assert "не помечена" in caught.value.detail


@pytest.mark.asyncio
async def test_replaces_and_reports_the_counts(monkeypatch):
    import app
    import kappa_entity_files as kef

    seen = {}

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: _Uploader())

    def _find(bids_id, dataset_ids=None):
        seen["dataset_ids"] = dataset_ids
        return [{"kappa_entity_id": "e1"}]
    monkeypatch.setattr(app, "find_by_bids_id", _find)

    async def _replace(**kwargs):
        seen["files"] = [p.name for p in kwargs["files"]]
        seen["entity_id"] = kwargs["entity_id"]
        return {"patched": 3, "added": 1, "deleted": 1,
                "failed": [], "delete_job": "succeeded"}
    monkeypatch.setattr(kef, "replace_entity_contents", _replace)
    # app.py imports kappa_run_log inside the endpoint (house style), so
    # the real module is what the local import resolves.
    monkeypatch.setattr("kappa_run_log.append", lambda *a, **k: None)
    monkeypatch.setattr("database.set_kappa_delivery",
                        lambda *a, **k: None)

    result = await app.replace_kappa_entity(
        "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)

    assert (result.patched, result.added, result.deleted) == (3, 1, 1)
    assert result.failed == []
    assert seen["entity_id"] == "e1"
    assert seen["files"] == ["t1.nii.gz"]
    # sub-NNN is unique only WITHIN a dataset; an unscoped lookup would
    # report another account's patient as this one.
    assert seen["dataset_ids"] == {351}


@pytest.mark.asyncio
async def test_refuses_without_a_kappa_session(monkeypatch):
    import app

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: None)

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_refuses_when_kappa_has_no_such_entity(monkeypatch):
    """Nothing to replace is not the same as a replacement that did
    nothing, and the operator needs to be able to tell them apart."""
    import app

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: _Uploader())
    monkeypatch.setattr(app, "find_by_bids_id", lambda *a, **k: [])

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)
    assert caught.value.status_code == 404


@pytest.mark.asyncio
async def test_refuses_when_the_session_is_not_on_disk(monkeypatch):
    """The run's own files are what the replacement sends. Without them
    there is nothing to send, and an empty set would mean emptying the
    entity in Kappa."""
    import app

    class _Empty(_Uploader):
        def _discover_sessions(self):
            return {}

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: _Empty())
    monkeypatch.setattr(app, "find_by_bids_id",
                        lambda *a, **k: [{"kappa_entity_id": "e1"}])

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)
    assert caught.value.status_code == 404
    assert "нечем заменять" in caught.value.detail


# --- The expert-mask count shown in the confirmation ------------------------

def _run_with_blocked(reason):
    from types import SimpleNamespace
    from datetime import datetime, timezone
    import json as _json
    return SimpleNamespace(
        run_id="run-1", output_path="/tmp/run", kappa_dataset_id=351,
        kappa_upload_status="needs_attention",
        kappa_upload_next_attempt=None,
        kappa_upload_detail=_json.dumps({
            "total": 1, "delivered": 0, "reason": reason,
            "blocked": [{"session": "sub-002_ses-001", "reason": reason,
                         "message": "msg"}],
            "attempts": 1, "first_failure_at": None,
            "last_attempt_at": datetime.now(timezone.utc).isoformat(),
        }),
    )


def test_a_superseding_session_carries_its_expert_mask_count(monkeypatch):
    """The operator is about to overwrite data an expert drew on. The
    number is what makes that consequence concrete before confirming."""
    import app

    monkeypatch.setattr(app, "find_by_bids_id",
                        lambda *a, **k: [{"kappa_entity_id": "e1"}])
    monkeypatch.setattr("mask_service.get_mask_history",
                        lambda eid: [{"source": "ai"}, {"source": "expert"},
                                     {"source": "expert"}])

    status = app._delivery_status(_run_with_blocked("supersedes_kappa"))

    assert status.blocked[0].expert_masks == 2


def test_other_reasons_do_not_pay_for_the_lookup(monkeypatch):
    """Two queries per blocked row on every history page, for a number
    nothing would display. The reason guard is load-bearing."""
    import app

    calls = []
    monkeypatch.setattr(app, "find_by_bids_id",
                        lambda *a, **k: calls.append(a) or [])

    status = app._delivery_status(_run_with_blocked("name_clash"))

    assert status.blocked[0].expert_masks == 0
    assert calls == [], "реестр опрошен там, где счётчик не нужен"


def test_a_failed_count_does_not_break_the_history(monkeypatch):
    """The count decorates a confirmation. The history list is how the
    operator finds the problem at all, so it must survive the decoration
    failing."""
    import app

    def _boom(*a, **k):
        raise RuntimeError("реестр недоступен")
    monkeypatch.setattr(app, "find_by_bids_id", _boom)

    status = app._delivery_status(_run_with_blocked("supersedes_kappa"))

    assert status.blocked[0].expert_masks == 0
    assert status.status == "needs_attention"


# --- The replacement has to move the run's delivery state -------------------
#
# Observed live on 2026-10-05: both replacements succeeded in Kappa (5 files
# patched, 1 deleted, delete job succeeded) and the run kept reading
# «требует внимания, 1 из 2». Nothing was wrong except the screen, which is
# indistinguishable from Kappa being down — and that is what it was reported
# as.

def _run_needing_replacement():
    from types import SimpleNamespace
    from datetime import datetime, timezone
    return SimpleNamespace(
        run_id="run-1", output_path="/tmp/run", lesion_type="glioblastoma",
        kappa_dataset_id=351,
        reprocessed_sessions=json.dumps(["sub-002_ses-001"]),
        kappa_upload_status="needs_attention",
        kappa_upload_next_attempt=None,
        kappa_upload_detail=json.dumps({
            "total": 2, "delivered": 1, "reason": "supersedes_kappa",
            "blocked": [{"session": "sub-002_ses-001",
                         "reason": "supersedes_kappa", "message": "m"}],
            "attempts": 0, "first_failure_at": None,
            "last_attempt_at": datetime(2026, 10, 5, 9, 40,
                                        tzinfo=timezone.utc).isoformat(),
        }),
    )


def _wire_successful_replacement(monkeypatch, result):
    import app
    import kappa_entity_files as kef

    run = _run_needing_replacement()
    written = {}

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: run)
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: _Uploader())
    monkeypatch.setattr(app, "find_by_bids_id",
                        lambda *a, **k: [{"kappa_entity_id": "e1"}])
    monkeypatch.setattr("kappa_run_log.append", lambda *a, **k: None)

    async def _replace(**kwargs):
        return result
    monkeypatch.setattr(kef, "replace_entity_contents", _replace)

    def _set(db, run_id, status, next_attempt, detail):
        written.update(status=status, next_attempt=next_attempt, detail=detail)
    monkeypatch.setattr("database.set_kappa_delivery", _set)
    return written


@pytest.mark.asyncio
async def test_a_complete_replacement_clears_the_blockage(monkeypatch):
    import app

    written = _wire_successful_replacement(monkeypatch, {
        "patched": 5, "added": 0, "deleted": 1,
        "failed": [], "delete_job": "succeeded",
    })

    await app.replace_kappa_entity(
        "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)

    assert written, "состояние доставки не обновлено — экран остался прежним"
    assert written["status"] == "done"
    assert written["detail"]["delivered"] == 2
    assert written["detail"]["blocked"] == []


@pytest.mark.asyncio
async def test_a_partial_replacement_leaves_the_session_blocked(monkeypatch):
    """Some files did not make it, so the patient in Kappa is now a mix of
    old and new. Calling that delivered would hide a worse state than the
    one we started from."""
    import app

    written = _wire_successful_replacement(monkeypatch, {
        "patched": 3, "added": 0, "deleted": 0,
        "failed": ["sub-002_ses-001_t2.nii.gz"], "delete_job": None,
    })

    await app.replace_kappa_entity(
        "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)

    assert written == {}, "частичная замена не должна считаться доставкой"


@pytest.mark.asyncio
async def test_a_delete_job_still_running_does_not_block_the_verdict(monkeypatch):
    """Every file this run owns is in place; only Kappa's own cleanup of
    files we no longer send is still going. That is not a reason to keep
    asking the operator for a decision."""
    import app

    written = _wire_successful_replacement(monkeypatch, {
        "patched": 5, "added": 0, "deleted": 1,
        "failed": [], "delete_job": "running",
    })

    await app.replace_kappa_entity(
        "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)

    assert written["status"] == "done"
