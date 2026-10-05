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
