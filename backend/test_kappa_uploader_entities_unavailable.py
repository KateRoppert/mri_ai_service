"""
KI-069: a dataset listing that could not be read is "Kappa unavailable",
never "the dataset is empty".

Every duplicate and name-clash check reads the dataset's entity listing. When
get_dataset_entities() failed it returned None, the uploader took that as an
empty dataset, and the checks switched themselves off: on 2026-10-12 every
delivery attempt re-POSTed sessions an earlier attempt had already delivered
(dataset 353 ended with 29 entities for 13 names), and on 2026-10-14 a listing
that failed for the hashes but worked for the names flagged all 11 delivered
sessions as name clashes.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

import kappa_client
import kappa_delivery
import kappa_uploader


def _uploader():
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="r1", output_path="/tmp", token="tok", user_id=26, user_type_id=3,
        lesion_type="glioblastoma", preprocessing_config_path=cfg, dataset_id=353,
    )


def _complete_gbm(session_key):
    return {
        "preprocessed": [Path(f"{session_key}_{m}.nii.gz") for m in ("t1", "t1c", "t2", "t2fl")],
        "masks": [Path(f"{session_key}_segmask.nii.gz")],
    }


def _row(name, hash_):
    return {"dsEntityName": name, "dsEntityInfo": {"study_hash": hash_}}


@pytest.fixture
def up(monkeypatch, tmp_path):
    u = _uploader()
    monkeypatch.setattr(u, "_allocation_db", tmp_path / "alloc.db", raising=False)
    monkeypatch.setattr(u, "_discover_sessions", lambda: {
        "sub-001_ses-001": _complete_gbm("sub-001_ses-001"),
        "sub-001_ses-002": _complete_gbm("sub-001_ses-002"),
    })
    monkeypatch.setattr(u, "_compute_study_hash",
                        lambda data: "h-" + data["masks"][0].name[:15])
    u.posted = []

    async def fake_upload_session(dataset_id, session_key, session_data):
        u.posted.append(session_key)
        return {"session": session_key, "success": True, "entity_id": "new"}

    async def fake_reconcile(dataset_id, session_key, session_data, study_hash, attempts=3):
        return "existing-id"

    monkeypatch.setattr(u, "_upload_session", fake_upload_session)
    monkeypatch.setattr(u, "_reconcile_session", fake_reconcile)
    return u


def _listing(monkeypatch, *responses):
    """get_dataset_entities returns these in order (the last one repeats)."""
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return responses[min(len(calls), len(responses)) - 1]

    monkeypatch.setattr(kappa_client, "get_dataset_entities", fake)
    return calls


@pytest.mark.asyncio
async def test_unreadable_listing_uploads_nothing_and_is_retried(monkeypatch, up):
    _listing(monkeypatch, None)

    report = await up.upload_results()

    assert up.posted == []
    assert report["error"] == "entities_unavailable"
    assert report["total"] == 2
    state = kappa_delivery.classify(report, None, {}, datetime.now(timezone.utc))
    assert state["status"] == "pending"
    assert state["detail"]["reason"] == "network"


@pytest.mark.asyncio
async def test_hashes_and_names_come_from_one_listing(monkeypatch, up):
    """14 Oct: first read failed, second worked -> every session a 'name
    clash'. With one read the two views cannot disagree."""
    calls = _listing(monkeypatch, None, [_row("sub-001_ses-001", "h-sub-001_ses-001")])

    report = await up.upload_results()

    assert len(calls) == 1
    assert report["error"] == "entities_unavailable"
    assert not any(s.get("error") == "name_clash" for s in report.get("sessions", []))


@pytest.mark.asyncio
async def test_delivered_sessions_are_duplicates_not_reposted(monkeypatch, up):
    calls = _listing(monkeypatch, [
        _row("sub-001_ses-001", "h-sub-001_ses-001"),
        _row("sub-001_ses-002", "h-sub-001_ses-002"),
    ])

    report = await up.upload_results()

    assert len(calls) == 1
    assert up.posted == []
    assert all(s["skipped_upload"] and s["success"] for s in report["sessions"])


@pytest.mark.asyncio
async def test_empty_dataset_still_uploads(monkeypatch, up):
    _listing(monkeypatch, [])

    report = await up.upload_results()

    assert sorted(up.posted) == ["sub-001_ses-001", "sub-001_ses-002"]
    assert report["uploaded"] == 2


@pytest.mark.asyncio
async def test_each_attempt_reads_the_listing_afresh(monkeypatch, up):
    """The cache lives for one attempt: a retry must see what Kappa holds now."""
    calls = _listing(monkeypatch, None, [])

    first = await up.upload_results()
    second = await up.upload_results()

    assert first["error"] == "entities_unavailable"
    assert second["uploaded"] == 2
    assert len(calls) == 2
