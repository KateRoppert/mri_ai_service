"""KI-035: a complete rerun completes an incomplete Kappa entity.

Entities uploaded before _readiness existed can lack a modality and the mask
(audit 2026-10-19: one in dataset 337, six in 249). A complete rerun of the
same study has the same study_hash and used to be skipped as a duplicate, so
the incomplete version stayed forever and the doctor never got a mask.
"""
from pathlib import Path

import pytest

import kappa_client
import kappa_entity_files
import kappa_uploader

KEY = "sub-214_ses-001"
HASH = "h-214"


def _uploader(superseding=frozenset()):
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="run-new", output_path="/tmp", token="tok", user_id=26,
        user_type_id=3, lesion_type="glioblastoma",
        preprocessing_config_path=cfg, dataset_id=249,
        superseding_sessions=superseding,
    )


def _session():
    return {
        "preprocessed": [Path(f"/out/{KEY}_{m}.nii.gz") for m in ("t1", "t1c", "t2", "t2fl")],
        "masks": [Path(f"/out/{KEY}_t1c_segmask.nii.gz")],
        "quality_reports": [], "volume_report": None, "lobar_report": None,
        "lesion_stats_report": None, "mcdonald_report": None,
        "lesion_labels_mask": None,
    }


def _entity(file_names, name=KEY, files_key=True, status=2):
    e = {"dsEntityId": "ent-214", "dsEntityName": name, "dsEntityStatus": status,
         "dsEntityInfo": {"study_hash": HASH}}
    if files_key:
        e["files"] = [{"fileId": f"id-{n}", "fileName": n} for n in file_names]
    return e


INCOMPLETE = [f"{KEY}_t1.nii.gz", f"{KEY}_t2.nii.gz", f"{KEY}_t2fl.nii.gz", "empty.json"]
COMPLETE = [f"{KEY}_{m}.nii.gz" for m in ("t1", "t1c", "t2", "t2fl")] + [f"{KEY}_t1c_segmask.nii.gz"]


@pytest.fixture
def harness(monkeypatch):
    def make(entity, replace_result=None, superseding=frozenset()):
        up = _uploader(superseding)
        monkeypatch.setattr(up, "_allocation_db", Path("/tmp/none.db"), raising=False)
        monkeypatch.setattr(up, "_discover_sessions", lambda: {KEY: _session()})
        monkeypatch.setattr(up, "_compute_study_hash", lambda data: HASH)
        h = {"posted": [], "replaced": [], "registered": [], "status": [], "info": []}

        async def listing(**kw):
            return [entity]

        async def post(dataset_id, session_key, session_data):
            h["posted"].append(session_key)
            return {"session": session_key, "success": True}

        async def replace(token, user_id, user_type_id, dataset_id, entity_id, files):
            h["replaced"].append((entity_id, sorted(p.name for p in files)))
            return replace_result or {"patched": 3, "added": 2, "deleted": 1,
                                      "failed": [], "delete_job": "succeeded"}

        async def status(**kw):
            h["status"].append(kw["status"])

        async def reconcile(*a, **k):
            return "ent-214"

        monkeypatch.setattr(kappa_client, "get_dataset_entities", listing)
        monkeypatch.setattr(up, "_upload_session", post)
        monkeypatch.setattr(up, "_reconcile_session", reconcile)
        monkeypatch.setattr(kappa_entity_files, "replace_entity_contents", replace)

        async def update_info(token, user_id, user_type_id, dataset_id, entity_id, info, remark):
            h["info"].append((entity_id, info))
            return True

        monkeypatch.setattr(kappa_entity_files, "update_entity_info", update_info)
        monkeypatch.setattr(kappa_uploader, "update_entity_status", status)
        monkeypatch.setattr(up, "_do_local_registration",
                            lambda name, data, eid, ds: h["registered"].append((name, eid, ds)))
        return up, h
    return make


@pytest.mark.asyncio
async def test_complete_entity_stays_a_duplicate(harness):
    up, h = harness(_entity(COMPLETE))

    report = await up.upload_results()

    s = report["sessions"][0]
    assert s["skipped_upload"] and s["success"]
    assert h["replaced"] == [] and h["posted"] == []


@pytest.mark.asyncio
async def test_incomplete_entity_is_completed_with_the_whole_session(harness):
    up, h = harness(_entity(INCOMPLETE))

    report = await up.upload_results()

    s = report["sessions"][0]
    assert s["success"] is True and s["completed"] is True
    assert s["entity_id"] == "ent-214"
    # every file of this run, not only the missing ones (KI-062: two runs'
    # registrations must not mix in one entity)
    assert h["replaced"] == [("ent-214", sorted(p.name for p in
                              _session()["preprocessed"] + _session()["masks"]))]
    assert h["posted"] == []
    assert h["status"] == [3]
    assert h["registered"] == [(KEY, "ent-214", 249)]
    assert report["uploaded"] == 1
    (eid, info), = h["info"]
    assert eid == "ent-214" and info["study_hash"] == HASH
    assert f"{KEY}_t1c.nii.gz" in info["data_files"]


@pytest.mark.asyncio
async def test_status_is_not_lowered(harness):
    up, h = harness(_entity(INCOMPLETE, status=5))

    await up.upload_results()

    assert h["status"] == []


@pytest.mark.asyncio
async def test_incomplete_entity_with_expert_mask_needs_a_human(harness):
    up, h = harness(_entity(INCOMPLETE + [f"{KEY}_t1c_segmask_v2.nii.gz"]))

    report = await up.upload_results()

    s = report["sessions"][0]
    assert s["success"] is False and s["error"] == "supersedes"
    assert h["replaced"] == [] and h["posted"] == []


@pytest.mark.asyncio
async def test_failed_completion_is_retried_not_reposted(harness):
    up, h = harness(_entity(INCOMPLETE),
                    replace_result={"patched": 3, "added": 0, "deleted": 0,
                                    "failed": [f"{KEY}_t1c.nii.gz"], "delete_job": None})

    report = await up.upload_results()

    s = report["sessions"][0]
    assert s["success"] is False and s["error"] == "upload failed"
    assert h["posted"] == [] and h["registered"] == []
    from datetime import datetime, timezone
    from kappa_delivery import classify
    assert classify(report, None, {}, datetime.now(timezone.utc))["status"] == "pending"


@pytest.mark.asyncio
async def test_failed_stale_file_delete_is_a_failure(harness):
    up, h = harness(_entity(INCOMPLETE),
                    replace_result={"patched": 3, "added": 2, "deleted": 1,
                                    "failed": [], "delete_job": "failed"})

    report = await up.upload_results()

    assert report["sessions"][0]["error"] == "upload failed"
    assert h["registered"] == []


@pytest.mark.asyncio
async def test_no_file_list_means_unknown_and_stays_a_duplicate(harness):
    """Never rewrite an entity on missing information."""
    up, h = harness(_entity([], files_key=False))

    report = await up.upload_results()

    assert report["sessions"][0]["skipped_upload"]
    assert h["replaced"] == []


@pytest.mark.asyncio
async def test_superseding_session_keeps_its_own_path(harness):
    up, h = harness(_entity(INCOMPLETE), superseding={KEY})

    report = await up.upload_results()

    assert report["sessions"][0]["error"] == "supersedes"
    assert h["replaced"] == []


@pytest.mark.asyncio
async def test_completeness_is_judged_by_the_entity_name(harness):
    """An entity can sit in Kappa under another sub-NNN (reconciled names);
    its files carry that name."""
    other = "sub-215_ses-001"
    files = [n.replace(KEY, other) for n in COMPLETE]
    up, h = harness(_entity(files, name=other))

    report = await up.upload_results()

    assert report["sessions"][0]["skipped_upload"]
    assert h["replaced"] == []


@pytest.mark.asyncio
async def test_incomplete_entity_under_another_name_needs_a_human(harness):
    """Files carry the local name: written into an entity named otherwise it
    would stay 'incomplete' and be rewritten on every attempt."""
    other = "sub-215_ses-001"
    up, h = harness(_entity([n.replace(KEY, other) for n in INCOMPLETE], name=other))

    report = await up.upload_results()

    assert report["sessions"][0]["error"] == "supersedes"
    assert h["replaced"] == []


@pytest.mark.asyncio
async def test_info_update_failure_does_not_fail_the_completion(harness, monkeypatch):
    up, h = harness(_entity(INCOMPLETE))

    async def refuse(*a, **k):
        return False

    monkeypatch.setattr(kappa_entity_files, "update_entity_info", refuse)

    report = await up.upload_results()

    assert report["sessions"][0]["completed"] is True
