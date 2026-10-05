"""A reprocessed session must not pass as a duplicate.

study_hash covers PatientID:StudyInstanceUID — the DICOM study, not the
images — so a rerun on a corrected modality set hashes the same and the
uploader reports it delivered. The correction then never reaches Kappa, and
nothing says so.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from kappa_uploader import KappaUploader


def _async(value):
    async def _inner(*a, **k):
        return value
    return _inner


def _uploader(tmp_path, superseding=frozenset()):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"preprocessing": {"steps": []}}),
                   encoding="utf-8")
    return KappaUploader(
        run_id="r1", output_path=str(tmp_path), token="tok",
        user_id=26, user_type_id=1, lesion_type="glioblastoma",
        preprocessing_config_path=str(cfg),
        dataset_id=351, superseding_sessions=superseding,
    )


def _stub_discovery(monkeypatch, up):
    monkeypatch.setattr(up, "_resolve_dataset_id", _async(351))
    monkeypatch.setattr(up, "_bind_pending_scope", lambda ds: None)
    monkeypatch.setattr(up, "_discover_sessions",
                        lambda: {"sub-002_ses-001": {"preprocessed": [],
                                                     "masks": []}})
    monkeypatch.setattr(up, "_compute_study_hash", lambda data: "h1")
    monkeypatch.setattr(up, "_get_existing_study_hashes", _async({"h1"}))
    monkeypatch.setattr(up, "_get_existing_entity_names",
                        _async({"sub-002_ses-001"}))


@pytest.mark.asyncio
async def test_a_superseding_session_is_reported_not_skipped(tmp_path, monkeypatch):
    up = _uploader(tmp_path, superseding={"sub-002_ses-001"})
    _stub_discovery(monkeypatch, up)

    result = await up.upload_results()

    entry = result["sessions"][0]
    assert entry["error"] == "supersedes"
    assert entry["success"] is False


@pytest.mark.asyncio
async def test_a_superseding_session_does_not_count_as_delivered(tmp_path, monkeypatch):
    """classify() counts 'duplicate' as delivered. If supersedes were
    rounded into that, the run would report done and the operator would
    never be offered the replacement."""
    from kappa_delivery import classify
    from datetime import datetime, timezone

    up = _uploader(tmp_path, superseding={"sub-002_ses-001"})
    _stub_discovery(monkeypatch, up)

    result = await up.upload_results()
    out = classify(result, None, {}, datetime(2026, 10, 2, tzinfo=timezone.utc))

    assert out["detail"]["delivered"] == 0


@pytest.mark.asyncio
async def test_an_ordinary_duplicate_still_passes(tmp_path, monkeypatch):
    """Nothing changes for a session nobody reprocessed."""
    up = _uploader(tmp_path)
    _stub_discovery(monkeypatch, up)
    monkeypatch.setattr(up, "_reconcile_session", _async("e1"))

    result = await up.upload_results()

    assert result["sessions"][0]["error"] != "supersedes"
    assert result["sessions"][0]["skipped_upload"] is True


# --- The wiring, not just the function -------------------------------------
#
# Earlier in this feature's history an identical shape of bug shipped: a
# unit test passed the value in directly, proving the function worked while
# the caller never supplied it. Both construction sites get a contract test.

def test_the_worker_passes_the_runs_list_to_the_uploader(tmp_path, monkeypatch):
    import kappa_delivery_worker as worker
    from types import SimpleNamespace
    import json

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"preprocessing": {"steps": []}}),
                   encoding="utf-8")
    monkeypatch.setattr(worker, "PREPROCESSING_CONFIG", cfg)

    run = SimpleNamespace(
        run_id="r1", output_path=str(tmp_path), lesion_type="glioblastoma",
        kappa_dataset_id=351,
        reprocessed_sessions=json.dumps(["sub-002_ses-001"]),
    )
    uploader = worker.build_uploader(
        run, {"kappa_token": "t", "user_id": 26, "user_type_id": 1})

    assert uploader is not None
    assert uploader.superseding_sessions == {"sub-002_ses-001"}


def test_the_monitor_passes_the_runs_list_to_the_uploader(tmp_path, monkeypatch):
    import pipeline_monitor as pm
    from types import SimpleNamespace
    import json

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"preprocessing": {"steps": []}}),
                   encoding="utf-8")
    monkeypatch.setattr(pm, "PREPROCESSING_CONFIG", cfg)

    run = SimpleNamespace(
        kappa_dataset_id=351,
        reprocessed_sessions=json.dumps(["sub-002_ses-001"]),
    )
    monkeypatch.setattr("kappa_auth.get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26,
                                     "user_type_id": 1})
    monkeypatch.setattr("database.get_pipeline_run", lambda db, rid: run)

    uploader = pm.pipeline_monitor._create_kappa_uploader(
        "r1", str(tmp_path), "sid", "glioblastoma")

    assert uploader is not None, "аплоадер не создался — проводка не проверена"
    assert uploader.superseding_sessions == {"sub-002_ses-001"}
