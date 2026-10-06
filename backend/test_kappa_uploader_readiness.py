"""
Only complete sessions go to Kappa, and the count is against what entered
processing, not against what happened to be on disk.

Run 30_09_1752 (MS, 6 sessions): stage 05 was OOM-killed mid-ses-005 (t1, t2
written, no t2fl) and never started ses-006. The uploader built its list from
whatever was in preprocessed/, uploaded ses-005 with two images and no mask,
never saw ses-006, and reported "5/5" instead of 4 of 6.
"""
from pathlib import Path

import pytest

import kappa_uploader


def _uploader(output_path, lesion_type="multiple_sclerosis", **kwargs):
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="r1",
        output_path=str(output_path),
        token="tok",
        user_id=26,
        user_type_id=3,
        lesion_type=lesion_type,
        preprocessing_config_path=cfg,
        **kwargs,
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _session(out: Path, sub: str, ses: str, modalities, mask=True, bids=True):
    """Lay out one session the way stages 01/05/06 do."""
    key = f"{sub}_{ses}"
    if bids:
        (out / "bids_organized" / sub / ses / "anat").mkdir(parents=True, exist_ok=True)
    for m in modalities:
        _touch(out / "preprocessed" / sub / ses / "anat" / f"{key}_{m}.nii.gz")
    if mask:
        seg = out / "segmentation" / sub / ses / "anat" / "multiple_sclerosis"
        _touch(seg / f"{key}_t1_segmask.nii.gz")
        _touch(seg / f"{key}_t1_segmask_native_t1.nii.gz")


def _the_30_09_1752_run(out: Path) -> None:
    for n in range(1, 5):
        _session(out, "sub-003", f"ses-00{n}", ["t1", "t2", "t2fl"])
    _session(out, "sub-003", "ses-005", ["t1", "t2"], mask=False)
    _session(out, "sub-003", "ses-006", [], mask=False)
    (out / "segmentation").mkdir(exist_ok=True)


# --- universe -----------------------------------------------------------

def test_universe_is_every_session_stage_01_produced(tmp_path):
    _the_30_09_1752_run(tmp_path)
    up = _uploader(tmp_path)

    universe = up._session_universe(up._discover_sessions())

    assert universe == [f"sub-003_ses-00{n}" for n in range(1, 7)]


def test_universe_ignores_the_incomplete_queue(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"])
    (tmp_path / "bids_organized" / "_incomplete" / "sub-002" / "ses-001" / "anat").mkdir(parents=True)
    up = _uploader(tmp_path)

    assert up._session_universe(up._discover_sessions()) == ["sub-001_ses-001"]


def test_universe_falls_back_to_preprocessed_without_bids(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"], bids=False)
    up = _uploader(tmp_path)

    assert up._session_universe(up._discover_sessions()) == ["sub-001_ses-001"]


# --- readiness ----------------------------------------------------------

def _ready(tmp_path, key, lesion_type="multiple_sclerosis"):
    up = _uploader(tmp_path, lesion_type=lesion_type)
    return up._readiness(key, up._discover_sessions().get(key))


def test_full_session_is_ready(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"])
    assert _ready(tmp_path, "sub-001_ses-001") == (True, None)


def test_missing_modality_and_mask_is_not_ready(tmp_path):
    _session(tmp_path, "sub-003", "ses-005", ["t1", "t2"], mask=False)
    (tmp_path / "segmentation").mkdir(exist_ok=True)
    ok, reason = _ready(tmp_path, "sub-003_ses-005")
    assert ok is False
    assert reason == "нет t2fl и маски сегментации"


def test_modalities_without_mask_is_not_ready(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"], mask=False)
    (tmp_path / "segmentation").mkdir(exist_ok=True)
    assert _ready(tmp_path, "sub-001_ses-001") == (False, "нет маски сегментации")


def test_session_with_nothing_after_stage_01_is_not_ready(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"])
    _session(tmp_path, "sub-001", "ses-002", [], mask=False)
    assert _ready(tmp_path, "sub-001_ses-002") == (False, "нет данных после предобработки")


def test_native_mask_alone_does_not_count(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"], mask=False)
    _touch(tmp_path / "segmentation" / "sub-001" / "ses-001" / "anat"
           / "sub-001_ses-001_t1_segmask_native_t1.nii.gz")
    ok, _ = _ready(tmp_path, "sub-001_ses-001")
    assert ok is False


def test_t1c_is_required_for_gbm_only(tmp_path):
    _session(tmp_path, "sub-001", "ses-001", ["t1", "t2", "t2fl"])
    assert _ready(tmp_path, "sub-001_ses-001", "multiple_sclerosis")[0] is True
    assert _ready(tmp_path, "sub-001_ses-001", "glioblastoma") == (False, "нет t1c")


# --- upload_results -----------------------------------------------------

@pytest.mark.asyncio
async def test_upload_sends_only_ready_sessions_and_counts_all(monkeypatch, tmp_path):
    _the_30_09_1752_run(tmp_path)
    up = _uploader(tmp_path, dataset_id=353)
    monkeypatch.setattr(up, "_allocation_db", tmp_path / "alloc.db", raising=False)
    monkeypatch.setattr(up, "_compute_study_hash", lambda data: None)

    async def nothing(dataset_id):
        return set()

    uploaded = []

    async def fake_upload_session(dataset_id, session_key, session_data):
        uploaded.append(session_key)
        return {"session": session_key, "success": True}

    monkeypatch.setattr(up, "_get_existing_study_hashes", nothing)
    monkeypatch.setattr(up, "_get_existing_entity_names", nothing)
    monkeypatch.setattr(up, "_upload_session", fake_upload_session)
    monkeypatch.setattr(kappa_uploader.asyncio, "sleep", _no_sleep)

    report = await up.upload_results()

    assert uploaded == [f"sub-003_ses-00{n}" for n in range(1, 5)]
    assert report["total"] == 6
    assert report["uploaded"] == 4
    skipped = {s["session"]: s for s in report["sessions"] if s.get("error") == "not_processed"}
    assert set(skipped) == {"sub-003_ses-005", "sub-003_ses-006"}
    assert skipped["sub-003_ses-005"]["success"] is False
    assert skipped["sub-003_ses-005"]["message"] == "нет t2fl и маски сегментации"
    assert skipped["sub-003_ses-006"]["message"] == "нет данных после предобработки"


@pytest.mark.asyncio
async def test_nothing_ready_makes_no_kappa_calls(monkeypatch, tmp_path):
    _session(tmp_path, "sub-001", "ses-001", [], mask=False)
    up = _uploader(tmp_path, dataset_id=353)
    monkeypatch.setattr(up, "_allocation_db", tmp_path / "alloc.db", raising=False)

    async def must_not_be_called(dataset_id):
        raise AssertionError("asked Kappa although nothing is ready")

    monkeypatch.setattr(up, "_get_existing_study_hashes", must_not_be_called)
    monkeypatch.setattr(up, "_get_existing_entity_names", must_not_be_called)

    report = await up.upload_results()

    assert report["total"] == 1
    assert report["uploaded"] == 0
    assert report["sessions"][0]["error"] == "not_processed"


async def _no_sleep(_seconds):
    return None
