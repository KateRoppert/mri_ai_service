"""Applying a desired set: what lands on disk, and what must not."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import pipeline_manager
from pipeline_manager import PipelineManager
from session_assignment import AssignmentError


def _run_dir(tmp_path, series, excluded, status="incomplete"):
    bids = tmp_path / "bids_organized"
    (bids / "_incomplete").mkdir(parents=True, exist_ok=True)
    mapping = {"patients": {"sub-001": {"original_id": "P1", "sessions": {
        "ses-001": {
            "original_date": "20230101",
            "status": status,
            "series": series,
            "excluded_series": excluded,
        },
    }}}}
    (bids / "dataset_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8")
    return tmp_path


def _mapping(tmp_path):
    return json.loads(
        (tmp_path / "bids_organized" / "dataset_mapping.json").read_text(
            encoding="utf-8")
    )


def _excluded(path):
    return {"original_path": path, "series_description": "d",
            "slice_count": 3, "detected_modality": None,
            "reason": "unrecognized"}


def test_a_rejected_set_leaves_the_mapping_byte_identical(tmp_path):
    """The whole point of validating first: a bad request must not apply
    half of itself."""
    _run_dir(tmp_path, {"t1": {"original_path": "/raw/a"}}, [_excluded("/raw/b")])
    before = (tmp_path / "bids_organized" / "dataset_mapping.json").read_bytes()

    with pytest.raises(AssignmentError):
        PipelineManager().apply_assignment(
            str(tmp_path), "sub-001", "ses-001",
            {"t1": "/raw/does-not-belong"}, "glioblastoma",
        )

    after = (tmp_path / "bids_organized" / "dataset_mapping.json").read_bytes()
    assert after == before


def test_an_unchanged_set_copies_nothing_and_flags_nothing(tmp_path, monkeypatch):
    _run_dir(tmp_path, {"t1": {"original_path": "/raw/a"}}, [])
    copied = []
    monkeypatch.setattr(
        pipeline_manager, "copy_and_anonymize_series",
        lambda *a, **k: copied.append(a) or 0,
    )

    result = PipelineManager().apply_assignment(
        str(tmp_path), "sub-001", "ses-001", {"t1": "/raw/a"}, "glioblastoma",
    )

    assert copied == []
    assert result["needs_reprocess"] is False


def test_clearing_a_modality_makes_the_session_incomplete_again(tmp_path):
    """A doctor is allowed to say "this is not the t1c"."""
    _run_dir(
        tmp_path,
        {"t1": {"original_path": "/raw/a"}, "t1c": {"original_path": "/raw/b"},
         "t2": {"original_path": "/raw/c"}, "t2fl": {"original_path": "/raw/d"}},
        [], status="complete",
    )
    (tmp_path / "bids_organized" / "sub-001" / "ses-001" / "anat" / "t1c").mkdir(
        parents=True)

    result = PipelineManager().apply_assignment(
        str(tmp_path), "sub-001", "ses-001",
        {"t1": "/raw/a", "t2": "/raw/c", "t2fl": "/raw/d"}, "glioblastoma",
    )

    assert result["status"] == "incomplete"
    assert result["needs_reprocess"] is True
    # The cleared series is not lost — it goes back to the pool.
    paths = [e["original_path"] for e in result["excluded_series"]]
    assert "/raw/b" in paths
    # And the session moved back out of the main tree.
    assert (tmp_path / "bids_organized" / "_incomplete" / "sub-001" / "ses-001").exists()


def test_the_flag_is_set_whenever_the_set_changed(tmp_path, monkeypatch):
    _run_dir(tmp_path, {}, [_excluded("/raw/b")])
    monkeypatch.setattr(
        pipeline_manager, "find_dicom_files", lambda p: [Path("/raw/b/1.dcm")])
    monkeypatch.setattr(
        pipeline_manager, "copy_and_anonymize_series", lambda *a, **k: 1)
    monkeypatch.setattr(
        PipelineManager, "_build_metadata_extractor", lambda self: object())

    result = PipelineManager().apply_assignment(
        str(tmp_path), "sub-001", "ses-001", {"t1": "/raw/b"}, "glioblastoma",
    )

    assert result["needs_reprocess"] is True
    assert _mapping(tmp_path)["patients"]["sub-001"]["sessions"]["ses-001"][
        "needs_reprocess"] is True
