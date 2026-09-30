"""The assignment endpoint: rejections are 400, not 500."""
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, "backend")


@pytest.mark.asyncio
async def test_a_bad_set_is_a_client_error(monkeypatch):
    """AssignmentError means the doctor sent something impossible, not that
    the service broke. A 500 would also hide the reason from the screen."""
    import app
    from models import AssignmentRequest
    from session_assignment import AssignmentError

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())

    def _reject(*a, **k):
        raise AssignmentError("Серия '/raw/x' не принадлежит этой сессии")

    monkeypatch.setattr(app.pipeline_manager, "apply_assignment", _reject)

    with pytest.raises(HTTPException) as caught:
        await app.save_assignment(
            "run-1", "sub-001", "ses-001",
            AssignmentRequest(assignments={"t1": "/raw/x"}), db=None,
        )
    assert caught.value.status_code == 400
    assert "не принадлежит" in caught.value.detail


@pytest.mark.asyncio
async def test_a_successful_save_returns_the_new_state(monkeypatch):
    import app
    from models import AssignmentRequest

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"
        kappa_dataset_id = None

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(
        app.pipeline_manager, "apply_assignment",
        lambda *a, **k: {
            "status": "complete",
            "selected": [{"modality": "t1", "series_description": "d",
                          "original_path": "/raw/a", "slice_count": 3}],
            "excluded_series": [],
            "needs_reprocess": True,
        },
    )

    result = await app.save_assignment(
        "run-1", "sub-001", "ses-001",
        AssignmentRequest(assignments={"t1": "/raw/a"}), db=None,
    )
    assert result.status == "complete"
    assert result.needs_reprocess is True
    assert result.selected[0].modality == "t1"


@pytest.mark.asyncio
async def test_warns_when_the_old_version_is_already_in_kappa(monkeypatch):
    """study_hash covers the DICOM study, not the images, so a reprocessed
    patient looks like a duplicate and Kappa keeps the old mask. Silence
    here would mean a correction that looks applied and is not."""
    import app
    from models import AssignmentRequest

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"
        kappa_dataset_id = 351

    seen = {}

    def _find(bids_id, dataset_ids=None):
        seen["dataset_ids"] = dataset_ids
        return [{"bids_id": bids_id, "kappa_entity_id": "e1"}]

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app, "find_by_bids_id", _find)
    monkeypatch.setattr(
        app.pipeline_manager, "apply_assignment",
        lambda *a, **k: {"status": "complete", "selected": [],
                         "excluded_series": [], "needs_reprocess": True},
    )

    result = await app.save_assignment(
        "run-1", "sub-001", "ses-001",
        AssignmentRequest(assignments={"t1": "/raw/a"}), db=None,
    )

    assert result.kappa_warning is not None
    assert "Kappa" in result.kappa_warning
    # sub-NNN is unique only WITHIN a dataset; an unscoped lookup would
    # report another account's patient as this one.
    assert seen["dataset_ids"] == {351}


@pytest.mark.asyncio
async def test_no_warning_when_nothing_changed(monkeypatch):
    import app
    from models import AssignmentRequest

    class _Run:
        output_path = "/tmp/run"
        lesion_type = "glioblastoma"
        kappa_dataset_id = 351

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(
        app.pipeline_manager, "apply_assignment",
        lambda *a, **k: {"status": "complete", "selected": [],
                         "excluded_series": [], "needs_reprocess": False},
    )

    result = await app.save_assignment(
        "run-1", "sub-001", "ses-001",
        AssignmentRequest(assignments={"t1": "/raw/a"}), db=None,
    )
    assert result.kappa_warning is None
