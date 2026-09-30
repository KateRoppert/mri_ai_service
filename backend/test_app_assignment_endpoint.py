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
