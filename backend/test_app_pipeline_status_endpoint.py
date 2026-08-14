import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _fake_run(run_id="run-1", status="completed", parent_run_id=None):
    return SimpleNamespace(
        run_id=run_id,
        input_path="/in",
        output_path="/out",
        status=status,
        current_stage=7,
        overall_progress=100.0,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error_message=None,
        lesion_type="glioblastoma",
        parent_run_id=parent_run_id,
    )


def test_status_response_includes_parent_run_id_when_set():
    run = _fake_run(parent_run_id="orig-run")

    with patch("app.get_pipeline_run", return_value=run), \
         patch("app.get_stage_executions", return_value=[]):
        response = client.get("/api/pipeline/status/run-1")

    assert response.status_code == 200
    assert response.json()["parent_run_id"] == "orig-run"


def test_status_response_parent_run_id_none_for_ordinary_run():
    run = _fake_run(parent_run_id=None)

    with patch("app.get_pipeline_run", return_value=run), \
         patch("app.get_stage_executions", return_value=[]):
        response = client.get("/api/pipeline/status/run-1")

    assert response.status_code == 200
    assert response.json()["parent_run_id"] is None
