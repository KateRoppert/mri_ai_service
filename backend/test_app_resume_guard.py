"""Resuming a stopped run must not silently mix two configurations."""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _stopped_run(run_id="run-1"):
    return SimpleNamespace(
        run_id=run_id,
        status="stopped",
        input_path="/in",
        output_path="/out",
        lesion_type="glioblastoma",
        config_path="/configs/config_run-1.yaml",
        current_stage=5,
    )


def _new_run():
    return SimpleNamespace(
        run_id="new",
        input_path="/in",
        output_path="/out",
        lesion_type="glioblastoma",
        created_at=datetime.now(timezone.utc),
    )


def test_stopped_run_can_be_resumed():
    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={}), \
         patch("app.diff_configs", return_value=[]), \
         patch("app.create_pipeline_run", return_value=_new_run()):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 200


def test_changed_settings_block_resume_and_are_reported():
    differences = [{"setting": "Удаление черепа", "was": "bet", "now": "hdbet"}]

    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={"steps": []}), \
         patch("app.diff_configs", return_value=differences):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 409
    assert response.json()["detail"]["differences"] == differences


def test_use_snapshot_resumes_despite_differences():
    differences = [{"setting": "Удаление черепа", "was": "bet", "now": "hdbet"}]

    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={"steps": []}), \
         patch("app.diff_configs", return_value=differences), \
         patch("app.create_pipeline_run", return_value=_new_run()):
        response = client.post(
            "/api/pipeline-runs/run-1/requeue", json={"use_snapshot": True}
        )

    assert response.status_code == 200


def test_running_run_still_rejected():
    running = _stopped_run()
    running.status = "running"

    with patch("app.get_pipeline_run", return_value=running):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 409
