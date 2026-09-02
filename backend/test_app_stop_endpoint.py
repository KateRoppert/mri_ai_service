"""Stop endpoint: kill the run, record why, never lie about the outcome."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _run(run_id="run-1", status="running", current_stage=5):
    return SimpleNamespace(
        run_id=run_id,
        status=status,
        current_stage=current_stage,
        output_path="/out",
        input_path="/in",
        lesion_type="glioblastoma",
    )


def test_404_when_run_unknown():
    with patch("app.get_pipeline_run", return_value=None):
        response = client.post("/api/pipeline-runs/nope/stop")

    assert response.status_code == 404


def test_409_when_run_already_finished():
    with patch("app.get_pipeline_run", return_value=_run(status="completed")):
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 409


def test_kills_process_group_and_records_state():
    proc = MagicMock()
    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=proc), \
         patch("app._kill_process_tree") as kill, \
         patch("app.update_pipeline_run") as update:
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    kill.assert_called_once_with(proc)
    kwargs = update.call_args.kwargs
    assert kwargs["status"] == "stopped"
    assert kwargs["stopped_at_stage"] == 5


def test_marks_stopped_even_when_process_already_gone():
    # After a backend restart the DB can still say "running" while no process
    # exists. The process is provably dead, so stop corrects the record
    # instead of failing.
    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=None), \
         patch("app._kill_process_tree") as kill, \
         patch("app.update_pipeline_run") as update:
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    kill.assert_not_called()
    assert update.call_args.kwargs["status"] == "stopped"
