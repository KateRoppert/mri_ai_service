import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


def _fake_original_run(run_id="orig-run", input_path="/in", output_path="/out", lesion_type="glioblastoma", status="completed"):
    return SimpleNamespace(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        lesion_type=lesion_type,
        status=status,
    )


def _fake_new_run(run_id="new-run", input_path="/in", output_path="/out", lesion_type="glioblastoma"):
    return SimpleNamespace(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        lesion_type=lesion_type,
        created_at=datetime.now(timezone.utc),
    )


def test_404_when_run_not_found():
    with patch("app.get_pipeline_run", return_value=None):
        response = client.post("/api/pipeline-runs/nonexistent-run/requeue")
    assert response.status_code == 404


def test_409_when_original_run_is_still_running():
    original = _fake_original_run(status="running")

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.create_pipeline_run") as mock_create, \
         patch("app.run_pipeline_background") as mock_bg, \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()) as mock_monitor:
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 409
    assert "выполняется" in response.json()["detail"]

    # must not start a second orchestrator over the same output_path
    mock_create.assert_not_called()
    mock_bg.assert_not_called()
    mock_monitor.assert_not_called()


def test_409_when_original_run_is_pending():
    original = _fake_original_run(status="pending")

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.create_pipeline_run") as mock_create:
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 409
    mock_create.assert_not_called()


def test_409_when_a_different_run_is_active_on_the_same_output_path():
    # Run A completed, but run B (its own earlier requeue-child) is still
    # running on the SAME output_path. Reopening A's review and requeuing it
    # again must not start a third orchestrator process over that path.
    original = _fake_original_run(run_id="run-a", status="completed")
    other_active_run = _fake_new_run(run_id="run-b", output_path=original.output_path)

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=other_active_run) as mock_active, \
         patch("app.create_pipeline_run") as mock_create, \
         patch("app.run_pipeline_background") as mock_bg, \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()) as mock_monitor:
        response = client.post("/api/pipeline-runs/run-a/requeue")

    assert response.status_code == 409
    assert "пут" in response.json()["detail"]

    mock_active.assert_called_once_with(mock_active.call_args[0][0], original.output_path)

    # must not start a second orchestrator over the same output_path
    mock_create.assert_not_called()
    mock_bg.assert_not_called()
    mock_monitor.assert_not_called()


def test_creates_new_run_with_same_paths_and_does_not_run_pipeline_synchronously():
    original = _fake_original_run()
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.create_pipeline_run", return_value=new_run) as mock_create, \
         patch("app.run_pipeline_background") as mock_bg, \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()) as mock_monitor:
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "new-run"
    assert body["status"] == "pending"
    assert body["lesion_type"] == "glioblastoma"

    # create_pipeline_run must be called with the ORIGINAL run's paths, not new ones
    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs["input_path"] == original.input_path
    assert kwargs["output_path"] == original.output_path
    assert kwargs["lesion_type"] == original.lesion_type

    # the actual pipeline must not run inside the test — it's scheduled as a
    # background task, which TestClient executes after returning the response,
    # so run_pipeline_background must be a stub here (real one launches a subprocess)
    mock_bg.assert_called_once()
    assert mock_bg.call_args[0][0] == "new-run"
    assert mock_bg.call_args[0][1] == original.input_path
    assert mock_bg.call_args[0][2] == original.output_path

    mock_monitor.assert_called_once()
    monitor_args = mock_monitor.call_args[0]
    assert monitor_args[0] == "new-run"
    assert monitor_args[2] is None  # kappa_session_id — no new Kappa context on requeue


def test_requeue_passes_parent_run_id_to_create_pipeline_run():
    original = _fake_original_run(run_id="orig-run")
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.create_pipeline_run", return_value=new_run) as mock_create, \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    _, kwargs = mock_create.call_args
    assert kwargs["parent_run_id"] == "orig-run"
