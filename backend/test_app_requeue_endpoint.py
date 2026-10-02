import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


def _fake_original_run(run_id="orig-run", input_path="/in", output_path="/out", lesion_type="glioblastoma", status="completed", kappa_dataset_id=None):
    return SimpleNamespace(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        kappa_dataset_id=kappa_dataset_id,
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


def test_requeue_passes_kappa_session_to_monitoring():
    """Without a Kappa session the monitor never builds an uploader
    (pipeline_monitor: `if kappa_session_id and lesion_type`), so a requeued
    run completes and silently never reaches Kappa. Real case: KA03 was
    completed manually after being incomplete, processed fine, and never
    uploaded.
    """
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=new_run), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()) as mock_monitor:
        response = client.post("/api/pipeline-runs/orig-run/requeue",
                               json={"kappa_session_id": "session-42"})

    assert response.status_code == 200
    args = mock_monitor.call_args[0]
    assert args[2] == "session-42", (
        f"kappa_session_id must reach start_monitoring, got {args[2]!r}")


def test_requeue_without_a_session_still_works():
    """CLI-ish/legacy callers that send no session must keep working — the
    run just has nothing to upload with, as before."""
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=new_run), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()) as mock_monitor:
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    assert mock_monitor.call_args[0][2] is None


def test_requeue_rebuilds_sessions_whose_set_changed():
    """Without this the correction changes nothing: skip_existing sees the
    old outputs and walks straight past the patient."""
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()
    purged = []

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=new_run), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               side_effect=lambda out: purged.append(out) or {"sub-001/ses-001": []}):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    assert purged == ["/out"], "помеченные сессии не очищены перед запуском"


def test_a_failed_purge_does_not_block_the_run():
    """Cleanup is housekeeping. Refusing to start the run because a stale
    folder could not be removed would be a worse outcome than the mess."""
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=new_run), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               side_effect=OSError("disk is read-only")):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200


def test_requeue_records_what_it_purged_on_the_new_run():
    """needs_reprocess is cleared by the purge itself, so by upload time
    nothing would remember that these sessions supersede what Kappa holds."""
    import json as _json

    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()
    captured = {}

    def _create(db, **kwargs):
        captured.update(kwargs)
        return new_run

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", side_effect=_create), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               return_value={"sub-002/ses-001": [], "sub-002/ses-002": []}):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    assert sorted(_json.loads(captured["reprocessed_sessions"])) == [
        "sub-002_ses-001", "sub-002_ses-002",
    ]


def test_requeue_without_a_purge_records_nothing():
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()
    captured = {}

    def _create(db, **kwargs):
        captured.update(kwargs)
        return new_run

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", side_effect=_create), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               return_value={}):
        client.post("/api/pipeline-runs/orig-run/requeue")

    assert captured.get("reprocessed_sessions") is None
