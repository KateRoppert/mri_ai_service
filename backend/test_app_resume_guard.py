"""Resuming a stopped run must not silently mix two configurations."""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml

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
         patch("app.load_config_snapshot", return_value={"steps": []}), \
         patch("app.diff_configs", return_value=[]), \
         patch("app.create_pipeline_run", return_value=_new_run()):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 200


def test_default_resume_rejects_missing_preprocessing_snapshot():
    """Default Resume must not live-start when the stop-time snapshot is gone."""
    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={}), \
         patch("app.create_pipeline_run", return_value=_new_run()) as mock_create, \
         patch("app.run_pipeline_background") as mock_bg, \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "snapshot_unavailable"
    assert "сохранённ" in detail["message"].lower() or "настройк" in detail["message"].lower()
    mock_create.assert_not_called()
    mock_bg.assert_not_called()


def test_changed_settings_block_resume_and_are_reported():
    differences = [{"setting": "Удаление черепа", "was": "bet", "now": "hdbet"}]

    with patch("app.get_pipeline_run", return_value=_stopped_run()), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={"steps": []}), \
         patch("app.diff_configs", return_value=differences):
        response = client.post("/api/pipeline-runs/run-1/requeue")

    assert response.status_code == 409
    assert response.json()["detail"]["differences"] == differences


def test_use_snapshot_resumes_despite_differences(tmp_path):
    differences = [{"setting": "Удаление черепа", "was": "bet", "now": "hdbet"}]
    retained = tmp_path / "config_run-1.yaml"
    retained.write_text("general: {}\nstages: {}\n", encoding="utf-8")
    preproc = tmp_path / "preprocessing_run-1.yaml"
    preproc.write_text("steps: []\n", encoding="utf-8")

    original = _stopped_run()
    original.config_path = str(retained)

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.load_config_snapshot", return_value={"steps": []}), \
         patch("app.diff_configs", return_value=differences), \
         patch("app.preprocessing_snapshot_path", return_value=preproc), \
         patch("app.create_pipeline_run", return_value=_new_run()), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()):
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


def test_use_snapshot_rejects_missing_runtime_config():
    """Choosing saved settings must not silently fall back to the live template."""
    original = _stopped_run(run_id="run-1")
    original.config_path = "/nonexistent/runtime_configs/config_run-1.yaml"

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run") as mock_create, \
         patch("app.run_pipeline_background") as mock_bg:
        response = client.post(
            "/api/pipeline-runs/run-1/requeue", json={"use_snapshot": True}
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "snapshot_unavailable"
    assert "сохранённ" in detail["message"].lower() or "настройк" in detail["message"].lower()
    mock_create.assert_not_called()
    mock_bg.assert_not_called()


def test_use_snapshot_rejects_missing_preprocessing_snapshot(tmp_path):
    """Pipeline snapshot without preprocessing copy must not mix two configs."""
    retained = tmp_path / "config_run-1.yaml"
    retained.write_text("general: {}\nstages: {}\n", encoding="utf-8")
    missing_preproc = tmp_path / "preprocessing_run-1.yaml"  # not created

    original = _stopped_run(run_id="run-1")
    original.config_path = str(retained)

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.preprocessing_snapshot_path", return_value=missing_preproc), \
         patch("app.create_pipeline_run") as mock_create, \
         patch("app.run_pipeline_background") as mock_bg:
        response = client.post(
            "/api/pipeline-runs/run-1/requeue", json={"use_snapshot": True}
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "snapshot_unavailable"
    mock_create.assert_not_called()
    mock_bg.assert_not_called()


def test_use_snapshot_passes_retained_runtime_config_to_pipeline(tmp_path):
    """«На прежних настройках» must hand the saved runtime YAML downstream."""
    retained = tmp_path / "config_run-1.yaml"
    retained.write_text("general: {}\nstages: {}\n", encoding="utf-8")
    preproc = tmp_path / "preprocessing_run-1.yaml"
    preproc.write_text("steps: []\n", encoding="utf-8")

    original = _stopped_run(run_id="run-1")
    original.config_path = str(retained)

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", return_value=_new_run()), \
         patch("app.preprocessing_snapshot_path", return_value=preproc), \
         patch("app.run_pipeline_background") as mock_bg, \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()):
        response = client.post(
            "/api/pipeline-runs/run-1/requeue", json={"use_snapshot": True}
        )

    assert response.status_code == 200
    kwargs = mock_bg.call_args.kwargs
    assert kwargs["snapshot_runtime_config"] == retained
    assert kwargs["preprocessing_snapshot"] == preproc


def test_create_runtime_config_from_snapshot_rewrites_preprocessing_paths(tmp_path):
    from pipeline_manager import PipelineManager

    retained = tmp_path / "config_run-1.yaml"
    retained.write_text(
        yaml.dump({
            "general": {
                "root_input_dir": "/old/in",
                "root_output_dir": "/old/out",
                "lesion_type": "multiple_sclerosis",
            },
            "stages": {
                "stage_05_preprocessing": {
                    "script": "scripts/05_preprocessing.py",
                    "args": {"config": "configs/preprocessing_config.yaml"},
                },
                "stage_07_inverse_transform": {
                    "script": "scripts/07_inverse_transform.py",
                    "args": {
                        "preprocessing-config": "configs/preprocessing_config.yaml",
                    },
                },
                "stage_06_segmentation": {
                    "script": "scripts/06_segmentation.py",
                    "args": {"config": "configs/segmentation_config.yaml"},
                },
            },
        }),
        encoding="utf-8",
    )
    preproc_snap = tmp_path / "preprocessing_run-1.yaml"
    preproc_snap.write_text("steps: []\n", encoding="utf-8")

    pm = PipelineManager()
    pm.pipeline_root = tmp_path
    out = pm.create_runtime_config_from_snapshot(
        run_id="new-run",
        input_path="/in",
        output_path="/out",
        snapshot_config_path=retained,
        lesion_type="glioblastoma",
        preprocessing_snapshot=preproc_snap,
    )

    with out.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["general"]["root_input_dir"] == "/in"
    assert cfg["general"]["root_output_dir"] == "/out"
    assert cfg["general"]["lesion_type"] == "glioblastoma"
    snap_abs = str(preproc_snap.resolve())
    assert cfg["stages"]["stage_05_preprocessing"]["args"]["config"] == snap_abs
    assert (
        cfg["stages"]["stage_07_inverse_transform"]["args"]["preprocessing-config"]
        == snap_abs
    )
    # unrelated config keys must stay untouched
    assert (
        cfg["stages"]["stage_06_segmentation"]["args"]["config"]
        == "configs/segmentation_config.yaml"
    )
