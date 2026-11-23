"""start_pipeline must not rebuild from the live template when a snapshot is required."""
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _manager(tmp_path):
    pm = PipelineManager()
    pm.pipeline_root = tmp_path
    pm.config_template = tmp_path / "pipeline_config.yaml"
    pm.orchestrator_script = tmp_path / "orchestrator.py"
    return pm


def _nonempty_input(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "patient-1").mkdir()
    return inp


def test_missing_snapshot_runtime_config_does_not_use_live_template(tmp_path):
    pm = _manager(tmp_path)
    inp = _nonempty_input(tmp_path)
    missing = tmp_path / "runtime_configs" / "config_old.yaml"

    with patch.object(pm, "create_runtime_config") as create_live, \
         patch("pipeline_manager.subprocess.Popen") as popen:
        result = pm.start_pipeline(
            "new-run",
            str(inp),
            str(tmp_path / "out"),
            snapshot_runtime_config=missing,
        )

    assert result is None
    create_live.assert_not_called()
    popen.assert_not_called()


def test_start_pipeline_records_who_started_the_run(tmp_path):
    pm = _manager(tmp_path)
    pm.config_template.write_text(
        "general:\n  root_input_dir: /in\n  root_output_dir: /out\nstages: {}\n",
        encoding="utf-8",
    )
    pm.orchestrator_script.write_text("", encoding="utf-8")
    inp = _nonempty_input(tmp_path)

    with patch("pipeline_manager.subprocess.Popen") as popen:
        popen.return_value.pid = 7
        result = pm.start_pipeline(
            "run-1", str(inp), str(tmp_path / "out"),
            kappa_user_name="e.roppert",
        )

    assert result is not None
    written = yaml.safe_load(
        (tmp_path / "runtime_configs" / "config_run-1.yaml").read_text(encoding="utf-8")
    )
    assert written["general"]["kappa_user_name"] == "e.roppert"


def test_resume_records_who_resumed_it_and_drops_the_previous_login(tmp_path):
    pm = _manager(tmp_path)
    pm.orchestrator_script.write_text("", encoding="utf-8")
    inp = _nonempty_input(tmp_path)
    retained = tmp_path / "config_old.yaml"
    retained.write_text(
        yaml.dump({
            "general": {
                "kappa_user_name": "test_med1",
                "lesion_type": "glioblastoma",
            },
            "stages": {},
        }),
        encoding="utf-8",
    )

    with patch("pipeline_manager.subprocess.Popen") as popen:
        popen.return_value.pid = 7
        result = pm.start_pipeline(
            "run-2", str(inp), str(tmp_path / "out"),
            snapshot_runtime_config=retained,
            kappa_user_name=None,
        )

    assert result is not None
    written = yaml.safe_load(
        (tmp_path / "runtime_configs" / "config_run-2.yaml").read_text(encoding="utf-8")
    )
    assert written["general"]["kappa_user_name"] is None


def test_missing_preprocessing_snapshot_does_not_start(tmp_path):
    pm = _manager(tmp_path)
    inp = _nonempty_input(tmp_path)
    retained = tmp_path / "config_old.yaml"
    retained.write_text(
        yaml.dump({
            "general": {"lesion_type": "glioblastoma"},
            "stages": {
                "stage_05_preprocessing": {
                    "script": "scripts/05_preprocessing.py",
                    "args": {"config": str(tmp_path / "preprocessing_old.yaml")},
                },
            },
        }),
        encoding="utf-8",
    )
    missing_pre = tmp_path / "preprocessing_old.yaml"

    with patch.object(pm, "create_runtime_config") as create_live, \
         patch.object(pm, "create_runtime_config_from_snapshot") as from_snap, \
         patch("pipeline_manager.subprocess.Popen") as popen:
        result = pm.start_pipeline(
            "new-run",
            str(inp),
            str(tmp_path / "out"),
            snapshot_runtime_config=retained,
            preprocessing_snapshot=missing_pre,
        )

    assert result is None
    create_live.assert_not_called()
    from_snap.assert_not_called()
    popen.assert_not_called()
