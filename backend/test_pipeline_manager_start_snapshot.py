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
