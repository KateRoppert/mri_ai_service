"""HD-BET runs through hdbet_runner.py with the CLI's interpreter.

The `hd-bet` CLI forks ~13 torch processes for one file (5.1 GB on a 1 mm
volume); the runner uses one worker of each kind (~2 GB), same outputs.
"""
import inspect
import os
import stat
from pathlib import Path

import pytest

from preprocessing_steps.skull_stripping import hdbet
from preprocessing_steps.skull_stripping.hdbet import HdBetStripper, _hdbet_python


def _fake_cli(tmp_path, monkeypatch, shebang):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    cli = bindir / "hd-bet"
    cli.write_text(f"{shebang}\nimport sys\n")
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return cli


def test_interpreter_from_absolute_shebang(tmp_path, monkeypatch):
    py = tmp_path / "python3.12"
    py.write_text("")
    py.chmod(0o755)
    _fake_cli(tmp_path, monkeypatch, f"#!{py}")
    assert _hdbet_python() == str(py)


def test_interpreter_from_env_shebang(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch, "#!/usr/bin/env hd-bet")  # resolvable name
    assert _hdbet_python() == str(tmp_path / "bin" / "hd-bet")


def test_unavailable_without_cli(monkeypatch):
    monkeypatch.setattr(hdbet.shutil, "which", lambda name: None)
    assert _hdbet_python() is None
    assert not HdBetStripper().is_available()


def test_unavailable_when_shebang_interpreter_missing(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch, "#!/nonexistent/python")
    assert not HdBetStripper().is_available()


class _Done:
    returncode = 0
    stdout = stderr = ""


@pytest.mark.parametrize("device, params, tta_off", [
    ("cuda", {}, False),
    ("cpu", {}, True),                        # CPU: TTA off by default
    ("cuda", {"disable_tta": True}, True),
    ("cpu", {"disable_tta": False}, False),
])
def test_command_runs_runner_with_cli_interpreter(tmp_path, monkeypatch, device, params, tta_off):
    monkeypatch.setattr(hdbet, "_hdbet_python", lambda: "/opt/fsl/python3")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("-o") + 1])
        out.write_text("img")
        Path(str(out)[:-7] + "_bet.nii.gz").write_text("mask")
        return _Done()

    monkeypatch.setattr(hdbet.subprocess, "run", fake_run)
    out = tmp_path / "t1_brain.nii.gz"
    mask = tmp_path / "masks" / "t1_mask.nii.gz"

    res = HdBetStripper().strip(tmp_path / "t1.nii.gz", out, mask, {**params, "device": device})

    cmd = seen["cmd"]
    assert cmd[:2] == ["/opt/fsl/python3", str(Path(hdbet.__file__).with_name("hdbet_runner.py"))]
    assert cmd[cmd.index("-device") + 1] == device
    assert ("--disable_tta" in cmd) == tta_off
    assert res["success"] and res["mask_path"] == str(mask) and mask.exists()


def test_runner_requires_nii_gz_output():
    from preprocessing_steps.skull_stripping import hdbet_runner
    with pytest.raises(SystemExit):
        hdbet_runner.main(["-i", "in.nii.gz", "-o", "out.nii"])


def test_hdbet_internals_used_by_runner_still_exist():
    """Fails on an image whose HD-BET/nnU-Net changed what the runner calls."""
    pred = pytest.importorskip("HD_BET.hd_bet_prediction")
    pytest.importorskip("HD_BET.checkpoint_download").maybe_download_parameters
    assert {"use_tta", "device"} <= set(inspect.signature(pred.get_hdbet_predictor).parameters)
    assert list(inspect.signature(pred.apply_bet).parameters) == ["img", "bet", "out_fname"]
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    accepted = set(inspect.signature(nnUNetPredictor.predict_from_files).parameters)
    assert {"num_processes_preprocessing", "num_processes_segmentation_export",
            "save_probabilities", "overwrite", "folder_with_segs_from_prev_stage",
            "num_parts", "part_id"} <= accepted
