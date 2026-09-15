from pathlib import Path

from preprocessing_steps.skull_stripping.deepbet import DeepBetStripper
from preprocessing_steps.skull_stripping import dispatcher


def test_registered_in_strippers():
    assert "deepbet" in dispatcher.STRIPPERS
    assert dispatcher.STRIPPERS["deepbet"]().name == "deepbet"


def test_uses_gpu_is_false():
    """DeepBET finishes in ~2.5 s on CPU, so it stays off the HD-BET device
    pool: claiming a GPU slot would block a tool that actually needs one."""
    assert DeepBetStripper.uses_gpu is False


def test_build_command_has_input_output_and_mask():
    cmd = DeepBetStripper().build_command(
        Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"), Path("/out/mask.nii.gz"), {})

    assert cmd[0] == "deepbet-cli"
    assert "-i" in cmd and "/in/t1.nii.gz" in cmd
    assert "-o" in cmd and "/out/t1.nii.gz" in cmd
    assert "-m" in cmd and "/out/mask.nii.gz" in cmd


def test_cpu_is_requested_by_default():
    """`-g` is DeepBET's flag for *avoiding* the GPU (it is the short form of
    --no_gpu, not of --gpu). Getting this backwards would silently put the
    tool on the card, so the default is pinned by a test."""
    cmd = DeepBetStripper().build_command(
        Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"), None, {})

    assert "-g" in cmd


def test_gpu_can_be_requested_explicitly():
    cmd = DeepBetStripper().build_command(
        Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"), None, {"use_gpu": True})

    assert "-g" not in cmd


def test_threshold_and_dilate_are_passed_through():
    cmd = DeepBetStripper().build_command(
        Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"), None,
        {"threshold": 0.4, "n_dilate": 2})

    assert "-t" in cmd and "0.4" in cmd
    assert "-d" in cmd and "2" in cmd


def test_tuning_params_are_omitted_when_unset():
    """An absent key must mean 'tool default', not an explicit value —
    the benchmark tunes these and production must not inherit a stray one."""
    cmd = DeepBetStripper().build_command(
        Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"), None, {})

    assert "-t" not in cmd and "-d" not in cmd


def test_nested_tool_params_are_read():
    """Stage 05 hands the whole params block; other wrappers accept the
    nested `tool_params` form too, so this one must not be the odd one out."""
    cmd = DeepBetStripper().build_command(
        Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"), None,
        {"tool_params": {"threshold": 0.6}})

    assert "-t" in cmd and "0.6" in cmd


def test_is_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(
        "preprocessing_steps.skull_stripping.deepbet.shutil.which", lambda _: None)

    assert DeepBetStripper().is_available() is False


def test_strip_returns_error_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "preprocessing_steps.skull_stripping.deepbet.shutil.which", lambda _: None)

    result = DeepBetStripper().strip(
        input_path=tmp_path / "in.nii.gz",
        output_path=tmp_path / "out.nii.gz",
        mask_path=tmp_path / "mask.nii.gz",
        params={})

    assert result["success"] is False
    assert "deepbet" in result["error"].lower()
