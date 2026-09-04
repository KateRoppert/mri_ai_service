from pathlib import Path

from preprocessing_steps.skull_stripping.synthstrip import SynthStripStripper
from preprocessing_steps.skull_stripping import dispatcher


def test_registered_in_strippers():
    assert "synthstrip" in dispatcher.STRIPPERS
    assert dispatcher.STRIPPERS["synthstrip"]().name == "synthstrip"


def test_uses_gpu_is_false():
    assert SynthStripStripper.uses_gpu is False


def test_build_command_includes_border_and_mask():
    s = SynthStripStripper()
    cmd = s.build_command(
        Path("/in/t1.nii.gz"),
        Path("/out/t1.nii.gz"),
        Path("/out/mask.nii.gz"),
        {"border": 2},
    )
    assert cmd[:1] == ["mri_synthstrip"]
    assert "-i" in cmd and "/in/t1.nii.gz" in cmd
    assert "-o" in cmd and "/out/t1.nii.gz" in cmd
    assert "-m" in cmd and "/out/mask.nii.gz" in cmd
    assert "-b" in cmd and "2" in cmd


def test_build_command_omits_border_when_unset():
    s = SynthStripStripper()
    cmd = s.build_command(
        Path("/in/t1.nii.gz"),
        Path("/out/t1.nii.gz"),
        Path("/out/mask.nii.gz"),
        {},
    )
    assert "-b" not in cmd


def test_is_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(
        "preprocessing_steps.skull_stripping.synthstrip.shutil.which",
        lambda _: None,
    )
    assert SynthStripStripper().is_available() is False


def test_strip_returns_error_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "preprocessing_steps.skull_stripping.synthstrip.shutil.which",
        lambda _: None,
    )
    result = SynthStripStripper().strip(
        input_path=tmp_path / "in.nii.gz",
        output_path=tmp_path / "out.nii.gz",
        mask_path=tmp_path / "mask.nii.gz",
        params={},
    )
    assert result["success"] is False
    assert "error" in result
