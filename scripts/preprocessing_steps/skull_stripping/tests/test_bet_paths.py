"""KI-049: BET on a path with a space, its error text, and what success means.

FSL `bet` is a shell wrapper that splits its arguments on spaces: under
"…/Рабочий стол/…" it printed "Error: input image … not valid" to stdout,
returned 1, and the pipeline logged an empty error and carried on with the
skull still on the image. Cyrillic alone is fine; the space breaks it.
"""
import os
import shutil
import stat

import nibabel as nib
import numpy as np
import pytest

from preprocessing_steps.skull_stripping import bet

needs_bet = pytest.mark.skipif(shutil.which("bet") is None, reason="FSL bet not installed")


def _head(path):
    """A bright ellipsoid 'brain' inside a dimmer shell 'skull', 2 mm voxels."""
    z, y, x = np.mgrid[-40:40, -48:48, -40:40].astype(float)
    r = (x / 32) ** 2 + (y / 40) ** 2 + (z / 32) ** 2
    vol = np.where(r < 0.7, 800.0, 0.0) + np.where((r >= 0.85) & (r < 1.0), 300.0, 0.0)
    vol += np.random.default_rng(0).normal(0, 10, vol.shape).clip(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), np.diag([2, 2, 2, 1])), path)
    return path


@pytest.fixture(autouse=True)
def _system_bet(monkeypatch):
    monkeypatch.setattr(bet, "FSL_DIR", None)
    monkeypatch.setattr(bet, "FSL_BIN_DIR", None)


@needs_bet
def test_output_root_with_a_space_works(tmp_path):
    root = tmp_path / "Рабочий стол" / "run 1"
    src = _head(root / "in" / "sub-001_ses-001_t1.nii.gz")
    out = root / "out" / "sub-001_ses-001_t1.nii.gz"
    mask = root / "masks" / "sub-001_ses-001_brain_mask.nii.gz"

    res = bet.run_bet(src, out, mask_path=mask)

    assert res["success"], res
    assert out.exists() and mask.exists()
    assert res["mask_path"] == str(mask)
    m = np.asarray(nib.load(mask).dataobj)
    assert 0 < m.sum() < m.size


@needs_bet
def test_no_temporary_files_are_left_next_to_the_output(tmp_path):
    src = _head(tmp_path / "a b" / "t1.nii.gz")
    out = tmp_path / "a b" / "t1_brain.nii.gz"

    bet.run_bet(src, out, mask_path=tmp_path / "a b" / "mask.nii.gz")

    assert sorted(p.name for p in (tmp_path / "a b").iterdir()) == \
        ["mask.nii.gz", "t1.nii.gz", "t1_brain.nii.gz"]


def test_a_failure_carries_bets_own_message(tmp_path, monkeypatch):
    """Real BET prints its error to stdout and leaves stderr empty."""
    _fake_bet(tmp_path, monkeypatch,
              'echo ""\necho "Error: input image $1 not valid"\nexit 1\n')
    src = tmp_path / "t1.nii.gz"
    src.write_bytes(b"x")

    res = bet.run_bet(src, tmp_path / "out.nii.gz", mask_path=tmp_path / "m.nii.gz")

    assert res["success"] is False
    assert "return code 1" in res["error"]
    assert "not valid" in res["error"]


def _fake_bet(tmp_path, monkeypatch, body):
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    script = bindir / "bet"
    script.write_text("#!/bin/sh\n"
                      'if [ "$1" = "-help" ]; then echo "Usage: bet <input> <output>"; exit 1; fi\n'
                      + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def test_exit_zero_without_output_is_a_failure(tmp_path, monkeypatch):
    _fake_bet(tmp_path, monkeypatch, "exit 0\n")
    src = tmp_path / "t1.nii.gz"
    src.write_bytes(b"x")

    res = bet.run_bet(src, tmp_path / "out.nii.gz", mask_path=tmp_path / "m.nii.gz")

    assert res["success"] is False
    assert "stripped image" in res["error"]


def test_exit_zero_without_mask_is_a_failure(tmp_path, monkeypatch):
    # writes the stripped image (2nd arg + .nii.gz) but no mask
    _fake_bet(tmp_path, monkeypatch, 'cp "$1" "$2.nii.gz"\nexit 0\n')
    src = tmp_path / "t1.nii.gz"
    src.write_bytes(b"x")

    res = bet.run_bet(src, tmp_path / "out.nii.gz", mask_path=tmp_path / "m.nii.gz")

    assert res["success"] is False
    assert "brain mask" in res["error"]
