"""Tests for utils/nifti_integrity.py."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.nifti_integrity import is_complete_nifti


@pytest.fixture
def complete_nifti(tmp_path):
    """A small but structurally real .nii.gz."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    path = tmp_path / "complete.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def test_complete_file_passes(complete_nifti):
    assert is_complete_nifti(complete_nifti) is True


def test_truncated_file_fails(complete_nifti, tmp_path):
    # The exact failure a killed process leaves behind: the write stopped
    # partway. Note nibabel.load() still succeeds on this — the header
    # survived — which is why existence and header checks are insufficient.
    truncated = tmp_path / "truncated.nii.gz"
    blob = complete_nifti.read_bytes()
    truncated.write_bytes(blob[: len(blob) // 2])

    assert is_complete_nifti(truncated) is False


def test_empty_file_fails(tmp_path):
    empty = tmp_path / "empty.nii.gz"
    empty.touch()

    assert is_complete_nifti(empty) is False


def test_missing_file_fails(tmp_path):
    assert is_complete_nifti(tmp_path / "nope.nii.gz") is False


def test_garbage_file_fails(tmp_path):
    garbage = tmp_path / "garbage.nii.gz"
    garbage.write_bytes(b"this is not a nifti file at all")

    assert is_complete_nifti(garbage) is False


def test_uncompressed_nifti_supported(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    path = tmp_path / "plain.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

    assert is_complete_nifti(path) is True


def test_truncated_uncompressed_nifti_fails(tmp_path):
    data = np.random.rand(8, 8, 8).astype(np.float32)
    path = tmp_path / "plain.nii"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    blob = path.read_bytes()
    path.write_bytes(blob[: len(blob) // 2])

    assert is_complete_nifti(path) is False
