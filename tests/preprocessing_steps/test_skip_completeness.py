"""Stage skip logic must not accept a truncated file as finished work."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

importlib = __import__("importlib")
stage05 = importlib.import_module("05_preprocessing")


def _write_nifti(path, truncate=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(8, 8, 8).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    if truncate:
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])


def test_complete_subject_is_skipped(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    for base in (in_dir, out_dir):
        _write_nifti(base / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz")

    is_processed, missing = stage05.check_subject_processed(
        in_dir, out_dir, "sub-001", "ses-001", ["t1"]
    )

    assert is_processed is True
    assert missing == []


def test_truncated_output_is_not_skipped(tmp_path):
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _write_nifti(in_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz")
    _write_nifti(
        out_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz",
        truncate=True,
    )

    is_processed, missing = stage05.check_subject_processed(
        in_dir, out_dir, "sub-001", "ses-001", ["t1"]
    )

    assert is_processed is False, "a truncated output must be recomputed"
    assert "t1" in missing
