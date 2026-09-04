"""Stage skip logic must not accept a truncated file as finished work."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

if "ants" not in sys.modules:
    sys.modules["ants"] = MagicMock()

importlib = __import__("importlib")
stage05 = importlib.import_module("05_preprocessing")
stage03 = importlib.import_module("03_convert_to_nifti")
stage07 = importlib.import_module("07_inverse_transform")


def _write_nifti(path, truncate=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(8, 8, 8).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    if truncate:
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])


def _converter():
    converter = stage03.NiftiConverter.__new__(stage03.NiftiConverter)
    return converter


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


def test_stage03_complete_nifti_is_skipped(tmp_path):
    out_dir = tmp_path / "nifti"
    _write_nifti(out_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz")

    assert _converter().check_output_exists(out_dir, "001", "001", "t1") is True


def test_stage03_truncated_nifti_is_not_skipped(tmp_path):
    out_dir = tmp_path / "nifti"
    _write_nifti(
        out_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.nii.gz",
        truncate=True,
    )
    (out_dir / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_t1.json").write_text("{}")

    assert _converter().check_output_exists(out_dir, "001", "001", "t1") is False


def test_stage07_complete_native_is_skipped(tmp_path):
    out_subdir = tmp_path / "sub-001" / "ses-001" / "anat" / "glioblastoma"
    _write_nifti(out_subdir / "sub-001_ses-001_t1_segmask_native_t1.nii.gz")

    assert stage07.has_complete_native_mask(out_subdir, "sub-001_ses-001_t1") is True


def test_stage07_truncated_native_is_not_skipped(tmp_path):
    out_subdir = tmp_path / "sub-001" / "ses-001" / "anat" / "glioblastoma"
    _write_nifti(
        out_subdir / "sub-001_ses-001_t1_segmask_native_t1.nii.gz",
        truncate=True,
    )
    (out_subdir / "sub-001_ses-001_t1_segmask_native_t1.json").write_text("{}")

    assert stage07.has_complete_native_mask(out_subdir, "sub-001_ses-001_t1") is False
