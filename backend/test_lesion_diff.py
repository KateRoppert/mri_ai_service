import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lesion_diff import compare_labeled_masks


def _save_labeled(path: Path, labels: dict, shape=(20, 20, 20)):
    """labels: {label_int: (slice_x, slice_y, slice_z)}"""
    data = np.zeros(shape, dtype=np.int16)
    for label, (sx, sy, sz) in labels.items():
        data[sx, sy, sz] = label
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_lesion_with_no_prior_overlap_is_new(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 3), slice(0, 3), slice(0, 3))})       # far away
    _save_labeled(curr_path, {1: (slice(10, 13), slice(10, 13), slice(10, 13))})  # new lesion

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1)

    assert result["new_count"] == 1
    statuses = {l["status"] for l in result["lesions"]}
    assert "new" in statuses


def test_prior_lesion_absent_in_current_is_resolved(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 3), slice(0, 3), slice(0, 3))})
    _save_labeled(curr_path, {})  # empty — lesion resolved

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1)

    assert result["resolved_count"] == 1
    resolved = [l for l in result["lesions"] if l["status"] == "resolved"][0]
    assert resolved["volume_cm3"] == 0.0
    assert resolved["previous_volume_cm3"] > 0.0


def test_matched_lesion_growing_by_relative_threshold(tmp_path):
    # Small lesion: 3x3x3=27 voxels (0.027 cm3) -> grows to 5x5x5=125 voxels
    # (0.125 cm3). Relative growth = (0.125-0.027)/0.027 ≈ 3.6 >> 0.20.
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(5, 8), slice(5, 8), slice(5, 8))})
    _save_labeled(curr_path, {1: (slice(5, 10), slice(5, 10), slice(5, 10))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["growing_count"] == 1
    assert result["new_count"] == 0


def test_matched_lesion_growing_by_absolute_threshold_with_small_relative_growth(tmp_path):
    # Large lesion: 10x10x10=1000 voxels (1.0 cm3) -> grows to 1100 voxels by
    # adding a 10x10x1 slab (100 voxels = 0.1 cm3). Relative growth = 10% < 20%,
    # but absolute growth 0.1 cm3 >= 0.03 cm3 threshold -> still "growing".
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})
    _save_labeled(curr_path, {1: (slice(0, 10), slice(0, 10), slice(0, 11))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["growing_count"] == 1
    assert result["stable_count"] == 0


def test_matched_lesion_with_negligible_change_is_stable(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    _save_labeled(prev_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})
    _save_labeled(curr_path, {1: (slice(0, 10), slice(0, 10), slice(0, 10))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    assert result["stable_count"] == 1
    assert result["growing_count"] == 0


def test_sub_threshold_components_excluded_from_both_sides(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    # A single 1mm voxel = 0.001 cm3, far below the 5mm3 (0.005 cm3) floor.
    _save_labeled(prev_path, {1: (slice(0, 1), slice(0, 1), slice(0, 1))})
    _save_labeled(curr_path, {1: (slice(15, 16), slice(15, 16), slice(15, 16))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=1, min_lesion_volume_mm3=5.0)

    assert result["lesions"] == []
    assert result["new_count"] == 0
    assert result["resolved_count"] == 0


def test_current_lesion_overlapping_two_prior_lesions_sums_previous_volume(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr.nii.gz"
    # Two separate prior lesions, each 3x3x3=27 voxels (0.027 cm3), adjacent
    # but not touching (gap at x=8). A merged current lesion spans both.
    _save_labeled(prev_path, {
        1: (slice(0, 3), slice(0, 3), slice(0, 3)),
        2: (slice(8, 11), slice(0, 3), slice(0, 3)),
    })
    _save_labeled(curr_path, {1: (slice(0, 11), slice(0, 3), slice(0, 3))})

    result = compare_labeled_masks(prev_path, curr_path, dilation_voxels=0)

    matched = [l for l in result["lesions"] if l["status"] in ("growing", "stable")][0]
    assert matched["previous_volume_cm3"] == pytest.approx(0.027 + 0.027, abs=1e-4)


def test_shape_mismatch_raises_value_error(tmp_path):
    prev_path = tmp_path / "prev.nii.gz"
    curr_path = tmp_path / "curr_different_shape.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.int16), np.eye(4)), str(prev_path))
    nib.save(nib.Nifti1Image(np.zeros((22, 22, 22), dtype=np.int16), np.eye(4)), str(curr_path))

    with pytest.raises(ValueError):
        compare_labeled_masks(prev_path, curr_path)
