import sys
from pathlib import Path

import numpy as np
import nibabel as nib

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from atlas_resample import resample_to_grid


def test_resamples_identical_grid_unchanged(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[3:6, 3:6, 3:6] = 7
    affine = np.eye(4)
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(atlas_path))

    result = resample_to_grid(atlas_path, target_affine=affine, target_shape=(10, 10, 10))

    assert result is not None
    assert result.shape == (10, 10, 10)
    assert result[4, 4, 4] == 7
    assert result[0, 0, 0] == 0


def test_resamples_finer_grid_to_coarser_target(tmp_path):
    # Atlas at 0.5mm voxels, target grid at 1mm voxels covering the same
    # physical space — a coarser target should still pick up the labeled region.
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[6:12, 6:12, 6:12] = 3
    atlas_affine = np.diag([0.5, 0.5, 0.5, 1.0])
    atlas_path = tmp_path / "atlas_fine.nii.gz"
    nib.save(nib.Nifti1Image(data, atlas_affine), str(atlas_path))

    target_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    result = resample_to_grid(atlas_path, target_affine=target_affine, target_shape=(10, 10, 10))

    assert result is not None
    assert result.shape == (10, 10, 10)
    # Physical center of the labeled region (~ voxel index 4-5 in 1mm space) is labeled.
    assert result[4, 4, 4] == 3


def test_returns_none_on_missing_atlas(tmp_path):
    missing = tmp_path / "does_not_exist.nii.gz"
    result = resample_to_grid(missing, target_affine=np.eye(4), target_shape=(5, 5, 5))
    assert result is None
