import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from lesion_stats import compute_lesion_stats, MIN_LESION_VOLUME_MM3


def _write_mask(tmp_path: Path, data: np.ndarray, voxel_mm=(1.0, 1.0, 1.0)) -> Path:
    affine = np.diag(list(voxel_mm) + [1.0])
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    path = tmp_path / "sub-001_ses-001_t1_segmask.nii.gz"
    nib.save(img, str(path))
    return path


def test_counts_two_separate_lesions_above_threshold(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1   # 27 voxels = 27 mm3 (1mm voxels) -> kept
    data[10:13, 10:13, 10:13] = 1  # another 27-voxel blob, far away -> kept
    mask_path = _write_mask(tmp_path, data)

    stats, labeled, affine, kept_labels = compute_lesion_stats(mask_path)

    assert stats["lesion_count"] == 2
    assert len(stats["lesion_volumes_cm3"]) == 2
    assert labeled.max() == 2
    assert kept_labels == {1, 2}


def test_sub_threshold_blob_excluded_from_count_but_kept_in_total(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[0, 0, 0] = 1  # 1 voxel = 1 mm3, below MIN_LESION_VOLUME_MM3 (5.0)
    data[10:13, 10:13, 10:13] = 1  # 27 mm3, kept
    mask_path = _write_mask(tmp_path, data)

    stats, labeled, affine, kept_labels = compute_lesion_stats(mask_path)

    assert stats["lesion_count"] == 1  # only the 27mm3 blob is "counted"
    total_voxels = int((data > 0).sum())
    assert stats["total_volume_cm3"] == pytest.approx(total_voxels / 1000.0, abs=1e-6)
    assert labeled.max() == 2  # both blobs still get a label in the saved array
    assert kept_labels == {2}  # label 1 is the sub-threshold 1-voxel speck at [0,0,0],
                                # which scipy.ndimage.label (raster-scan order) assigns
                                # label 1 before the 10:13 block gets label 2
