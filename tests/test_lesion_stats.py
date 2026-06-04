# tests/test_lesion_stats.py
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import importlib.util

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "stage08", str(Path(__file__).parent.parent / "scripts" / "08_lobar_localization.py")
)
stage08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage08)


def _make_mask(tmp_path):
    # 1mm isotropic, two separate blobs: 8 voxels (kept) and 1 voxel (noise).
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1   # 8 voxels -> 0.008 cm3 (>= 3 mm3, kept)
    data[7, 7, 7] = 1         # 1 voxel  -> 0.001 cm3 (< 3 mm3, filtered)
    p = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    return p


def test_min_size_filter_excludes_noise_from_count(tmp_path):
    mask = _make_mask(tmp_path)
    stats, labeled, affine = stage08.compute_lesion_stats(mask)

    # The 1-voxel speck (<3 mm3) is excluded from the count and per-lesion list.
    assert stats["lesion_count"] == 1
    by_label = stats["lesion_volumes_by_label"]
    assert len(by_label) == 1
    assert round(next(iter(by_label.values())), 3) == 0.008
    assert stats["lesion_volumes_cm3"] == [0.008]
    # mean is over kept lesions only.
    assert round(stats["mean_lesion_volume_cm3"], 3) == 0.008

    # total_volume_cm3 is the FULL burden — it still includes the speck.
    assert round(stats["total_volume_cm3"], 3) == 0.009

    # The saved labeled array keeps EVERY component (both blobs); only the
    # stats/hover map drops the sub-threshold one.
    assert set(int(x) for x in np.unique(labeled) if x != 0) == {1, 2}


def test_display_list_sorted_descending(tmp_path):
    # Three kept blobs of different sizes — list must be sorted desc.
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1    # 27 voxels
    data[10:12, 10:12, 10:12] = 1  # 8 voxels
    data[15:17, 15:18, 15:17] = 1  # 12 voxels
    p = tmp_path / "mask3.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))

    stats, _, _ = stage08.compute_lesion_stats(p)
    assert stats["lesion_count"] == 3
    assert stats["lesion_volumes_cm3"] == sorted(stats["lesion_volumes_cm3"], reverse=True)
