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
    # 1mm isotropic, two separate blobs: 8 voxels and 1 voxel
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1   # 8 voxels -> 0.008 cm3
    data[7, 7, 7] = 1         # 1 voxel  -> 0.001 cm3
    p = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    return p


def test_compute_lesion_stats_returns_label_map(tmp_path):
    mask = _make_mask(tmp_path)
    stats, labeled, affine = stage08.compute_lesion_stats(mask)

    assert stats["lesion_count"] == 2
    # label->volume map has one entry per lesion, keyed by string label
    by_label = stats["lesion_volumes_by_label"]
    assert len(by_label) == 2
    assert set(round(v, 3) for v in by_label.values()) == {0.008, 0.001}
    # labeled array carries integer labels matching the map keys
    assert set(int(x) for x in np.unique(labeled) if x != 0) == {int(k) for k in by_label}
    # display list stays sorted descending
    assert stats["lesion_volumes_cm3"] == sorted(stats["lesion_volumes_cm3"], reverse=True)
