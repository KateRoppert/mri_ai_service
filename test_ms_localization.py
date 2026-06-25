import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from ms_localization import MSZoneAnalyzer


def _save(path: Path, data: np.ndarray, affine=None):
    affine = affine if affine is not None else np.eye(4)
    nib.save(nib.Nifti1Image(data.astype(np.int16), affine), str(path))


@pytest.fixture
def zone_atlases(tmp_path):
    """20x20x20 1mm grid. Ventricle = a 2x2x2 block at [8:12,8:12,8:12].
    Cortex = a full-plane shell at x=0 and x=19. Infratentorial = a 2x2x2
    block at [4:6,4:6,4:6]. All three are placed with at least a 2-voxel
    gap between them (and between infratentorial and the cortex planes),
    so that with dilation_voxels=1 (1-voxel dilation on each zone) no two
    zones overlap by accident — each test's lesion placement is what
    decides which single zone gets touched."""
    shape = (20, 20, 20)

    ventricle = np.zeros(shape, dtype=np.int16)
    ventricle[8:12, 8:12, 8:12] = 1
    ventricle_path = tmp_path / "ventricles.nii.gz"
    _save(ventricle_path, ventricle)

    cortex = np.zeros(shape, dtype=np.int16)
    cortex[0, :, :] = 1
    cortex[19, :, :] = 1
    cortex_path = tmp_path / "cortex.nii.gz"
    _save(cortex_path, cortex)

    infratentorial = np.zeros(shape, dtype=np.int16)
    infratentorial[4:6, 4:6, 4:6] = 1
    infratentorial_path = tmp_path / "infratentorial.nii.gz"
    _save(infratentorial_path, infratentorial)

    return {"ventricle": ventricle_path, "cortex": cortex_path, "infratentorial": infratentorial_path}


def test_lesion_touching_ventricle_is_periventricular(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[9:11, 9:11, 7:9] = 1  # touches the ventricle block at z=8
    mask_path = tmp_path / "sub-001_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["periventricular"]["lesion_count"] == 1
    assert report["zones"]["juxtacortical"]["lesion_count"] == 0
    assert report["zones"]["spinal_cord"] == {"supported": False}


def test_lesion_touching_cortex_is_juxtacortical(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[0:2, 5:7, 5:7] = 1  # touches cortex shell at x=0
    mask_path = tmp_path / "sub-002_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["juxtacortical"]["lesion_count"] == 1
    assert report["zones"]["periventricular"]["lesion_count"] == 0


def test_lesion_in_infratentorial_zone(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[4:6, 4:6, 4:6] = 1  # exactly the infratentorial block
    mask_path = tmp_path / "sub-003_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["infratentorial"]["lesion_count"] == 1


def test_lesion_touching_nothing_is_deep_white_matter(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[14:16, 14:16, 14:16] = 1  # isolated, far from all three zones
    mask_path = tmp_path / "sub-004_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["deep_white_matter"]["lesion_count"] == 1


def test_periventricular_takes_priority_over_juxtacortical(tmp_path):
    # Dedicated atlases (not the shared `zone_atlases` fixture) so the
    # ventricle and cortex zones are close enough for one lesion to touch
    # both directly — this is what actually exercises the priority order.
    shape = (20, 20, 20)

    ventricle = np.zeros(shape, dtype=np.int16)
    ventricle[5:7, 5:7, 5:7] = 1  # z in [5,7)
    ventricle_path = tmp_path / "ventricles.nii.gz"
    _save(ventricle_path, ventricle)

    cortex = np.zeros(shape, dtype=np.int16)
    cortex[5:7, 5:7, 8:10] = 1  # z in [8,10) — adjacent gap, not overlapping ventricle
    cortex_path = tmp_path / "cortex.nii.gz"
    _save(cortex_path, cortex)

    infratentorial = np.zeros(shape, dtype=np.int16)
    infratentorial[0:2, 0:2, 0:2] = 1  # far from everything
    infratentorial_path = tmp_path / "infratentorial.nii.gz"
    _save(infratentorial_path, infratentorial)

    # Lesion spans z=[6,9): touches ventricle at z=6 AND cortex at z=8.
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:7, 5:7, 6:9] = 1
    mask_path = tmp_path / "sub-005_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=ventricle_path,
        cortex_atlas_path=cortex_path,
        infratentorial_atlas_path=infratentorial_path,
        dilation_voxels=0,  # isolate the priority logic from dilation tolerance
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["periventricular"]["lesion_count"] == 1
    assert report["zones"]["juxtacortical"]["lesion_count"] == 0


def test_save_report_writes_json(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[14:16, 14:16, 14:16] = 1
    mask_path = tmp_path / "sub-006_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)
    out_path = tmp_path / "out" / "report.json"

    assert analyzer.save_report(report, out_path) is True
    assert out_path.exists()
