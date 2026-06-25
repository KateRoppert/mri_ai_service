"""
McDonald zone classification for multiple_sclerosis lesions.

Classifies each connected-component lesion into one McDonald 2017 zone —
periventricular, juxtacortical, or infratentorial — falling back to
deep_white_matter when a lesion touches none of those, with a fixed
hierarchy (periventricular > juxtacortical > infratentorial) matching
McDonald's clinical priority order. Spinal cord is not supported (no
spine-registration infrastructure in this pipeline) and is surfaced in the
report as explicitly unsupported, not silently omitted.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, label as ndimage_label

from anatomical_analyzer_base import AnatomicalAnalyzerBase
from atlas_resample import resample_to_grid

logger = logging.getLogger(__name__)

ZONE_ORDER = ("periventricular", "juxtacortical", "infratentorial", "deep_white_matter")


class MSZoneAnalyzer(AnatomicalAnalyzerBase):
    """Classifies MS lesions into McDonald zones using standard-space atlases."""

    REPORT_SUFFIX = "_mcdonald_report.json"

    def __init__(
        self,
        ventricle_atlas_path: Path,
        cortex_atlas_path: Path,
        infratentorial_atlas_path: Path,
        dilation_voxels: int = 1,
    ):
        self.ventricle_atlas_path = Path(ventricle_atlas_path)
        self.cortex_atlas_path = Path(cortex_atlas_path)
        self.infratentorial_atlas_path = Path(infratentorial_atlas_path)
        self.dilation_voxels = dilation_voxels

    def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
        try:
            mask_nii = nib.load(str(mask_path))
            mask_data = np.asarray(mask_nii.dataobj)
            voxel_vol_mm3 = float(np.prod(mask_nii.header.get_zooms()[:3]))
            voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
            target_affine = mask_nii.affine
            target_shape = mask_nii.shape[:3]

            ventricle = resample_to_grid(self.ventricle_atlas_path, target_affine, target_shape)
            cortex = resample_to_grid(self.cortex_atlas_path, target_affine, target_shape)
            infratentorial = resample_to_grid(self.infratentorial_atlas_path, target_affine, target_shape)
            if ventricle is None or cortex is None or infratentorial is None:
                logger.error(f"Failed to resample one or more zone atlases for {mask_path.name}")
                return None

            structure = np.ones((3, 3, 3)) if self.dilation_voxels > 0 else None
            ventricle_zone = (ventricle > 0)
            cortex_zone = (cortex > 0)
            infratentorial_zone = (infratentorial > 0)
            for _ in range(self.dilation_voxels):
                ventricle_zone = binary_dilation(ventricle_zone, structure=structure)
                cortex_zone = binary_dilation(cortex_zone, structure=structure)
                infratentorial_zone = binary_dilation(infratentorial_zone, structure=structure)

            binary = (mask_data > 0).astype(np.uint8)
            labeled, n_components = ndimage_label(binary)

            zone_counts = {zone: 0 for zone in ZONE_ORDER}
            zone_volumes_cm3 = {zone: 0.0 for zone in ZONE_ORDER}
            lesion_zones_by_label = {}

            for i in range(1, n_components + 1):
                lesion_voxels = (labeled == i)
                voxel_count = int(lesion_voxels.sum())
                volume_cm3 = voxel_count * voxel_vol_cm3

                if (lesion_voxels & ventricle_zone).any():
                    zone = "periventricular"
                elif (lesion_voxels & cortex_zone).any():
                    zone = "juxtacortical"
                elif (lesion_voxels & infratentorial_zone).any():
                    zone = "infratentorial"
                else:
                    zone = "deep_white_matter"

                zone_counts[zone] += 1
                zone_volumes_cm3[zone] = round(zone_volumes_cm3[zone] + volume_cm3, 4)
                lesion_zones_by_label[str(i)] = zone

            zones_report = {
                zone: {"lesion_count": zone_counts[zone], "total_volume_cm3": zone_volumes_cm3[zone]}
                for zone in ZONE_ORDER
            }
            zones_report["spinal_cord"] = {"supported": False}

            return {
                "mask_file": mask_path.name,
                "total_lesion_count": n_components,
                "zones": zones_report,
                "lesion_zones_by_label": lesion_zones_by_label,
            }

        except Exception as e:
            logger.error(f"Failed to analyze {mask_path.name}: {e}")
            return None

    def save_report(self, report: Dict, output_path: Path) -> bool:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save report to {output_path}: {e}")
            return False
