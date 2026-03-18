"""
Lobar localization analysis module.

Computes overlap between segmentation mask and lobar atlas to determine
anatomical localization of brain lesions by lobe.
"""

import json
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LobarAnalyzer:
    """Analyzes anatomical localization of lesions using a lobar atlas."""

    def __init__(self, atlas_path: Path, mapping_path: Path, 
                 seg_classes: Optional[Dict] = None):
        """
        Args:
            atlas_path: Path to lobar atlas NIfTI (in same space as segmentation)
            mapping_path: Path to lobar_mapping.json
            seg_classes: Dict of {label_int: {name_en, name_ru}} for segmentation classes
        """
        self.atlas_path = atlas_path
        self.mapping = self._load_mapping(mapping_path)
        self.seg_classes = seg_classes or {
            1: {"name_en": "NCR", "name_ru": "Некротическое ядро"},
            2: {"name_en": "ED",  "name_ru": "Перитуморальный отёк"},
            3: {"name_en": "NET", "name_ru": "Неусиливающаяся опухоль"},
            4: {"name_en": "ET",  "name_ru": "Усиливающаяся опухоль"},
        }

        # Load atlas once
        logger.info(f"Loading lobar atlas: {atlas_path}")
        atlas_nii = nib.load(str(atlas_path))
        self.atlas_data = np.asarray(atlas_nii.dataobj).astype(int)
        self.atlas_voxel_volume = float(np.prod(atlas_nii.header.get_zooms()[:3]))
        logger.info(f"Atlas shape: {self.atlas_data.shape}, "
                     f"voxel volume: {self.atlas_voxel_volume:.4f} mm³")

        # Build lobe label sets from mapping
        self.lobe_labels = self._build_lobe_labels()

    @staticmethod
    def _load_mapping(mapping_path: Path) -> Dict:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_lobe_labels(self) -> Dict[str, set]:
        """Build {lobe_name: set(atlas_labels)} from mapping."""
        lobe_labels = {}
        for label_str, region_info in self.mapping["regions"].items():
            lobe = region_info["lobe"]
            if lobe not in lobe_labels:
                lobe_labels[lobe] = set()
            lobe_labels[lobe].add(int(label_str))
        return lobe_labels

    def _resample_atlas_to_mask(self, mask_nii) -> Optional[np.ndarray]:
        """Resample atlas to match mask grid using nearest-neighbor."""
        try:
            from scipy.ndimage import affine_transform

            atlas_nii = nib.load(str(self.atlas_path))
            atlas_data = np.asarray(atlas_nii.dataobj).astype(np.float64)

            # Compute voxel-to-voxel mapping: mask voxel -> atlas voxel
            # mask_voxel -> world: mask_affine
            # world -> atlas_voxel: inv(atlas_affine)
            mask_affine = mask_nii.affine
            atlas_affine = atlas_nii.affine

            # Combined: atlas_voxel = inv(atlas_affine) @ mask_affine @ mask_voxel
            transform = np.linalg.inv(atlas_affine) @ mask_affine

            # affine_transform uses: output[o] = input[matrix @ o + offset]
            matrix = transform[:3, :3]
            offset = transform[:3, 3]

            resampled = affine_transform(
                atlas_data,
                matrix,
                offset=offset,
                output_shape=mask_nii.shape[:3],
                order=0,  # nearest-neighbor
                mode='constant',
                cval=0
            )

            resampled = resampled.astype(int)
            logger.info(f"  Resampled atlas: {resampled.shape}, "
                        f"non-zero: {(resampled > 0).sum()}")
            return resampled

        except Exception as e:
            logger.error(f"Failed to resample atlas: {e}")
            return None

    def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
        """
        Analyze a single segmentation mask against the lobar atlas.
        
        Args:
            mask_path: Path to segmentation mask NIfTI
            
        Returns:
            Dict with localization report or None on failure
        """
        try:
            mask_nii = nib.load(str(mask_path))
            mask_data = np.asarray(mask_nii.dataobj).astype(int)
            mask_voxel_volume = float(np.prod(mask_nii.header.get_zooms()[:3]))

            # Resample atlas to mask grid if shapes differ
            atlas_data = self.atlas_data
            if mask_data.shape != atlas_data.shape:
                logger.info(
                    f"  Resampling atlas {atlas_data.shape} -> {mask_data.shape}")
                atlas_data = self._resample_atlas_to_mask(mask_nii)
                if atlas_data is None:
                    return None

            # Overall lesion stats
            lesion_mask = mask_data > 0
            total_lesion_voxels = int(lesion_mask.sum())

            if total_lesion_voxels == 0:
                logger.warning(f"No lesion voxels in {mask_path.name}")
                return self._empty_report(mask_path, mask_voxel_volume)

            # Per-lobe, per-class analysis
            lobe_results = {}
            for lobe_name, atlas_labels in self.lobe_labels.items():
                lobe_info = self.mapping["lobes"][lobe_name]

                # Create binary mask for this lobe
                lobe_mask = np.isin(atlas_data, list(atlas_labels))

                # Overlap with full lesion
                overlap = lesion_mask & lobe_mask
                lobe_voxels = int(overlap.sum())

                if lobe_voxels == 0:
                    continue

                # Per-class breakdown
                class_breakdown = {}
                for cls_label, cls_info in self.seg_classes.items():
                    cls_mask = (mask_data == cls_label) & lobe_mask
                    cls_voxels = int(cls_mask.sum())
                    if cls_voxels > 0:
                        class_breakdown[str(cls_label)] = {
                            "name_en": cls_info["name_en"],
                            "name_ru": cls_info["name_ru"],
                            "voxel_count": cls_voxels,
                            "volume_mm3": round(cls_voxels * mask_voxel_volume, 2),
                            "volume_cm3": round(cls_voxels * mask_voxel_volume / 1000, 4),
                        }

                lobe_results[lobe_name] = {
                    "name_en": lobe_info["name_en"],
                    "name_ru": lobe_info["name_ru"],
                    "color": lobe_info["color"],
                    "total_voxels": lobe_voxels,
                    "total_volume_mm3": round(lobe_voxels * mask_voxel_volume, 2),
                    "total_volume_cm3": round(lobe_voxels * mask_voxel_volume / 1000, 4),
                    "percent_of_lesion": round(lobe_voxels / total_lesion_voxels * 100, 2),
                    "classes": class_breakdown,
                }

            # Sort by volume descending
            lobe_results = dict(
                sorted(lobe_results.items(),
                       key=lambda x: x[1]["total_voxels"], reverse=True)
            )

            report = {
                "mask_file": mask_path.name,
                "atlas_name": self.mapping["atlas_name"],
                "voxel_volume_mm3": round(mask_voxel_volume, 4),
                "total_lesion_voxels": total_lesion_voxels,
                "total_lesion_volume_mm3": round(total_lesion_voxels * mask_voxel_volume, 2),
                "total_lesion_volume_cm3": round(total_lesion_voxels * mask_voxel_volume / 1000, 4),
                "lobes": lobe_results,
            }

            affected_lobes = [v["name_ru"] for v in lobe_results.values()]
            logger.info(f"  {mask_path.name}: lesion in {', '.join(affected_lobes)}")

            return report

        except Exception as e:
            logger.error(f"Failed to analyze {mask_path.name}: {e}")
            return None

    def _empty_report(self, mask_path: Path, voxel_volume: float) -> Dict:
        return {
            "mask_file": mask_path.name,
            "atlas_name": self.mapping["atlas_name"],
            "voxel_volume_mm3": round(voxel_volume, 4),
            "total_lesion_voxels": 0,
            "total_lesion_volume_mm3": 0.0,
            "total_lesion_volume_cm3": 0.0,
            "lobes": {},
        }

    def save_report(self, report: Dict, output_path: Path) -> bool:
        """Save report as JSON."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.debug(f"  Report saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report to {output_path}: {e}")
            return False