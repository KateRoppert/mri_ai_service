#!/usr/bin/env python3
"""
One-time data-prep: register MS McDonald-zone atlases (lateral ventricles,
brainstem+cerebellum) from native FSL MNI152 space into each template listed
in lobar_atlas_config.yaml (SRI24, MNI152_FSL, MNI152_ICBM).

Prerequisite: run scripts/fetch_ms_zone_atlases.sh first to populate
data/templates/ms_zones_raw/.

Output (committed to git, same convention as the existing cortical atlas):
  data/templates/ms_zones/ventricles_<TEMPLATE>.nii.gz
  data/templates/ms_zones/infratentorial_<TEMPLATE>.nii.gz

Usage:
    python scripts/register_ms_zone_atlases.py
"""

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "preprocessing_steps"))
from registration import register_to_atlas, apply_transform  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "templates" / "ms_zones_raw"
OUT_DIR = PROJECT_ROOT / "data" / "templates" / "ms_zones"
WORK_DIR = PROJECT_ROOT / "data" / "templates" / "ms_zones_work"  # intermediates, not committed

# Voxel values in HarvardOxford-sub-maxprob-thr25-1mm.nii.gz.
# CONFIRMED: real "index"+1 values from
# `grep -i "ventricle" data/templates/ms_zones_raw/HarvardOxford-Subcortical.xml`
# (Task 1, Step 3 of the implementation plan).
LEFT_LATERAL_VENTRICLE = 3   # CONFIRMED against HarvardOxford-Subcortical.xml
RIGHT_LATERAL_VENTRICLE = 14  # CONFIRMED against HarvardOxford-Subcortical.xml
BRAIN_STEM = 8                # CONFIRMED against HarvardOxford-Subcortical.xml

# Source template the raw atlases are natively aligned to (FSL's own MNI152).
SOURCE_TEMPLATE_T1 = PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz"

# Target templates: name -> (template T1 path, needs_registration)
# MNI152_FSL is the atlas's native space (SOURCE_TEMPLATE_T1 itself) — no
# registration needed, just binarize and copy.
TARGETS = {
    "MNI152_FSL": (PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz", False),
    "SRI24": (PROJECT_ROOT / "data" / "templates" / "sri24_t1.nii.gz", True),
    "MNI152_ICBM": (PROJECT_ROOT / "data" / "templates" / "mni_icbm152_t1_tal_nlin_sym_09a.nii", True),
}


def binarize_labels(data: np.ndarray, label_values: set) -> np.ndarray:
    """Return a 0/1 int array: 1 where `data` matches any value in `label_values`."""
    return np.isin(data, list(label_values)).astype(np.int16)


def build_zone_masks() -> tuple:
    """Binarize the raw HarvardOxford-sub + Cerebellum atlases in native MNI152_FSL space."""
    sub_img = nib.load(str(RAW_DIR / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"))
    sub_data = np.asarray(sub_img.dataobj)

    cerebellum_img = nib.load(str(RAW_DIR / "Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz"))
    cerebellum_data = np.asarray(cerebellum_img.dataobj)

    ventricle_mask = binarize_labels(sub_data, {LEFT_LATERAL_VENTRICLE, RIGHT_LATERAL_VENTRICLE})
    # Infratentorial = brainstem (from HarvardOxford-sub) OR any cerebellar
    # lobule (every non-zero label in Cerebellum-MNIflirt is cerebellum).
    brainstem_mask = binarize_labels(sub_data, {BRAIN_STEM})
    cerebellum_mask = (cerebellum_data > 0).astype(np.int16)
    infratentorial_mask = np.clip(brainstem_mask + cerebellum_mask, 0, 1).astype(np.int16)

    return ventricle_mask, infratentorial_mask, sub_img.affine


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    ventricle_mask, infratentorial_mask, native_affine = build_zone_masks()
    ventricle_path = WORK_DIR / "ventricles_native.nii.gz"
    infratentorial_path = WORK_DIR / "infratentorial_native.nii.gz"
    nib.save(nib.Nifti1Image(ventricle_mask, native_affine), str(ventricle_path))
    nib.save(nib.Nifti1Image(infratentorial_mask, native_affine), str(infratentorial_path))
    logger.info(f"Native-space binary masks written to {WORK_DIR}")

    for template_name, (template_t1, needs_registration) in TARGETS.items():
        if not needs_registration:
            for src, label in [(ventricle_path, "ventricles"), (infratentorial_path, "infratentorial")]:
                dest = OUT_DIR / f"{label}_{template_name}.nii.gz"
                dest.write_bytes(src.read_bytes())
                logger.info(f"{template_name}: copied {label} (native space) -> {dest}")
            continue

        transform_path = WORK_DIR / f"mni152fsl_to_{template_name}_transform.mat"
        warped_t1_path = WORK_DIR / f"mni152fsl_to_{template_name}_t1.nii.gz"

        logger.info(f"Registering MNI152_FSL T1 -> {template_name} ...")
        result = register_to_atlas(
            moving_path=SOURCE_TEMPLATE_T1,
            atlas_path=template_t1,
            output_path=warped_t1_path,
            transform_path=transform_path,
            registration_type="Affine",  # two different population templates -> allow scale/shear
        )
        if not result["success"]:
            raise RuntimeError(f"Registration to {template_name} failed: {result.get('error')}")

        for src, label in [(ventricle_path, "ventricles"), (infratentorial_path, "infratentorial")]:
            dest = OUT_DIR / f"{label}_{template_name}.nii.gz"
            apply_result = apply_transform(
                moving_path=src,
                fixed_path=template_t1,
                transform_path=transform_path,
                output_path=dest,
                interpolation="nearestNeighbor",
            )
            if not apply_result["success"]:
                raise RuntimeError(f"Applying transform for {label}/{template_name} failed: {apply_result.get('error')}")
            logger.info(f"{template_name}: registered {label} -> {dest}")

    logger.info(f"Done. Zone atlases written to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
