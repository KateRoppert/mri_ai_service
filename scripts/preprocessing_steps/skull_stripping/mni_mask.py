"""
Atlas-baseline skull stripping: apply a fixed MNI152 brain mask.

Two variants: ``strict`` (no dilation) and ``loose`` (~2 mm / 2-voxel
dilation at 1 mm). Inputs are assumed registered to the production atlas
(MNI152_FSL / MNI152_T1_1mm); the mask is resampled onto the input grid
if the affine or shape differs.

Uses the FSL ``MNI152_T1_1mm_brain_mask``, not a threshold on the T1
template (that T1 still has skull). Enclosed cavities in that mask (and
speckles from nearest-neighbour resample onto an ANTs affine) are filled
so the applied envelope is solid.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_dilation, binary_fill_holes

from .base import SkullStripperBase, apply_brain_mask

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES = _REPO_ROOT / "data" / "templates"
_DEFAULT_BRAIN_MASK = _TEMPLATES / "MNI152_T1_1mm_brain_mask.nii.gz"


def resolve_mask_template(override: Optional[PathLike] = None) -> Optional[Path]:
    """First existing MNI152 1mm brain mask, not the skull-on T1."""
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.append(_DEFAULT_BRAIN_MASK)
    fsl = os.environ.get("FSLDIR")
    if fsl:
        candidates.append(
            Path(fsl) / "data" / "standard" / "MNI152_T1_1mm_brain_mask.nii.gz"
        )
    candidates.extend((
        Path("/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"),
        Path("/usr/share/fsl/6.0/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"),
    ))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _merged_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = params or {}
    nested = params.get("tool_params")
    nested = nested if isinstance(nested, dict) else {}
    merged = dict(nested)
    for key, value in params.items():
        if key != "tool_params":
            merged[key] = value
    return merged


class MniMaskStripper(SkullStripperBase):
    """Apply a (optionally dilated) MNI152 brain mask."""

    name = "mni_mask"
    uses_gpu = False

    def is_available(self) -> bool:
        return resolve_mask_template() is not None

    @staticmethod
    def _dilate(mask: np.ndarray, radius_vox: int) -> np.ndarray:
        binary = mask > 0
        if radius_vox <= 0:
            return binary.astype(np.uint8)
        return binary_dilation(binary, iterations=int(radius_vox)).astype(np.uint8)

    def strip(
        self,
        input_path: Path,
        output_path: Path,
        mask_path: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        try:
            tp = _merged_params(params)
            variant = str(tp.get("variant", "strict")).lower()
            radius = 0 if variant != "loose" else int(tp.get("dilation_vox", 2))
            template_path = resolve_mask_template(tp.get("mask_template"))
            if template_path is None:
                raise RuntimeError(
                    "MNI152 brain mask not found. Expected "
                    f"{_DEFAULT_BRAIN_MASK} or $FSLDIR/data/standard/"
                    "MNI152_T1_1mm_brain_mask.nii.gz"
                )

            in_img = nib.load(str(input_path))
            tmpl_img = nib.load(str(template_path))
            if (
                tmpl_img.shape != in_img.shape
                or not np.allclose(tmpl_img.affine, in_img.affine, atol=1e-4)
            ):
                tmpl_img = resample_from_to(tmpl_img, in_img, order=0)

            mask = self._dilate(np.asanyarray(tmpl_img.dataobj), radius)
            # FSL's brain mask (and NN resample onto an ANTs affine) leave
            # enclosed cavities; skull-stripping wants a solid envelope.
            mask = binary_fill_holes(mask > 0).astype(np.uint8)
            if mask_path is None:
                mask_path = output_path.with_name(
                    output_path.name.replace(".nii.gz", "_mask.nii.gz")
                )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(
                nib.Nifti1Image(mask, in_img.affine, in_img.header),
                str(mask_path),
            )

            apply_res = apply_brain_mask(input_path, mask_path, output_path)
            if not apply_res.get("success"):
                raise RuntimeError(apply_res.get("error", "apply_brain_mask failed"))

            processing_time = time.time() - start
            logger.info(
                "MNI mask (%s, radius=%d) completed in %.2f seconds",
                variant,
                radius,
                processing_time,
            )
            return {
                "success": True,
                "output_path": str(output_path),
                "mask_path": str(mask_path),
                "processing_time": processing_time,
            }
        except Exception as e:
            logger.error("MNI mask failed on %s: %s", Path(input_path).name, e)
            return {"success": False, "error": str(e)}
