"""
Shared nearest-neighbor resampling of a standard-space label atlas onto a
target voxel grid. Used by LobarAnalyzer (cortical lobes) and MSZoneAnalyzer
(McDonald zones) — both overlay a fixed standard-space atlas with a subject
mask that may be on a slightly different grid (resolution/shape mismatch
after independent registration runs).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

logger = logging.getLogger(__name__)


def resample_to_grid(
    atlas_path: Path,
    target_affine: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    """
    Nearest-neighbor resample a standard-space atlas NIfTI onto a target grid.

    Args:
        atlas_path: path to the standard-space label atlas NIfTI.
        target_affine: affine of the grid to resample onto (e.g. a subject mask's).
        target_shape: shape of the grid to resample onto.

    Returns:
        Resampled int array of `target_shape`, or None on failure.
    """
    try:
        atlas_nii = nib.load(str(atlas_path))
        atlas_data = np.asarray(atlas_nii.dataobj).astype(np.float64)
        atlas_affine = atlas_nii.affine

        # Combined: atlas_voxel = inv(atlas_affine) @ target_affine @ target_voxel
        transform = np.linalg.inv(atlas_affine) @ target_affine
        matrix = transform[:3, :3]
        offset = transform[:3, 3]

        resampled = affine_transform(
            atlas_data,
            matrix,
            offset=offset,
            output_shape=target_shape,
            order=0,  # nearest-neighbor
            mode='constant',
            cval=0,
        )
        return resampled.astype(int)

    except Exception as e:
        logger.error(f"Failed to resample atlas {atlas_path}: {e}")
        return None
