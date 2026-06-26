"""
Cross-session MS lesion diffing: classifies each lesion in the current
session as new, growing, stable, or resolved relative to the patient's
previous session.

Both sessions' labeled masks (*_segmask_labels.nii.gz, written by Stage 08
for every multiple_sclerosis mask) are already on the same voxel grid —
each session is independently registered to the same standard-space atlas
during preprocessing — so no inter-session registration is needed here,
just a direct per-component overlap comparison.

MIN_LESION_VOLUME_MM3 intentionally duplicates the value in
scripts/lesion_stats.py rather than importing it: backend/ and scripts/ are
two independently-deployed layers (a long-running FastAPI service vs. a
per-run pipeline subprocess) and the project does not cross-import between
them anywhere today. It is the same clinical noise-floor concept in both
places — keep the two literals in sync if the value ever changes.

Caveat on previous_volume_cm3: it is a per-lesion attribution, not a
partition of prior burden. If dilation_voxels bridges two current lesions
into the same previous lesion (more likely at higher dilation_voxels), that
previous lesion's volume is summed into both current lesions independently
— matched_prev_labels still marks it matched once, so new/stable/resolved
counts stay correct, but summing previous_volume_cm3 across a session pair's
lesions can double-count and will not reconcile to the true prior total.
"""

import logging
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

MIN_LESION_VOLUME_MM3 = 5.0


def compare_labeled_masks(
    prev_path: Path,
    curr_path: Path,
    *,
    growth_threshold_relative: float = 0.20,
    growth_threshold_absolute_cm3: float = 0.03,
    dilation_voxels: int = 1,
    min_lesion_volume_mm3: float = MIN_LESION_VOLUME_MM3,
) -> Dict:
    prev_img = nib.load(str(prev_path))
    curr_img = nib.load(str(curr_path))
    prev_data = np.asarray(prev_img.dataobj)
    curr_data = np.asarray(curr_img.dataobj)

    if prev_data.shape != curr_data.shape:
        raise ValueError(
            f"Shape mismatch comparing {prev_path.name} ({prev_data.shape}) "
            f"vs {curr_path.name} ({curr_data.shape}) — sessions are not on "
            f"the same grid; cannot diff without registration."
        )

    voxel_vol_cm3 = float(np.prod(np.abs(np.diag(curr_img.affine[:3, :3])))) / 1000.0
    min_cm3 = min_lesion_volume_mm3 / 1000.0
    structure = np.ones((3, 3, 3))

    def _volume_cm3(data: np.ndarray, label: int) -> float:
        return round(int((data == label).sum()) * voxel_vol_cm3, 4)

    prev_labels = sorted(
        int(l) for l in np.unique(prev_data) if l != 0 and _volume_cm3(prev_data, int(l)) >= min_cm3
    )
    curr_labels = sorted(
        int(l) for l in np.unique(curr_data) if l != 0 and _volume_cm3(curr_data, int(l)) >= min_cm3
    )
    prev_labels_set = set(prev_labels)

    lesions = []
    matched_prev_labels = set()

    for label in curr_labels:
        curr_mask = (curr_data == label)
        curr_vol = _volume_cm3(curr_data, label)

        dilated = curr_mask
        for _ in range(dilation_voxels):
            dilated = binary_dilation(dilated, structure=structure)

        overlapping_prev = sorted(
            int(l) for l in np.unique(prev_data[dilated]) if l != 0 and int(l) in prev_labels_set
        )

        if not overlapping_prev:
            lesions.append({
                "label": label, "status": "new",
                "volume_cm3": curr_vol, "previous_volume_cm3": 0.0,
            })
            continue

        matched_prev_labels.update(overlapping_prev)
        prev_vol = round(sum(_volume_cm3(prev_data, l) for l in overlapping_prev), 4)

        growth_abs = curr_vol - prev_vol
        growth_rel = (growth_abs / prev_vol) if prev_vol > 0 else float("inf")
        is_growing = (growth_rel >= growth_threshold_relative) or (growth_abs >= growth_threshold_absolute_cm3)
        status = "growing" if is_growing else "stable"

        lesions.append({
            "label": label, "status": status,
            "volume_cm3": curr_vol, "previous_volume_cm3": prev_vol,
        })

    for label in prev_labels:
        if label not in matched_prev_labels:
            lesions.append({
                "label": label, "status": "resolved",
                "volume_cm3": 0.0, "previous_volume_cm3": _volume_cm3(prev_data, label),
            })

    counts = {"new": 0, "growing": 0, "stable": 0, "resolved": 0}
    for lesion in lesions:
        counts[lesion["status"]] += 1

    return {
        "new_count": counts["new"],
        "growing_count": counts["growing"],
        "stable_count": counts["stable"],
        "resolved_count": counts["resolved"],
        "lesions": lesions,
    }
