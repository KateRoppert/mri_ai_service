"""
Per-lesion connected-component statistics for binary lesion masks (MS).

Counts individual lesions (connected components), computes per-lesion and
total burden volumes. Used by Stage 08 for multiple_sclerosis masks, where
each connected component corresponds to one clinically distinct lesion
(unlike GBM, which is analyzed as lobar regions of a single tumor mass).
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label

# Components smaller than this are treated as noise for counting purposes —
# they are excluded from lesion_count, the per-lesion list, and hover lookup.
# They still contribute to total_volume_cm3 (full lesion burden). 5 mm³ matches
# what is visually counted as a discrete lesion (sub-visible 3-4 voxel specks
# are dropped); see KI-037 for the upstream determinism this mitigates.
MIN_LESION_VOLUME_MM3 = 5.0


def compute_lesion_stats(mask_path: Path) -> Tuple[dict, np.ndarray, np.ndarray, set]:
    """
    Count connected components (individual lesions) in a binary mask.
    Used for MS where each component = one lesion.

    Components below MIN_LESION_VOLUME_MM3 are treated as noise: excluded from
    lesion_count / lesion_volumes_cm3 / lesion_volumes_by_label, but their
    volume is still included in total_volume_cm3 (full burden).

    Returns (stats_dict, labeled_array, affine, kept_labels):
      stats_dict: lesion_count, total_volume_cm3, mean_lesion_volume_cm3,
                  lesion_volumes_cm3 (sorted desc, kept lesions, for display/table),
                  lesion_volumes_by_label ({str(label): volume_cm3}, kept, for hover).
      labeled_array: int array, EVERY component its own integer label 1..N
                     (not filtered — the saved mask keeps all blobs; only the
                     stats/hover map drop sub-threshold ones).
      affine: source affine (to save the labeled mask).
      kept_labels: set[int] of label IDs that passed the volume filter — the
                   single source of truth for "what counts as a lesion," shared
                   with MSZoneAnalyzer so the sibling *_mcdonald_report.json
                   agrees exactly on lesion counts and label numbering.
    """
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)
    voxel_vol_mm3 = float(np.prod(np.abs(np.diag(img.affine[:3, :3]))))
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
    min_cm3 = MIN_LESION_VOLUME_MM3 / 1000.0

    binary = (data > 0).astype(np.uint8)
    labeled, n_components = ndimage_label(binary)

    all_volumes = {}
    for i in range(1, n_components + 1):
        voxel_count = int(np.sum(labeled == i))
        all_volumes[str(i)] = round(voxel_count * voxel_vol_cm3, 4)

    total = round(sum(all_volumes.values()), 4)  # full burden, unfiltered

    kept_by_label = {lbl: v for lbl, v in all_volumes.items() if v >= min_cm3}
    kept_volumes = sorted(kept_by_label.values(), reverse=True)
    count = len(kept_volumes)
    mean = round(sum(kept_volumes) / count, 4) if count > 0 else 0.0

    stats = {
        "lesion_count": count,
        "total_volume_cm3": total,
        "mean_lesion_volume_cm3": mean,
        "lesion_volumes_cm3": kept_volumes,
        "lesion_volumes_by_label": kept_by_label,
    }
    kept_labels = {int(lbl) for lbl in kept_by_label}
    return stats, labeled.astype(np.int16), img.affine, kept_labels
