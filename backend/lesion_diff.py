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

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

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


def coregistration_dice(
    prev_ref_path: Union[str, Path],
    curr_ref_path: Union[str, Path],
) -> Optional[float]:
    """Brain-overlap Dice between two sessions' preprocessed reference images.

    The longitudinal diff is only meaningful when the two sessions occupy the same
    anatomical space. Preprocessed images are skull-stripped (background exactly 0),
    so ``data > 0`` is the brain mask, and the Dice of the two brain masks measures
    how well the sessions are co-registered — independent of lesion biology.

    Returns:
        Dice in [0, 1]; 0.0 if the two grids differ (definitely not aligned);
        None if a file is missing/unreadable (cannot assess — caller should not
        block on this).
    """
    prev_ref_path, curr_ref_path = Path(prev_ref_path), Path(curr_ref_path)
    if not prev_ref_path.exists() or not curr_ref_path.exists():
        return None
    try:
        prev = np.asarray(nib.load(str(prev_ref_path)).dataobj)
        curr = np.asarray(nib.load(str(curr_ref_path)).dataobj)
    except Exception as e:
        logger.warning(f"coregistration_dice: failed to load references: {e}")
        return None

    if prev.shape != curr.shape:
        return 0.0

    prev_brain = prev > 0
    curr_brain = curr > 0
    denom = int(prev_brain.sum()) + int(curr_brain.sum())
    if denom == 0:
        return None
    inter = int((prev_brain & curr_brain).sum())
    return 2.0 * inter / denom


def _diff_cache_key(prev_path: Path, curr_path: Path, params: Dict) -> str:
    """Deterministic cache key over the two mask files and the diff parameters.

    Includes each file's mtime (ns), so an expert re-saving an edited mask
    invalidates the entry automatically without an explicit purge.
    """
    def sig(p: Path) -> str:
        st = Path(p).stat()
        return f"{p}:{st.st_mtime_ns}:{st.st_size}"

    payload = "|".join([
        sig(prev_path),
        sig(curr_path),
        f"gr={params.get('growth_threshold_relative')}",
        f"ga={params.get('growth_threshold_absolute_cm3')}",
        f"dv={params.get('dilation_voxels')}",
        f"mv={params.get('min_lesion_volume_mm3')}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_compare_labeled_masks(
    prev_path: Union[str, Path],
    curr_path: Union[str, Path],
    cache_dir: Union[str, Path],
    **kwargs,
) -> Dict:
    """compare_labeled_masks with a persistent JSON cache.

    The diff of two label masks is deterministic given the files and parameters,
    so it is computed once and reused. Loading full-resolution masks and running
    the per-label numpy passes is expensive (~seconds on 0.35 mm SibBMS volumes);
    without this cache every clinical-report open recomputed all session pairs
    from scratch.

    The cache key (see _diff_cache_key) embeds file mtimes, so an edited mask is
    picked up automatically. Writes are atomic (temp file + os.replace) so a
    concurrent reader never observes a half-written entry.
    """
    prev_path = Path(prev_path)
    curr_path = Path(curr_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _diff_cache_key(prev_path, curr_path, kwargs)
    cache_file = cache_dir / f"{key}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Ignoring corrupt diff cache {cache_file.name}: {e}")

    result = compare_labeled_masks(prev_path, curr_path, **kwargs)

    tmp_file = cache_file.with_suffix(".json.tmp")
    try:
        tmp_file.write_text(json.dumps(result), encoding="utf-8")
        os.replace(tmp_file, cache_file)
    except OSError as e:
        # A cache-write failure must not break the request — just log it.
        logger.warning(f"Failed to write diff cache {cache_file.name}: {e}")
    return result
