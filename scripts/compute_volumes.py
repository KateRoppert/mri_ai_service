# -*- coding: utf-8 -*-
"""
Compute tumor volumes from multi-class segmentation masks.

Reads a NIfTI segmentation mask, counts voxels per class,
computes volumes in mm³ and cm³ using voxel dimensions from the header,
and saves a plain-text report next to the mask.

Label mapping (BraTS / Siberian convention):
  0 — Background
  1 — NCR (Necrotic core)
  2 — ED  (Edema)
  3 — NET (Non-enhancing tumor)
  4 — ET  (Enhancing tumor)

Usage as standalone:
  python compute_volumes.py /path/to/mask_segmask.nii.gz

Usage as library (from 06_segmentation.py):
  from compute_volumes import compute_and_save_volume_report
  report_path = compute_and_save_volume_report(mask_path)
"""

import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import nibabel as nib
except ImportError:
    raise ImportError(
        "nibabel is required for volume computation. "
        "Install it with: pip install nibabel"
    )

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Label definitions
# ------------------------------------------------------------------
LABEL_MAP = {
    1: "NCR (Necrotic core)",
    2: "ED (Edema)",
    3: "NET (Non-enhancing tumor)",
    4: "ET (Enhancing tumor)",
}


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------

def compute_volumes(mask_path: Path) -> dict:
    """
    Compute per-class and total tumor volumes from a segmentation mask.

    Args:
        mask_path: Path to *_segmask.nii.gz file.

    Returns:
        Dictionary with keys:
          - mask_file        : str, filename of the mask
          - voxel_size_mm    : tuple(float, float, float)
          - voxel_volume_mm3 : float
          - classes          : dict[int, dict] with voxel_count, volume_mm3, volume_cm3
          - total_tumor      : dict with voxel_count, volume_mm3, volume_cm3
    """
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj, dtype=np.int16)
    header = img.header

    # Voxel dimensions in mm (first 3 elements of pixdim)
    pixdim = header.get_zooms()[:3]
    voxel_vol_mm3 = float(np.prod(pixdim))

    results = {
        "mask_file": mask_path.name,
        "voxel_size_mm": tuple(round(float(d), 4) for d in pixdim),
        "voxel_volume_mm3": round(voxel_vol_mm3, 6),
        "classes": {},
        "total_tumor": {},
    }

    for label, name in LABEL_MAP.items():
        count = int(np.sum(data == label))
        vol_mm3 = count * voxel_vol_mm3
        vol_cm3 = vol_mm3 / 1000.0
        results["classes"][label] = {
            "name": name,
            "voxel_count": count,
            "volume_mm3": round(vol_mm3, 2),
            "volume_cm3": round(vol_cm3, 4),
        }

    # Clinical metrics computed AFTER the loop so all classes are populated.
    # ET (label 4) was missing on the last iteration when clinical was inside
    # the loop, causing CE+ to be always 0 (bug).
    #
    # RANO / BraTS nomenclature:
    #   ET  (label 4) = Enhancing Tumor = CE+
    #   NCR (label 1) + NET (label 3) = CE-
    #   Tumor Core (TC) = ET + NCR + NET  — the clinical "tumor volume"
    #   ED  (label 2) = Peritumoral edema — reported separately, not in TC
    #
    # "Total tumor" here means TC (without edema), consistent with RANO and
    # with what the frontend shows as "Суммарная опухоль".
    def _v(label_int):
        return results["classes"].get(label_int, {}).get("voxel_count", 0)

    et_vox  = _v(4)   # ET  — CE+
    ncr_vox = _v(1)   # NCR — CE- component
    net_vox = _v(3)   # NET — CE- component
    ed_vox  = _v(2)   # ED  — edema, separate metric

    ce_neg_vox      = ncr_vox + net_vox
    tumor_core_vox  = et_vox + ce_neg_vox   # TC = ET+NCR+NET

    results["clinical"] = {
        "ce_positive": {
            "name": "CE+ (Enhancing)",
            "name_ru": "Контраст-позитивная (CE+)",
            "voxel_count": et_vox,
            "volume_mm3": round(et_vox * voxel_vol_mm3, 2),
            "volume_cm3": round(et_vox * voxel_vol_mm3 / 1000, 4),
        },
        "ce_negative": {
            "name": "CE− (Non-enhancing)",
            "name_ru": "Контраст-негативная (CE−)",
            "voxel_count": ce_neg_vox,
            "volume_mm3": round(ce_neg_vox * voxel_vol_mm3, 2),
            "volume_cm3": round(ce_neg_vox * voxel_vol_mm3 / 1000, 4),
        },
        "tumor_core": {
            "name": "Total tumor (CE+ + CE−)",
            "name_ru": "Суммарная опухоль (CE+ + CE−)",
            "voxel_count": tumor_core_vox,
            "volume_mm3": round(tumor_core_vox * voxel_vol_mm3, 2),
            "volume_cm3": round(tumor_core_vox * voxel_vol_mm3 / 1000, 4),
        },
        "edema": {
            "name": "Peritumoral edema",
            "name_ru": "Перитуморальный отёк",
            "voxel_count": ed_vox,
            "volume_mm3": round(ed_vox * voxel_vol_mm3, 2),
            "volume_cm3": round(ed_vox * voxel_vol_mm3 / 1000, 4),
        },
    }

    # total_tumor = Tumor Core (TC), without edema — this is what RANO
    # and the frontend refer to as "Суммарная опухоль".
    tc_mm3 = tumor_core_vox * voxel_vol_mm3
    results["total_tumor"] = {
        "voxel_count": tumor_core_vox,
        "volume_mm3": round(tc_mm3, 2),
        "volume_cm3": round(tc_mm3 / 1000.0, 4),
    }

    return results


# ------------------------------------------------------------------
# Report formatting
# ------------------------------------------------------------------

def format_report(volumes: dict) -> str:
    """
    Format volume data as a human-readable plain-text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("TUMOR VOLUME REPORT")
    lines.append("=" * 60)
    lines.append(f"Mask file:    {volumes['mask_file']}")
    lines.append(f"Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Voxel size:   {volumes['voxel_size_mm'][0]:.3f} x "
                 f"{volumes['voxel_size_mm'][1]:.3f} x "
                 f"{volumes['voxel_size_mm'][2]:.3f} mm")
    lines.append(f"Voxel volume: {volumes['voxel_volume_mm3']:.4f} mm³")
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"{'Class':<35} {'Voxels':>8}  {'mm³':>12}  {'cm³':>10}")
    lines.append("-" * 60)

    for label in sorted(volumes["classes"]):
        cls = volumes["classes"][label]
        lines.append(
            f"  {label}. {cls['name']:<31} {cls['voxel_count']:>8}  "
            f"{cls['volume_mm3']:>12.2f}  {cls['volume_cm3']:>10.4f}"
        )

    lines.append("-" * 60)
    total = volumes["total_tumor"]
    lines.append(
        f"  {'TUMOR CORE (NCR+NET+ET)':<34} {total['voxel_count']:>8}  "
        f"{total['volume_mm3']:>12.2f}  {total['volume_cm3']:>10.4f}"
    )

    # Clinical summary
    if "clinical" in volumes:
        lines.append("")
        lines.append("CLINICAL SUMMARY (RANO):")
        lines.append("-" * 60)
        for key in ["ce_positive", "ce_negative", "tumor_core", "edema"]:
            cls = volumes["clinical"][key]
            lines.append(
                f"  {cls['name']:<34} {cls['voxel_count']:>8}  "
                f"{cls['volume_mm3']:>12.2f}  {cls['volume_cm3']:>10.4f}"
            )
        lines.append("-" * 60)
        
    lines.append("=" * 60)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Save report
# ------------------------------------------------------------------

def compute_and_save_volume_report(
    mask_path: Path,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Compute volumes and save a text report.

    Args:
        mask_path:   Path to *_segmask.nii.gz
        output_path: Where to save the report.
                     Default: same directory as mask, replacing
                     '_segmask.nii.gz' with '_volume_report.txt'.

    Returns:
        Path to saved report, or None on error.
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        logger.error(f"Mask file not found: {mask_path}")
        return None

    if output_path is None:
        report_name = mask_path.name.replace("_segmask.nii.gz", "_volume_report.txt")
        output_path = mask_path.parent / report_name

    try:
        logger.info(f"Computing volumes for: {mask_path.name}")
        volumes = compute_volumes(mask_path)
        report_text = format_report(volumes)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

        # Сохраняем JSON-версию рядом с текстовой
        json_path = output_path.with_suffix(".json")
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(volumes, f, ensure_ascii=False, indent=2)
        logger.info(f"Volume report JSON saved: {json_path}")

        logger.info(f"Volume report saved: {output_path}")
        logger.info(
            f"  Total tumor volume: {volumes['total_tumor']['volume_cm3']:.4f} cm³ "
            f"({volumes['total_tumor']['voxel_count']} voxels)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Failed to compute volumes for {mask_path.name}: {e}")
        logger.exception("Full traceback:")
        return None


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute tumor volumes from a segmentation mask."
    )
    parser.add_argument(
        "mask_path",
        type=Path,
        help="Path to *_segmask.nii.gz file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output report path (default: next to mask as *_volume_report.txt)",
    )
    args = parser.parse_args()

    # Basic logging for standalone use
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    result = compute_and_save_volume_report(args.mask_path, args.output)

    if result:
        # Print report to stdout as well
        print()
        print(result.read_text(encoding="utf-8"))
        sys.exit(0)
    else:
        sys.exit(1)