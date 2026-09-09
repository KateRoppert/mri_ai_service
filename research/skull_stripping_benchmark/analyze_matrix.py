#!/usr/bin/env python3
"""
Build a defensible reference from the tool matrix and score the tools on it.

There is no manual ground truth for these subjects, and the two obvious
substitutes both fail:

  * the MNI atlas mask — measured 2026-09-07, it ranks a mask containing the
    patient's eyes *first* (DSC 0.997) and correct HD-BET masks at 0.84–0.87,
    because rigid registration cannot scale an atlas onto a head;
  * the cascade's own output — the cascade picked HD-BET for all twelve
    subjects, so scoring HD-BET against it returns 1.000 by construction.

What is left is agreement between independent tools. The reference for
scoring tool X is built from the *other* tools only, so no tool ever
contributes to its own reference. STAPLE is used rather than a plain majority
vote because it estimates each rater's sensitivity and specificity instead of
trusting them equally — which matters here, where at least one tool produces
frank garbage on some subjects.

The limitation has to be stated wherever these numbers are used: consensus is
biased toward what the tools agree on, so an error they all share (a sliver of
dura, say) becomes "truth". Only manual annotation removes that, and this
measures agreement, not accuracy.
"""

import argparse
import csv
import itertools
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    if total == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / total)


def load_masks(masks_dir: Path, subject: str, session: str, tools: list) -> dict:
    """Every tool's mask for one subject, as boolean arrays."""
    import nibabel as nib

    out = {}
    for tool in tools:
        path = masks_dir / tool / f"{subject}_{session}_mask.nii.gz"
        if path.exists():
            out[tool] = nib.load(str(path)).get_fdata() > 0
    return out


def staple_reference(masks: list) -> np.ndarray:
    """STAPLE consensus of several binary masks, thresholded at 0.5."""
    import SimpleITK as sitk

    images = [sitk.GetImageFromArray(m.astype(np.uint8)) for m in masks]
    probability = sitk.GetArrayFromImage(sitk.STAPLE(images, 1.0))
    return probability > 0.5


def analyse(matrix_dir: Path, out_dir: Path, tools: list):
    masks_dir = matrix_dir / "masks"
    rows = list(csv.DictReader((matrix_dir / "tool_matrix.csv").open()))
    subjects = sorted({(r["subject"], r["session"]) for r in rows})

    pairwise = defaultdict(list)
    loo_rows = []

    for subject, session in subjects:
        masks = load_masks(masks_dir, subject, session, tools)
        if len(masks) < 3:
            logger.warning("%s: only %d masks, skipped (leave-one-out needs 3 "
                           "others)", subject, len(masks))
            continue

        for left, right in itertools.combinations(sorted(masks), 2):
            pairwise[(left, right)].append(dice(masks[left], masks[right]))

        for tool in sorted(masks):
            others = [m for name, m in masks.items() if name != tool]
            if len(others) < 2:
                continue
            reference = staple_reference(others)
            loo_rows.append({
                "subject": subject,
                "session": session,
                "tool": tool,
                "dsc_vs_loo_consensus": round(dice(masks[tool], reference), 4),
                "reference_ml": round(float(reference.sum()) / 1000.0, 1),
                "n_reference_tools": len(others),
            })
        logger.info("%s: %d tools scored", subject, len(masks))

    out_dir.mkdir(parents=True, exist_ok=True)
    loo_path = out_dir / "consensus_dsc.csv"
    with loo_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(loo_rows[0]))
        writer.writeheader()
        writer.writerows(loo_rows)
    logger.info("%d rows -> %s", len(loo_rows), loo_path)

    print("\nDSC против консенсуса остальных инструментов (leave-one-out)")
    print("-" * 58)
    by_tool = defaultdict(list)
    for row in loo_rows:
        by_tool[row["tool"]].append(row["dsc_vs_loo_consensus"])
    for tool, values in sorted(by_tool.items(), key=lambda kv: -np.mean(kv[1])):
        arr = np.array(values)
        print(f"{tool:<12} среднее {arr.mean():.3f}   медиана {np.median(arr):.3f}"
              f"   мин {arr.min():.3f}   n={len(arr)}")

    print("\nПопарное согласие инструментов (средний DSC)")
    print("-" * 58)
    for (left, right), values in sorted(pairwise.items(),
                                        key=lambda kv: -np.mean(kv[1])):
        print(f"{left:<12} vs {right:<12} {np.mean(values):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tools", nargs="+",
                        default=["hdbet", "synthstrip", "bet", "mni_mask"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    analyse(args.matrix_dir, args.out_dir, args.tools)


if __name__ == "__main__":
    main()
