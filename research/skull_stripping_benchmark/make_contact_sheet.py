#!/usr/bin/env python3
"""
Contact sheet for visual review of cascade skull-stripping results.

Metrics alone cannot settle whether a mask is good: on the dropbox_33
calibration sample HD-BET and SynthStrip both scored clean on every gate,
yet the two tools disagree about where the brain ends. Calibration needs a
human verdict to compare the gates against, and reviewing volumes one at a
time in a viewer does not scale. This renders every subject as a row of
slices in one image, so a whole sample is reviewed in a single glance.

The background is the *skull-on* volume, not the stripped output. That
matters: leftover skull and cut-off brain are only visible against the
anatomy that was removed. The pipeline already saves what is needed —
`transformations/` keeps both the brain mask and the T1→atlas affine, so
the raw NIfTI can be pushed into atlas space in under a second per subject
instead of re-running Stage 05.

Usage:
    python make_contact_sheet.py \
        --run-dir   /path/to/run \
        --traces-dir /path/to/traces \
        --out       contact_sheet.png
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from atlas_space import DEFAULT_ATLAS, head_paths, to_atlas_space

logger = logging.getLogger(__name__)

# Axial levels sampled across the mask, as fractions of its z-extent. The
# extent is stretched downward first (see SUBMASK_MM) so the lowest panel
# lands near the orbits, where leftover eye tissue shows up.
AXIAL_FRACTIONS = (0.05, 0.25, 0.45, 0.62, 0.80)
SUBMASK_MM = 20


def outline(mask_slice: np.ndarray) -> np.ndarray:
    """One-voxel boundary of a 2D mask — drawn instead of a filled overlay
    so the anatomy underneath stays readable."""
    if not mask_slice.any():
        return mask_slice
    return mask_slice & ~ndimage.binary_erosion(mask_slice)


def pick_axial_levels(mask: np.ndarray) -> list:
    """Axial slice indices spanning the mask, reaching below it far enough
    to include the orbits."""
    zs = np.where(mask.any(axis=(0, 1)))[0]
    if zs.size == 0:
        return [mask.shape[2] // 2] * len(AXIAL_FRACTIONS)
    lo = max(0, int(zs.min()) - SUBMASK_MM)
    hi = int(zs.max())
    return [int(round(lo + f * (hi - lo))) for f in AXIAL_FRACTIONS]


def _panels(head: np.ndarray, mask: np.ndarray):
    """(image, mask) slice pairs for one subject: several axial levels plus
    a mid coronal and mid sagittal, which is where a cut hemisphere or a
    missing cerebellum is most obvious."""
    for z in pick_axial_levels(mask):
        yield head[:, :, z], mask[:, :, z]
    y = mask.shape[1] // 2
    yield head[:, y, :], mask[:, y, :]
    x = mask.shape[0] // 2
    yield head[x, :, :], mask[x, :, :]


def draw_row(axes, head: np.ndarray, mask: np.ndarray, label: str):
    for ax, (img, msk) in zip(axes, _panels(head, mask)):
        img = np.rot90(img)
        msk = np.rot90(msk).astype(bool)
        # Window on head voxels only; background zeros would otherwise drag
        # the low end down and wash out grey/white contrast.
        head_vox = img[img > 0]
        vmax = np.percentile(head_vox, 99.5) if head_vox.size else 1.0
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        edge = outline(msk)
        overlay = np.zeros(img.shape + (4,))
        overlay[edge] = (1.0, 0.15, 0.15, 1.0)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel(label, rotation=0, ha="right", va="center",
                       fontsize=8, labelpad=8, family="monospace")


def row_label(trace: dict) -> str:
    """Subject plus the facts a reviewer needs while looking at the row:
    which tool won, how big the mask is, and whether anything was flagged."""
    accepted = next((a for a in trace["attempts"] if a["decision"] == "accepted"), None)
    m = accepted["metrics"] if accepted else {}
    flags = trace["selected"].get("review_flags") or []
    switches = len(trace["attempts"]) - 1
    parts = [
        trace["subject"],
        trace["selected"]["tool"],
        f"{m.get('mask_volume_ml', 0):.0f} ml",
    ]
    if switches:
        parts.append(f"{switches} retry")
    if flags:
        parts.append("FLAG: " + ",".join(flags))
    return "\n".join(parts)


def cascade_rows(run_dir: Path, traces_dir: Path, atlas: Path, modality: str):
    """Rows for one cascade run: whichever mask the cascade actually shipped."""
    traces = sorted(traces_dir.glob("*.json"))
    if not traces:
        raise SystemExit(f"No traces in {traces_dir}")
    for trace_file in traces:
        trace = json.loads(trace_file.read_text())
        sub, ses = trace["subject"], trace["session"]
        raw, aff, msk = head_paths(run_dir, sub, ses, modality)
        missing = [p for p in (raw, aff, msk) if not p.exists()]
        if missing:
            logger.warning("%s: skipped, missing %s", sub, [p.name for p in missing])
            continue
        yield (row_label(trace), to_atlas_space(raw, aff, atlas),
               nib.load(str(msk)).get_fdata() > 0)
        logger.info("%s: rendered (%s)", sub, trace["selected"]["tool"])


def matrix_rows(matrix_dir: Path, tool: str, modality: str):
    """Rows for one tool out of the tool matrix.

    Reads the skull-on volume the matrix already wrote rather than resampling
    again — every tool in the matrix was given that exact file, so this shows
    the mask over the true input rather than a re-derived approximation.
    """
    rows_by_key = {}
    matrix_csv = matrix_dir / "tool_matrix.csv"
    if matrix_csv.exists():
        with matrix_csv.open() as handle:
            for row in csv.DictReader(handle):
                rows_by_key[(row["subject"], row["session"], row["tool"])] = row

    for mask_path in sorted((matrix_dir / "masks" / tool).glob("*_mask.nii.gz")):
        sub, ses = mask_path.name.split("_")[0], mask_path.name.split("_")[1]
        head_path = matrix_dir / "heads" / f"{sub}_{ses}_{modality}.nii.gz"
        if not head_path.exists():
            logger.warning("%s: no skull-on volume at %s", sub, head_path)
            continue
        yield (matrix_label(sub, tool, rows_by_key.get((sub, ses, tool))),
               nib.load(str(head_path)).get_fdata(),
               nib.load(str(mask_path)).get_fdata() > 0)
        logger.info("%s: rendered (%s)", sub, tool)


def matrix_label(subject: str, tool: str, row: dict) -> str:
    """Subject plus this tool's verdict — a reviewer comparing tools needs to
    see what the gate said about the very mask they are looking at."""
    if not row:
        return f"{subject}\n{tool}"
    parts = [subject, tool, f"{float(row['mask_volume_ml']):.0f} ml"]
    leak = float(row["leak_fraction"] or 0) * 100
    parts.append(f"leak {leak:.3f}%")
    if row["valid"] != "True":
        parts.append("REJECTED")
    elif row["review_flags"]:
        parts.append("FLAG: " + row["review_flags"].replace(";", ","))
    return "\n".join(parts)


def build(rows, out_path: Path, title: str, per_subject_dir: Path = None):
    rows = list(rows)
    if not rows:
        raise SystemExit("Nothing to render")

    ncols = len(AXIAL_FRACTIONS) + 2
    fig, axgrid = plt.subplots(len(rows), ncols,
                               figsize=(1.6 * ncols, 1.6 * len(rows)),
                               squeeze=False)
    for axes, (label, head, mask) in zip(axgrid, rows):
        draw_row(axes, head, mask, label)
    fig.suptitle(f"{title} ({len(rows)} subjects)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info("Contact sheet: %s", out_path)

    if per_subject_dir:
        per_subject_dir.mkdir(parents=True, exist_ok=True)
        for label, head, mask in rows:
            sub = label.split("\n")[0]
            f, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.4), squeeze=False)
            draw_row(axes[0], head, mask, label)
            f.suptitle(label.replace("\n", "  |  "), fontsize=11)
            f.tight_layout(rect=(0, 0, 1, 0.94))
            f.savefig(per_subject_dir / f"{sub}.png", dpi=110)
            plt.close(f)
        logger.info("Per-subject sheets: %s", per_subject_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path,
                   help="Pipeline run directory (holds nifti/ and transformations/)")
    p.add_argument("--traces-dir", type=Path,
                   help="Directory of *_cascade.json traces")
    p.add_argument("--matrix-dir", type=Path,
                   help="Tool-matrix directory (heads/, masks/, tool_matrix.csv)")
    p.add_argument("--tool", help="With --matrix-dir: which tool to render")
    p.add_argument("--out", type=Path, required=True, help="Output PNG")
    p.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    p.add_argument("--modality", default="t1",
                   help="Reference modality the mask was built on")
    p.add_argument("--per-subject-dir", type=Path, default=None,
                   help="Also write one full-size sheet per subject here")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if a.matrix_dir:
        if not a.tool:
            raise SystemExit("--matrix-dir needs --tool")
        rows = matrix_rows(a.matrix_dir, a.tool, a.modality)
        title = f"{a.tool} — mask over skull-on T1"
    else:
        if not (a.run_dir and a.traces_dir):
            raise SystemExit("Need --run-dir and --traces-dir (or --matrix-dir)")
        rows = cascade_rows(a.run_dir, a.traces_dir, a.atlas, a.modality)
        title = "Skull stripping — cascade output over skull-on T1"
    build(rows, a.out, title, a.per_subject_dir)


if __name__ == "__main__":
    main()
