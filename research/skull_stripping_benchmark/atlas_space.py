"""
Rebuild the skull-on volume a Stage 05 mask was made from.

Stage 05 strips the brain *after* registration, so its mask lives in atlas
space while the only skull-on copy of the patient is the raw NIfTI in native
space. Anything that has to see what the stripper removed — visual review,
leakage metrics, running a second tool on the same input — needs the two in
the same space.

Re-running Stage 05 to get there costs minutes per subject. It is not
necessary: `transformations/` already holds the T1→atlas affine, so the raw
NIfTI resamples into atlas space in well under a second. Reorientation
earlier in the stage permutes the array but preserves world coordinates, and
ANTs transforms act in world space, so the saved affine applies to the
untouched NIfTI just as well as to the reoriented one.

Verified on the dropbox_33 calibration sample: after the resample, 0.0000 of
the brain mask lands on background, and mean intensity inside the mask is
roughly double the head tissue outside it.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS = PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz"


def head_paths(run_dir: Path, subject: str, session: str, modality: str = "t1"):
    """Where Stage 05 leaves the three files this module needs."""
    anat = f"{subject}/{session}/anat"
    return (
        run_dir / "nifti" / anat / f"{subject}_{session}_{modality}.nii.gz",
        run_dir / "transformations" / anat / f"{subject}_{session}_{modality}_to_atlas.mat",
        run_dir / "transformations" / anat / f"{subject}_{session}_brain_mask.nii.gz",
    )


def to_atlas_space(raw_path: Path, affine_path: Path,
                   atlas_path: Path = DEFAULT_ATLAS, out_path: Path = None):
    """Resample a native-space volume into atlas space using a saved affine.

    Returns the array. Pass ``out_path`` to also write it — tools that take a
    file path (every stripper CLI) need it on disk.
    """
    import ants  # heavy import, only this path needs it

    moved = ants.apply_transforms(
        fixed=ants.image_read(str(atlas_path)),
        moving=ants.image_read(str(raw_path)),
        transformlist=[str(affine_path)],
        interpolator="linear",
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(moved, str(out_path))
    return moved.numpy()


def alignment_ok(head: np.ndarray, mask: np.ndarray) -> float:
    """Share of mask voxels that landed on background — a sanity check the
    caller can assert on. Should be ~0; anything larger means the affine and
    the mask are not from the same subject or the same stage."""
    if not mask.any():
        return 1.0
    return float((head[mask] == 0).mean())
