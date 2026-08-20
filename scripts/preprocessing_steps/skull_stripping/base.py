"""
Skull stripper plugin contract.

Every skull stripping tool implements SkullStripperBase, so Stage 05 can
switch tools from config without knowing anything tool-specific. Per the
Этап 5.5 design spec (docs/superpowers/specs/2026-06-15-skull-stripping-
research-design.md §8), this is also the seam the future MAS agents plug
into: each tool ships a manifest describing its compute needs and
preferences, which a coordinator will later use for routing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import logging
import yaml

logger = logging.getLogger(__name__)

# Manifests live outside the scripts tree, next to the other service
# manifests, so a tool's metadata stays in one place whether it runs as a
# plugin today or as its own container later.
MANIFEST_ROOT = (
    Path(__file__).resolve().parents[3] / "services" / "skull-stripping"
)


class SkullStripperBase(ABC):
    """
    One skull stripping tool.

    Implementations do exactly one thing: given an input volume, produce a
    skull-stripped image and a binary brain mask. Applying that mask to the
    other modalities is the caller's job (see apply_brain_mask), because
    that step is identical for every tool.
    """

    #: short identifier, must match the manifest directory name
    name: str = "unnamed"

    @abstractmethod
    def strip(self,
              input_path: Path,
              output_path: Path,
              mask_path: Optional[Path] = None,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run brain extraction on a single volume.

        Returns a dict with at least:
            success: bool
            output_path: str          — skull-stripped image
            mask_path: str | None     — binary brain mask
            processing_time: float    — seconds
            error: str                — present only when success is False
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Whether this tool can actually run right now: binary installed,
        package importable, weights present. Checked before use so the
        dispatcher can fall back instead of failing mid-run.
        """
        raise NotImplementedError

    @property
    def manifest(self) -> Dict[str, Any]:
        """
        Tool metadata from services/skull-stripping/{name}/manifest.yaml.
        Missing manifest is not fatal — it only carries metadata (compute
        requirements, MAS routing hints), never behaviour.
        """
        manifest_path = MANIFEST_ROOT / self.name / "manifest.yaml"
        if not manifest_path.is_file():
            logger.debug("no manifest for skull stripper %r at %s",
                         self.name, manifest_path)
            return {}
        try:
            with manifest_path.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("could not read manifest for %r: %s", self.name, e)
            return {}


# ---------------------------------------------------------------------------
# Shared operation
# ---------------------------------------------------------------------------
# Applying an existing brain mask to a volume is identical no matter which
# tool produced the mask, so it lives here rather than in any one plugin.
# Carried over verbatim from the original skull_stripping.py.

import nibabel as nib  # noqa: E402


def apply_brain_mask(
    input_path: Path,
    mask_path: Path,
    output_path: Path
) -> dict:
    """
    Apply brain mask to an image.
    
    Args:
        input_path: Path to input NIfTI file
        mask_path: Path to brain mask file
        output_path: Path to save masked image
    
    Returns:
        dict: Information about masking operation
    """
    try:
        logger.info(f"Applying brain mask to {input_path.name}")
        
        # Load images
        img = nib.load(input_path)
        mask = nib.load(mask_path)
        
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
        
        # Check shapes match
        if img_data.shape != mask_data.shape:
            raise ValueError(
                f"Image shape {img_data.shape} does not match mask shape {mask_data.shape}"
            )
        
        # Apply mask (element-wise multiplication)
        masked_data = img_data * (mask_data > 0)
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save masked image with original header
        masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
        nib.save(masked_img, output_path)
        
        logger.info(f"Saved masked image to {output_path}")
        
        return {
            "success": True,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error applying mask to {input_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
