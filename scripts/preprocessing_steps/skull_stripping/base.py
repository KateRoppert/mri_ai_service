"""Shared base class, manifest loading, and modality-masking helpers for skull strippers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# services/skull-stripping/<name>/manifest.yaml relative to project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_ROOT = _PROJECT_ROOT / "services" / "skull-stripping"


def load_manifest(name: str) -> dict:
    """Load services/skull-stripping/<name>/manifest.yaml; return {} if absent."""
    manifest_path = _MANIFEST_ROOT / name / "manifest.yaml"
    if not manifest_path.exists():
        logger.debug(f"Manifest not found for '{name}': {manifest_path}")
        return {}
    with open(manifest_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_gpu_memory_mb() -> float:
    """Currently used VRAM in MB across GPU 0, or 0.0 if pynvml/GPU unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 2)
    except Exception:
        return 0.0


def apply_brain_mask(input_path: Path, mask_path: Path, output_path: Path) -> dict:
    """Multiply an image by a binary mask and save it (preserves header/affine)."""
    try:
        logger.info(f"Applying brain mask to {input_path.name}")
        img = nib.load(input_path)
        mask = nib.load(mask_path)
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
        if img_data.shape != mask_data.shape:
            raise ValueError(
                f"Image shape {img_data.shape} != mask shape {mask_data.shape}"
            )
        masked_data = img_data * (mask_data > 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(masked_data, img.affine, img.header), output_path)
        logger.info(f"Saved masked image to {output_path}")
        return {"success": True, "output_path": str(output_path)}
    except Exception as e:
        logger.error(f"Error applying mask to {input_path.name}: {e}")
        return {"success": False, "error": str(e)}


def compare_before_after_stripping(original_path: Path, stripped_path: Path) -> None:
    """
    Compare statistics before and after skull stripping.

    Args:
        original_path: Path to original NIfTI file
        stripped_path: Path to skull-stripped NIfTI file
    """
    print("=" * 70)
    print("SKULL STRIPPING COMPARISON REPORT")
    print("=" * 70)

    # Load images
    orig_img = nib.load(original_path)
    strip_img = nib.load(stripped_path)

    orig_data = orig_img.get_fdata()
    strip_data = strip_img.get_fdata()

    # Calculate statistics
    orig_nonzero = np.sum(orig_data > 0)
    strip_nonzero = np.sum(strip_data > 0)

    orig_volume_voxels = orig_nonzero
    strip_volume_voxels = strip_nonzero

    # Calculate voxel volume in mm³
    voxel_dims = orig_img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_dims)

    orig_volume_mm3 = orig_volume_voxels * voxel_volume_mm3
    strip_volume_mm3 = strip_volume_voxels * voxel_volume_mm3

    removed_volume_mm3 = orig_volume_mm3 - strip_volume_mm3
    removed_percent = (removed_volume_mm3 / orig_volume_mm3) * 100

    print(f"\nOriginal image: {original_path.name}")
    print(f"  Non-zero voxels: {orig_nonzero:,}")
    print(f"  Volume: {orig_volume_mm3:,.0f} mm³")

    print(f"\nSkull-stripped image: {stripped_path.name}")
    print(f"  Non-zero voxels: {strip_nonzero:,}")
    print(f"  Volume: {strip_volume_mm3:,.0f} mm³")

    print("\n" + "-" * 70)
    print("CHANGES:")
    print("-" * 70)
    print(f"  Removed volume: {removed_volume_mm3:,.0f} mm³ ({removed_percent:.1f}%)")

    # Check if reasonable
    if 20 <= removed_percent <= 50:
        print(f"  ✓ Removal percentage looks reasonable (20-50%)")
    elif removed_percent < 20:
        print(f"  ⚠ Warning: Low removal percentage, skull might not be fully removed")
    else:
        print(f"  ⚠ Warning: High removal percentage, brain might be over-stripped")

    # Brain tissue statistics
    orig_brain = orig_data[strip_data > 0]  # Original intensities in brain region
    strip_brain = strip_data[strip_data > 0]  # Stripped intensities

    print("\n" + "-" * 70)
    print("BRAIN TISSUE STATISTICS:")
    print("-" * 70)
    print(f"  Mean intensity (original): {np.mean(orig_brain):.2f}")
    print(f"  Mean intensity (stripped): {np.mean(strip_brain):.2f}")
    print(f"  Std intensity (original):  {np.std(orig_brain):.2f}")
    print(f"  Std intensity (stripped):  {np.std(strip_brain):.2f}")

    print("=" * 70)


class SkullStripperBase(ABC):
    """Contract every skull stripping tool implements.

    strip() returns:
        {success, output_path, mask_path, processing_time, vram_used_gb, error?}
    """

    name: str = "base"

    @abstractmethod
    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    def manifest(self) -> dict:
        return load_manifest(self.name)
