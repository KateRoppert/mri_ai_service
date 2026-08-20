"""
Skull stripping step for MRI preprocessing.

Replaces the former single-module skull_stripping.py with a plugin package
(Этап 5.5 spec §8): the tool is selected by `method:` in
preprocessing_config.yaml instead of being hardcoded, and each tool ships a
manifest describing its compute needs — the seam the planned MAS skull
stripping agents plug into.

Existing imports keep working:

    from preprocessing_steps.skull_stripping import (
        setup_fsl_environment, check_fsl_installed, process_subject_skull_stripping
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np

from .base import SkullStripperBase, apply_brain_mask
from .bet import (
    BetStripper,
    check_fsl_installed,
    get_bet_command,
    get_fsl_env,
    run_bet,
    setup_fsl_environment,
)
from .dispatcher import (
    STRIPPERS,
    SkullStripperUnavailable,
    get_stripper,
    get_tool_params,
)
from .hdbet import HdBetStripper

logger = logging.getLogger(__name__)

__all__ = [
    # plugin API
    "SkullStripperBase",
    "BetStripper",
    "HdBetStripper",
    "STRIPPERS",
    "SkullStripperUnavailable",
    "get_stripper",
    "get_tool_params",
    # backward-compatible surface of the original module
    "setup_fsl_environment",
    "get_fsl_env",
    "get_bet_command",
    "check_fsl_installed",
    "run_bet",
    "apply_brain_mask",
    "process_subject_skull_stripping",
    "compare_before_after_stripping",
]


def process_subject_skull_stripping(
    subject_dir: Path,
    output_dir: Path,
    transform_dir: Path,
    modalities: list,
    params: dict
) -> dict:
    """
    Process all modalities for a subject (skull stripping step).

    Workflow:
    1. Create brain mask on reference modality, using the configured tool
    2. Apply that mask to all other modalities

    Only step 1 is tool-specific; masking the remaining modalities is the
    same arithmetic whichever tool produced the mask.

    Args:
        subject_dir: Path to subject directory (BIDS structure)
        output_dir: Path to output directory for skull-stripped images
        transform_dir: Where the brain mask is stored
        modalities: List of modality suffixes to process
        params: Skull stripping parameters (see preprocessing_config.yaml)

    Returns:
        dict: Processing results for each modality
    """
    results: Dict[str, Any] = {}

    # Extract subject and session from path
    subject_id = subject_dir.parent.parent.name  # sub-XXX
    session_id = subject_dir.parent.name          # ses-XXX

    logger.info(f"Processing {subject_id}/{session_id} - Skull Stripping")

    reference_modality = params.get("reference_modality", "t1c")

    # Pick the tool before touching any files, so an unavailable tool falls
    # back (or fails) up front rather than halfway through a subject.
    try:
        stripper = get_stripper(params)
    except SkullStripperUnavailable as e:
        logger.error("Skull stripping unavailable: %s", e)
        return {"success": False, "error": str(e)}

    tool_params = get_tool_params(params, stripper)

    # Step 1: Create brain mask on reference modality
    logger.info(f"Step 1: Creating brain mask on {reference_modality} using {stripper.name}")

    ref_pattern = f"{subject_id}_{session_id}_{reference_modality}.nii.gz"
    ref_files = list(subject_dir.glob(ref_pattern))

    if not ref_files:
        error_msg = f"Reference modality {reference_modality} not found"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    ref_file = ref_files[0]

    # Output paths
    ref_output = output_dir / subject_id / session_id / "anat" / ref_pattern

    # Save brain mask to transformations directory
    mask_pattern = f"{subject_id}_{session_id}_brain_mask.nii.gz"
    mask_path = transform_dir / subject_id / session_id / "anat" / mask_pattern

    strip_result = stripper.strip(
        input_path=ref_file,
        output_path=ref_output,
        mask_path=mask_path,
        params=tool_params,
    )
    # Record which tool produced this, so a run's provenance survives a
    # later config change.
    strip_result.setdefault("method", stripper.name)

    results[reference_modality] = strip_result

    if not strip_result["success"]:
        logger.error(f"Failed to create brain mask on {reference_modality}")
        return results

    # Step 2: Apply mask to other modalities
    logger.info("Step 2: Applying brain mask to other modalities")

    apply_to_all = params.get("apply_to_all", True)

    if apply_to_all:
        for modality in modalities:
            if modality == reference_modality:
                continue  # Already processed

            modal_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
            modal_files = list(subject_dir.glob(modal_pattern))

            if not modal_files:
                logger.warning(f"Modality {modality} not found, skipping")
                results[modality] = {"success": False, "error": "File not found"}
                continue

            modal_file = modal_files[0]
            modal_output = output_dir / subject_id / session_id / "anat" / modal_pattern

            mask_result = apply_brain_mask(
                input_path=modal_file,
                mask_path=mask_path,
                output_path=modal_output
            )

            results[modality] = mask_result

    # Optional cleanup
    if params.get("cleanup", True):
        # Remove temporary BET files (e.g., _mesh files)
        cleanup_patterns = ["*_mesh.vtk", "*_skull.nii.gz", "*_outskin_mesh.off"]
        for pattern in cleanup_patterns:
            for temp_file in subject_dir.parent.rglob(pattern):
                try:
                    temp_file.unlink()
                    logger.debug(f"Cleaned up {temp_file}")
                except Exception:
                    pass

    return results


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
