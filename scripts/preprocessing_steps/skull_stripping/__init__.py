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
from typing import Any, Dict, Optional

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
    build_cascade_order,
    get_stripper,
    get_tool_params,
    try_stripper,
)
from .hdbet import HdBetStripper
from .synthstrip import SynthStripStripper
from .gpu_pool import acquire_device, resolve_devices
from .validation import (
    cascade_decision_message,
    format_mask_check_line,
    mask_metrics,
    validate_mask,
    validation_gate_banner,
)

logger = logging.getLogger(__name__)

__all__ = [
    # plugin API
    "SkullStripperBase",
    "BetStripper",
    "HdBetStripper",
    "SynthStripStripper",
    "STRIPPERS",
    "SkullStripperUnavailable",
    "get_stripper",
    "get_tool_params",
    "build_cascade_order",
    "try_stripper",
    "validate_mask",
    "mask_metrics",
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
    params: dict,
    gpu_pool=None,
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

    # Step 1: Create brain mask on reference modality
    order = build_cascade_order(params)
    vcfg = dict(params.get("validation") or {})
    validation_enabled = vcfg.get("enabled", True)
    logger.info(
        "Step 1: Creating brain mask on %s; cascade %s",
        reference_modality,
        order,
    )
    logger.info("%s", validation_gate_banner(vcfg))

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

    extra_gpu: Dict[str, Any] = {}
    if "disable_tta" in params:
        extra_gpu["disable_tta"] = params["disable_tta"]

    strip_result: Optional[Dict[str, Any]] = None
    used: Optional[str] = None
    accepted = False

    for i, name in enumerate(order, start=1):
        remaining = order[i:]
        cand = try_stripper(name)
        if cand is None:
            logger.info(
                "Cascade attempt %d/%d: skip %r (unknown or unavailable); remaining %s",
                i,
                len(order),
                name,
                remaining,
            )
            continue

        logger.info("Cascade attempt %d/%d: running %r", i, len(order), name)
        tool_params = get_tool_params(params, cand)

        def _run_strip(tool, tool_params_local):
            return tool.strip(
                input_path=ref_file,
                output_path=ref_output,
                mask_path=mask_path,
                params=tool_params_local,
            )

        if getattr(cand, "uses_gpu", False):
            # Pin GPU tools only for this candidate. CPU fallback must not
            # hold a pool slot (Stage 05 serialises GPU jobs through it).
            if gpu_pool is not None:
                with acquire_device(gpu_pool) as device:
                    logger.info("Skull stripping (%s) on device %s", cand.name, device)
                    strip_result = _run_strip(
                        cand, {**tool_params, "device": device, **extra_gpu}
                    )
            else:
                device = resolve_devices(params)[0]
                logger.info("Skull stripping (%s) on device %s", cand.name, device)
                strip_result = _run_strip(
                    cand, {**tool_params, "device": device, **extra_gpu}
                )
        else:
            strip_result = _run_strip(cand, tool_params)

        strip_result.setdefault("method", cand.name)
        used = cand.name

        if not strip_result.get("success"):
            logger.info(
                "%s",
                cascade_decision_message(
                    name=name,
                    accepted=False,
                    remaining=remaining,
                    strip_failed=True,
                    validation_enabled=validation_enabled,
                ),
            )
            continue

        if not validation_enabled:
            logger.info(
                "%s",
                cascade_decision_message(
                    name=name,
                    accepted=True,
                    remaining=remaining,
                    validation_enabled=False,
                ),
            )
            accepted = True
            break

        check = validate_mask(mask_path, **vcfg)
        strip_result["mask_validation"] = check
        logger.info("Cascade %r mask metrics: %s", name, format_mask_check_line(check))
        if check["valid"]:
            logger.info(
                "%s",
                cascade_decision_message(
                    name=name,
                    accepted=True,
                    remaining=remaining,
                    check=check,
                    validation_enabled=True,
                ),
            )
            accepted = True
            break
        logger.info(
            "%s",
            cascade_decision_message(
                name=name,
                accepted=False,
                remaining=remaining,
                check=check,
                validation_enabled=True,
            ),
        )

    if not accepted:
        error = f"No usable stripper in cascade {order}"
        logger.error("Failed to create brain mask on %s: %s", reference_modality, error)
        payload: Dict[str, Any] = {
            "success": False,
            "error": error,
            "method": used,
        }
        if strip_result:
            payload = {**strip_result, **payload}
        results[reference_modality] = payload
        return results

    logger.info("Skull stripper used: %s (cascade %s)", used, order)
    results[reference_modality] = strip_result

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
