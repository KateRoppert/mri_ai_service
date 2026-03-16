"""
Registration step for MRI preprocessing.
Uses ANTsPy for rigid registration to SRI24 atlas and cross-modality registration.
"""

import logging
from pathlib import Path
import ants
import nibabel as nib
import numpy as np
import urllib.request
import shutil
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

def load_ants_image(path: Path):
    """
    Load image with ANTsPy (compatible with different versions).
    """
    try:
        # Try newer API
        return ants.image_read(str(path))
    except AttributeError:
        # Try older API
        try:
            nib_img = nib.load(str(path))
            return ants.from_nibabel(nib_img)
        except:
            # Last resort: use ANTsImage constructor
            return ants.ANTsImage(str(path))


def save_ants_image(image, path: Path):
    """
    Save image with ANTsPy (compatible with different versions).
    """
    try:
        # Try newer API
        ants.image_write(image, str(path))
    except AttributeError:
        # Try older API
        image.to_file(str(path))


def download_sri24_atlas(cache_dir: Path, atlas_url: str, atlas_filename: str) -> Path:
    """
    Download SRI24 atlas if not already cached.
    
    Args:
        cache_dir: Directory to cache the atlas
        atlas_url: URL to download atlas from
        atlas_filename: Filename for the atlas
    
    Returns:
        Path: Path to the atlas file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    atlas_path = cache_dir / atlas_filename
    
    if atlas_path.exists():
        logger.info(f"Atlas already cached at {atlas_path}")
        return atlas_path
    
    logger.info(f"Downloading SRI24 atlas from {atlas_url}")
    logger.info(f"Saving to {atlas_path}")
    
    try:
        # Download with progress
        with urllib.request.urlopen(atlas_url) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(atlas_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Atlas downloaded successfully to {atlas_path}")
        return atlas_path
        
    except Exception as e:
        logger.error(f"Failed to download atlas: {str(e)}")
        if atlas_path.exists():
            atlas_path.unlink()  # Remove partial download
        raise


def register_to_atlas(
    moving_path: Path,
    atlas_path: Path,
    output_path: Path,
    transform_path: Path,
    registration_type: str = "Rigid",
    output_resolution: list = [1.0, 1.0, 1.0],
    metric: str = "MI",
    metric_weight: float = 1.0,
    number_of_iterations: int = 1000,
    convergence_threshold: float = 1e-6,
    convergence_window_size: int = 10,
    smoothing_sigmas: list = [3, 2, 1, 0],
    shrink_factors: list = [8, 4, 2, 1],
    use_histogram_matching: bool = True
) -> dict:
    """
    Register moving image to atlas using ANTsPy.
    
    Args:
        moving_path: Path to moving image (will be registered to atlas)
        atlas_path: Path to fixed atlas image
        output_path: Path to save registered image
        transform_path: Path to save transformation matrix
        registration_type: Type of registration (default: "Rigid" for 6 DOF)
        output_resolution: Output resolution in mm (default: [1.0, 1.0, 1.0])
        metric: Similarity metric (default: "MI" - Mutual Information)
        metric_weight: Weight for the metric (default: 1.0)
        number_of_iterations: Number of iterations (default: 1000)
        convergence_threshold: Convergence threshold (default: 1e-6)
        convergence_window_size: Window size for convergence (default: 10)
        smoothing_sigmas: Smoothing sigmas for multi-resolution (default: [3,2,1,0])
        shrink_factors: Shrink factors for multi-resolution (default: [8,4,2,1])
        use_histogram_matching: Use histogram matching (default: True)
    
    Returns:
        dict: Information about the registration
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Registering {moving_path.name} to atlas")
        
        # Load images with ANTs
        fixed = load_ants_image(str(atlas_path))
        moving = load_ants_image(str(moving_path))
        
        logger.debug(f"Fixed (atlas) shape: {fixed.shape}, spacing: {fixed.spacing}")
        logger.debug(f"Moving image shape: {moving.shape}, spacing: {moving.spacing}")
        
        # Perform registration
        logger.info(f"Performing {registration_type} registration with {metric} metric")
        
        registration_result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=registration_type,
            grad_step=0.1,
            flow_sigma=3,
            total_sigma=0,
            aff_metric=metric,
            aff_sampling=32,
            syn_metric=metric,
            syn_sampling=32,
            reg_iterations=(number_of_iterations,) * len(shrink_factors),
            aff_iterations=(number_of_iterations,) * len(shrink_factors),
            aff_shrink_factors=tuple(shrink_factors),
            aff_smoothing_sigmas=tuple(smoothing_sigmas),
            verbose=False
        )
        
        # Get registered image
        warped_moving = registration_result['warpedmovout']
        
        # Resample to desired resolution
        if output_resolution != list(warped_moving.spacing):
            logger.info(f"Resampling to {output_resolution} mm isotropic")
            warped_moving = ants.resample_image(
                warped_moving,
                resample_params=output_resolution,
                use_voxels=False,
                interp_type=1  # Linear interpolation
            )
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save registered image
        save_ants_image(warped_moving, str(output_path))
        logger.info(f"Saved registered image to {output_path}")
        
        # Save transformation matrix
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ANTs saves transforms in the registration result
        # For rigid registration, we have one affine transform
        forward_transforms = registration_result['fwdtransforms']
        
        if forward_transforms:
            # Copy the transform file to our desired location
            transform_file = Path(forward_transforms[0])
            shutil.copy(transform_file, transform_path)
            logger.info(f"Saved transformation to {transform_path}")
        else:
            logger.warning("No forward transforms found in registration result")
        
        processing_time = time.time() - start_time
        logger.info(f"Registration completed in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "transform_path": str(transform_path),
            "output_path": str(output_path),
            "processing_time": processing_time,
            "metric_value": registration_result.get('metric_value', None)
        }
        
    except Exception as e:
        logger.error(f"Error during registration of {moving_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def apply_transform(
    moving_path: Path,
    fixed_path: Path,
    transform_path: Path,
    output_path: Path,
    interpolation: str = "Linear"
) -> dict:
    """
    Apply existing transformation to an image.
    
    Args:
        moving_path: Path to image to transform
        fixed_path: Path to reference (fixed) image
        transform_path: Path to transformation file
        output_path: Path to save transformed image
        interpolation: Interpolation method (default: "Linear")
    
    Returns:
        dict: Information about the transformation application
    """
    try:
        logger.info(f"Applying transform to {moving_path.name}")
        
        # Load images
        fixed = load_ants_image(str(fixed_path))
        moving = load_ants_image(str(moving_path))
        
        # Set interpolation type
        interp_map = {
            "Linear": 1,
            "NearestNeighbor": 0,
            "BSpline": 3,
            "Gaussian": 4
        }
        interp_type = interp_map.get(interpolation, 1)
        
        # Apply transformation
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[str(transform_path)],
            interpolator=interpolation.lower(),
            whichtoinvert=[False]
        )
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save transformed image
        save_ants_image(warped, str(output_path))
        logger.info(f"Saved transformed image to {output_path}")
        
        return {
            "success": True,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error applying transform to {moving_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def register_modalities(
    reference_path: Path,
    moving_path: Path,
    output_path: Path,
    transform_path: Path,
    registration_type: str = "Rigid"
) -> dict:
    """
    Register one modality to another (e.g., T1 to T1c).
    
    Args:
        reference_path: Path to reference (fixed) image
        moving_path: Path to moving image
        output_path: Path to save registered image
        transform_path: Path to save transformation
        registration_type: Type of registration (default: "Rigid")
    
    Returns:
        dict: Information about the registration
    """
    try:
        logger.info(f"Registering {moving_path.name} to {reference_path.name}")
        
        # Load images
        fixed = load_ants_image(str(reference_path))
        moving = load_ants_image(str(moving_path))
        
        # Perform registration
        registration_result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=registration_type,
            verbose=False
        )
        
        # Get registered image
        warped_moving = registration_result['warpedmovout']
        
        # Create output directories
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save registered image
        save_ants_image(warped_moving, str(output_path))
        logger.info(f"Saved registered image to {output_path}")
        
        # Save transformation
        forward_transforms = registration_result['fwdtransforms']
        if forward_transforms:
            transform_file = Path(forward_transforms[0])
            shutil.copy(transform_file, transform_path)
            logger.info(f"Saved transformation to {transform_path}")
        
        return {
            "success": True,
            "transform_path": str(transform_path),
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error registering {moving_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def process_subject_registration(
    subject_dir: Path,
    output_dir: Path,
    transform_dir: Path,
    atlas_path: Path,
    modalities: list,
    params: dict,
    bias_corrected_dir: Path = None
) -> dict:
    """
    Process all modalities for a subject (registration step).
    
    Workflow (following UPENN-GBM protocol):
    1. Register T1c (bias-corrected) to atlas
    2. Register other modalities (bias-corrected) to T1c
    3. Apply transformations to original (non-bias-corrected) images
    
    Args:
        subject_dir: Path to subject directory with original images
        output_dir: Path to output directory for registered images
        transform_dir: Path to save transformation matrices
        atlas_path: Path to atlas image
        modalities: List of modality suffixes
        params: Registration parameters
        bias_corrected_dir: Path to bias-corrected images (for registration only)
    
    Returns:
        dict: Processing results for each modality
    """
    results = {}
    
    # Extract subject and session from path
    subject_id = subject_dir.parent.parent.name  # sub-XXX
    session_id = subject_dir.parent.name          # ses-XXX
    
    logger.info(f"Processing {subject_id}/{session_id} - Registration")
    
    reference_modality = params.get("reference_modality", "t1c")
    
    # Determine which directory has the images to use for computing registration
    # (bias-corrected if available, otherwise original)
    if bias_corrected_dir is not None:
        registration_source_dir = bias_corrected_dir / subject_id / session_id / "anat"
        logger.info("Using bias-corrected images for registration computation")
    else:
        registration_source_dir = subject_dir
        logger.info("Using original images for registration computation")
    
    # Step 1: Register reference modality (T1c) to atlas
    logger.info(f"Step 1: Registering {reference_modality} to atlas")
    
    ref_pattern = f"{subject_id}_{session_id}_{reference_modality}.nii.gz"
    ref_files = list(registration_source_dir.glob(ref_pattern))
    
    if not ref_files:
        error_msg = f"Reference modality {reference_modality} not found"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    ref_file = ref_files[0]
    
    # Output paths for reference modality
    ref_output = output_dir / subject_id / session_id / "anat" / ref_pattern
    ref_transform = transform_dir / subject_id / session_id / "anat" / f"{subject_id}_{session_id}_{reference_modality}_to_atlas.mat"
    
    # Register reference to atlas
    ref_result = register_to_atlas(
        moving_path=ref_file,
        atlas_path=atlas_path,
        output_path=ref_output,
        transform_path=ref_transform,
        registration_type=params.get("registration_type", "Rigid"),
        output_resolution=params.get("output_resolution", [1.0, 1.0, 1.0]),
        metric=params.get("metric", "MI"),
        number_of_iterations=params.get("number_of_iterations", 1000),
        smoothing_sigmas=params.get("smoothing_sigmas", [3, 2, 1, 0]),
        shrink_factors=params.get("shrink_factors", [8, 4, 2, 1]),
        use_histogram_matching=params.get("use_histogram_matching", True)
    )
    
    results[reference_modality] = ref_result
    
    if not ref_result["success"]:
        logger.error(f"Failed to register reference modality {reference_modality}")
        return results
    
    # Step 2: Register other modalities to reference modality
    logger.info(f"Step 2: Registering other modalities to {reference_modality}")
    
    for modality in modalities:
        if modality == reference_modality:
            continue  # Already processed
        
        modal_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
        modal_files = list(registration_source_dir.glob(modal_pattern))
        
        if not modal_files:
            logger.warning(f"Modality {modality} not found, skipping")
            results[modality] = {"success": False, "error": "File not found"}
            continue
        
        modal_file = modal_files[0]
        
        # Register to reference modality (using bias-corrected reference)
        modal_output_temp = output_dir / subject_id / session_id / "anat" / f"temp_{modal_pattern}"
        modal_transform = transform_dir / subject_id / session_id / "anat" / f"{subject_id}_{session_id}_{modality}_to_{reference_modality}.mat"
        
        modal_result = register_modalities(
            reference_path=ref_file,
            moving_path=modal_file,
            output_path=modal_output_temp,
            transform_path=modal_transform,
            registration_type=params.get("registration_type", "Rigid")
        )
        
        if not modal_result["success"]:
            logger.error(f"Failed to register {modality}")
            results[modality] = modal_result
            continue
        
        # Step 3: Apply combined transformation to original (non-bias-corrected) image
        logger.info(f"Step 3: Applying combined transforms to original {modality}")

        # Get original image
        original_file = subject_dir / modal_pattern
        modal_output_final = output_dir / subject_id / session_id / "anat" / modal_pattern

        # Load atlas as fixed image
        atlas_img = load_ants_image(atlas_path)

        # Load original (non-bias-corrected) moving image
        original_img = load_ants_image(original_file)

        # Apply COMBINED transformation: modality → T1c → atlas
        # ANTs applies transforms in REVERSE order (last to first)
        # So we list: [T1c→atlas, modality→T1c]
        # This applies: modality→T1c first, then T1c→atlas
        combined_result = ants.apply_transforms(
            fixed=atlas_img,
            moving=original_img,
            transformlist=[
                str(ref_transform),      # Applied SECOND: T1c → atlas
                str(modal_transform)     # Applied FIRST: modality → T1c
            ],
            interpolator="linear",
            whichtoinvert=[False, False]  # Don't invert any transforms
        )

        # Save final result
        save_ants_image(combined_result, modal_output_final)

        # Clean up temp file
        if modal_output_temp.exists():
            modal_output_temp.unlink()

        logger.info(f"Saved final registered {modality} to {modal_output_final}")

        results[modality] = {
            "success": True,
            "transform_to_ref": str(modal_transform),
            "transform_to_atlas": str(ref_transform),
            "output_path": str(modal_output_final)
        }
    
    # Also apply reference transformation to original (non-bias-corrected) reference image
    if bias_corrected_dir is not None:
        logger.info(f"Applying transform to original {reference_modality}")
        original_ref = subject_dir / ref_pattern
        apply_transform(
            moving_path=original_ref,
            fixed_path=atlas_path,
            transform_path=ref_transform,
            output_path=ref_output
        )
    
    return results

def inverse_transform_mask(
    mask_path: Path,
    reference_path: Path,
    transform_paths: list,
    output_path: Path
) -> dict:
    """
    Apply inverse transformations to bring segmentation mask back to native space.
    
    Uses NearestNeighbor interpolation to preserve integer class labels.
    
    Args:
        mask_path: Path to segmentation mask in atlas space
        reference_path: Path to native-space image (defines output geometry)
        transform_paths: List of transform .mat files in FORWARD order
                         (e.g., [t1_to_atlas.mat] for T1,
                          [t1_to_atlas.mat, t1c_to_t1.mat] for T1c)
        output_path: Path to save native-space mask
    
    Returns:
        dict with success status and output path
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Inverse transform: {mask_path.name} -> {output_path.name}")
        logger.info(f"  Reference (native): {reference_path.name}")
        logger.info(f"  Transforms to invert: {[p.name for p in transform_paths]}")
        
        # Load images
        mask_img = load_ants_image(mask_path)
        reference_img = load_ants_image(reference_path)
        
        # ANTs applies transforms in REVERSE list order.
        # We receive forward transforms [t1_to_atlas, t1c_to_t1].
        # To go backwards (atlas → T1 → T1c), ANTs needs:
        #   transformlist = [t1c_to_t1, t1_to_atlas]  (reversed)
        #   whichtoinvert = [True, True]               (invert all)
        reversed_transforms = [str(p) for p in reversed(transform_paths)]
        invert_flags = [True] * len(reversed_transforms)
        
        # Apply inverse transforms
        native_mask = ants.apply_transforms(
            fixed=reference_img,
            moving=mask_img,
            transformlist=reversed_transforms,
            interpolator="nearestNeighbor",
            whichtoinvert=invert_flags
        )
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_ants_image(native_mask, str(output_path))
        
        processing_time = time.time() - start_time
        logger.info(f"  Saved native mask to {output_path} ({processing_time:.2f}s)")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error in inverse transform for {mask_path.name}: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def inverse_transform_subject_masks(
    mask_path: Path,
    nifti_dir: Path,
    transform_dir: Path,
    output_dir: Path,
    subject_id: str,
    session_id: str,
    reference_modality: str = "t1",
    modalities: list = None
) -> dict:
    """
    Inverse-transform segmentation mask to native space of each modality.
    
    For each modality, applies the appropriate inverse chain:
      - T1:   invert [t1_to_atlas]
      - T1c:  invert [t1_to_atlas, t1c_to_t1]
      - T2:   invert [t1_to_atlas, t2_to_t1]
      - T2fl: invert [t1_to_atlas, t2fl_to_t1]
    
    Args:
        mask_path: Path to *_segmask.nii.gz in atlas space
        nifti_dir: Root nifti/ directory (contains native-space images)
        transform_dir: Root transformations/ directory
        output_dir: Where to save native masks
        subject_id: e.g., "sub-001"
        session_id: e.g., "ses-001"
        reference_modality: Modality registered directly to atlas (default: "t1")
        modalities: List of modalities to process (default: all 4)
    
    Returns:
        dict: results per modality
    """
    if modalities is None:
        modalities = ["t1", "t1c", "t2", "t2fl"]
    
    results = {}
    
    # Paths to transforms
    transform_subdir = transform_dir / subject_id / session_id / "anat"
    nifti_subdir = nifti_dir / subject_id / session_id / "anat"
    output_subdir = output_dir / subject_id / session_id / "anat"
    
    # Atlas transform (always needed)
    atlas_transform = transform_subdir / f"{subject_id}_{session_id}_{reference_modality}_to_atlas.mat"
    
    if not atlas_transform.exists():
        error_msg = f"Atlas transform not found: {atlas_transform}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    for modality in modalities:
        # Find native-space reference image
        native_ref = nifti_subdir / f"{subject_id}_{session_id}_{modality}.nii.gz"
        
        if not native_ref.exists():
            logger.warning(f"  Native image not found for {modality}, skipping: {native_ref}")
            results[modality] = {"success": False, "error": "Native image not found"}
            continue
        
        # Build forward transform chain
        if modality == reference_modality:
            # T1: only atlas transform
            forward_transforms = [atlas_transform]
        else:
            # Other modalities: atlas + cross-modality
            cross_transform = transform_subdir / f"{subject_id}_{session_id}_{modality}_to_{reference_modality}.mat"
            
            if not cross_transform.exists():
                logger.warning(f"  Cross-modality transform not found for {modality}, skipping: {cross_transform}")
                results[modality] = {"success": False, "error": "Cross-modality transform not found"}
                continue
            
            forward_transforms = [atlas_transform, cross_transform]
        
        # Output path
        mask_stem = mask_path.name.replace("_segmask.nii.gz", "")
        output_path = output_subdir / f"{mask_stem}_segmask_native_{modality}.nii.gz"
        
        # Apply inverse transform
        result = inverse_transform_mask(
            mask_path=mask_path,
            reference_path=native_ref,
            transform_paths=forward_transforms,
            output_path=output_path
        )
        
        results[modality] = result
    
    # Summary
    successful = sum(1 for r in results.values() if r.get("success"))
    logger.info(f"Inverse transform complete for {subject_id}/{session_id}: "
                f"{successful}/{len(modalities)} modalities")
    
    return results

if __name__ == "__main__":
    # Test the registration step
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Download atlas:  python registration.py download <cache_dir> <url>")
        print("  Register to atlas: python registration.py atlas <moving> <atlas> <output> <transform>")
        print("  Register modalities: python registration.py modality <fixed> <moving> <output> <transform>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "download":
        if len(sys.argv) < 4:
            print("Usage: python registration.py download <cache_dir> <url>")
            sys.exit(1)
        
        cache_dir = Path(sys.argv[2])
        url = sys.argv[3]
        filename = "sri24_t1.nii.gz"
        
        atlas_path = download_sri24_atlas(cache_dir, url, filename)
        print(f"Atlas downloaded to: {atlas_path}")
        
    elif command == "atlas":
        if len(sys.argv) < 6:
            print("Usage: python registration.py atlas <moving> <atlas> <output> <transform>")
            sys.exit(1)
        
        moving = Path(sys.argv[2])
        atlas = Path(sys.argv[3])
        output = Path(sys.argv[4])
        transform = Path(sys.argv[5])
        
        result = register_to_atlas(moving, atlas, output, transform)
        print(f"\nResult: {result}")
        
    elif command == "modality":
        if len(sys.argv) < 6:
            print("Usage: python registration.py modality <fixed> <moving> <output> <transform>")
            sys.exit(1)
        
        fixed = Path(sys.argv[2])
        moving = Path(sys.argv[3])
        output = Path(sys.argv[4])
        transform = Path(sys.argv[5])
        
        result = register_modalities(fixed, moving, output, transform)
        print(f"\nResult: {result}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)