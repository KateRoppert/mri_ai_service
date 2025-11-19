"""
N4 Bias Field Correction step for MRI preprocessing.
Uses SimpleITK's N4BiasFieldCorrectionImageFilter.
"""

import logging
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def apply_n4_bias_correction(
    input_path: Path,
    output_path: Path,
    shrink_factor: int = 4,
    n_iterations: list = [50, 50, 50, 50],
    convergence_threshold: float = 0.001,
    save_bias_field: bool = False,
    bias_field_path: Path = None
) -> dict:
    """
    Apply N4 bias field correction to a NIfTI image using SimpleITK.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save corrected NIfTI file
        shrink_factor: Shrink factor for processing speed (default: 4)
        n_iterations: Number of iterations per level (default: [50, 50, 50, 50])
        convergence_threshold: Convergence threshold (default: 0.001)
        save_bias_field: Whether to save the estimated bias field
        bias_field_path: Path to save bias field (if save_bias_field=True)
    
    Returns:
        dict: Information about the correction
            - success: bool
            - bias_field_saved: bool
            - processing_time: float
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Applying N4 bias correction to {input_path.name}")
        
        # Load image with nibabel to preserve header information
        nib_img = nib.load(input_path)
        nib_data = nib_img.get_fdata()
        
        # Convert to SimpleITK image
        sitk_img = sitk.GetImageFromArray(nib_data.transpose(2, 1, 0))

        # Convert spacing to list of floats
        spacing = [float(x) for x in nib_img.header.get_zooms()[:3]]
        sitk_img.SetSpacing(spacing[::-1])  # Reverse for ITK convention

        # Convert origin to list of floats  
        origin = [float(x) for x in nib_img.affine[:3, 3]]
        sitk_img.SetOrigin(origin[::-1])  # Reverse for ITK convention
        
        # Convert to float for processing
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
        
        # Create mask (non-zero voxels)
        # This is important for brain images to avoid bias from background
        mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        
        # Apply shrink factor to speed up processing
        if shrink_factor > 1:
            sitk_img_shrunk = sitk.Shrink(sitk_img, [shrink_factor] * sitk_img.GetDimension())
            mask_shrunk = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())
        else:
            sitk_img_shrunk = sitk_img
            mask_shrunk = mask
        
        # Initialize N4 bias field corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(n_iterations)
        corrector.SetConvergenceThreshold(convergence_threshold)
        
        logger.debug(f"N4 parameters: iterations={n_iterations}, "
                    f"threshold={convergence_threshold}")
        
        # Execute correction on shrunken image
        logger.debug("Running N4 bias field correction...")
        corrected_img_shrunk = corrector.Execute(sitk_img_shrunk, mask_shrunk)
        
        # Get log bias field
        log_bias_field_shrunk = corrector.GetLogBiasFieldAsImage(sitk_img_shrunk)
        
        # Resample log bias field to original resolution if we used shrinking
        if shrink_factor > 1:
            log_bias_field = sitk.Resample(log_bias_field_shrunk, sitk_img)
        else:
            log_bias_field = log_bias_field_shrunk
        
        # Apply bias field correction to original resolution image
        bias_field = sitk.Exp(log_bias_field)
        corrected_img = sitk.Divide(sitk_img, bias_field)
        
        # Convert back to numpy array
        corrected_data = sitk.GetArrayFromImage(corrected_img).transpose(2, 1, 0)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save corrected image with original header
        corrected_nib = nib.Nifti1Image(corrected_data, nib_img.affine, nib_img.header)
        nib.save(corrected_nib, output_path)
        logger.info(f"Saved bias-corrected image to {output_path}")
        
        # Optionally save bias field
        bias_field_saved = False
        if save_bias_field and bias_field_path is not None:
            bias_field_data = sitk.GetArrayFromImage(bias_field).transpose(2, 1, 0)
            bias_field_nib = nib.Nifti1Image(bias_field_data, nib_img.affine, nib_img.header)
            bias_field_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(bias_field_nib, bias_field_path)
            logger.info(f"Saved bias field to {bias_field_path}")
            bias_field_saved = True
        
        processing_time = time.time() - start_time
        logger.info(f"N4 correction completed in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "bias_field_saved": bias_field_saved,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error applying N4 correction to {input_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def process_subject_bias_correction(
    subject_dir: Path,
    output_dir: Path,
    modalities: list,
    params: dict,
    temp_dir: Path = None
) -> dict:
    """
    Process all modalities for a subject (bias correction step).
    
    Args:
        subject_dir: Path to subject directory (BIDS structure)
        output_dir: Path to output directory for bias-corrected images
        modalities: List of modality suffixes to process
        params: Dictionary of bias correction parameters
        temp_dir: Optional temporary directory for bias-corrected images
                  (if using for registration only)
    
    Returns:
        dict: Processing results for each modality
    """
    results = {}
    
    # Extract subject and session from path
    subject_id = subject_dir.parent.parent.name  # sub-XXX
    session_id = subject_dir.parent.name          # ses-XXX
    
    logger.info(f"Processing {subject_id}/{session_id} - Bias Correction")
    
    # Determine if we're using bias correction for registration only
    use_for_registration_only = params.get("use_for_registration_only", True)
    
    # If using for registration only, save to temp directory
    if use_for_registration_only and temp_dir is not None:
        actual_output_dir = temp_dir / subject_id / session_id / "anat"
        logger.info("Saving bias-corrected images to temporary directory (for registration only)")
    else:
        actual_output_dir = output_dir / subject_id / session_id / "anat"
        logger.info("Saving bias-corrected images to output directory")
    
    for modality in modalities:
        # Find input file
        input_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
        input_files = list(subject_dir.glob(input_pattern))
        
        if not input_files:
            logger.warning(f"No file found for modality {modality} with pattern {input_pattern}")
            results[modality] = {"success": False, "error": "File not found"}
            continue
        
        input_file = input_files[0]
        
        # Create output path
        output_file = actual_output_dir / input_pattern
        
        # Optionally save bias field
        save_bias_field = params.get("save_bias_field", False)
        bias_field_file = None
        if save_bias_field:
            bias_field_pattern = f"{subject_id}_{session_id}_{modality}_bias_field.nii.gz"
            bias_field_file = actual_output_dir / bias_field_pattern
        
        # Process
        result = apply_n4_bias_correction(
            input_file,
            output_file,
            shrink_factor=params.get("shrink_factor", 4),
            n_iterations=params.get("n_iterations", [50, 50, 50, 50]),
            convergence_threshold=params.get("convergence_threshold", 0.001),
            save_bias_field=save_bias_field,
            bias_field_path=bias_field_file
        )
        
        result["output_path"] = str(output_file)
        if save_bias_field and bias_field_file:
            result["bias_field_path"] = str(bias_field_file)
        
        results[modality] = result
    
    return results


def compare_before_after(original_path: Path, corrected_path: Path) -> None:
    """
    Compare statistics of original and bias-corrected images.
    
    Args:
        original_path: Path to original NIfTI file
        corrected_path: Path to bias-corrected NIfTI file
    """
    print("=" * 70)
    print("BIAS CORRECTION COMPARISON REPORT")
    print("=" * 70)
    
    # Load images
    orig_img = nib.load(original_path)
    corr_img = nib.load(corrected_path)
    
    orig_data = orig_img.get_fdata()
    corr_data = corr_img.get_fdata()
    
    # Get brain mask (non-zero voxels)
    mask = orig_data > 0
    
    # Calculate statistics for brain region
    orig_brain = orig_data[mask]
    corr_brain = corr_data[mask]
    
    print(f"\nOriginal image: {original_path.name}")
    print(f"  Mean (brain):     {np.mean(orig_brain):.2f}")
    print(f"  Std (brain):      {np.std(orig_brain):.2f}")
    print(f"  Min/Max (brain):  {np.min(orig_brain):.2f} / {np.max(orig_brain):.2f}")
    print(f"  CV (brain):       {np.std(orig_brain)/np.mean(orig_brain):.4f}")
    
    print(f"\nCorrected image: {corrected_path.name}")
    print(f"  Mean (brain):     {np.mean(corr_brain):.2f}")
    print(f"  Std (brain):      {np.std(corr_brain):.2f}")
    print(f"  Min/Max (brain):  {np.min(corr_brain):.2f} / {np.max(corr_brain):.2f}")
    print(f"  CV (brain):       {np.std(corr_brain)/np.mean(corr_brain):.4f}")
    
    # Calculate improvement
    orig_cv = np.std(orig_brain) / np.mean(orig_brain)
    corr_cv = np.std(corr_brain) / np.mean(corr_brain)
    cv_change = ((corr_cv - orig_cv) / orig_cv) * 100
    
    print("\n" + "-" * 70)
    print("CHANGES:")
    print("-" * 70)
    print(f"  Coefficient of Variation change: {cv_change:+.2f}%")
    if cv_change < 0:
        print(f"  ✓ Bias correction reduced intensity variation (good)")
    else:
        print(f"  ⚠ Bias correction increased intensity variation (unusual)")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test the bias correction step
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Correct:  python bias_correction.py correct <input_file> <output_file>")
        print("  Compare:  python bias_correction.py compare <original_file> <corrected_file>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "correct":
        if len(sys.argv) < 4:
            print("Usage: python bias_correction.py correct <input_file> <output_file>")
            sys.exit(1)
        
        input_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3])
        
        result = apply_n4_bias_correction(input_path, output_path)
        print(f"\nResult: {result}")
        
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python bias_correction.py compare <original_file> <corrected_file>")
            sys.exit(1)
        
        original_path = Path(sys.argv[2])
        corrected_path = Path(sys.argv[3])
        
        compare_before_after(original_path, corrected_path)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)