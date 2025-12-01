"""
Skull stripping step for MRI preprocessing.
Uses FSL BET (Brain Extraction Tool) to remove non-brain tissue.
"""

import logging
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
import os

logger = logging.getLogger(__name__)

# Global FSL configuration
FSL_DIR = None
FSL_BIN_DIR = None


def setup_fsl_environment(fsl_dir: str = None):
    """
    Setup FSL environment variables.
    
    Args:
        fsl_dir: Path to FSL installation directory (e.g., /usr/local/fsl or /path/to/fsl/share/fsl)
                 If None, assumes FSL is in PATH
    """
    global FSL_DIR, FSL_BIN_DIR
    
    if fsl_dir and fsl_dir.strip():
        fsl_path = Path(fsl_dir)
        
        # If path points to a file (like bet or fsl executable), get parent directory
        if fsl_path.is_file():
            fsl_path = fsl_path.parent
        
        # Now fsl_path should be the bin directory, so go up one level to get FSL_DIR
        if fsl_path.name == "bin":
            FSL_DIR = fsl_path.parent
            FSL_BIN_DIR = fsl_path
        else:
            # Assume it's the root FSL directory
            FSL_DIR = fsl_path
            FSL_BIN_DIR = FSL_DIR / "bin"
        
        # Verify bin directory exists
        if not FSL_BIN_DIR.exists():
            logger.error(f"FSL bin directory not found: {FSL_BIN_DIR}")
            FSL_DIR = None
            FSL_BIN_DIR = None
            return
        
        # Verify bet exists
        bet_path = FSL_BIN_DIR / "bet"
        if not bet_path.exists():
            logger.error(f"BET executable not found: {bet_path}")
            FSL_DIR = None
            FSL_BIN_DIR = None
            return
        
        logger.info(f"✓ FSL configured successfully")
        logger.info(f"  FSLDIR: {FSL_DIR}")
        logger.info(f"  BIN: {FSL_BIN_DIR}")
        logger.info(f"  BET: {bet_path}")
    else:
        FSL_DIR = None
        FSL_BIN_DIR = None
        logger.info("Using FSL from system PATH")

def get_fsl_env() -> dict:
    """
    Get environment variables for FSL commands.
    
    Returns:
        dict: Environment variables
    """
    env = os.environ.copy()
    
    if FSL_DIR is not None:
        env['FSLDIR'] = str(FSL_DIR)
        env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        
        # Add FSL bin to PATH
        if FSL_BIN_DIR is not None:
            current_path = env.get('PATH', '')
            env['PATH'] = f"{FSL_BIN_DIR}:{current_path}"
        
        # Add FSL lib to LD_LIBRARY_PATH if exists
        fsl_lib = FSL_DIR / "lib"
        if fsl_lib.exists():
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{fsl_lib}:{current_ld_path}"
    
    return env


def get_bet_command() -> str:
    """
    Get the BET command path.
    
    Returns:
        str: Path to BET executable
    """
    if FSL_BIN_DIR is not None:
        return str(FSL_BIN_DIR / "bet")
    else:
        return "bet"


def check_fsl_installed() -> bool:
    """
    Check if FSL is installed and available.
    
    Returns:
        bool: True if FSL is available
    """
    try:
        bet_cmd = get_bet_command()
        env = get_fsl_env()
        
        logger.debug(f"Testing BET command: {bet_cmd}")
        logger.debug(f"Environment FSLDIR: {env.get('FSLDIR', 'not set')}")
        logger.debug(f"Environment PATH: {env.get('PATH', 'not set')[:200]}...")
        
        result = subprocess.run(
            [bet_cmd, '-help'],
            capture_output=True,
            timeout=5,
            env=env,
            text=True
        )
        
        logger.debug(f"BET return code: {result.returncode}")
        logger.debug(f"BET stdout: {result.stdout[:200] if result.stdout else 'empty'}")
        logger.debug(f"BET stderr: {result.stderr[:200] if result.stderr else 'empty'}")
        
        # BET returns code 1 for -help, but outputs usage info
        # Check if output contains expected BET usage text
        output = result.stdout + result.stderr
        is_valid = "Usage:" in output and "bet" in output.lower()
        
        if is_valid:
            logger.debug("✓ BET is working correctly")
        else:
            logger.debug("✗ BET output doesn't contain expected usage text")
        
        return is_valid
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"FSL check timeout: {e}")
        return False
    except FileNotFoundError as e:
        logger.error(f"FSL BET executable not found: {e}")
        return False
    except Exception as e:
        logger.error(f"FSL check failed with exception: {e}")
        return False


def run_bet(
    input_path: Path,
    output_path: Path,
    mask_path: Path = None,
    fractional_intensity: float = 0.5,
    vertical_gradient: float = 0.0,
    generate_mask: bool = True
) -> dict:
    """
    Run FSL BET for brain extraction.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save skull-stripped image
        mask_path: Path to save brain mask (optional)
        fractional_intensity: Fractional intensity threshold (0-1), default 0.5
        vertical_gradient: Vertical gradient in fractional intensity threshold (-1 to 1)
        generate_mask: Whether to generate brain mask
    
    Returns:
        dict: Information about the skull stripping
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Running BET on {input_path.name}")
        
        # Check if FSL is installed
        if not check_fsl_installed():
            raise RuntimeError(
                f"FSL BET not found. "
                f"FSL_DIR: {FSL_DIR}, FSL_BIN_DIR: {FSL_BIN_DIR}. "
                f"Please check FSL installation path in config."
            )
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get BET command and environment
        bet_cmd = get_bet_command()
        env = get_fsl_env()
        
        # Build BET command
        cmd = [
            bet_cmd,
            str(input_path),
            str(output_path),
            '-f', str(fractional_intensity),
            '-g', str(vertical_gradient)
        ]
        
        # Add mask generation flag
        if generate_mask:
            cmd.append('-m')
        
        # Add robust brain center estimation
        cmd.append('-R')

        logger.debug(f"BET command: {' '.join(cmd)}")
        logger.debug(f"FSLDIR: {env.get('FSLDIR', 'not set')}")
        
        # Run BET with FSL environment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            env=env
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"BET failed with return code {result.returncode}: {result.stderr}")
        
        logger.debug(f"BET output: {result.stdout}")
        
        # BET creates mask with _mask suffix automatically
        auto_mask_path = output_path.parent / f"{output_path.stem.replace('.nii', '')}_mask.nii.gz"
        
        mask_created = False
        if generate_mask and auto_mask_path.exists():
            # Move mask to desired location if specified
            if mask_path is not None and mask_path != auto_mask_path:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                auto_mask_path.rename(mask_path)
                logger.info(f"Saved brain mask to {mask_path}")
                mask_created = True
            else:
                mask_path = auto_mask_path
                logger.info(f"Brain mask created at {auto_mask_path}")
                mask_created = True
        
        processing_time = time.time() - start_time
        logger.info(f"BET completed in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "mask_path": str(mask_path) if mask_created else None,
            "processing_time": processing_time
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"BET timeout on {input_path.name}")
        return {
            "success": False,
            "error": "BET timeout (exceeded 5 minutes)"
        }
    except Exception as e:
        logger.error(f"Error running BET on {input_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


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
    1. Create brain mask on reference modality (e.g., T1c)
    2. Apply mask to all modalities
    
    Args:
        subject_dir: Path to subject directory (BIDS structure)
        output_dir: Path to output directory for skull-stripped images
        modalities: List of modality suffixes to process
        params: Skull stripping parameters
    
    Returns:
        dict: Processing results for each modality
    """
    results = {}
    
    # Extract subject and session from path
    subject_id = subject_dir.parent.parent.name  # sub-XXX
    session_id = subject_dir.parent.name          # ses-XXX
    
    logger.info(f"Processing {subject_id}/{session_id} - Skull Stripping")
    
    reference_modality = params.get("reference_modality", "t1c")
    
    # Step 1: Create brain mask on reference modality
    logger.info(f"Step 1: Creating brain mask on {reference_modality}")
    
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
    
    # Run BET on reference modality
    bet_result = run_bet(
        input_path=ref_file,
        output_path=ref_output,
        mask_path=mask_path,
        fractional_intensity=params.get("fractional_intensity", 0.5),
        vertical_gradient=params.get("vertical_gradient", 0.0),
        generate_mask=True
    )
    
    results[reference_modality] = bet_result
    
    if not bet_result["success"]:
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
            
            # Apply mask
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
                except:
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


if __name__ == "__main__":
    # Test the skull stripping step
    import sys
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Try to load FSL path from config
    config_path = Path(__file__).parent.parent.parent / "configs" / "preprocessing_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            fsl_config = config.get('fsl', {})
            fsl_dir = fsl_config.get('fsl_dir', '')
            if fsl_dir:
                setup_fsl_environment(fsl_dir)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Check FSL:  python skull_stripping.py check [fsl_dir]")
        print("  Run BET:    python skull_stripping.py bet <input> <output> [fractional_intensity] [vertical_gradient] [fsl_dir]")
        print("  Apply mask: python skull_stripping.py mask <input> <mask> <output>")
        print("  Compare:    python skull_stripping.py compare <original> <stripped>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check":
        # Set debug level for check command
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Allow FSL dir as argument
        if len(sys.argv) > 2:
            setup_fsl_environment(sys.argv[2])
        
        if check_fsl_installed():
            print("✓ FSL BET is installed and available")
            print(f"  FSL_DIR: {FSL_DIR}")
            print(f"  FSL_BIN_DIR: {FSL_BIN_DIR}")
            # Test version
            bet_cmd = get_bet_command()
            env = get_fsl_env()
            result = subprocess.run([bet_cmd], capture_output=True, text=True, env=env)
            print("\n" + result.stderr[:400])  # Print first part of help
        else:
            print("✗ FSL BET not found")
            print(f"  Searched at: {FSL_BIN_DIR if FSL_BIN_DIR else 'system PATH'}")
            print("\nDebug info above should show why BET check failed")
            print("\nPlease:")
            print("  1. Install FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation")
            print("  2. Or specify FSL path in configs/preprocessing_config.yaml")
            print("  3. Or run: python skull_stripping.py check /path/to/fsl")
        
    elif command == "bet":
        if len(sys.argv) < 4:
            print("Usage: python skull_stripping.py bet <input> <output> [fractional_intensity] [vertical_gradient]")
            sys.exit(1)
        
        input_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3])
        frac_intensity = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        vert_gradient = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        
        result = run_bet(
            input_path, 
            output_path,
            fractional_intensity=frac_intensity,
            vertical_gradient=vert_gradient
        )
        print(f"\nResult: {result}")
        
    elif command == "mask":
        if len(sys.argv) < 5:
            print("Usage: python skull_stripping.py mask <input> <mask> <output>")
            sys.exit(1)
        
        input_path = Path(sys.argv[2])
        mask_path = Path(sys.argv[3])
        output_path = Path(sys.argv[4])
        
        result = apply_brain_mask(input_path, mask_path, output_path)
        print(f"\nResult: {result}")
        
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python skull_stripping.py compare <original> <stripped>")
            sys.exit(1)
        
        original_path = Path(sys.argv[2])
        stripped_path = Path(sys.argv[3])
        
        compare_before_after_stripping(original_path, stripped_path)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)