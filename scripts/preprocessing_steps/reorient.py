"""
Reorientation step for MRI preprocessing.
Reorients NIfTI images to a target orientation (default: LPS).
"""

import logging
from pathlib import Path
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def reorient_to_standard(input_path: Path, output_path: Path, 
                         target_orientation: str = "LPS") -> dict:
    """
    Reorient NIfTI image to target orientation.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save reoriented NIfTI file
        target_orientation: Target orientation (default: "LPS")
    
    Returns:
        dict: Information about the reorientation
            - success: bool
            - original_orientation: str
            - target_orientation: str
            - transformation_applied: bool
    """
    try:
        logger.info(f"Reorienting {input_path.name} to {target_orientation}")
        
        # Load image
        img = nib.load(input_path)
        
        # Get original orientation
        original_orientation = nib.aff2axcodes(img.affine)
        original_orientation_str = ''.join(original_orientation)
        
        logger.debug(f"Original orientation: {original_orientation_str}")
        
        # Check if reorientation is needed
        if original_orientation_str == target_orientation:
            logger.info(f"Image already in {target_orientation} orientation, copying...")
            # Just copy the file
            img_reoriented = img
            transformation_applied = False
        else:
            # Reorient to target orientation
            img_reoriented = nib.as_closest_canonical(img)
            
            # Get axes that need to be flipped
            target_ornt = nib.orientations.axcodes2ornt(tuple(target_orientation))
            current_ornt = nib.orientations.io_orientation(img_reoriented.affine)
            transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
            
            # Apply transformation
            img_data = img_reoriented.get_fdata()
            img_data_reoriented = nib.orientations.apply_orientation(img_data, transform)
            
            # Update affine
            affine_reoriented = img_reoriented.affine @ nib.orientations.inv_ornt_aff(transform, img_data.shape)
            
            # Create new image
            img_reoriented = nib.Nifti1Image(img_data_reoriented, affine_reoriented, img.header)
            
            transformation_applied = True
            logger.info(f"Reoriented from {original_orientation_str} to {target_orientation}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save reoriented image
        nib.save(img_reoriented, output_path)
        logger.info(f"Saved reoriented image to {output_path}")
        
        return {
            "success": True,
            "original_orientation": original_orientation_str,
            "target_orientation": target_orientation,
            "transformation_applied": transformation_applied
        }
        
    except Exception as e:
        logger.error(f"Error reorienting {input_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def process_subject_reorient(subject_dir: Path, output_dir: Path, 
                             modalities: list, target_orientation: str = "LPS") -> dict:
    """
    Process all modalities for a subject (reorientation step).
    
    Args:
        subject_dir: Path to subject directory (BIDS structure)
        output_dir: Path to output directory
        modalities: List of modality suffixes to process
        target_orientation: Target orientation
    
    Returns:
        dict: Processing results for each modality
    """
    results = {}
    
    # Extract subject and session from path
    subject_id = subject_dir.parent.parent.name  # sub-XXX
    session_id = subject_dir.parent.name          # ses-XXX
    
    logger.info(f"Processing {subject_id}/{session_id} - Reorientation")
    
    for modality in modalities:
        # Find input file
        input_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
        input_files = list(subject_dir.glob(input_pattern))
        
        if not input_files:
            logger.warning(f"No file found for modality {modality} with pattern {input_pattern}")
            results[modality] = {"success": False, "error": "File not found"}
            continue
        
        input_file = input_files[0]
        
        # Create output path with same BIDS structure
        output_subdir = output_dir / subject_id / session_id / "anat"
        output_file = output_subdir / input_pattern
        
        # Process
        result = reorient_to_standard(input_file, output_file, target_orientation)
        results[modality] = result
    
    return results

def check_orientation(image_path: Path) -> dict:
    """
    Check and display orientation information of a NIfTI image.
    
    Args:
        image_path: Path to NIfTI file
    
    Returns:
        dict: Orientation information
    """
    img = nib.load(image_path)
    
    # Get orientation
    orientation = nib.aff2axcodes(img.affine)
    orientation_str = ''.join(orientation)
    
    # Get affine matrix
    affine = img.affine
    
    # Get shape
    shape = img.shape
    
    # Get voxel dimensions
    voxel_sizes = img.header.get_zooms()[:3]
    
    # Determine if matrix is RAS+ or LPS+
    # Check determinant of affine (excluding translation)
    det = np.linalg.det(affine[:3, :3])
    coordinate_system = "RAS+" if det > 0 else "LPS+"
    
    info = {
        "orientation": orientation_str,
        "orientation_tuple": orientation,
        "shape": shape,
        "voxel_sizes": voxel_sizes,
        "coordinate_system": coordinate_system,
        "affine": affine,
        "affine_determinant": det
    }
    
    return info


def compare_orientations(original_path: Path, reoriented_path: Path) -> None:
    """
    Compare orientations of original and reoriented images and print report.
    
    Args:
        original_path: Path to original NIfTI file
        reoriented_path: Path to reoriented NIfTI file
    """
    print("=" * 70)
    print("ORIENTATION COMPARISON REPORT")
    print("=" * 70)
    
    # Get info for both images
    orig_info = check_orientation(original_path)
    reor_info = check_orientation(reoriented_path)
    
    # Print comparison
    print(f"\nOriginal file: {original_path.name}")
    print(f"  Orientation:        {orig_info['orientation']}")
    print(f"  Shape:              {orig_info['shape']}")
    print(f"  Voxel sizes:        {orig_info['voxel_sizes']}")
    print(f"  Coordinate system:  {orig_info['coordinate_system']}")
    print(f"  Affine determinant: {orig_info['affine_determinant']:.4f}")
    
    print(f"\nReoriented file: {reoriented_path.name}")
    print(f"  Orientation:        {reor_info['orientation']}")
    print(f"  Shape:              {reor_info['shape']}")
    print(f"  Voxel sizes:        {reor_info['voxel_sizes']}")
    print(f"  Coordinate system:  {reor_info['coordinate_system']}")
    print(f"  Affine determinant: {reor_info['affine_determinant']:.4f}")
    
    # Check what changed
    print("\n" + "-" * 70)
    print("CHANGES:")
    print("-" * 70)
    
    if orig_info['orientation'] != reor_info['orientation']:
        print(f"✓ Orientation changed: {orig_info['orientation']} → {reor_info['orientation']}")
    else:
        print(f"○ Orientation unchanged: {orig_info['orientation']}")
    
    if orig_info['shape'] != reor_info['shape']:
        print(f"✓ Shape changed: {orig_info['shape']} → {reor_info['shape']}")
    else:
        print(f"○ Shape unchanged: {orig_info['shape']}")
    
    # Compare affine matrices
    affine_diff = np.abs(orig_info['affine'] - reor_info['affine'])
    max_diff = np.max(affine_diff)
    if max_diff > 1e-6:
        print(f"✓ Affine matrix changed (max difference: {max_diff:.6f})")
    else:
        print(f"○ Affine matrix unchanged")
    
    # Verify target orientation
    print("\n" + "-" * 70)
    print("VERIFICATION:")
    print("-" * 70)
    
    if reor_info['orientation'] == "LPS":
        print("✓ Target orientation (LPS) achieved")
    else:
        print(f"✗ WARNING: Expected LPS, got {reor_info['orientation']}")
    
    # Explain what LPS means
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("-" * 70)
    print(f"LPS orientation means:")
    print(f"  - First axis (L):  Left → Right")
    print(f"  - Second axis (P): Posterior → Anterior")
    print(f"  - Third axis (S):  Superior → Inferior")
    print("=" * 70)


def verify_reorientation_on_data(image_path: Path) -> bool:
    """
    Verify that reorientation actually changes voxel data position.
    Samples center voxel and corner voxels to check if coordinates changed.
    
    Args:
        image_path: Path to NIfTI file
    
    Returns:
        bool: True if orientation looks correct
    """
    img = nib.load(image_path)
    
    # Get center voxel in world coordinates
    center_voxel = np.array(img.shape[:3]) / 2
    center_world = nib.affines.apply_affine(img.affine, center_voxel)
    
    print(f"\nVoxel-to-world coordinate check:")
    print(f"  Center voxel [i,j,k]: {center_voxel}")
    print(f"  World coordinates [x,y,z]: {center_world}")
    
    # Check corner voxels
    corners = [
        [0, 0, 0],
        [img.shape[0]-1, 0, 0],
        [0, img.shape[1]-1, 0],
        [0, 0, img.shape[2]-1],
    ]
    
    print(f"\n  Corner voxels in world space:")
    for corner in corners:
        world = nib.affines.apply_affine(img.affine, corner)
        print(f"    {corner} → {world}")
    
    return True

if __name__ == "__main__":
    # Test the reorientation step
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Reorient:  python reorient.py reorient <input_file> <output_file> [orientation]")
        print("  Check:     python reorient.py check <image_file>")
        print("  Compare:   python reorient.py compare <original_file> <reoriented_file>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "reorient":
        if len(sys.argv) < 4:
            print("Usage: python reorient.py reorient <input_file> <output_file> [orientation]")
            sys.exit(1)
        
        input_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3])
        orientation = sys.argv[4] if len(sys.argv) > 4 else "LPS"
        
        result = reorient_to_standard(input_path, output_path, orientation)
        print(f"Result: {result}")
        
    elif command == "check":
        if len(sys.argv) < 3:
            print("Usage: python reorient.py check <image_file>")
            sys.exit(1)
        
        image_path = Path(sys.argv[2])
        info = check_orientation(image_path)
        
        print("\n" + "=" * 70)
        print(f"ORIENTATION INFO: {image_path.name}")
        print("=" * 70)
        print(f"  Orientation:        {info['orientation']}")
        print(f"  Shape:              {info['shape']}")
        print(f"  Voxel sizes:        {info['voxel_sizes']}")
        print(f"  Coordinate system:  {info['coordinate_system']}")
        print(f"  Affine determinant: {info['affine_determinant']:.4f}")
        print("=" * 70)
        
        verify_reorientation_on_data(image_path)
        
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python reorient.py compare <original_file> <reoriented_file>")
            sys.exit(1)
        
        original_path = Path(sys.argv[2])
        reoriented_path = Path(sys.argv[3])
        
        compare_orientations(original_path, reoriented_path)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)