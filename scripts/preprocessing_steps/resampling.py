"""
Resampling preprocessing step for MRI data.
Resamples images to target resolution without registration to atlas.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import ants

logger = logging.getLogger(__name__)


def resample_image(
    image: ants.ANTsImage,
    target_spacing: List[float],
    interpolation: str = 'linear'
) -> ants.ANTsImage:
    """
    Resample image to target spacing.
    
    Args:
        image: Input ANTs image
        target_spacing: Target spacing [x, y, z] in mm
        interpolation: Interpolation method ('linear', 'nearestNeighbor', 'bSpline')
    
    Returns:
        Resampled ANTs image
    """
    # Map interpolation string to ANTs integer codes
    interp_map = {
        'linear': 1,
        'nearestNeighbor': 0,
        'nearestneighbor': 0,
        'nearest': 0,
        'bSpline': 3,
        'bspline': 3
    }
    
    interp_code = interp_map.get(interpolation.lower(), 1)
    
    # Calculate new shape based on target spacing
    old_spacing = image.spacing
    old_shape = image.shape
    
    new_shape = [
        int(round(old_dim * old_sp / new_sp))
        for old_dim, old_sp, new_sp in zip(old_shape, old_spacing, target_spacing)
    ]
    
    logger.debug(f"Resampling from {old_shape} @ {old_spacing} to {new_shape} @ {target_spacing}")
    
    # Resample using ANTs
    resampled = ants.resample_image(
        image,
        new_shape,
        use_voxels=True,
        interp_type=interp_code
    )
    
    # Verify spacing
    actual_spacing = resampled.spacing
    logger.debug(f"Actual output spacing: {actual_spacing}")
    
    return resampled


def process_subject_resampling(
    subject_dir: Path,
    output_dir: Path,
    modalities: List[str],
    params: dict
) -> Dict[str, dict]:
    """
    Resample all modalities for a subject to target resolution.
    
    Args:
        subject_dir: Path to subject's anat directory (input)
        output_dir: Output directory for resampled data
        modalities: List of modalities to process
        params: Resampling parameters from config
            - output_resolution: [x, y, z] in mm
            - interpolation: 'linear', 'nearestNeighbor', or 'bSpline' (optional)
    
    Returns:
        Dictionary with results for each modality
    """
    results = {}
    
    # Extract parameters
    target_spacing = params.get('output_resolution', [1.0, 1.0, 1.0])
    interpolation = params.get('interpolation', 'linear')
    
    # Get subject and session IDs from path
    session_id = subject_dir.parent.name
    subject_id = subject_dir.parent.parent.name
    
    logger.info(f"Resampling {subject_id}/{session_id} to {target_spacing}mm")
    
    # Create output directory
    output_anat = output_dir / subject_id / session_id / "anat"
    output_anat.mkdir(parents=True, exist_ok=True)
    
    # Process each modality
    for modality in modalities:
        try:
            # Find input file
            input_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
            input_files = list(subject_dir.glob(input_pattern))
            
            if not input_files:
                logger.debug(f"Skipping {modality}: file not found")
                results[modality] = {
                    "success": False,
                    "error": "File not found",
                    "skipped": True
                }
                continue
            
            input_file = input_files[0]
            output_file = output_anat / input_pattern
            
            logger.debug(f"Processing {modality}: {input_file.name}")
            
            # Load image
            image = ants.image_read(str(input_file))
            
            # Resample
            resampled = resample_image(
                image=image,
                target_spacing=target_spacing,
                interpolation=interpolation
            )
            
            # Save
            ants.image_write(resampled, str(output_file))
            
            results[modality] = {
                "success": True,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "input_shape": image.shape,
                "output_shape": resampled.shape,
                "input_spacing": image.spacing,
                "output_spacing": resampled.spacing
            }
            
            logger.debug(f"✓ {modality}: {image.shape}@{image.spacing} → {resampled.shape}@{resampled.spacing}")
            
        except Exception as e:
            logger.error(f"✗ Failed to resample {modality}: {str(e)}")
            results[modality] = {
                "success": False,
                "error": str(e)
            }
    
    # Summary
    successful = sum(1 for r in results.values() if r.get('success', False))
    logger.info(f"Resampling: {successful}/{len(results)} modalities successful")
    
    return results