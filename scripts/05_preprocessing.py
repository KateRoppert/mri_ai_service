"""
MRI Brain Preprocessing Pipeline for UPENN-GBM dataset.
Performs reorientation, bias correction, registration to atlas, and skull stripping.

Usage:
    python 05_preprocessing.py <input_dir> <output_dir> [options]
"""

import argparse
import logging
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Import preprocessing steps
from preprocessing_steps.reorient import process_subject_reorient
from preprocessing_steps.bias_correction import process_subject_bias_correction
from preprocessing_steps.registration import (
    download_sri24_atlas,
    process_subject_registration,
    load_ants_image
)
from preprocessing_steps.skull_stripping import (
    setup_fsl_environment,
    check_fsl_installed,
    process_subject_skull_stripping
)

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path = None, level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True
    )


def load_config(config_path: Path) -> dict:
    """
    Load preprocessing configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Configuration loaded: {json.dumps(config, indent=2)}")
    return config


def find_subjects(input_dir: Path, max_subjects: int = None) -> List[Tuple[Path, str, str]]:
    """
    Find all subjects and sessions in input directory.
    
    Args:
        input_dir: Input directory with BIDS structure
        max_subjects: Maximum number of subjects to process
    
    Returns:
        List of tuples: (anat_dir, subject_id, session_id)
    """
    logger.info(f"Scanning for subjects in {input_dir}")
    
    subjects = []
    
    # Find all subject directories
    for subject_dir in sorted(input_dir.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Find all session directories
        for session_dir in sorted(subject_dir.glob("ses-*")):
            if not session_dir.is_dir():
                continue
            
            session_id = session_dir.name
            
            # Check for anat directory
            anat_dir = session_dir / "anat"
            if not anat_dir.exists():
                logger.warning(f"No anat directory found for {subject_id}/{session_id}")
                continue
            
            subjects.append((anat_dir, subject_id, session_id))
            
            if max_subjects and len(subjects) >= max_subjects:
                break
        
        if max_subjects and len(subjects) >= max_subjects:
            break
    
    logger.info(f"Found {len(subjects)} subject/session pairs")
    
    return subjects


def prepare_atlas(config: dict) -> Path:
    """
    Download and prepare SRI24 atlas.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Path: Path to atlas file
    """
    atlas_config = config.get('atlas', {})
    
    cache_dir = Path(atlas_config.get('cache_dir', 'data/atlases'))
    atlas_url = atlas_config.get('url', '')
    atlas_filename = atlas_config.get('filename', 'sri24_t1.nii.gz')
    
    logger.info("Preparing atlas...")
    
    try:
        atlas_path = download_sri24_atlas(cache_dir, atlas_url, atlas_filename)
        
        # Verify atlas is readable
        test_img = load_ants_image(atlas_path)
        logger.info(f"✓ Atlas ready: {atlas_path}")
        logger.info(f"  Shape: {test_img.shape}, Spacing: {test_img.spacing}")
        
        return atlas_path
        
    except Exception as e:
        logger.error(f"Failed to prepare atlas: {e}")
        raise


def process_single_subject(
    anat_dir: Path,
    subject_id: str,
    session_id: str,
    output_dir: Path,
    transform_dir: Path,
    temp_dir: Path,
    atlas_path: Path,
    config: dict,
    modalities: List[str]
) -> dict:
    """
    Process a single subject through all preprocessing steps.
    
    Args:
        anat_dir: Path to subject's anat directory
        subject_id: Subject ID (e.g., 'sub-001')
        session_id: Session ID (e.g., 'ses-001')
        output_dir: Output directory for preprocessed data
        transform_dir: Output directory for transformations
        temp_dir: Temporary directory
        atlas_path: Path to atlas file
        config: Configuration dictionary
        modalities: List of modalities to process
    
    Returns:
        dict: Processing results
    """
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info(f"Processing {subject_id}/{session_id}")
    logger.info("=" * 70)
    
    results = {
        "subject_id": subject_id,
        "session_id": session_id,
        "success": False,
        "steps": {},
        "errors": []
    }
    
    try:
        # Create temporary directories
        temp_reoriented = temp_dir / "reoriented"
        temp_bias_corrected = temp_dir / "bias_corrected"
        
        # Get step configurations
        steps_config = {step['name']: step for step in config.get('steps', [])}
        
        # STEP 1: Reorientation
        step_name = "reorient"
        if steps_config.get(step_name, {}).get('enabled', True):
            logger.info("Step 1/4: Reorientation to LPS")
            step_start = time.time()
            
            step_params = steps_config.get(step_name, {}).get('params', {})
            
            reorient_results = process_subject_reorient(
                subject_dir=anat_dir,
                output_dir=temp_reoriented,
                modalities=modalities,
                target_orientation=step_params.get('target_orientation', 'LPS')
            )
            
            results['steps']['reorient'] = {
                "success": all(r.get('success', False) for r in reorient_results.values()),
                "results": reorient_results,
                "time": time.time() - step_start
            }
            
            # Check if reference modality succeeded
            reference_modality = config.get('steps', [{}])[2].get('params', {}).get('reference_modality', 't1c')
            if not reorient_results.get(reference_modality, {}).get('success', False):
                raise RuntimeError(f"Reference modality {reference_modality} failed reorientation")
            
            logger.info(f"✓ Reorientation completed in {results['steps']['reorient']['time']:.1f}s")
        else:
            logger.info("Step 1/4: Reorientation - SKIPPED (disabled in config)")
            # If skipped, use original data
            temp_reoriented = anat_dir.parent.parent
        
        # STEP 2: Bias Correction (for registration only)
        step_name = "bias_correction"
        if steps_config.get(step_name, {}).get('enabled', True):
            logger.info("Step 2/4: N4 Bias Correction (for registration)")
            step_start = time.time()
            
            step_params = steps_config.get(step_name, {}).get('params', {})
            
            # Get reoriented data location
            reoriented_anat = temp_reoriented / subject_id / session_id / "anat"
            
            bias_results = process_subject_bias_correction(
                subject_dir=reoriented_anat,
                output_dir=temp_bias_corrected,
                modalities=modalities,
                params=step_params,
                temp_dir=temp_bias_corrected
            )
            
            results['steps']['bias_correction'] = {
                "success": all(r.get('success', False) for r in bias_results.values()),
                "results": bias_results,
                "time": time.time() - step_start
            }
            
            logger.info(f"✓ Bias correction completed in {results['steps']['bias_correction']['time']:.1f}s")
        else:
            logger.info("Step 2/4: Bias Correction - SKIPPED (disabled in config)")
            temp_bias_corrected = None
        
        # STEP 3: Registration to Atlas
        step_name = "registration"
        if steps_config.get(step_name, {}).get('enabled', True):
            logger.info("Step 3/4: Registration to SRI24 Atlas")
            step_start = time.time()
            
            step_params = steps_config.get(step_name, {}).get('params', {})
            
            # Get reoriented data (original, non-bias-corrected)
            reoriented_anat = temp_reoriented / subject_id / session_id / "anat"
            
            registration_results = process_subject_registration(
                subject_dir=reoriented_anat,
                output_dir=output_dir,
                transform_dir=transform_dir,
                atlas_path=atlas_path,
                modalities=modalities,
                params=step_params,
                bias_corrected_dir=temp_bias_corrected if temp_bias_corrected else None
            )
            
            results['steps']['registration'] = {
                "success": registration_results.get(step_params.get('reference_modality', 't1c'), {}).get('success', False),
                "results": registration_results,
                "time": time.time() - step_start
            }
            
            if not results['steps']['registration']['success']:
                raise RuntimeError("Registration failed for reference modality")
            
            logger.info(f"✓ Registration completed in {results['steps']['registration']['time']:.1f}s")
        else:
            logger.info("Step 3/4: Registration - SKIPPED (disabled in config)")
        
        # STEP 4: Skull Stripping
        step_name = "skull_stripping"
        if steps_config.get(step_name, {}).get('enabled', True):
            logger.info("Step 4/4: Skull Stripping")
            step_start = time.time()
            
            step_params = steps_config.get(step_name, {}).get('params', {})
            
            # Get registered data
            registered_anat = output_dir / subject_id / session_id / "anat"
            
            skull_strip_results = process_subject_skull_stripping(
                subject_dir=registered_anat,
                output_dir=output_dir,
                modalities=modalities,
                params=step_params
            )
            
            results['steps']['skull_stripping'] = {
                "success": all(r.get('success', False) for r in skull_strip_results.values()),
                "results": skull_strip_results,
                "time": time.time() - step_start
            }
            
            logger.info(f"✓ Skull stripping completed in {results['steps']['skull_stripping']['time']:.1f}s")
        else:
            logger.info("Step 4/4: Skull Stripping - SKIPPED (disabled in config)")
        
        # STEP 5: Cleanup temporary files
        logger.info("Step 5/4: Cleaning up temporary files")
        try:
            subject_temp_reoriented = temp_reoriented / subject_id
            subject_temp_bias = temp_bias_corrected / subject_id if temp_bias_corrected else None
            
            if subject_temp_reoriented.exists():
                shutil.rmtree(subject_temp_reoriented)
                logger.debug(f"Removed {subject_temp_reoriented}")
            
            if subject_temp_bias and subject_temp_bias.exists():
                shutil.rmtree(subject_temp_bias)
                logger.debug(f"Removed {subject_temp_bias}")
            
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        # Mark as successful
        results['success'] = True
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        logger.info(f"✓ Completed {subject_id}/{session_id} in {processing_time:.1f}s")
        
    except Exception as e:
        logger.error(f"✗ Failed to process {subject_id}/{session_id}: {str(e)}")
        results['errors'].append(str(e))
        results['success'] = False
    
    return results


def main():
    """Main preprocessing pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="MRI Brain Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with NIfTI files in BIDS format"
    )
    
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for preprocessed data"
    )
    
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "preprocessing_config.yaml",
        help="Path to preprocessing configuration file"
    )
    
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, level="INFO")
    
    logger.info("=" * 70)
    logger.info("MRI BRAIN PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file:      {args.config}")
    
    pipeline_start_time = time.time()
    
    try:
        # Validate input directory
        if not args.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Setup FSL environment
        fsl_config = config.get('fsl', {})
        fsl_dir = fsl_config.get('fsl_dir', '')
        if fsl_dir:
            setup_fsl_environment(fsl_dir)
        
        # Check FSL
        if not check_fsl_installed():
            raise RuntimeError(
                "FSL BET not found. Please install FSL or configure FSL path in config."
            )
        logger.info("✓ FSL BET is available")
        
        # Prepare atlas
        atlas_path = prepare_atlas(config)
        
        # Create output directories
        preprocessed_dir = args.output_dir / "preprocessed"
        transform_dir = args.output_dir / "transformations"
        temp_dir = args.output_dir / "temp"
        
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Output directories created")
        
        # Get modalities from config
        modalities = config.get('modalities', ['t1c', 't1', 't2', 't2fl'])
        logger.info(f"Processing modalities: {', '.join(modalities)}")
        
        # Find subjects
        subjects = find_subjects(args.input_dir, args.max_subjects)
        
        if not subjects:
            logger.error("No subjects found to process")
            return 1
        
        if args.max_subjects:
            logger.info(f"Limited to {args.max_subjects} subjects (--max-subjects)")
        
        # Process each subject
        all_results = []
        successful = 0
        failed = 0
        
        for idx, (anat_dir, subject_id, session_id) in enumerate(subjects, 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Subject {idx}/{len(subjects)}")
            logger.info(f"{'=' * 70}")
            
            result = process_single_subject(
                anat_dir=anat_dir,
                subject_id=subject_id,
                session_id=session_id,
                output_dir=preprocessed_dir,
                transform_dir=transform_dir,
                temp_dir=temp_dir,
                atlas_path=atlas_path,
                config=config,
                modalities=modalities
            )
            
            all_results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Cleanup temp directory
        try:
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                shutil.rmtree(temp_dir)
                logger.info("✓ Removed empty temporary directory")
        except Exception as e:
            logger.warning(f"Could not remove temp directory: {e}")
        
        # Save summary
        summary_path = args.output_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "pipeline_version": "1.0",
                "total_subjects": len(subjects),
                "successful": successful,
                "failed": failed,
                "total_time": time.time() - pipeline_start_time,
                "results": all_results
            }, f, indent=2)
        
        logger.info(f"✓ Summary saved to {summary_path}")
        
        # Final statistics
        total_time = time.time() - pipeline_start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING PIPELINE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total subjects:    {len(subjects)}")
        logger.info(f"Successful:        {successful}")
        logger.info(f"Failed:            {failed}")
        logger.info(f"Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
        if successful > 0:
            logger.info(f"Average time/subject: {total_time/len(subjects):.1f}s")
        logger.info("=" * 70)
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())