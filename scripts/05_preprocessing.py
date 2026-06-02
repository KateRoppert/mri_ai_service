"""
MRI Brain Preprocessing Pipeline for UPENN-GBM dataset.
Performs reorientation, bias correction, registration to atlas, and skull stripping.

Usage:
    python 05_preprocessing.py <input_dir> <output_dir> [options]
"""

import argparse
import logging
import sys
import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Allow imports from the project root (utils/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from pipeline_validator import InputOutputValidator

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
from preprocessing_steps.resampling import process_subject_resampling

logger = logging.getLogger(__name__)

# Cores reserved for the OS, display server, and IDE on a workstation.
# Prevents ANTs from starving the desktop environment and avoids RAM pressure
# from too many concurrent ITK threads on machines with ≤32 GB RAM.
_OS_RESERVED_CORES = 2


def calculate_optimal_parallelism(
    n_subjects: int, cpu_count: int, max_workers: int
) -> tuple[int, int]:
    """
    Calculate actual worker count and threads per worker for parallel mode.

    `max_workers` (from config) is treated as a ceiling — the user's resource cap.
    The formula then reduces it further if subjects or CPU count make a smaller
    number more efficient:

      effective_cpu       = cpu_count - _OS_RESERVED_CORES   # leave headroom for OS/IDE
      workers_by_subjects = min(max_workers, n_subjects)      # no idle processes
      workers_by_cpu      = max(1, effective_cpu // 8)        # keep >=8 threads/worker
      actual_workers      = min(workers_by_subjects, workers_by_cpu)
      threads             = effective_cpu // actual_workers

    N4 bias correction (ITK) and ANTs registration both benefit from ~8-12 threads;
    giving a worker fewer than 8 threads leaves most of the allocated CPUs idle.
    """
    effective_cpu = max(4, cpu_count - _OS_RESERVED_CORES)
    workers_by_subjects = min(max_workers, n_subjects)
    workers_by_cpu = max(1, effective_cpu // 8)
    actual_workers = max(1, min(workers_by_subjects, workers_by_cpu))
    threads = max(1, effective_cpu // actual_workers)
    return actual_workers, threads


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

def check_subject_processed(
    input_dir: Path,
    output_dir: Path,
    subject_id: str,
    session_id: str,
    modalities: list
) -> tuple[bool, list]:
    """
    Check if subject has already been processed.
    Compares input and output: if all input modalities exist in output, 
    subject is considered processed.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        subject_id: Subject ID
        session_id: Session ID
        modalities: List of all possible modalities
    
    Returns:
        tuple: (is_processed: bool, missing_on_output: list)
    """
    input_anat = input_dir / subject_id / session_id / "anat"
    output_anat = output_dir / subject_id / session_id / "anat"
    
    # Find which modalities exist on INPUT
    input_modalities = []
    for modality in modalities:
        filename = f"{subject_id}_{session_id}_{modality}.nii.gz"
        input_file = input_anat / filename
        
        if input_file.exists():
            input_modalities.append(modality)
    
    # If no input modalities found, cannot be processed
    if not input_modalities:
        return False, []
    
    # Check if all INPUT modalities exist on OUTPUT
    missing_on_output = []
    for modality in input_modalities:
        filename = f"{subject_id}_{session_id}_{modality}.nii.gz"
        output_file = output_anat / filename
        
        if not output_file.exists():
            missing_on_output.append(modality)
    
    # Processed if no missing modalities on output
    is_processed = len(missing_on_output) == 0
    
    return is_processed, missing_on_output

def process_single_subject(
    anat_dir: Path,
    subject_id: str,
    session_id: str,
    output_dir: Path,
    transform_dir: Path,
    temp_dir: Path,
    atlas_path: Path,
    config: dict,
    modalities: List[str],
    lesion_type: str = 'glioblastoma'
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

    # Early check: verify reference modality exists BEFORE starting any processing
    reference_modality = None
    for step in config.get('steps', []):
        if step['name'] == 'registration':
            reference_modality = step.get('params', {}).get('reference_modality')
            break

    if reference_modality is None:
        # Derive from lesion_types.yaml rather than hardcoding 't1c'
        try:
            reference_modality = load_lesion_type_config(lesion_type)['reference_modality']
        except KeyError:
            reference_modality = 't1'

    ref_pattern = f"{subject_id}_{session_id}_{reference_modality}.nii.gz"
    ref_files = list(anat_dir.glob(ref_pattern))

    if not ref_files:
        error_msg = f"Reference modality '{reference_modality}' not found (file: {ref_pattern})"
        logger.warning(f"⊙ Skipping {subject_id}/{session_id}: {error_msg}")
        results['success'] = False
        results['skipped'] = True
        results['skip_reason'] = f"missing_reference_modality_{reference_modality}"
        results['errors'].append(error_msg)
        return results
    
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
            
            # Check if reference modality succeeded (reference_modality resolved above from steps config)
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
                "success": registration_results.get(step_params.get('reference_modality', reference_modality), {}).get('success', False),
                "results": registration_results,
                "time": time.time() - step_start
            }
            
            if not results['steps']['registration']['success']:
                raise RuntimeError("Registration failed for reference modality")
            
            logger.info(f"✓ Registration completed in {results['steps']['registration']['time']:.1f}s")
        else:
            logger.info("Step 3/4: Registration - SKIPPED (disabled in config)")


        # STEP 3.5: Resampling (if registration is disabled)
        step_name = "resampling"
        if steps_config.get(step_name, {}).get('enabled', False):
            logger.info("Step 3.5/4: Resampling to target resolution")
            step_start = time.time()
            
            step_params = steps_config.get(step_name, {}).get('params', {})
            
            # Get input data (either from reorient or previous step)
            if steps_config.get('registration', {}).get('enabled', False):
                logger.warning("Both registration and resampling enabled - using registration output")
                input_anat = output_dir / subject_id / session_id / "anat"
            else:
                input_anat = temp_reoriented / subject_id / session_id / "anat"
            
            resampling_results = process_subject_resampling(
                subject_dir=input_anat,
                output_dir=output_dir,
                modalities=modalities,
                params=step_params
            )
            
            results['steps']['resampling'] = {
                "success": all(r.get('success', False) for r in resampling_results.values() if not r.get('skipped')),
                "results": resampling_results,
                "time": time.time() - step_start
            }
            
            logger.info(f"✓ Resampling completed in {results['steps']['resampling']['time']:.1f}s")
        else:
            logger.info("Step 3.5/4: Resampling - SKIPPED (disabled in config)")
        
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
                transform_dir=transform_dir,
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
        
        # Cleanup temporary files
        logger.info("Cleaning up temporary files")
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

def process_subject_wrapper(args_tuple):
    """
    Wrapper for parallel processing of subjects.
    Unpacks arguments and calls process_single_subject.
    Sets thread limits and creates isolated temp directory for this worker.
    
    Args:
        args_tuple: Tuple of arguments for process_single_subject
    
    Returns:
        dict: Processing results
    """
    import traceback
    
    (anat_dir, subject_id, session_id, output_dir, transform_dir,
     base_temp_dir, atlas_path, config, modalities, lesion_type, threads_per_worker) = args_tuple
    
    # Set thread limits for this worker (only in parallel mode)
    if threads_per_worker is not None:
        os.environ['ANTS_RANDOM_SEED'] = '42'
        os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads_per_worker)
        os.environ['ANTS_NUMBER_OF_THREADS'] = str(threads_per_worker)
    
    # Create isolated temp directory for this worker
    worker_id = os.getpid()
    worker_temp_dir = base_temp_dir / f"worker_{worker_id}"
    worker_temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = process_single_subject(
            anat_dir=anat_dir,
            subject_id=subject_id,
            session_id=session_id,
            output_dir=output_dir,
            transform_dir=transform_dir,
            temp_dir=worker_temp_dir,  # Use worker-specific temp dir
            atlas_path=atlas_path,
            config=config,
            modalities=modalities,
            lesion_type=lesion_type
        )
        
        # Cleanup worker temp directory for this subject
        subject_temp = worker_temp_dir / "reoriented" / subject_id
        if subject_temp.exists():
            shutil.rmtree(subject_temp, ignore_errors=True)
        
        subject_temp = worker_temp_dir / "bias_corrected" / subject_id
        if subject_temp.exists():
            shutil.rmtree(subject_temp, ignore_errors=True)
        
        return result
        
    except Exception as e:
        logger.error(f"Worker exception for {subject_id}/{session_id}: {e}", exc_info=True)
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "success": False,
            "errors": [str(e)],
            "traceback": traceback.format_exc()
        }
    finally:
        # Try to cleanup empty worker temp directory
        try:
            if worker_temp_dir.exists() and not any(worker_temp_dir.rglob('*')):
                worker_temp_dir.rmdir()
        except:
            pass


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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable detailed performance monitoring and save metrics"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['sequential', 'parallel'],
        default='sequential',
        help="Processing mode: sequential or parallel (default: sequential)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel mode (default: 1)"
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results_preprocessing"),
        help="Directory to save benchmark results (default: ./results_preprocessing)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable input-output validation and generate completeness report"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects that have already been processed (resume interrupted processing)"
    )

    parser.add_argument(
        '--lesion-type',
        type=str,
        default='glioblastoma',
        choices=['glioblastoma', 'multiple_sclerosis'],
        help='Type of brain lesion — determines which modalities to process'
    )

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file, level="INFO")

    cpu_count = os.cpu_count() or 4

    logger.info(f"System: {cpu_count} CPU cores available")
    logger.info(f"Mode: {args.mode}")

    if args.mode == 'sequential':
        # Sequential: give all cores to the single worker
        threads_per_worker = cpu_count
        logger.info(f"Workers: 1 | Threads: {threads_per_worker} (all cores)")
        os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads_per_worker)
        os.environ['ANTS_NUMBER_OF_THREADS'] = str(threads_per_worker)
    else:
        # Parallel: actual workers/threads calculated after subject discovery
        threads_per_worker = 1  # placeholder; overwritten before executor starts
        logger.info(f"Workers (configured cap): {args.workers} | Threads: auto (calculated after subject scan)")
    
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
        preprocessed_dir = args.output_dir
        transform_dir = args.output_dir.parent / "transformations" 
        temp_dir = args.output_dir.parent / "temp" 

        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Output directories created")
        
        # Resolve modalities from lesion_types.yaml; fall back to preprocessing config
        try:
            lt_config = load_lesion_type_config(args.lesion_type)
            modalities = lt_config['required_modalities']
            logger.info(
                f"Lesion type: {args.lesion_type}, processing modalities: {modalities}"
            )
        except KeyError:
            modalities = config.get('modalities', ['t1c', 't1', 't2', 't2fl'])
            logger.warning(f"Unknown lesion_type '{args.lesion_type}', using config modalities")
        
        # Find subjects
        subjects = find_subjects(args.input_dir, max_subjects=None)

        if not subjects:
            logger.error("No subjects found to process")
            return 1

        if args.max_subjects and len(subjects) > args.max_subjects:
            subjects = subjects[:args.max_subjects]
            logger.info(f"Limited to first {args.max_subjects} subjects (--max-subjects)")
        
        # Process each subject
        all_results = []
        successful = 0
        failed = 0
        skipped = 0

        # Initialize performance monitoring if benchmark mode
        monitor = None
        benchmark_logger = None

        if args.benchmark:
            monitor = PerformanceMonitor(enabled=True)
            benchmark_logger = BenchmarkLogger(args.results_dir)
            
            # Clear output directory for clean benchmark
            if preprocessed_dir.exists():
                logger.warning(f"Clearing output directory for benchmark: {preprocessed_dir}")
                shutil.rmtree(preprocessed_dir)
                preprocessed_dir.mkdir(parents=True)
            if transform_dir.exists():
                shutil.rmtree(transform_dir)
                transform_dir.mkdir(parents=True)
            
            monitor.start()
            logger.info("Performance monitoring started")

        # Prepare base arguments (threads slot is None for parallel — filled after subject scan)
        threads_for_parallel = None if args.mode == 'parallel' else None

        processing_args = [
            (anat_dir, subject_id, session_id, preprocessed_dir, transform_dir,
            temp_dir, atlas_path, config, modalities, args.lesion_type, threads_for_parallel)
            for anat_dir, subject_id, session_id in subjects
        ]

        # Sequential or parallel processing
        if args.mode == 'sequential':
            logger.info(f"Processing mode: SEQUENTIAL")
            if args.skip_existing:
                logger.info("Skip-existing mode: ENABLED")
            
            for idx, args_tuple in enumerate(processing_args, 1):
                anat_dir, subject_id, session_id = args_tuple[0], args_tuple[1], args_tuple[2]
                
                logger.info(f"\n{'=' * 70}")
                logger.info(f"Subject {idx}/{len(subjects)}")
                logger.info(f"{'=' * 70}")
                
                # Check if already processed
                if args.skip_existing:
                    is_processed, missing_mods = check_subject_processed(
                        args.input_dir, preprocessed_dir, subject_id, session_id, modalities
                    )
                    
                    if is_processed:
                        logger.info(f"⊙ {subject_id}/{session_id} already processed, skipping")
                        all_results.append({
                            "subject_id": subject_id,
                            "session_id": session_id,
                            "success": True,
                            "skipped": True
                        })
                        skipped += 1
                        continue
                    elif missing_mods:
                        logger.info(f"→ {subject_id}/{session_id} partially processed, missing: {', '.join(missing_mods)}")
                
                result = process_subject_wrapper(args_tuple)
                all_results.append(result)
                
                if result['success']:
                    successful += 1
                elif result.get('skipped'):
                    # Subject was skipped due to missing data
                    logger.warning(f"⊙ Skipped: {result.get('skip_reason', 'unknown reason')}")
                    failed += 1  # Count as failed since not processed
                else:
                    failed += 1

        elif args.mode == 'parallel':
            logger.info("Processing mode: PARALLEL")
            if args.skip_existing:
                logger.info("Skip-existing mode: ENABLED")

                # Filter out already processed subjects
                filtered_args = []
                skipped_count = 0

                for args_tuple in processing_args:
                    subject_id, session_id = args_tuple[1], args_tuple[2]

                    is_processed, missing_mods = check_subject_processed(
                        args.input_dir, preprocessed_dir, subject_id, session_id, modalities
                    )

                    if is_processed:
                        logger.info(f"⊙ {subject_id}/{session_id} already processed, skipping")
                        all_results.append({
                            "subject_id": subject_id,
                            "session_id": session_id,
                            "success": True,
                            "skipped": True
                        })
                        skipped += 1
                        skipped_count += 1
                    else:
                        if missing_mods:
                            logger.info(f"→ {subject_id}/{session_id} partially processed, missing: {', '.join(missing_mods)}")
                        filtered_args.append(args_tuple)

                logger.info(f"Skipped {skipped_count} already processed subjects")
                logger.info(f"Processing {len(filtered_args)} remaining subjects")

                processing_args = filtered_args

            if processing_args:
                # Calculate actual parallelism now that we know the real workload
                actual_workers, threads_per_worker = calculate_optimal_parallelism(
                    n_subjects=len(processing_args),
                    cpu_count=cpu_count,
                    max_workers=args.workers,
                )

                reason_parts = []
                if actual_workers < args.workers:
                    if actual_workers == len(processing_args):
                        reason_parts.append(f"subjects={len(processing_args)}")
                    if actual_workers == max(1, cpu_count // 8):
                        reason_parts.append(f"CPU-optimal={max(1, cpu_count // 8)} for {cpu_count} cores")
                    reason = f" (cap={args.workers} → {', '.join(reason_parts)})"
                else:
                    reason = ""

                logger.info(
                    f"Workers: {actual_workers}{reason} | "
                    f"Threads per worker: {threads_per_worker} | "
                    f"Total threads: {actual_workers * threads_per_worker}/{cpu_count}"
                )

                # Inject the calculated thread count into every args tuple (last slot)
                processing_args = [t[:-1] + (threads_per_worker,) for t in processing_args]

                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    # Submit all jobs
                    future_to_subject = {
                        executor.submit(process_subject_wrapper, args_tuple): (args_tuple[1], args_tuple[2])
                        for args_tuple in processing_args
                    }
                    
                    # Process completed jobs
                    for idx, future in enumerate(as_completed(future_to_subject), 1):
                        subject_id, session_id = future_to_subject[future]
                        
                        try:
                            result = future.result()
                            all_results.append(result)
                            
                            if result['success']:
                                successful += 1
                                logger.info(f"✓ [{idx}/{len(processing_args)}] {subject_id}/{session_id} completed")
                            elif result.get('skipped'):
                                failed += 1
                                logger.warning(f"⊙ [{idx}/{len(processing_args)}] {subject_id}/{session_id} skipped: {result.get('skip_reason', 'unknown')}")
                            else:
                                failed += 1
                                logger.error(f"✗ [{idx}/{len(processing_args)}] {subject_id}/{session_id} failed")
                                
                        except Exception as e:
                            failed += 1
                            logger.error(f"✗ [{idx}/{len(processing_args)}] {subject_id}/{session_id} exception: {e}")
                            all_results.append({
                                "subject_id": subject_id,
                                "session_id": session_id,
                                "success": False,
                                "errors": [str(e)]
                            })
            else:
                logger.info("All subjects already processed, nothing to do")

        # Stop performance monitoring
        if monitor:
            monitor.stop()
            logger.info("Performance monitoring stopped")
        
        # Cleanup temp directory
        try:
            if temp_dir.exists():
                # Remove worker temp directories
                for worker_dir in temp_dir.glob("worker_*"):
                    if worker_dir.is_dir():
                        shutil.rmtree(worker_dir, ignore_errors=True)
                        logger.debug(f"Removed worker directory: {worker_dir.name}")
                
                # Remove any remaining empty subdirectories
                for subdir in temp_dir.iterdir():
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                        logger.debug(f"Removed empty subdirectory: {subdir.name}")
                
                # Now remove temp_dir if empty
                if not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
                    logger.info("✓ Removed temporary directory")
                else:
                    logger.warning(f"Temporary directory not empty, keeping: {temp_dir}")
                    logger.debug(f"Remaining contents: {list(temp_dir.iterdir())}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory: {e}")
        
        # Processing time (excludes validation) — used for benchmark metrics
        processing_time = time.time() - pipeline_start_time

        # Save benchmark metrics if enabled
        if args.benchmark and monitor and benchmark_logger:
            system_metrics = monitor.get_metrics()
            
            # Generate experiment ID
            experiment_id = f"preproc_{args.mode}_w{args.workers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create metrics object
            metrics = ExperimentMetrics(
                experiment_id=experiment_id,
                timestamp=datetime.now().isoformat(),
                mode=args.mode,
                workers=args.workers if args.mode == 'parallel' else 1,
                total_series=len(subjects),
                successful=successful,
                failed=failed,
                skipped=skipped,
                total_time=processing_time,
                time_per_series=processing_time / len(subjects) if len(subjects) > 0 else 0,
                throughput=len(subjects) / processing_time if processing_time > 0 else 0,
                cpu_avg=system_metrics.get('cpu_avg'),
                cpu_max=system_metrics.get('cpu_max'),
                memory_avg_mb=system_metrics.get('memory_avg_mb'),
                memory_peak_mb=system_metrics.get('memory_peak_mb')
            )
            
            # Calculate speedup and efficiency vs baseline
            if args.mode == 'sequential' and args.workers == 1:
                # This IS the baseline
                metrics.speedup = 1.0
                metrics.efficiency = 1.0
            else:
                # Compare to baseline
                baseline_time = benchmark_logger.get_baseline_time()
                if baseline_time and baseline_time > 0:
                    metrics.speedup = baseline_time / metrics.time_per_series
                    metrics.efficiency = metrics.speedup / metrics.workers
                else:
                    logger.warning("No baseline found. Run sequential mode with 1 worker first.")
            
            # Log metrics
            benchmark_logger.log_metrics(metrics)
            
            logger.info(f"✓ Benchmark metrics saved to {args.results_dir}")

        # Validation: compare input and output structures
        if args.validate:
            logger.info("\n" + "=" * 70)
            logger.info("VALIDATING INPUT-OUTPUT CORRESPONDENCE")
            logger.info("=" * 70)
            
            try:
                validator = InputOutputValidator(logger=logger)
                
                # Scan input structure
                logger.info("Scanning input structure...")
                input_structure = validator.scan_structure(
                    directory=args.input_dir,
                    format_type='bids-nifti'
                )
                
                # Scan output structure
                logger.info("Scanning output structure...")
                output_structure = validator.scan_structure(
                    directory=preprocessed_dir,
                    format_type='bids-nifti'
                )
                
                # Compare structures
                logger.info("Comparing input and output structures...")
                comparison_result = validator.compare_structures(
                    input_structure=input_structure,
                    output_structure=output_structure
                )
                
                # Generate report
                report_path = validator.generate_incomplete_report(
                    comparison_result=comparison_result,
                    stage_name="05_preprocessing",
                    output_dir=args.output_dir,
                    filename="preprocessing_incomplete_data.json"
                )
                
                # Log summary
                stats = comparison_result['statistics']
                logger.info(f"\n{'=' * 70}")
                logger.info("VALIDATION SUMMARY")
                logger.info(f"{'=' * 70}")
                logger.info(f"Total patients:      {stats['total_patients']}")
                logger.info(f"Complete patients:   {stats['complete_patients']}")
                logger.info(f"Incomplete patients: {stats['incomplete_patients']}")
                logger.info(f"Total sessions:      {stats['total_sessions']}")
                logger.info(f"Complete sessions:   {stats['complete_sessions']}")
                logger.info(f"Incomplete sessions: {stats['incomplete_sessions']}")
                logger.info(f"Success rate:        {stats['success_rate_percent']}%")
                logger.info(f"{'=' * 70}")
                
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}", exc_info=True)
        
        # Final statistics
        total_time = time.time() - pipeline_start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING PIPELINE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total subjects:    {len(subjects)}")
        logger.info(f"Successful:        {successful}")
        logger.info(f"Skipped:           {skipped}")
        logger.info(f"Failed:            {failed}")
        logger.info(f"Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
        if successful > 0:
            logger.info(f"Average time/subject: {total_time/len(subjects):.1f}s")
        logger.info("=" * 70)

        # Add validation info if enabled
        if args.validate and 'comparison_result' in locals():
            logger.info(f"\nValidation:")
            logger.info(f"  Success rate:       {comparison_result['statistics']['success_rate_percent']}%")
            logger.info(f"  Complete sessions:  {comparison_result['statistics']['complete_sessions']}/{comparison_result['statistics']['total_sessions']}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())