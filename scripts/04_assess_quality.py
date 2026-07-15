#!/usr/bin/env python3
"""
Quality Assessment Script (Baseline Version)
Assesses quality of NIfTI brain MRI images.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pipeline_validator import InputOutputValidator

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config

import numpy as np
import yaml

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Error: nibabel not installed. Install with: pip install nibabel")
    sys.exit(1)

# Import quality metrics modules
from quality_metrics import (
    create_foreground_mask,
    SNRMetric,
    CNRMetric,
    EFCMetric,
    FBERMetric,
    GradientSharpnessMetric,
    VoxelAnisotropyMetric,
    IntensityVarianceMetric,
    CoefficientOfVariationMetric,
    AVAILABLE_METRICS
)

from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics


def _load_image_data(nifti_path):
    """Load a NIfTI file's voxel data as float32.

    nibabel's ``get_fdata()`` defaults to float64. On high-resolution volumes
    (e.g. SibBMS at 0.35 mm, ~231M voxels) that is ~1.85 GB for a single array
    before any metric runs, and downstream metrics (gradient_sharpness) multiply
    it into several full-size temporaries — enough to OOM a worker. float32
    halves every downstream array and is more than precise enough for the
    statistical quality metrics (source images are int16).

    Returns:
        (img, data): the nibabel image (for header access) and the float32 array.
    """
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    return img, data


class QualityAssessor:
    """Handles quality assessment of NIfTI images."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize assessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Category counters
        self.category_counts = {
            'GOOD': 0,
            'ACCEPTABLE': 0,
            'POOR': 0
        }
        
        # Initialize metric calculators
        self.metrics = {
            'snr': SNRMetric(),
            'cnr': CNRMetric(),
            'efc': EFCMetric(),
            'fber': FBERMetric(),
            'gradient_sharpness': GradientSharpnessMetric(),
            'voxel_anisotropy': VoxelAnisotropyMetric(),
            'intensity_variance': IntensityVarianceMetric(),
            'coefficient_of_variation': CoefficientOfVariationMetric()
        }

    def find_nifti_images(self, input_dir: Path, input_format: str = 'bids') -> List[Tuple[Path, str, str]]:
        """
        Find all NIfTI images based on input format.
        
        Args:
            input_dir: Root directory with NIfTI files
            input_format: Format of input directory ('bids' or 'upenn-flat')
            
        Returns:
            List of tuples (nifti_path, patient_id, modality)
        """
        if input_format == 'bids':
            return self.find_nifti_images_bids(input_dir)
        elif input_format == 'upenn-flat':
            return self.find_nifti_images_upenn_flat(input_dir)
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    
    def find_nifti_images_bids(self, input_dir: Path) -> List[Tuple[Path, str, str]]:
        """
        Find all NIfTI images in BIDS structure.
        
        Args:
            input_dir: Root directory with BIDS NIfTI structure
            
        Returns:
            List of tuples (nifti_path, patient_id, session_id, modality)
        """
        images = []
        
        # BIDS structure: sub-<ID>/ses-<session>/anat/*.nii.gz
        for subject_dir in sorted(input_dir.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name.replace("ses-", "")
                
                # Look in anat directory
                anat_dir = session_dir / "anat"
                if not anat_dir.exists():
                    continue
                
                # Find NIfTI files
                for nifti_file in sorted(anat_dir.glob("*.nii.gz")):
                    # Extract modality from filename.
                    # Format: sub-<ID>_ses-<session>_<modality>[_extras].nii.gz
                    # Always use index 2 — dcm2niix may append _Eq_1, _e2, etc.
                    # which would make parts[-1] return a spurious suffix, not the modality.
                    parts = nifti_file.stem.replace('.nii', '').split('_')
                    if len(parts) >= 3:
                        modality = parts[2]  # 3rd token is always the modality
                        images.append((nifti_file, patient_id, session_id, modality))
        
        self.logger.info(f"Found {len(images)} NIfTI images")
        return images
    
    def find_nifti_images_upenn_flat(self, input_dir: Path) -> List[Tuple[Path, str, str]]:
        """
        Find all NIfTI images in UPENN-flat structure.
        
        Structure: UPENN-GBM-XXXXX_YY/UPENN-GBM-XXXXX_YY_MODALITY_unstripped.nii.gz
        
        Args:
            input_dir: Root directory with UPENN-flat structure
            
        Returns:
            List of tuples (nifti_path, patient_id, session_id, modality)
        """
        images = []
        
        # Mapping from UPENN naming to standard modality names
        modality_map = {
            'FLAIR': 't2fl',
            'T1GD': 't1c',
            'T1': 't1',
            'T2': 't2'
        }
        
        # Iterate through patient directories
        for patient_dir in sorted(input_dir.glob("UPENN-GBM-*")):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name  # e.g., "UPENN-GBM-00001_11"
            
            # Find all NIfTI files in patient directory
            for nifti_file in sorted(patient_dir.glob("*.nii.gz")):
                # Parse filename: UPENN-GBM-00001_11_FLAIR_unstripped.nii.gz
                filename = nifti_file.stem.replace('.nii', '')
                
                # Split and extract modality (3rd part from end: ..._MODALITY_unstripped)
                parts = filename.split('_')
                if len(parts) >= 2:
                    # Find modality part (usually second to last before "unstripped")
                    if 'unstripped' in parts:
                        modality_idx = parts.index('unstripped') - 1
                    else:
                        modality_idx = -1
                    
                    modality_upenn = parts[modality_idx]
                    
                    # Map to standard modality name
                    modality = modality_map.get(modality_upenn, modality_upenn.lower())

                    # UPENN-flat has no session structure — use a fixed session ID
                    # so the tuple matches (path, patient_id, session_id, modality)
                    images.append((nifti_file, patient_id, "001", modality))
                    self.logger.debug(f"Found: {patient_id} / {modality_upenn} -> {modality}")
        
        self.logger.info(f"Found {len(images)} NIfTI images in UPENN-flat format")
        return images
    
    def normalize_metric(self, value: float, thresholds: Dict, 
                        lower_is_better: bool = False) -> float:
        """
        Normalize metric value to 0-1 scale based on thresholds.
        
        Args:
            value: Raw metric value
            thresholds: Dict with 'good', 'acceptable', 'poor' thresholds
            lower_is_better: If True, lower values are better (like EFC)
            
        Returns:
            Normalized score (0-1)
        """
        good = thresholds['good']
        poor = thresholds['poor']
        
        if lower_is_better:
            # For metrics like EFC where lower is better
            if value <= good:
                return 1.0
            elif value >= poor:
                return 0.0
            else:
                return (poor - value) / (poor - good)
        else:
            # For metrics like SNR/CNR where higher is better
            if value >= good:
                return 1.0
            elif value <= poor:
                return 0.0
            else:
                return (value - poor) / (good - poor)
    
    def calculate_quality_score(self, metrics: Dict, modality: str) -> Tuple[float, Dict]:
        """
        Calculate overall quality score and normalized metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            modality: Modality name (for thresholds)
            
        Returns:
            Tuple of (quality_score, normalized_metrics)
        """
        # Get thresholds for this modality
        if modality not in self.config['thresholds']:
            self.logger.warning(f"No thresholds for modality '{modality}', using t1")
            modality = 't1'
        
        thresholds = self.config['thresholds'][modality]
        weights = self.config['weights']
        
        # Define which metrics have lower_is_better=True
        lower_is_better_metrics = {'efc', 'voxel_anisotropy', 'coefficient_of_variation'}
        
        # Normalize all metrics dynamically
        normalized = {}
        for metric_name, metric_value in metrics.items():
            if metric_name in thresholds:
                lower_is_better = metric_name in lower_is_better_metrics
                normalized[metric_name] = self.normalize_metric(
                    metric_value, thresholds[metric_name], lower_is_better=lower_is_better
                )
        
        # Calculate weighted score (0-1) for all available metrics
        score = 0.0
        for metric_name, weight in weights.items():
            if metric_name in normalized:
                score += weight * normalized[metric_name]
        
        # Convert to 0-100 scale
        score *= 100
        
        return score, normalized
    
    def get_quality_category(self, score: float) -> str:
        """
        Determine quality category based on score.
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            Category name: "GOOD", "ACCEPTABLE", or "POOR"
        """
        categories = self.config['categories']
        
        if score >= categories['good']['min']:
            return "GOOD"
        elif score >= categories['acceptable']['min']:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def assess_image(self, nifti_path: Path, patient_id: str, session_id: str,
                    modality: str, output_dir: Path, skip_existing: bool = True) -> bool:
        """
        Assess quality of a single NIfTI image.
        
        Args:
            nifti_path: Path to NIfTI file
            patient_id: Patient identifier
            session_id: Session identifier
            modality: Modality name
            output_dir: Output directory for reports
            skip_existing: Skip if output file already exists
            
        Returns:
            True if successful, False otherwise
        """
        self.stats['total_images'] += 1

        # Check skip before loading — avoids computing 8 metrics for nothing
        report_dir = output_dir / f"sub-{patient_id}" / f"ses-{session_id}" / "anat"
        report_file = report_dir / f"sub-{patient_id}_ses-{session_id}_{modality}_quality.json"
        if skip_existing and report_file.exists():
            self.logger.debug(f"Skipping {patient_id}/{modality}: output already exists")
            self.stats['skipped'] += 1
            return True

        try:
            # Load NIfTI image (float32 to bound peak memory — see _load_image_data)
            img, data = _load_image_data(nifti_path)
            img_header = img.header

            # Create foreground mask
            mask_method = self.config.get('foreground_mask', {}).get('method', 'otsu')
            fg_mask = create_foreground_mask(data, method=mask_method)

            # Calculate metrics using modular calculators
            calculated_metrics = {}
            for metric_name, metric_calculator in self.metrics.items():
                if metric_name == 'voxel_anisotropy':
                    # Voxel anisotropy needs header
                    calculated_metrics[metric_name] = metric_calculator.calculate(data, fg_mask, img_header)
                else:
                    calculated_metrics[metric_name] = metric_calculator.calculate(data, fg_mask)

            # Calculate quality score
            score, normalized = self.calculate_quality_score(calculated_metrics, modality)
            category = self.get_quality_category(score)

            # Update category counts
            self.category_counts[category] += 1

            # Create report
            report = {
                'file': nifti_path.name,
                'patient_id': patient_id,
                'modality': modality,
                'quality_score': round(score, 2),
                'quality_category': category,
                'metrics': {k: round(v, 3) for k, v in calculated_metrics.items()},
                'normalized_scores': {k: round(v, 3) for k, v in normalized.items()}
            }

            report_dir.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.debug(
                f"Assessed {patient_id}/{modality}: {category} (score: {score:.1f})"
            )
            self.stats['successful'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Failed to assess {nifti_path}: {e}")
            self.stats['failed'] += 1
            return False
    
    def _process_sequential(self, images, output_dir, skip_existing=True):
        """Process images sequentially."""
        for nifti_path, patient_id, session_id, modality in images:
            self.logger.info(f"Assessing: {patient_id}/ses-{session_id} - {modality}")
            self.assess_image(nifti_path, patient_id, session_id, modality, output_dir, skip_existing)
    
    def _process_parallel(self, images, output_dir, workers, skip_existing=True):
        """Process images in parallel using multiprocessing."""
        self.logger.info(f"Starting parallel processing with {workers} workers")
        
        # Prepare arguments for workers
        tasks = [
            (nifti_path, patient_id, session_id, modality, output_dir, self.config, skip_existing)
            for nifti_path, patient_id, session_id, modality in images
        ]
        
        # Update total_images counter BEFORE processing
        self.stats['total_images'] = len(images)
        
        # Process in parallel.
        # ProcessPoolExecutor (not multiprocessing.Pool) so that a worker killed
        # mid-flight — e.g. by the cgroup OOM killer — surfaces as
        # BrokenProcessPool instead of deadlocking. Pool.map() waits forever for
        # a result the dead worker will never return, which previously hung the
        # stage indefinitely with idle workers at ~0% CPU.
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(self._assess_image_wrapper, tasks))
        except BrokenProcessPool:
            self.logger.error(
                f"Quality assessment aborted: a worker process was killed while "
                f"processing {len(tasks)} images with {workers} workers. The most "
                f"likely cause is out-of-memory — lower 'workers' for "
                f"stage_04_quality or free RAM, then retry."
            )
            raise
        
        # Aggregate results from workers
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        category_counts = {'GOOD': 0, 'ACCEPTABLE': 0, 'POOR': 0}

        for success, category in results:
            if success:
                if category == "SKIPPED":
                    skipped_count += 1
                    continue
                successful_count += 1
                if category in category_counts:
                    category_counts[category] += 1
            else:
                failed_count += 1

        self.stats['successful'] = successful_count
        self.stats['failed'] = failed_count
        self.stats['skipped'] = skipped_count

        for category, count in category_counts.items():
            self.category_counts[category] = count

    @staticmethod
    def _assess_image_wrapper(args):
        """Static wrapper for multiprocessing."""
        nifti_path, patient_id, session_id, modality, output_dir, config, skip_existing = args
        
        try:
            # Check skip before loading — avoids computing 8 metrics for nothing
            report_dir = output_dir / f"sub-{patient_id}" / f"ses-{session_id}" / "anat"
            report_file = report_dir / f"sub-{patient_id}_ses-{session_id}_{modality}_quality.json"
            if skip_existing and report_file.exists():
                return True, "SKIPPED"

            # Create metrics calculators
            metrics_calc = {
                'snr': SNRMetric(),
                'cnr': CNRMetric(),
                'efc': EFCMetric(),
                'fber': FBERMetric(),
                'gradient_sharpness': GradientSharpnessMetric(),
                'voxel_anisotropy': VoxelAnisotropyMetric(),
                'intensity_variance': IntensityVarianceMetric(),
                'coefficient_of_variation': CoefficientOfVariationMetric()
            }

            # Load NIfTI image (float32 to bound peak memory — see _load_image_data)
            img, data = _load_image_data(nifti_path)
            img_header = img.header

            # Create foreground mask
            mask_method = config.get('foreground_mask', {}).get('method', 'otsu')
            fg_mask = create_foreground_mask(data, method=mask_method)
            
            # Calculate metrics
            calculated_metrics = {}
            for metric_name, metric_calculator in metrics_calc.items():
                if metric_name == 'voxel_anisotropy':
                    calculated_metrics[metric_name] = metric_calculator.calculate(data, fg_mask, img_header)
                else:
                    calculated_metrics[metric_name] = metric_calculator.calculate(data, fg_mask)
            
            # Get thresholds for this modality
            modality_key = modality if modality in config['thresholds'] else 't1'
            thresholds = config['thresholds'][modality_key]
            weights = config['weights']
            
            # Define which metrics have lower_is_better=True
            lower_is_better_metrics = {'efc', 'voxel_anisotropy', 'coefficient_of_variation'}
            
            # Normalize metrics
            normalized = {}
            for metric_name, metric_value in calculated_metrics.items():
                if metric_name in thresholds:
                    good = thresholds[metric_name]['good']
                    poor = thresholds[metric_name]['poor']
                    lower_is_better = metric_name in lower_is_better_metrics
                    
                    if lower_is_better:
                        if metric_value <= good:
                            normalized[metric_name] = 1.0
                        elif metric_value >= poor:
                            normalized[metric_name] = 0.0
                        else:
                            normalized[metric_name] = (poor - metric_value) / (poor - good)
                    else:
                        if metric_value >= good:
                            normalized[metric_name] = 1.0
                        elif metric_value <= poor:
                            normalized[metric_name] = 0.0
                        else:
                            normalized[metric_name] = (metric_value - poor) / (good - poor)
            
            # Calculate score
            score = 0.0
            for metric_name, weight in weights.items():
                if metric_name in normalized:
                    score += weight * normalized[metric_name]
            score *= 100
            
            # Determine category
            categories = config['categories']
            if score >= categories['good']['min']:
                category = "GOOD"
            elif score >= categories['acceptable']['min']:
                category = "ACCEPTABLE"
            else:
                category = "POOR"
            
            # Create report
            report = {
                'file': nifti_path.name,
                'patient_id': patient_id,
                'modality': modality,
                'quality_score': round(score, 2),
                'quality_category': category,
                'metrics': {k: round(v, 3) for k, v in calculated_metrics.items()},
                'normalized_scores': {k: round(v, 3) for k, v in normalized.items()}
            }
            
            # Save report (report_dir already created above for skip check)
            report_dir.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            return True, category

        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
            return False, "UNKNOWN"
    
    def run(self, input_dir: Path, output_dir: Path, max_subjects: Optional[int] = None, 
            input_format: str = 'bids', benchmark: bool = False, 
            results_dir: Optional[Path] = None, mode: str = 'sequential', workers: int = 1,
            skip_existing: bool = True):
        """
        Run quality assessment on all images.
        
        Args:
            input_dir: Input directory with NIfTI files
            output_dir: Output directory for quality reports
            max_subjects: Maximum number of subjects to process (optional)
            input_format: Format of input directory ('bids' or 'upenn-flat')
            benchmark: Enable performance monitoring
            results_dir: Directory to save benchmark results
            mode: Processing mode ('sequential' or 'parallel')
            workers: Number of workers for parallel processing
            skip_existing: Skip processing if output file exists
        """
        start_time = time.time()
        
        # Setup benchmark monitoring if requested
        monitor = None
        benchmark_logger = None
        if benchmark:
            monitor = PerformanceMonitor(enabled=True)
            results_dir.mkdir(parents=True, exist_ok=True)
            benchmark_logger = BenchmarkLogger(results_dir)
            monitor.start()
        
        self.logger.info("Starting quality assessment")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Input format: {input_format}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Processing mode: {mode}")
        if mode == 'parallel':
            self.logger.info(f"Workers: {workers}")
        
        # Find all images
        images = self.find_nifti_images(input_dir, input_format)
        
        if max_subjects is not None:
            unique_patients = []
            filtered_images = []
            for nifti_path, patient_id, session_id, modality in images:
                if patient_id not in unique_patients:
                    if len(unique_patients) >= max_subjects:
                        break
                    unique_patients.append(patient_id)
                if patient_id in unique_patients:
                    filtered_images.append((nifti_path, patient_id, session_id, modality))

            # Only log when actual filtering happened
            if len(filtered_images) < len(images):
                self.logger.info(
                    f"Limited to first {max_subjects} subjects: "
                    f"{len(filtered_images)} images (was {len(images)})"
                )
            images = filtered_images
        
        if not images:
            self.logger.warning("No NIfTI images found")
            return
        
        # Process images based on mode
        if mode == 'sequential':
            self._process_sequential(images, output_dir, skip_existing)
        else:  # parallel
            self._process_parallel(images, output_dir, workers, skip_existing)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / self.stats['total_images'] if self.stats['total_images'] > 0 else 0
        
        # Handle benchmark results
        if benchmark and monitor:
            monitor.stop()
            
            # Calculate metrics
            throughput = self.stats['total_images'] / elapsed_time if elapsed_time > 0 else 0
            
            # Get performance metrics
            perf_metrics = monitor.get_metrics()
            
            # Calculate speedup and efficiency BEFORE creating metrics
            if mode == 'sequential' and workers == 1:
                # This is the baseline experiment
                speedup = 1.0
                efficiency = 1.0
            else:
                # Get baseline for comparison
                baseline_time = benchmark_logger.get_baseline_time() if benchmark_logger else None
                if baseline_time and avg_time_per_image > 0:
                    speedup = baseline_time / avg_time_per_image
                    efficiency = speedup / workers
                else:
                    # No baseline found, but still record values for future comparison
                    speedup = None
                    efficiency = None

            # Create experiment metrics
            experiment_id = f"{mode}_{workers}workers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metrics = ExperimentMetrics(
                experiment_id=experiment_id,
                timestamp=datetime.now().isoformat(),
                mode=mode,
                workers=workers,
                total_series=self.stats['total_images'],
                successful=self.stats['successful'],
                failed=self.stats['failed'],
                skipped=self.stats['skipped'],
                total_time=elapsed_time,
                time_per_series=avg_time_per_image,
                throughput=throughput,
                cpu_avg=perf_metrics.get('cpu_avg'),
                cpu_max=perf_metrics.get('cpu_max'),
                memory_avg_mb=perf_metrics.get('memory_avg_mb'),
                memory_peak_mb=perf_metrics.get('memory_peak_mb'),
                speedup=speedup,
                efficiency=efficiency
            )
            
            # Save metrics
            benchmark_logger.log_metrics(metrics)
            
            self.logger.info("=" * 60)
            self.logger.info("Performance Metrics:")
            self.logger.info(f"Throughput: {throughput:.3f} images/sec")
            if perf_metrics.get('cpu_avg'):
                self.logger.info(f"CPU Average: {perf_metrics['cpu_avg']:.2f}%")
                self.logger.info(f"CPU Peak: {perf_metrics['cpu_max']:.2f}%")
            if perf_metrics.get('memory_avg_mb'):
                self.logger.info(f"Memory Average: {perf_metrics['memory_avg_mb']:.2f} MB")
                self.logger.info(f"Memory Peak: {perf_metrics['memory_peak_mb']:.2f} MB")
            if speedup is not None:
                self.logger.info(f"Speedup: {speedup:.3f}x")
                if efficiency is not None:
                    self.logger.info(f"Efficiency: {efficiency:.3f}")
            self.logger.info("=" * 60)
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Assessment completed")
        self.logger.info(f"Total images: {self.stats['total_images']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"GOOD: {self.category_counts['GOOD']}")
        self.logger.info(f"ACCEPTABLE: {self.category_counts['ACCEPTABLE']}")
        self.logger.info(f"POOR: {self.category_counts['POOR']}")
        self.logger.info(f"Total time: {elapsed_time:.2f}s")
        self.logger.info(f"Average time per image: {avg_time_per_image:.3f}s")
        self.logger.info("=" * 60)

        self.logger.info("Validating input-output correspondence...")
        
        try:
            validator = InputOutputValidator(logger=self.logger)
            
            # Сканируем входную структуру (NIfTI)
            self.logger.info("Scanning input structure (NIfTI)...")
            input_structure = validator.scan_structure(input_dir, format_type='bids-nifti')
            self.logger.info(f"Found {len(input_structure)} patients in input")
            
            # Сканируем выходную структуру (качество JSON)
            self.logger.info("Scanning output structure (quality metrics)...")
            output_structure = validator.scan_structure(output_dir, format_type='bids-quality')
            self.logger.info(f"Found {len(output_structure)} patients in output")
            
            # Сравниваем структуры
            self.logger.info("Comparing structures...")
            comparison_result = validator.compare_structures(input_structure, output_structure)
            
            # Генерируем отчет
            validator.generate_incomplete_report(
                comparison_result=comparison_result,
                stage_name="04_assess_quality",
                output_dir=output_dir
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}", exc_info=True)
        
        self.logger.info("=" * 60)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("quality_assessor")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Assess quality of NIfTI brain MRI images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with NIfTI files in BIDS format"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for quality reports"
    )
    
    # Optional arguments
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "quality_config.yaml",
        help="Path to quality configuration file"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=['bids', 'upenn-flat'],
        default='bids',
        help="Input directory structure format: 'bids' (default) or 'upenn-flat'"
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
        default=Path("results_quality"),
        help="Directory to save benchmark results (default: ./results_quality)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip processing if output file already exists (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing: clear output directory before starting"
    )
    parser.add_argument(
        '--lesion-type',
        type=str,
        default='glioblastoma',
        choices=['glioblastoma', 'multiple_sclerosis'],
        help='Type of brain lesion — determines expected modalities'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not args.config.exists():
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_file)

    # Handle --force flag: clear output directory
    if args.force:
        if args.output_dir.exists():
            import shutil
            logger_temp = logging.getLogger("temp")
            logger_temp.addHandler(logging.StreamHandler(sys.stdout))
            logger_temp.setLevel(logging.INFO)
            logger_temp.info(f"--force flag: clearing output directory {args.output_dir}")
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load lesion type config and expected modalities
        try:
            lt_config = load_lesion_type_config(args.lesion_type)
            expected_modalities = set(lt_config['required_modalities'])
        except KeyError:
            logger.warning(f"Unknown lesion_type '{args.lesion_type}', using all modalities")
            expected_modalities = {'t1', 't1c', 't2', 't2fl'}

        logger.info(f"Lesion type: {args.lesion_type}, expected modalities: {sorted(expected_modalities)}")

        # Determine number of workers
        if args.mode == 'parallel':
            if args.workers == 1:
                args.workers = cpu_count()
                logger.info(f"Workers not specified for parallel mode, using CPU count: {args.workers}")
            elif args.workers > cpu_count():
                logger.warning(f"Requested {args.workers} workers but only {cpu_count()} CPUs available")
        else:
            args.workers = 1  # Sequential mode always uses 1 worker
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create assessor and run
        assessor = QualityAssessor(config, logger)
        assessor.run(
            args.input_dir, 
            args.output_dir,
            max_subjects=args.max_subjects,
            input_format=args.input_format,
            benchmark=args.benchmark,
            results_dir=args.results_dir if args.benchmark else None,
            mode=args.mode,
            workers=args.workers,
            skip_existing=args.skip_existing
        )
        
    except KeyboardInterrupt:
        logger.info("\nAssessment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()