#!/usr/bin/env python3
"""
DICOM Metadata Extraction Script (Baseline Version)
Extracts metadata from DICOM files organized in BIDS format.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pydicom
import yaml
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from datetime import datetime
from multiprocessing import Pool, cpu_count


class MetadataExtractor:
    """Handles DICOM metadata extraction."""
    
    def __init__(self, tags_config: Dict, logger: logging.Logger):
        """
        Initialize extractor.
        
        Args:
            tags_config: Dictionary with DICOM tags configuration
            logger: Logger instance
        """
        self.tags_config = tags_config
        self.logger = logger
        self.stats = {
            'total_series': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        self.monitor = None
        self.benchmark_mode = False
    
    def load_tags_config(self, config_path: Path) -> Dict:
        """Load DICOM tags configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load tags config: {e}")
            raise
    
    def find_dicom_series(self, input_dir: Path) -> List[Tuple[Path, str, str, str]]:
        """
        Find all DICOM series in BIDS structure.
        
        Args:
            input_dir: Root directory with BIDS structure
            
        Returns:
            List of tuples (series_path, patient_id, session_id, modality)
        """
        series_list = []

        subject_dirs = sorted(input_dir.glob("sub-*"))
        
        # BIDS structure: sub-<patientID>/ses-<session>/anat/<modality>/
        for subject_dir in subject_dirs:
            if not subject_dir.is_dir():
                continue
                
            patient_id = subject_dir.name.replace("sub-", "")
            
            # Look for session directories
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                
                # Look for anat directory within session
                anat_dir = session_dir / "anat"
                if not anat_dir.exists():
                    continue
                
                # Look for modality directories within anat
                for modality_dir in anat_dir.iterdir():
                    if not modality_dir.is_dir():
                        continue
                    
                    # Check if there are DICOM files directly
                    dicom_files = list(modality_dir.glob("*.dcm")) + \
                                list(modality_dir.glob("*.DCM")) + \
                                list(modality_dir.glob("*.dicom"))
                    
                    if dicom_files:
                        modality = modality_dir.name
                        series_list.append((modality_dir, patient_id, session_id, modality))
                    else:
                        # Check subdirectories for DICOM files
                        for subdir in modality_dir.iterdir():
                            if not subdir.is_dir():
                                continue
                            
                            dicom_files = list(subdir.glob("*.dcm")) + \
                                        list(subdir.glob("*.DCM")) + \
                                        list(subdir.glob("*.dicom"))
                            
                            if dicom_files:
                                # Use subdirectory name as modality hint
                                modality = subdir.name
                                series_list.append((subdir, patient_id, session_id, modality))
        
        self.logger.info(f"Found {len(series_list)} DICOM series")
        return series_list
    
    def get_first_dicom_file(self, series_path: Path) -> Optional[Path]:
        """
        Get the first DICOM file from a series directory.
        
        Args:
            series_path: Path to directory containing DICOM files
            
        Returns:
            Path to first DICOM file or None
        """
        # Try different extensions
        for pattern in ["*.dcm", "*.DCM", "*.dicom", "*"]:
            dicom_files = sorted(series_path.glob(pattern))
            # Filter out non-DICOM files when using wildcard
            if pattern == "*":
                dicom_files = [f for f in dicom_files if f.is_file() and 
                              not f.suffix in ['.json', '.txt', '.yaml', '.yml']]
            
            if dicom_files:
                return dicom_files[0]
        
        return None
    
    def extract_metadata(self, dicom_path: Path) -> Optional[Dict]:
        """
        Extract metadata from a single DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Dictionary with extracted metadata or None on failure
        """
        try:
            # Read DICOM file
            dcm = pydicom.dcmread(str(dicom_path), force=True)
            
            metadata = {
                'source_file': str(dicom_path.name),
                'extraction_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Extract tags from all groups
            required_missing = []
            
            for group_name, tags in self.tags_config.items():
                metadata[group_name] = {}
                
                for tag_name, tag_info in tags.items():
                    group, element, required = tag_info
                    tag_address = (int(group, 16), int(element, 16))
                    
                    try:
                        if tag_address in dcm:
                            value = dcm[tag_address].value
                            
                            # Convert to JSON-serializable format
                            if hasattr(value, 'tolist'):  # numpy array
                                value = value.tolist()
                            elif isinstance(value, bytes):
                                value = value.decode('utf-8', errors='ignore')
                            elif hasattr(value, '__iter__') and not isinstance(value, str):
                                value = list(value)
                            
                            metadata[group_name][tag_name] = value
                        else:
                            metadata[group_name][tag_name] = None
                            if required:
                                required_missing.append(tag_name)
                    except Exception as e:
                        self.logger.warning(f"Error extracting tag {tag_name}: {e}")
                        metadata[group_name][tag_name] = None
                        if required:
                            required_missing.append(tag_name)
            
            # Check if critical tags are missing
            if required_missing:
                self.logger.warning(
                    f"Missing required tags in {dicom_path}: {required_missing}"
                )
                return None
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to read DICOM file {dicom_path}: {e}")
            return None
    
    def save_metadata(self, metadata: Dict, output_dir: Path, 
                      patient_id: str, session: str, modality: str) -> bool:
        """
        Save metadata to JSON file following BIDS naming convention.
        
        Args:
            metadata: Extracted metadata dictionary
            output_dir: Root output directory
            patient_id: Patient identifier
            session: Session identifier (e.g., "01")
            modality: Modality name (e.g., "T1", "T2")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create BIDS-compliant directory structure
            subject_dir = output_dir / f"sub-{patient_id}"
            session_dir = subject_dir / f"ses-{session}"
            anat_dir = session_dir / "anat"
            modality_dir = anat_dir / modality
            modality_dir.mkdir(parents=True, exist_ok=True)
            
            # BIDS naming: sub-<label>_ses-<label>_<modality>.json
            filename = f"sub-{patient_id}_ses-{session}_{modality}.json"
            output_path = modality_dir / filename
            
            # Save JSON
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved metadata to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            return False
    
    def process_series(self, series_path: Path, patient_id: str, 
                       modality: str, output_dir: Path, session: str = "001") -> bool:
        """
        Process a single DICOM series.
        
        Args:
            series_path: Path to series directory
            patient_id: Patient identifier
            modality: Modality name
            output_dir: Output directory
            session: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        self.stats['total_series'] += 1
        
        # Find first DICOM file
        dicom_file = self.get_first_dicom_file(series_path)
        
        if dicom_file is None:
            self.logger.warning(f"No DICOM files found in {series_path}")
            self.stats['skipped'] += 1
            return False
        
        # Extract metadata
        metadata = self.extract_metadata(dicom_file)
        
        if metadata is None:
            self.logger.warning(f"Failed to extract metadata from {series_path}")
            self.stats['failed'] += 1
            return False
        
        # Save metadata
        success = self.save_metadata(metadata, output_dir, patient_id, session, modality)
        
        if success:
            self.stats['successful'] += 1
            return True
        else:
            self.stats['failed'] += 1
            return False
        
    @staticmethod
    def process_series_wrapper(args):
        """
        Static wrapper for multiprocessing.
        
        Args:
            args: Tuple of (series_path, patient_id, modality, output_dir, session, tags_config)
            
        Returns:
            Tuple of (success, stats_update)
        """
        series_path, patient_id, modality, output_dir, session, tags_config = args
        
        # Create a temporary extractor (no logger in worker processes)
        import logging
        temp_logger = logging.getLogger(f"worker_{patient_id}")
        temp_logger.setLevel(logging.WARNING)  # Minimize worker logging
        
        temp_extractor = MetadataExtractor(tags_config, temp_logger)
        
        # Process the series
        success = temp_extractor.process_series(series_path, patient_id, modality, output_dir, session)
        
        # Return success and stats update
        return (success, temp_extractor.stats)
    
    def run(self, input_dir: Path, output_dir: Path, max_subjects: Optional[int] = None, 
        benchmark: bool = False, results_dir: Optional[Path] = None,
        mode: str = 'sequential', workers: int = 1):
        """
        Run metadata extraction on all series.
        
        Args:
            input_dir: Input directory with BIDS structure
            output_dir: Output directory for JSON files
            max_subjects: Maximum number of subjects to process (optional)
            benchmark: Enable detailed performance monitoring
            results_dir: Directory to save benchmark results
            mode: Processing mode ('sequential' or 'parallel')
            workers: Number of worker processes (for parallel mode)
        """

        self.benchmark_mode = benchmark

        if benchmark:
            self.monitor = PerformanceMonitor(enabled=True)
            self.monitor.start()
            self.logger.info("Performance monitoring enabled")

        start_time = time.time()
        
        self.logger.info(f"Starting metadata extraction")
        self.logger.info(f"Mode: {mode}" + (f" ({workers} workers)" if mode == 'parallel' else ""))
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Find all series
        series_list = self.find_dicom_series(input_dir)

        if max_subjects is not None:
            # Limit to first N subjects
            unique_patients = []
            filtered_series = []
            for series_path, patient_id, session_id, modality in series_list:
                if patient_id not in unique_patients:
                    if len(unique_patients) >= max_subjects:
                        break
                    unique_patients.append(patient_id)
                if patient_id in unique_patients:
                    filtered_series.append((series_path, patient_id, session_id, modality))
            series_list = filtered_series
            self.logger.info(f"Limited to first {max_subjects} subjects ({len(unique_patients)} found)")
        
        if not series_list:
            self.logger.warning("No DICOM series found")
            return
        
        # Process series based on mode
        if mode == 'sequential':
            self._process_sequential(series_list, output_dir)
        elif mode == 'parallel':
            self._process_parallel(series_list, output_dir, workers)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_time_per_series = elapsed_time / self.stats['total_series'] if self.stats['total_series'] > 0 else 0
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Extraction completed")
        self.logger.info(f"Total series: {self.stats['total_series']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"Total time: {elapsed_time:.2f}s")
        self.logger.info(f"Average time per series: {avg_time_per_series:.3f}s")
        self.logger.info("=" * 60)

        # Save benchmark results
        if self.benchmark_mode and self.monitor:
            self.monitor.stop()
            perf_metrics = self.monitor.get_metrics()
            
            # Calculate throughput
            throughput = self.stats['successful'] / elapsed_time if elapsed_time > 0 else 0

            # Initialize speedup/efficiency
            baseline_time = None
            speedup = None
            efficiency = None
            
            # Get baseline and calculate speedup
            if results_dir:
                benchmark_logger = BenchmarkLogger(results_dir) 
                baseline_time = benchmark_logger.get_baseline_time()
                
                self.logger.debug(f"Baseline time: {baseline_time}")  # Отладка
                
                # Calculate speedup
                if mode == 'sequential' and workers == 1:
                    # This IS the baseline
                    speedup = 1.0
                    efficiency = 1.0
                elif baseline_time is not None: 
                    # Compare to baseline
                    speedup = baseline_time / avg_time_per_series
                    efficiency = speedup / workers if workers > 0 else None
                else:
                    self.logger.warning("No baseline found. Run with --mode sequential first to establish baseline.")
            
            # Create metrics object
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metrics = ExperimentMetrics(
                experiment_id=experiment_id,
                timestamp=datetime.now().isoformat(),
                mode=mode,          
                workers=workers,
                total_series=self.stats['total_series'],
                successful=self.stats['successful'],
                failed=self.stats['failed'],
                skipped=self.stats['skipped'],
                total_time=elapsed_time,
                time_per_series=avg_time_per_series,
                throughput=throughput,
                cpu_avg=perf_metrics.get('cpu_avg'),
                cpu_max=perf_metrics.get('cpu_max'),
                memory_avg_mb=perf_metrics.get('memory_avg_mb'),
                memory_peak_mb=perf_metrics.get('memory_peak_mb'),
                speedup=speedup, 
                efficiency=efficiency
            )
            
            # Log to file
            if results_dir:
                benchmark_logger.log_metrics(metrics)  # ⬅️ Используем тот же объект
                
                self.logger.info("=" * 60)
                self.logger.info("Performance Metrics:")
                self.logger.info(f"Throughput: {throughput:.3f} series/sec")
                if perf_metrics.get('cpu_avg'):
                    self.logger.info(f"CPU Average: {perf_metrics['cpu_avg']:.2f}%")
                    self.logger.info(f"CPU Peak: {perf_metrics['cpu_max']:.2f}%")
                if perf_metrics.get('memory_avg_mb'):
                    self.logger.info(f"Memory Average: {perf_metrics['memory_avg_mb']:.2f} MB")
                    self.logger.info(f"Memory Peak: {perf_metrics['memory_peak_mb']:.2f} MB")
                if speedup is not None:  # ⬅️ Явная проверка
                    self.logger.info(f"Speedup: {speedup:.3f}x")
                    if efficiency is not None:
                        self.logger.info(f"Efficiency: {efficiency:.3f}")
                self.logger.info("=" * 60)

    def _process_sequential(self, series_list, output_dir):
        """Process series sequentially."""
        for series_path, patient_id, session_id, modality in series_list:
            self.logger.info(f"Processing: {patient_id}/ses-{session_id} - {modality}")
            self.process_series(series_path, patient_id, modality, output_dir, session_id)
    
    def _process_parallel(self, series_list, output_dir, workers):
        """Process series in parallel using multiprocessing."""
        self.logger.info(f"Starting parallel processing with {workers} workers")
        
        # Prepare arguments for workers
        tasks = [
            (series_path, patient_id, modality, output_dir, session_id, self.tags_config)
            for series_path, patient_id, session_id, modality in series_list
        ]
        
        # Process in parallel
        with Pool(processes=workers) as pool:
            results = pool.map(self.process_series_wrapper, tasks)
        
        # Aggregate results
        for success, worker_stats in results:
            self.stats['total_series'] += 1
            if success:
                self.stats['successful'] += 1
            # Note: worker stats already updated, we just count here


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("metadata_extractor")
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
        description='Extract metadata from DICOM files in BIDS format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with DICOM files in BIDS format"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for JSON metadata"
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
        default=Path(__file__).parent / "config" / "dicom_tags.yaml",
        help="Path to DICOM tags configuration file"
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
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory to save benchmark results (default: ./results)"
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
        help="Number of worker processes for parallel mode (default: CPU count)"
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
    
    try:
        # Load tags configuration
        with open(args.config, 'r') as f:
            tags_config = yaml.safe_load(f)

        # Determine number of workers
        if args.mode == 'parallel':
            if args.workers is None:
                args.workers = cpu_count()
                logger.info(f"Workers not specified, using CPU count: {args.workers}")
            elif args.workers > cpu_count():
                logger.warning(f"Requested {args.workers} workers but only {cpu_count()} CPUs available")
        else:
            args.workers = 1  # Sequential mode always uses 1 worker
        
        # Create extractor and run
        extractor = MetadataExtractor(tags_config, logger)
        extractor.run(
            args.input_dir, 
            args.output_dir, 
            max_subjects=args.max_subjects,
            benchmark=args.benchmark,
            results_dir=args.results_dir if args.benchmark else None,
            mode=args.mode,
            workers=args.workers 
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()