"""
DICOM to NIfTI Conversion Script (Baseline Version)
Converts DICOM files to NIfTI format using dcm2niix.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from pipeline_validator import InputOutputValidator

def process_series_worker(args):
    """
    Worker function for multiprocessing conversion.
    
    Args:
        args: Tuple of (series_path, patient_id, modality, output_dir, session, dcm2niix_cmd)
    
    Returns:
        Tuple of (success, stats)
    """
    series_path, patient_id, modality, output_dir, session, dcm2niix_base_cmd = args
    
    # Create temporary converter instance for this process
    import logging
    temp_logger = logging.getLogger(f"converter_worker_{patient_id}_{modality}")
    temp_logger.setLevel(logging.WARNING)  # Minimize logging in workers
    
    temp_converter = NiftiConverter.__new__(NiftiConverter)
    temp_converter.logger = temp_logger
    temp_converter.dcm2niix_base_cmd = dcm2niix_base_cmd
    temp_converter.stats = {'total_series': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
    
    success, error = temp_converter.convert_series(
    series_path,
    patient_id,
    modality,
    output_dir,
    session
    )

    return {
        "success": success,
        "stats": temp_converter.stats,
        "patient": patient_id,
        "session": session,
        "modality": modality,
        "series_path": str(series_path),
        "error": error,
    }


class NiftiConverter:
    """Handles DICOM to NIfTI conversion using dcm2niix."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize converter.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.stats = {
            'total_series': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        self.failed_cases = []

        # Performance monitoring
        self.monitor = None
        self.benchmark_mode = False
        
        # dcm2niix base command with optimal parameters
        self.dcm2niix_base_cmd = [
            'dcm2niix',
            '-b', 'n',      # No JSON (we have metadata from stage 1)
            '-z', 'y',      # Gzip compression
            '-9', '6',      # Compression level 6 (balance)
            '-v', '0',      # Quiet mode
            '-w', '0',      # Skip existing files
            '-m', 'y'       # Merge 2D slices
        ]
        
        # Check if dcm2niix is available
        self._check_dcm2niix()
    
    def _check_dcm2niix(self):
        """Check if dcm2niix is installed and available."""
        try:
            result = subprocess.run(
                ['dcm2niix', '-h'],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError
            self.logger.info("dcm2niix found and ready")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.error(
                "dcm2niix not found! Please install it:\n"
                "  Ubuntu/Debian: sudo apt-get install dcm2niix\n"
                "  macOS: brew install dcm2niix\n"
                "  From source: https://github.com/rordenlab/dcm2niix"
            )
            sys.exit(1)
    
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
        
        # BIDS structure: sub-<patientID>/ses-<session>/<modality>/
        for subject_dir in subject_dirs:
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            # Look for session directories
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                
                # Look for modality directories
                for modality_type_dir in session_dir.iterdir():
                    if not modality_type_dir.is_dir():
                        continue
                    
                    # Check if there are DICOM files
                    dicom_files = list(modality_type_dir.glob("*.dcm")) + \
                                  list(modality_type_dir.glob("*.DCM")) + \
                                  list(modality_type_dir.glob("*.dicom"))
                    
                    if dicom_files:
                        modality = modality_type_dir.name
                        series_list.append((modality_type_dir, patient_id, session_id, modality))

                    else:
                        # Check subdirectories
                        for subdir in modality_type_dir.iterdir():
                            if not subdir.is_dir():
                                continue
                            
                            dicom_files = list(subdir.glob("*.dcm")) + \
                                          list(subdir.glob("*.DCM")) + \
                                          list(subdir.glob("*.dicom"))
                            
                            if dicom_files:
                                modality = subdir.name
                                series_list.append((subdir, patient_id, session_id, modality))
        
        self.logger.info(f"Found {len(series_list)} DICOM series")
        return series_list
    
    def check_output_exists(self, output_dir: Path, patient_id: str, 
                           session: str, modality: str) -> bool:
        """
        Check if output NIfTI file already exists.
        
        Args:
            output_dir: Root output directory
            patient_id: Patient identifier
            session: Session identifier
            modality: Modality name
            
        Returns:
            True if file exists, False otherwise
        """
        # Expected BIDS path
        subject_dir = output_dir / f"sub-{patient_id}"
        session_dir = subject_dir / f"ses-{session}"
        anat_dir = session_dir / "anat"
        
        # Expected filename
        expected_file = anat_dir / f"sub-{patient_id}_ses-{session}_{modality}.nii.gz"
        
        return expected_file.exists()
    
    def convert_series(self, series_path: Path, patient_id: str, 
                      modality: str, output_dir: Path, session: str = "001") -> Tuple[bool, Optional[str]]:
        """
        Convert a single DICOM series to NIfTI.
        
        Args:
            series_path: Path to DICOM series directory
            patient_id: Patient identifier
            modality: Modality name
            output_dir: Root output directory
            session: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        self.stats['total_series'] += 1
        
        # Check if already exists
        if self.check_output_exists(output_dir, patient_id, session, modality):
            self.logger.info(f"Skipping {patient_id}/{modality} (already exists)")
            self.stats['skipped'] += 1
            return True, None
        
        # Create output directory structure
        subject_dir = output_dir / f"sub-{patient_id}"
        session_dir = subject_dir / f"ses-{session}"
        anat_dir = session_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare dcm2niix command
        # Output filename pattern: sub-<ID>_ses-<session>_<modality>
        filename_pattern = f"sub-{patient_id}_ses-{session}_{modality}"
        
        cmd = self.dcm2niix_base_cmd + [
            '-f', filename_pattern,     # Filename pattern
            '-o', str(anat_dir),        # Output directory
            str(series_path)            # Input directory
        ]
        
        # Run dcm2niix
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 60 second timeout per series
            )
            
            if result.returncode == 0:
                # Check if output file was created
                expected_file = anat_dir / f"{filename_pattern}.nii.gz"
                if expected_file.exists():
                    # dcm2niix sometimes produces extra files (e.g. _Eq_1, _e2) when a
                    # DICOM series contains multiple equivalent groups (common in MPR series).
                    # These don't fit our naming convention and confuse downstream stages.
                    for extra in anat_dir.glob(f"{filename_pattern}_*.nii.gz"):
                        extra.unlink()
                        self.logger.warning(
                            f"Removed dcm2niix artifact: {extra.name} "
                            f"(DICOM series may contain multiple equivalent acquisitions)"
                        )
                    self.stats['successful'] += 1
                    self.logger.debug(f"Successfully converted {patient_id}/{modality}")
                    return True, None
                else:
                    self.logger.warning(
                        f"dcm2niix returned 0 but no output file: {patient_id}/{modality}"
                    )
                    reason = "dcm2niix finished successfully but output file was not created"

                    self.stats['failed'] += 1

                    return False, reason
            else:
                reason = (
                    f"dcm2niix exited with code {result.returncode}: "
                    f"{result.stderr.strip()}"
                )

                self.logger.error(
                    f"Failed to convert {patient_id}/{modality}: {reason}"
                )

                self.stats['failed'] += 1

                return False, reason
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout converting {patient_id}/{modality}")
            self.stats['failed'] += 1
            return False, "Timeout (300 seconds)"
        except Exception as e:
            self.logger.error(f"Error converting {patient_id}/{modality}: {e}")
            self.stats['failed'] += 1
            return False, str(e)
    def run(self, input_dir: Path, output_dir: Path, max_subjects: Optional[int] = None,
            benchmark: bool = False, results_dir: Optional[Path] = None,
            mode: str = 'sequential', workers: int = 1) -> None:
        """
        Run the conversion process.
        
        Args:
            input_dir: Input directory with DICOM files
            output_dir: Output directory for NIfTI files
            max_subjects: Maximum number of subjects to process (optional)
            benchmark: Enable performance monitoring
            results_dir: Directory to save benchmark results (optional)
            mode: Processing mode ('sequential' or 'parallel')
            workers: Number of worker processes for parallel mode
        """
        start_time = time.time()
        
        self.benchmark_mode = benchmark
        benchmark_logger = None
        
        # Setup performance monitoring
        if benchmark:
            if results_dir:
                results_dir.mkdir(parents=True, exist_ok=True)
                benchmark_logger = BenchmarkLogger(results_dir)
            self.monitor = PerformanceMonitor()
            self.monitor.start()
        
        self.logger.info(f"Starting conversion: {input_dir} -> {output_dir}")
        self.logger.info(f"Mode: {mode}, Workers: {workers if mode == 'parallel' else 1}")
        
        # Find all DICOM series
        series_list = self.find_dicom_series(input_dir)
        
        if not series_list:
            self.logger.warning("No DICOM series found!")
            return
        
        # Limit subjects if specified
        if max_subjects:
            original_count = len(series_list)
            # Group by patient to limit subjects, not series
            patients = {}
            for series_path, patient_id, session_id, modality in series_list:
                if patient_id not in patients:
                    patients[patient_id] = []
                patients[patient_id].append((series_path, patient_id, session_id, modality))

            limited_patients = list(patients.keys())[:max_subjects]
            series_list = []
            for patient_id in limited_patients:
                series_list.extend(patients[patient_id])

            # Only log when the limit actually removes subjects
            if len(limited_patients) < len(patients):
                self.logger.info(
                    f"Limited to {max_subjects} subjects: {len(series_list)} series "
                    f"(was {original_count})"
                )
        
        # Process series
        if mode == 'parallel' and workers > 1:
            self._process_parallel(series_list, output_dir, workers)
        else:
            self._process_sequential(series_list, output_dir)
        
        # Stop performance monitoring
        perf_metrics = None
        if benchmark and self.monitor:
            self.monitor.stop()
            perf_metrics = self.monitor.get_metrics()
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        total_processed = self.stats['successful'] + self.stats['failed']
        avg_time_per_series = elapsed_time / total_processed if total_processed > 0 else 0
        throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate speedup and efficiency — only meaningful in benchmark mode
        speedup = None
        efficiency = None

        if benchmark:
            baseline_time = None

            # Load baseline from BenchmarkLogger CSV (same source as Stage 01)
            if benchmark_logger:
                try:
                    baseline_time = benchmark_logger.get_baseline_time()
                    if baseline_time:
                        self.logger.debug(f"Loaded baseline time: {baseline_time:.2f}s")
                except Exception as e:
                    self.logger.warning(f"Failed to load baseline: {e}")

            if mode == 'sequential' and workers == 1:
                # Sequential mode is the baseline
                speedup = 1.0
                efficiency = 1.0
            elif mode == 'parallel' and workers > 1:
                if baseline_time is not None:
                    speedup = baseline_time / elapsed_time if elapsed_time > 0 else 0
                    efficiency = speedup / workers if workers > 0 else 0
                else:
                    self.logger.warning(
                        "No baseline found! Run with --benchmark --mode sequential first."
                    )

        # Final statistics
        if self.failed_cases:
            self.logger.error("=" * 60)
            self.logger.error(f"FAILED CONVERSIONS ({len(self.failed_cases)}):")

            for case in self.failed_cases:
                self.logger.error(
                    f"\n"
                    f"Patient : {case['patient']}\n"
                    f"Session : {case['session']}\n"
                    f"Modality: {case['modality']}\n"
                    f"Series  : {case['series_path']}\n"
                    f"Reason  : {case['error']}"
                )

            self.logger.error("=" * 60)
        self.logger.info("=" * 60)
        self.logger.info("CONVERSION SUMMARY:")
        self.logger.info(f"Total series: {self.stats['total_series']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"Total time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Average time per series: {avg_time_per_series:.3f} seconds")
        self.logger.info(f"Throughput: {throughput:.3f} series/second")
        if speedup is not None:
            self.logger.info(f"Parallel speedup: {speedup:.3f}x")
            self.logger.info(f"Parallel efficiency: {efficiency:.3f}")
        self.logger.info("=" * 60)

        self.logger.info("Validating input-output correspondence...")
        
        try:
            validator = InputOutputValidator(logger=self.logger)
            
            # Сканируем входную структуру (DICOM в BIDS)
            self.logger.info("Scanning input structure (DICOM)...")
            input_structure = validator.scan_structure(input_dir, format_type='bids-dicom')
            self.logger.info(f"Found {len(input_structure)} patients in input")
            
            # Сканируем выходную структуру (NIfTI)
            self.logger.info("Scanning output structure (NIfTI)...")
            output_structure = validator.scan_structure(output_dir, format_type='bids-nifti')
            self.logger.info(f"Found {len(output_structure)} patients in output")
            
            # Сравниваем структуры
            self.logger.info("Comparing structures...")
            comparison_result = validator.compare_structures(input_structure, output_structure)
            
            # Генерируем отчет
            validator.generate_incomplete_report(
                comparison_result=comparison_result,
                stage_name="03_convert_to_nifti",
                output_dir=output_dir
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}", exc_info=True)
        
        self.logger.info("=" * 60)
       
        # Save benchmark results
        if benchmark and benchmark_logger:
            metrics = ExperimentMetrics(
                experiment_id=f"{mode}_workers_{workers}",
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
            if benchmark_logger:
                benchmark_logger.log_metrics(metrics)
                
                self.logger.info("=" * 60)
                self.logger.info("Performance Metrics:")
                self.logger.info(f"Throughput: {throughput:.3f} series/sec")
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

    def _process_sequential(self, series_list, output_dir):
        """Process series sequentially."""
        for series_path, patient_id, session_id, modality in series_list:
            self.logger.info(f"Converting: {patient_id}/ses-{session_id} - {modality}")
            success, error = self.convert_series(
                series_path,
                patient_id,
                modality,
                output_dir,
                session_id
            )

            if not success:
                self.failed_cases.append({
                    "patient": patient_id,
                    "session": session_id,
                    "modality": modality,
                    "series_path": str(series_path),
                    "error": error
                })
    
    def _process_parallel(self, series_list, output_dir, workers):
        """Process series in parallel using multiprocessing."""
        self.logger.info(f"Starting parallel processing with {workers} workers")

        # Prepare arguments for workers
        tasks = [
            (series_path, patient_id, modality, output_dir, session_id, self.dcm2niix_base_cmd)
            for series_path, patient_id, session_id, modality in series_list
        ]

        # Process in parallel
        with Pool(processes=workers) as pool:
            results = pool.map(process_series_worker, tasks)

        # Aggregate stats from each worker's returned stats dict.
        # Workers track successful/failed/skipped independently; we must
        # sum them here — checking only `success` misses failed and skipped.
        for result in results:

            worker_stats = result["stats"]

            self.stats['total_series'] += worker_stats['total_series']
            self.stats['successful'] += worker_stats['successful']
            self.stats['failed'] += worker_stats['failed']
            self.stats['skipped'] += worker_stats['skipped']

            if not result["success"]:
                self.failed_cases.append({
                    "patient": result["patient"],
                    "session": result["session"],
                    "modality": result["modality"],
                    "series_path": result["series_path"],
                    "error": result["error"],
                })


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("nifti_converter")
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
        description='Convert DICOM files to NIfTI format using dcm2niix',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mutually exclusive group for main modes
    mode_group = parser.add_mutually_exclusive_group()
    
    # Regular conversion arguments
    mode_group.add_argument(
        "input_dir",
        type=Path,
        nargs='?',
        help="Input directory with DICOM files in BIDS format"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs='?',
        help="Output directory for NIfTI files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory before processing (for benchmarks)"
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
        help="Number of worker processes for parallel mode (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate regular conversion arguments
    if not args.input_dir or not args.output_dir:
        print("Error: input_dir and output_dir are required for conversion mode")
        parser.print_help()
        sys.exit(1)
        
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_file)

    # Handle --force flag
    if args.force and args.output_dir.exists():
        logger.warning(f"Removing existing subjects from: {args.output_dir}")
        # Remove only sub-* directories, keep results/ and other folders
        for item in args.output_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                logger.debug(f"Removing {item}")
                shutil.rmtree(item)
            elif item.is_file() and item.suffix in ['.nii', '.gz', '.json']:
                # Remove any stray nifti/json files in root
                logger.debug(f"Removing {item}")
                item.unlink()
    
    try:
        # Determine number of workers
        if args.mode == 'parallel':
            if args.workers == 1:
                args.workers = cpu_count()
                logger.info(f"Workers not specified for parallel mode, using CPU count: {args.workers}")
            elif args.workers > cpu_count():
                logger.warning(f"Requested {args.workers} workers but only {cpu_count()} CPUs available")
        else:
            args.workers = 1  # Sequential mode always uses 1 worker

        # Create converter and run
        converter = NiftiConverter(logger)

        converter.run(
            args.input_dir,
            args.output_dir,
            max_subjects=args.max_subjects,
            benchmark=args.benchmark,
            results_dir=args.results_dir if args.benchmark else None,
            mode=args.mode,
            workers=args.workers
        )
        
    except KeyboardInterrupt:
        logger.info("\nConversion interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()