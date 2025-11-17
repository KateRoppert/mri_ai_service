#!/usr/bin/env python3
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


class DatasetValidator:
    """Validates dataset completeness and generates report."""
    
    def __init__(self, expected_modalities: Set[str] = {'t1', 't1c', 't2', 't2fl'}):
        self.expected_modalities = expected_modalities
        self.patients_data = defaultdict(lambda: defaultdict(set))
        
    def add_series(self, patient_id: str, session: str, modality: str):
        """Add found series to validation data."""
        self.patients_data[patient_id][session].add(modality.lower())
    
    def scan_converted_data(self, output_dir: Path):
        """Scan already converted NIfTI files for validation."""
        self.patients_data = defaultdict(lambda: defaultdict(set))  # Reset data
        
        # BIDS structure: output_dir/sub-*/ses-*/anat/*.nii.gz
        subject_dirs = sorted(output_dir.glob("sub-*"))
        
        for subject_dir in subject_dirs:
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            # Look for session directories
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                anat_dir = session_dir / "anat"
                
                if not anat_dir.exists():
                    continue
                
                # Find NIfTI files
                nifti_files = list(anat_dir.glob("*.nii.gz"))
                
                # Extract modalities from filenames
                # Pattern: sub-{patient_id}_ses-{session_id}_{modality}.nii.gz
                for nifti_file in nifti_files:
                    filename = nifti_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        modality = parts[-1]  # Last part is modality
                        self.patients_data[patient_id][session_id].add(modality.lower())
    
    def validate_and_report(self, output_dir: Path) -> Dict:
        """Validate dataset and create incomplete data report."""
        total_patients = len(self.patients_data)
        total_sessions = sum(len(sessions) for sessions in self.patients_data.values())
        
        complete_patients = 0
        complete_sessions = 0
        incomplete_details = []
        
        for patient_id, sessions in self.patients_data.items():
            patient_complete = True
            
            for session_id, modalities in sessions.items():
                missing_modalities = self.expected_modalities - modalities
                
                if missing_modalities:
                    patient_complete = False
                    # Extract session date from session_id if available
                    session_display = session_id
                    
                    incomplete_details.append({
                        'patient_id': patient_id,
                        'session_id': session_id,
                        'session_display': session_display,
                        'missing': sorted(missing_modalities),
                        'available': sorted(modalities)
                    })
                else:
                    complete_sessions += 1
            
            if patient_complete:
                complete_patients += 1
        
        # Generate report
        report_data = {
            'total_patients': total_patients,
            'complete_patients': complete_patients,
            'incomplete_patients': total_patients - complete_patients,
            'total_sessions': total_sessions,
            'complete_sessions': complete_sessions,
            'incomplete_sessions': total_sessions - complete_sessions,
            'incomplete_details': incomplete_details
        }
        
        self._write_report(output_dir, report_data)
        return report_data
    
    def _write_report(self, output_dir: Path, data: Dict):
        """Write incomplete data report to file."""
        report_file = output_dir / "incomplete_data.txt"
        
        with open(report_file, 'w') as f:
            f.write("=== Incomplete Data Report ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total patients: {data['total_patients']}\n")
            f.write(f"Complete patients: {data['complete_patients']} ")
            f.write(f"({data['complete_patients']/data['total_patients']*100:.1f}%)\n" if data['total_patients'] > 0 else "(0.0%)\n")
            f.write(f"Incomplete patients: {data['incomplete_patients']} ")
            f.write(f"({data['incomplete_patients']/data['total_patients']*100:.1f}%)\n" if data['total_patients'] > 0 else "(0.0%)\n")
            f.write(f"Total sessions: {data['total_sessions']}\n")
            f.write(f"Complete sessions: {data['complete_sessions']} ")
            f.write(f"({data['complete_sessions']/data['total_sessions']*100:.1f}%)\n" if data['total_sessions'] > 0 else "(0.0%)\n")
            f.write(f"Incomplete sessions: {data['incomplete_sessions']} ")
            f.write(f"({data['incomplete_sessions']/data['total_sessions']*100:.1f}%)\n" if data['total_sessions'] > 0 else "(0.0%)\n")
            f.write("\n=== Details ===\n")
            
            for detail in data['incomplete_details']:
                f.write(f"Patient: sub-{detail['patient_id']}\n")
                f.write(f"  Session: ses-{detail['session_id']}\n")
                f.write(f"    Missing: {', '.join(detail['missing'])}\n")
                f.write(f"    Available: {', '.join(detail['available'])}\n")


class ConversionValidator:
    """Validates conversion completeness by comparing input DICOM with output NIfTI."""
    
    def __init__(self):
        self.input_data = defaultdict(lambda: defaultdict(set))
        self.output_data = defaultdict(lambda: defaultdict(set))
    
    def scan_dicom_data(self, input_dir: Path):
        """Scan input DICOM directory structure."""
        self.input_data = defaultdict(lambda: defaultdict(set))
        
        subject_dirs = sorted(input_dir.glob("sub-*"))
        print(f"Scanning DICOM: Found {len(subject_dirs)} subjects in {input_dir}")
        
        for subject_dir in subject_dirs:
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                # НЕ нормализуем session_id - используем как есть!
                
                for modality_type_dir in session_dir.iterdir():
                    if not modality_type_dir.is_dir():
                        continue
                    
                    # Check if there are DICOM files
                    dicom_files = list(modality_type_dir.glob("*.dcm")) + \
                                  list(modality_type_dir.glob("*.DCM")) + \
                                  list(modality_type_dir.glob("*.dicom"))
                    
                    if dicom_files:
                        modality = modality_type_dir.name
                        self.input_data[patient_id][session_id].add(modality.lower())
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
                                self.input_data[patient_id][session_id].add(modality.lower())
    
    def scan_nifti_data(self, output_dir: Path):
        """Scan output NIfTI directory structure."""
        self.output_data = defaultdict(lambda: defaultdict(set))
        
        subject_dirs = sorted(output_dir.glob("sub-*"))
        print(f"Scanning NIfTI: Found {len(subject_dirs)} subjects in {output_dir}")
        
        for subject_dir in subject_dirs:
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                # НЕ нормализуем session_id - используем как есть!
                anat_dir = session_dir / "anat"
                
                if not anat_dir.exists():
                    continue
                
                # Find NIfTI files
                nifti_files = list(anat_dir.glob("*.nii.gz"))
                
                # Extract modalities from filenames
                for nifti_file in nifti_files:
                    filename = nifti_file.stem.replace('.nii', '')
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        modality = parts[-1]
                        self.output_data[patient_id][session_id].add(modality.lower())
    
    def compare_and_report(self, output_dir: Path) -> Dict:
        """Compare input and output data and create conversion report."""
        # Calculate statistics
        input_patients_count = len(self.input_data)
        input_sessions_count = 0
        input_modalities_count = 0
        
        for patient_id, sessions in self.input_data.items():
            input_sessions_count += len(sessions)
            for session_id, modalities_set in sessions.items():
                input_modalities_count += len(modalities_set)
        
        output_patients_count = len(self.output_data)
        output_sessions_count = 0
        output_modalities_count = 0
        
        for patient_id, sessions in self.output_data.items():
            output_sessions_count += len(sessions)
            for session_id, modalities_set in sessions.items():
                output_modalities_count += len(modalities_set)
        
        # Find losses
        missing_patients = []
        missing_sessions = []
        missing_modalities = []
        
        for patient_id, patient_input_sessions in self.input_data.items():
            if patient_id not in self.output_data:
                # Entire patient missing
                patient_modalities_total = sum(len(modalities_set) for modalities_set in patient_input_sessions.values())
                missing_patients.append({
                    'patient_id': patient_id,
                    'sessions': len(patient_input_sessions),
                    'modalities': patient_modalities_total
                })
            else:
                # Check sessions
                output_sessions_for_patient = self.output_data[patient_id]
                
                for session_id, session_input_modalities_set in patient_input_sessions.items():
                    if session_id not in output_sessions_for_patient:
                        # Entire session missing
                        missing_sessions.append({
                            'patient_id': patient_id,
                            'session_id': session_id,
                            'modalities': len(session_input_modalities_set)
                        })
                    else:
                        # Check modalities
                        session_output_modalities_set = output_sessions_for_patient[session_id]
                        lost_modalities_set = session_input_modalities_set - session_output_modalities_set
                        
                        if lost_modalities_set:
                            missing_modalities.append({
                                'patient_id': patient_id,
                                'session_id': session_id,
                                'missing': sorted(lost_modalities_set),
                                'expected': sorted(session_input_modalities_set),
                                'found': sorted(session_output_modalities_set)
                            })
        
        # Debug: Check variable types and data alignment
        print(f"Input: {input_patients_count} patients, {input_sessions_count} sessions, {input_modalities_count} modalities")
        print(f"Output: {output_patients_count} patients, {output_sessions_count} sessions, {output_modalities_count} modalities")
        
        # Debug: Show sample data structures
        if self.input_data and self.output_data:
            sample_patient = list(self.input_data.keys())[0]
            print(f"Sample patient {sample_patient}:")
            print(f"  Input sessions: {dict(self.input_data[sample_patient])}")
            if sample_patient in self.output_data:
                print(f"  Output sessions: {dict(self.output_data[sample_patient])}")
            else:
                print(f"  Output sessions: MISSING")
        
        # Generate report data
        report_data = {
            'input_patients': input_patients_count,
            'input_sessions': input_sessions_count,
            'input_modalities': input_modalities_count,
            'output_patients': output_patients_count,
            'output_sessions': output_sessions_count,
            'output_modalities': output_modalities_count,
            'missing_patients': missing_patients,
            'missing_sessions': missing_sessions,
            'missing_modalities': missing_modalities,
            'conversion_success_rate': {
                'patients': (output_patients_count / input_patients_count * 100) if input_patients_count > 0 else 0,
                'sessions': (output_sessions_count / input_sessions_count * 100) if input_sessions_count > 0 else 0,
                'modalities': (output_modalities_count / input_modalities_count * 100) if input_modalities_count > 0 else 0
            }
        }
        
        self._write_comparison_report(output_dir, report_data)
        return report_data
    
    def _write_comparison_report(self, output_dir: Path, data: Dict):
        """Write conversion comparison report to file."""
        report_file = output_dir / "conversion_validation.txt"
        
        with open(report_file, 'w') as f:
            f.write("=== Conversion Validation Report ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== Input DICOM Summary ===\n")
            f.write(f"Patients: {data['input_patients']}\n")
            f.write(f"Sessions: {data['input_sessions']}\n")
            f.write(f"Modalities: {data['input_modalities']}\n\n")
            
            f.write("=== Output NIfTI Summary ===\n")
            f.write(f"Patients: {data['output_patients']}\n")
            f.write(f"Sessions: {data['output_sessions']}\n")
            f.write(f"Modalities: {data['output_modalities']}\n\n")
            
            f.write("=== Conversion Success Rate ===\n")
            f.write(f"Patients: {data['conversion_success_rate']['patients']:.1f}% ")
            f.write(f"({data['output_patients']}/{data['input_patients']})\n")
            f.write(f"Sessions: {data['conversion_success_rate']['sessions']:.1f}% ")
            f.write(f"({data['output_sessions']}/{data['input_sessions']})\n")
            f.write(f"Modalities: {data['conversion_success_rate']['modalities']:.1f}% ")
            f.write(f"({data['output_modalities']}/{data['input_modalities']})\n\n")
            
            f.write("=== Conversion Losses ===\n")
            
            if data['missing_patients']:
                f.write(f"\nMissing Patients ({len(data['missing_patients'])}):\n")
                for patient in data['missing_patients']:
                    f.write(f"  sub-{patient['patient_id']} ")
                    f.write(f"({patient['sessions']} sessions, {patient['modalities']} modalities)\n")
            
            if data['missing_sessions']:
                f.write(f"\nMissing Sessions ({len(data['missing_sessions'])}):\n")
                for session in data['missing_sessions']:
                    f.write(f"  sub-{session['patient_id']}/ses-{session['session_id']} ")
                    f.write(f"({session['modalities']} modalities)\n")
            
            if data['missing_modalities']:
                f.write(f"\nMissing Modalities ({len(data['missing_modalities'])}):\n")
                for item in data['missing_modalities']:
                    f.write(f"  sub-{item['patient_id']}/ses-{item['session_id']}\n")
                    f.write(f"    Missing: {', '.join(item['missing'])}\n")
                    f.write(f"    Expected: {', '.join(item['expected'])}\n")
                    f.write(f"    Found: {', '.join(item['found'])}\n")
            
            if not any([data['missing_patients'], data['missing_sessions'], data['missing_modalities']]):
                f.write("\n🎉 No conversion losses detected! All data converted successfully.\n")


def validate_conversion(input_dir: Path, output_dir: Path):
    """Validate conversion by comparing input DICOM with output NIfTI."""
    logger = logging.getLogger("conversion_validator")
    
    logger.info(f"Comparing DICOM input: {input_dir}")
    logger.info(f"With NIfTI output: {output_dir}")
    
    validator = ConversionValidator()
    
    logger.info("Scanning input DICOM data...")
    validator.scan_dicom_data(input_dir)
    
    logger.info("Scanning output NIfTI data...")
    validator.scan_nifti_data(output_dir)
    
    logger.info("Comparing and generating report...")
    comparison_report = validator.compare_and_report(output_dir)
    
    logger.info("=" * 60)
    logger.info("CONVERSION VALIDATION SUMMARY:")
    logger.info(f"Input:  {comparison_report['input_patients']} patients, "
               f"{comparison_report['input_sessions']} sessions, "
               f"{comparison_report['input_modalities']} modalities")
    logger.info(f"Output: {comparison_report['output_patients']} patients, "
               f"{comparison_report['output_sessions']} sessions, "
               f"{comparison_report['output_modalities']} modalities")
    logger.info(f"Success rate: "
               f"P:{comparison_report['conversion_success_rate']['patients']:.1f}% "
               f"S:{comparison_report['conversion_success_rate']['sessions']:.1f}% "
               f"M:{comparison_report['conversion_success_rate']['modalities']:.1f}%")
    
    total_losses = (len(comparison_report['missing_patients']) + 
                   len(comparison_report['missing_sessions']) + 
                   len(comparison_report['missing_modalities']))
    
    if total_losses > 0:
        logger.warning(f"⚠️  Found {total_losses} conversion issues!")
        logger.warning(f"Missing: {len(comparison_report['missing_patients'])} patients, "
                      f"{len(comparison_report['missing_sessions'])} sessions, "
                      f"{len(comparison_report['missing_modalities'])} modalities")
    else:
        logger.info("✅ Perfect conversion! No data losses detected.")
    
    logger.info(f"Detailed report saved to: {output_dir}/conversion_validation.txt")
    logger.info("=" * 60)
    
    return comparison_report


def validate_converted_dataset(output_dir: Path, expected_modalities: Set[str] = {'t1', 't1c', 't2', 't2fl'}):
    """Validate already converted dataset (legacy function for backward compatibility)."""
    logger = logging.getLogger("dataset_validator")
    
    validator = DatasetValidator(expected_modalities)
    validator.scan_converted_data(output_dir)
    
    logger.info(f"Validating converted dataset in: {output_dir}")
    validation_report = validator.validate_and_report(output_dir)
    
    logger.info("=" * 60)
    logger.info("CONVERTED DATASET VALIDATION:")
    if validation_report['total_patients'] > 0:
        logger.info(f"Complete patients: {validation_report['complete_patients']}/{validation_report['total_patients']} "
                   f"({validation_report['complete_patients']/validation_report['total_patients']*100:.1f}%)")
        logger.info(f"Complete sessions: {validation_report['complete_sessions']}/{validation_report['total_sessions']} "
                   f"({validation_report['complete_sessions']/validation_report['total_sessions']*100:.1f}%)")
    else:
        logger.info("No patients found in dataset!")
    logger.info(f"Report saved to: {output_dir}/incomplete_data.txt")
    logger.info("=" * 60)
    
    return validation_report

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
    
    success = temp_converter.convert_series(series_path, patient_id, modality, output_dir, session)
    return success, temp_converter.stats


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

        # Performance monitoring
        self.monitor = None
        self.benchmark_mode = False
        
        # Dataset validation
        self.validator = DatasetValidator()
        
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
                        # Add to validator
                        self.validator.add_series(patient_id, session_id, modality)
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
                                # Add to validator
                                self.validator.add_series(patient_id, session_id, modality)
        
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
                      modality: str, output_dir: Path, session: str = "001") -> bool:
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
            return True
        
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
                    self.stats['successful'] += 1
                    self.logger.debug(f"Successfully converted {patient_id}/{modality}")
                    return True
                else:
                    self.logger.warning(
                        f"dcm2niix returned 0 but no output file: {patient_id}/{modality}"
                    )
                    self.stats['failed'] += 1
                    return False
            else:
                self.logger.error(
                    f"Failed to convert {patient_id}/{modality}: "
                    f"returncode={result.returncode}\n{result.stderr}"
                )
                self.stats['failed'] += 1
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout converting {patient_id}/{modality}")
            self.stats['failed'] += 1
            return False
        except Exception as e:
            self.logger.error(f"Error converting {patient_id}/{modality}: {e}")
            self.stats['failed'] += 1
            return False
    # @staticmethod
    # def process_series_wrapper(self, args):
    #     """
    #     Wrapper for multiprocessing conversion.
        
    #     Args:
    #         args: Tuple of (series_path, patient_id, modality, output_dir, session, dcm2niix_cmd)
        
    #     Returns:
    #         Tuple of (success, stats)
    #     """
    #     series_path, patient_id, modality, output_dir, session, dcm2niix_base_cmd = args
        
    #     # Create temporary converter instance for this process
    #     temp_logger = logging.getLogger(f"converter_worker_{patient_id}_{modality}")
    #     temp_converter = NiftiConverter.__new__(NiftiConverter)
    #     temp_converter.logger = temp_logger
    #     temp_converter.dcm2niix_base_cmd = dcm2niix_base_cmd
    #     temp_converter.stats = {'total_series': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        
    #     success = temp_converter.convert_series(series_path, patient_id, modality, output_dir, session)
    #     return success, temp_converter.stats
    
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
            
            # Take first N patients
            limited_patients = list(patients.keys())[:max_subjects]
            series_list = []
            for patient_id in limited_patients:
                series_list.extend(patients[patient_id])
            
            self.logger.info(f"Limited to {max_subjects} subjects: {len(series_list)} series "
                           f"(was {original_count})")
        
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
        
        # Calculate speedup and efficiency for parallel mode
        speedup = None
        efficiency = None
        if mode == 'parallel' and workers > 1:
            # Theoretical baseline: sequential time
            sequential_time_estimate = avg_time_per_series * total_processed
            speedup = sequential_time_estimate / elapsed_time if elapsed_time > 0 else 0
            efficiency = speedup / workers if workers > 0 else 0
        
        # Final statistics
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

        # Generate validation report
        self.logger.info("Generating dataset validation report...")
        validation_report = self.validator.validate_and_report(output_dir)

        # Log validation summary
        self.logger.info("=" * 60)
        self.logger.info("DATASET VALIDATION SUMMARY:")
        if validation_report['total_patients'] > 0:
            self.logger.info(f"Complete patients: {validation_report['complete_patients']}/{validation_report['total_patients']} "
                           f"({validation_report['complete_patients']/validation_report['total_patients']*100:.1f}%)")
            self.logger.info(f"Complete sessions: {validation_report['complete_sessions']}/{validation_report['total_sessions']} "
                           f"({validation_report['complete_sessions']/validation_report['total_sessions']*100:.1f}%)")
        else:
            self.logger.info("No patients found for validation!")
        self.logger.info(f"Incomplete data report saved to: {output_dir}/incomplete_data.txt")
        self.logger.info("=" * 60)

    def _process_sequential(self, series_list, output_dir):
        """Process series sequentially."""
        for series_path, patient_id, session_id, modality in series_list:
            self.logger.info(f"Converting: {patient_id}/ses-{session_id} - {modality}")
            self.convert_series(series_path, patient_id, modality, output_dir, session_id)
    
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
        
        # Aggregate results
        for success, worker_stats in results:
            self.stats['total_series'] += 1
            if success:
                self.stats['successful'] += 1
            # Note: worker stats already counted internally


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
    
    # Validation-only mode
    mode_group.add_argument(
        "--validate-only",
        type=Path,
        nargs=2,
        metavar=('INPUT_DIR', 'OUTPUT_DIR'),
        help="Compare input DICOM with output NIfTI (provide input_dir output_dir)"
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
        default=Path("results_conversion"),
        help="Directory to save benchmark results (default: ./results_conversion)"
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
        "--expected-modalities",
        type=str,
        nargs='+',
        default=['t1', 't1c', 't2', 't2fl'],
        help="Expected modalities for validation (default: t1 t1c t2 t2fl)"
    )
    
    args = parser.parse_args()
    
    # Handle validation-only mode
    if args.validate_only:
        input_dir, output_dir = args.validate_only
        
        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)
        if not output_dir.exists():
            print(f"Error: Output directory does not exist: {output_dir}")
            sys.exit(1)
        
        logger = setup_logging(args.log_file)
        validate_conversion(input_dir, output_dir)
        sys.exit(0)
    
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
        # Set expected modalities if specified
        if args.expected_modalities:
            converter.validator.expected_modalities = set(args.expected_modalities)

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