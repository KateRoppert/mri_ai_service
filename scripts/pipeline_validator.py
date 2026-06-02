"""
Pipeline Validator Module
Validates input-output correspondence for pipeline stages.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class InputOutputValidator:
    """
    Validates that input and output structures match in terms of
    patients, sessions, and modalities count.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize validator.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)

    def scan_structure(self, directory: Path, format_type: str = 'bids-dicom') -> Dict[str, Dict[str, Set[str]]]:
        """
        Scan directory structure and extract patient/session/modality information.
        
        Args:
            directory: Root directory to scan
            format_type: Type of structure to scan:
                - 'bids-dicom': sub-*/ses-*/anat/{modality}/ (DICOM files)
                - 'bids-nifti': sub-*/ses-*/anat/*.nii.gz
                - 'bids-metadata': sub-*/ses-*/metadata/*.json
                - 'bids-quality': sub-*/ses-*/quality/*.json
        
        Returns:
            Nested dictionary: {patient_id: {session_id: {modality1, modality2, ...}}}
        """
        structure = defaultdict(lambda: defaultdict(set))
        
        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return dict(structure)
        
        # Scan based on format type
        if format_type == 'bids-dicom':
            structure = self._scan_bids_dicom(directory)
        elif format_type == 'bids-nifti':
            structure = self._scan_bids_nifti(directory)
        elif format_type == 'bids-metadata':
            structure = self._scan_bids_metadata(directory)
        elif format_type == 'bids-quality':
            structure = self._scan_bids_quality(directory)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        # Convert defaultdict to regular dict for cleaner output
        return {
            patient_id: {
                session_id: modalities 
                for session_id, modalities in sessions.items()
            }
            for patient_id, sessions in structure.items()
        }

    def _scan_bids_dicom(self, directory: Path) -> Dict[str, Dict[str, Set[str]]]:
        """
        Scan BIDS DICOM structure: sub-*/ses-*/anat/{modality}/
        
        Args:
            directory: Root directory
            
        Returns:
            Structure dictionary
        """
        structure = defaultdict(lambda: defaultdict(set))
        
        for subject_dir in sorted(directory.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                anat_dir = session_dir / "anat"
                
                if not anat_dir.exists():
                    continue
                
                # Each subdirectory in anat is a modality
                for modality_dir in anat_dir.iterdir():
                    if not modality_dir.is_dir():
                        continue
                    
                    # Check if directory contains DICOM files
                    dicom_files = list(modality_dir.glob("*.dcm")) + \
                                 list(modality_dir.glob("*.DCM")) + \
                                 list(modality_dir.rglob("*.dcm"))  # Check nested dirs too
                    
                    if dicom_files:
                        modality = modality_dir.name.lower()
                        structure[patient_id][session_id].add(modality)
        
        return structure

    def _scan_bids_nifti(self, directory: Path) -> Dict[str, Dict[str, Set[str]]]:
        """
        Scan BIDS NIfTI structure: sub-*/ses-*/anat/*.nii.gz
        
        Args:
            directory: Root directory
            
        Returns:
            Structure dictionary
        """
        structure = defaultdict(lambda: defaultdict(set))
        
        for subject_dir in sorted(directory.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                anat_dir = session_dir / "anat"
                
                if not anat_dir.exists():
                    continue
                
                # Find NIfTI files
                for nifti_file in anat_dir.glob("*.nii.gz"):
                    # Extract modality from filename.
                    # Expected format: sub-{patient}_ses-{session}_{modality}.nii.gz
                    # dcm2niix may append extra suffixes (_Eq_1, _e2, etc.) producing
                    # sub-{patient}_ses-{session}_{modality}_Eq_1.nii.gz — modality is
                    # always the 3rd underscore-separated token (index 2), never the last.
                    filename = nifti_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
                    parts = filename.split('_')

                    if len(parts) >= 3:
                        modality = parts[2].lower()  # 3rd token: sub-X_ses-Y_{modality}[_extras]
                        structure[patient_id][session_id].add(modality)
        
        return structure

    def _scan_bids_metadata(self, directory: Path) -> Dict[str, Dict[str, Set[str]]]:
        """
        Scan BIDS metadata structure: sub-*/ses-*/anat/{modality}/*.json
        (or sub-*/ses-*/metadata/*.json as alternative)
        
        Args:
            directory: Root directory
            
        Returns:
            Structure dictionary
        """
        structure = defaultdict(lambda: defaultdict(set))
        
        for subject_dir in sorted(directory.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                
                # Try two possible locations:
                # 1. sub-*/ses-*/anat/{modality}/*.json (primary)
                # 2. sub-*/ses-*/metadata/*.json (alternative)
                
                anat_dir = session_dir / "anat"
                metadata_dir = session_dir / "metadata"
                
                # Check anat directory first
                if anat_dir.exists():
                    # Look for modality subdirectories
                    for modality_dir in anat_dir.iterdir():
                        if not modality_dir.is_dir():
                            continue
                        
                        # Find JSON files in modality directory
                        json_files = list(modality_dir.glob("*.json"))
                        
                        if json_files:
                            modality = modality_dir.name.lower()
                            structure[patient_id][session_id].add(modality)
                
                # Check metadata directory as fallback
                if metadata_dir.exists():
                    for json_file in metadata_dir.glob("*.json"):
                        # Extract modality from filename
                        # Expected format: sub-{patient}_ses-{session}_{modality}_metadata.json
                        filename = json_file.stem
                        parts = filename.split('_')
                        
                        # Find modality (part before _metadata)
                        if len(parts) >= 3:
                            # Remove 'metadata' suffix if present
                            if parts[-1] == 'metadata':
                                modality = parts[-2].lower()
                            else:
                                modality = parts[-1].lower()
                            structure[patient_id][session_id].add(modality)
        
        return structure

    def _scan_bids_quality(self, directory: Path) -> Dict[str, Dict[str, Set[str]]]:
        """
        Scan BIDS quality assessment structure.
        Quality JSON files can be in two locations:
        - sub-*/ses-*/anat/*_quality.json (primary)
        - sub-*/ses-*/quality/*.json (alternative)
        
        Args:
            directory: Root directory
            
        Returns:
            Structure dictionary
        """
        structure = defaultdict(lambda: defaultdict(set))
        
        for subject_dir in sorted(directory.glob("sub-*")):
            if not subject_dir.is_dir():
                continue
            
            patient_id = subject_dir.name.replace("sub-", "")
            
            for session_dir in sorted(subject_dir.glob("ses-*")):
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name.replace("ses-", "")
                
                # Try two possible locations:
                # 1. sub-*/ses-*/anat/*_quality.json (primary location)
                anat_dir = session_dir / "anat"
                if anat_dir.exists():
                    for json_file in anat_dir.glob("*_quality.json"):
                        # Extract modality from filename
                        # Expected format: sub-{patient}_ses-{session}_{modality}_quality.json
                        filename = json_file.stem
                        parts = filename.split('_')
                        
                        # Find modality (part before _quality)
                        if len(parts) >= 4 and parts[-1] == 'quality':
                            modality = parts[-2].lower()
                            structure[patient_id][session_id].add(modality)
                
                # 2. sub-*/ses-*/quality/*.json (alternative location)
                quality_dir = session_dir / "quality"
                if quality_dir.exists():
                    for json_file in quality_dir.glob("*.json"):
                        # Extract modality from filename
                        # Expected format: sub-{patient}_ses-{session}_{modality}_quality.json
                        filename = json_file.stem
                        parts = filename.split('_')
                        
                        # Find modality (part before _quality)
                        if len(parts) >= 3:
                            # Remove 'quality' suffix if present
                            if parts[-1] == 'quality':
                                modality = parts[-2].lower()
                            else:
                                modality = parts[-1].lower()
                            structure[patient_id][session_id].add(modality)
        
        return structure

    def compare_structures(
        self, 
        input_structure: Dict[str, Dict[str, Set[str]]], 
        output_structure: Dict[str, Dict[str, Set[str]]]
    ) -> Dict:
        """
        Compare input and output structures to find mismatches.
        
        Args:
            input_structure: Input structure from scan_structure()
            output_structure: Output structure from scan_structure()
        
        Returns:
            Dictionary with comparison results including incomplete data
        """
        incomplete_data = []
        complete_patients = 0
        complete_sessions = 0
        total_patients = len(input_structure)
        total_sessions = 0
        
        # Check all patients from input
        for patient_id, input_sessions in input_structure.items():
            patient_has_issues = False
            patient_incomplete_sessions = []
            
            for session_id, input_modalities in input_sessions.items():
                total_sessions += 1
                
                # Check if session exists in output
                if patient_id not in output_structure:
                    patient_has_issues = True
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'input_modalities': sorted(list(input_modalities)),
                        'output_modalities': [],
                        'missing_in_output': sorted(list(input_modalities)),
                        'reason': 'patient_missing_in_output'
                    })
                    continue
                
                if session_id not in output_structure[patient_id]:
                    patient_has_issues = True
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'input_modalities': sorted(list(input_modalities)),
                        'output_modalities': [],
                        'missing_in_output': sorted(list(input_modalities)),
                        'reason': 'session_missing_in_output'
                    })
                    continue
                
                output_modalities = output_structure[patient_id][session_id]
                
                # Check if modalities match
                missing_in_output = input_modalities - output_modalities
                extra_in_output = output_modalities - input_modalities
                
                if missing_in_output or extra_in_output:
                    patient_has_issues = True
                    session_info = {
                        'session_id': session_id,
                        'input_modalities': sorted(list(input_modalities)),
                        'output_modalities': sorted(list(output_modalities)),
                    }
                    
                    if missing_in_output:
                        session_info['missing_in_output'] = sorted(list(missing_in_output))
                    
                    if extra_in_output:
                        session_info['extra_in_output'] = sorted(list(extra_in_output))
                    
                    session_info['reason'] = 'modality_mismatch'
                    patient_incomplete_sessions.append(session_info)
                else:
                    complete_sessions += 1
            
            if patient_has_issues:
                incomplete_data.append({
                    'patient_id': patient_id,
                    'incomplete_sessions': patient_incomplete_sessions
                })
            else:
                complete_patients += 1
        
        # Calculate statistics
        incomplete_patients = len(incomplete_data)
        incomplete_sessions = total_sessions - complete_sessions
        
        success_rate = (complete_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        return {
            'statistics': {
                'total_patients': total_patients,
                'complete_patients': complete_patients,
                'incomplete_patients': incomplete_patients,
                'total_sessions': total_sessions,
                'complete_sessions': complete_sessions,
                'incomplete_sessions': incomplete_sessions,
                'success_rate_percent': round(success_rate, 2)
            },
            'incomplete_data': incomplete_data
        }
    
    def validate_segmentation_completion(
        self, 
        input_structure: Dict[str, Dict[str, Set[str]]], 
        output_structure: Dict[str, Dict[str, Set[str]]]
    ) -> Dict:
        """
        Validate segmentation stage completion.
        
        Checks that each input session has corresponding output data.
        Does not compare modalities (input: 4 modalities -> output: 1 mask is OK).
        
        Args:
            input_structure: Input structure from scan_structure()
            output_structure: Output structure from scan_structure()
        
        Returns:
            Dictionary with comparison results
        """
        incomplete_data = []
        complete_patients = 0
        complete_sessions = 0
        total_patients = len(input_structure)
        total_sessions = 0
        
        # Check all patients from input
        for patient_id, input_sessions in input_structure.items():
            patient_has_issues = False
            patient_incomplete_sessions = []
            
            for session_id, input_modalities in input_sessions.items():
                total_sessions += 1
                
                # Check if patient exists in output
                if patient_id not in output_structure:
                    patient_has_issues = True
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'input_modalities': sorted(list(input_modalities)),
                        'output_data': False,
                        'reason': 'patient_missing_in_output'
                    })
                    continue
                
                # Check if session exists in output
                if session_id not in output_structure[patient_id]:
                    patient_has_issues = True
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'input_modalities': sorted(list(input_modalities)),
                        'output_data': False,
                        'reason': 'session_missing_in_output'
                    })
                    continue
                
                # Check if ANY output data exists
                output_modalities = output_structure[patient_id][session_id]
                if not output_modalities:
                    patient_has_issues = True
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'input_modalities': sorted(list(input_modalities)),
                        'output_data': False,
                        'reason': 'no_output_data'
                    })
                else:
                    # Session has output - count as complete
                    complete_sessions += 1
            
            if patient_has_issues:
                incomplete_data.append({
                    'patient_id': patient_id,
                    'incomplete_sessions': patient_incomplete_sessions
                })
            else:
                complete_patients += 1
        
        # Calculate statistics
        incomplete_patients = len(incomplete_data)
        incomplete_sessions = total_sessions - complete_sessions
        success_rate = (complete_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        return {
            'statistics': {
                'total_patients': total_patients,
                'complete_patients': complete_patients,
                'incomplete_patients': incomplete_patients,
                'total_sessions': total_sessions,
                'complete_sessions': complete_sessions,
                'incomplete_sessions': incomplete_sessions,
                'success_rate_percent': round(success_rate, 2)
            },
            'incomplete_data': incomplete_data
        }

    def generate_incomplete_report(
        self,
        comparison_result: Dict,
        stage_name: str,
        output_dir: Path,
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate and save incomplete data report.
        
        Args:
            comparison_result: Result from compare_structures()
            stage_name: Name of the pipeline stage (e.g., "02_extract_metadata")
            output_dir: Directory to save the report
            filename: Optional custom filename (default: {stage_name}_incomplete_data.json)
        
        Returns:
            Path to saved report file
        """
        # Create incomplete_data directory if it doesn't exist
        incomplete_dir = output_dir / "incomplete_data"
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stage': stage_name,
            'statistics': comparison_result['statistics'],
            'incomplete_data': comparison_result['incomplete_data']
        }
        
        # Determine filename
        if filename is None:
            filename = f"{stage_name}_incomplete_data.json"
        
        report_path = incomplete_dir / filename
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Incomplete data report saved to: {report_path}")
        self.logger.info(f"Success rate: {report['statistics']['success_rate_percent']}%")
        self.logger.info(
            f"Complete: {report['statistics']['complete_sessions']}/{report['statistics']['total_sessions']} sessions"
        )
        
        if report['statistics']['incomplete_patients'] > 0:
            self.logger.warning(
                f"Found {report['statistics']['incomplete_patients']} patients with incomplete data"
            )
        
        return report_path


class CompletenessValidator:
    """
    Validates that patients/sessions have all required modalities.
    Used for Stage 1 (reorganize_folders).
    """
    
    def __init__(
        self, 
        required_modalities: Set[str] = {'t1', 't1c', 't2', 't2fl'},
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize validator.
        
        Args:
            required_modalities: Set of required modality names
            logger: Logger instance (optional)
        """
        self.required_modalities = {m.lower() for m in required_modalities}
        self.logger = logger or logging.getLogger(__name__)

    def check_completeness(
        self, 
        structure: Dict[str, Dict[str, Set[str]]]
    ) -> Dict:
        """
        Check if all patients/sessions have required modalities.
        
        Args:
            structure: Structure from InputOutputValidator.scan_structure()
        
        Returns:
            Dictionary with completeness results
        """
        incomplete_data = []
        complete_patients = 0
        complete_sessions = 0
        total_patients = len(structure)
        total_sessions = 0
        
        for patient_id, sessions in structure.items():
            patient_has_issues = False
            patient_incomplete_sessions = []
            
            for session_id, modalities in sessions.items():
                total_sessions += 1
                
                # Normalize modality names to lowercase
                modalities_lower = {m.lower() for m in modalities}
                missing_modalities = self.required_modalities - modalities_lower
                
                if missing_modalities:
                    patient_has_issues = True
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'available_modalities': sorted(list(modalities_lower)),
                        'missing_modalities': sorted(list(missing_modalities))
                    })
                else:
                    complete_sessions += 1
            
            if patient_has_issues:
                incomplete_data.append({
                    'patient_id': patient_id,
                    'incomplete_sessions': patient_incomplete_sessions
                })
            else:
                complete_patients += 1
        
        # Calculate statistics
        incomplete_patients = len(incomplete_data)
        incomplete_sessions = total_sessions - complete_sessions
        success_rate = (complete_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        return {
            'statistics': {
                'total_patients': total_patients,
                'complete_patients': complete_patients,
                'incomplete_patients': incomplete_patients,
                'total_sessions': total_sessions,
                'complete_sessions': complete_sessions,
                'incomplete_sessions': incomplete_sessions,
                'success_rate_percent': round(success_rate, 2),
                'required_modalities': sorted(list(self.required_modalities))
            },
            'incomplete_data': incomplete_data
        }

    def generate_completeness_report(
        self,
        completeness_result: Dict,
        stage_name: str,
        output_dir: Path,
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate and save completeness report.
        
        Args:
            completeness_result: Result from check_completeness()
            stage_name: Name of the pipeline stage
            output_dir: Directory to save the report
            filename: Optional custom filename
        
        Returns:
            Path to saved report file
        """
        # Create incomplete_data directory if it doesn't exist
        incomplete_dir = output_dir / "incomplete_data"
        incomplete_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stage': stage_name,
            'statistics': completeness_result['statistics'],
            'incomplete_data': completeness_result['incomplete_data']
        }
        
        # Determine filename
        if filename is None:
            filename = f"{stage_name}_incomplete_data.json"
        
        report_path = incomplete_dir / filename
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Completeness report saved to: {report_path}")
        self.logger.info(f"Success rate: {report['statistics']['success_rate_percent']}%")
        self.logger.info(
            f"Complete: {report['statistics']['complete_sessions']}/{report['statistics']['total_sessions']} sessions"
        )
        
        if report['statistics']['incomplete_patients'] > 0:
            self.logger.warning(
                f"Found {report['statistics']['incomplete_patients']} patients with incomplete modalities"
            )
        
        return report_path