import os
import shutil
import pydicom
import pydicom.dataelem
import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
from functools import wraps


# --- Data Classes ---
@dataclass
class SeriesInfo:
    uid: str
    files: List[str]
    series_number: int
    study_datetime: datetime
    first_dataset: pydicom.Dataset
    protocol_name: str = ""
    series_desc: str = ""


@dataclass
class StudyInfo:
    uid: str
    series: Dict[str, SeriesInfo]
    study_datetime: datetime


@dataclass
class PatientData:
    original_id: str
    studies: Dict[str, StudyInfo]


# --- Configuration ---
MODALITY_KEYWORDS = {
    't1c': {
        'keywords': ['ce', 't1', 'contrast', 'gad', 'postcontrast', 't1c', 't1+c', 't1-ce', 
                    't1contrast', 't1 gd', 'ce-t1', 't1 post', 'gd t1', 'mdc', 'with contrast', '+gd', 'post gd'],
        'forbidden': ['mpr'],
        'priority_order': ['tfe', 'tse']
    },
    't1': {
        'keywords': ['t1', 't1w', 't1 weighted', 'spgr', 'mprage', 'tfl', 'bravo'],
        'forbidden': ['ce', 'mpr', 'contrast'],
        'priority_order': ['tfe', 'tse']
    },
    't2fl': {
        'keywords': ['flair', 't2fl', 'fluid attenuated inversion recovery', 'ir_fse', 'darkfluid'],
        'forbidden': ['mpr'],
        'priority_order': []
    },
    't2': {
        'keywords': ['t2', 't2w', 't2 weighted', 'tse', 'fse', 't2 tse', 't2 fse', 'haste'],
        'forbidden': ['mpr'],
        'priority_order': ['axi', 'sag', 'cor']
    }
}

TECHNICAL_PARAMS = {
    't2fl': {'ti_min': 1500, 'tr_min': 4000, 'te_min': 70},
    't1': {'tr_max': 1000, 'te_max': 30},
    't1c': {'tr_max': 1200, 'te_max': 30},
    't2': {'tr_min': 2000, 'te_min': 70}
}

# 2021-2022 specific keywords
LEGACY_KEYWORDS_2021_2022 = {
    't1c': {'required': ['ce', 't1'], 'forbidden': ['mpr'], 'prefer_order': ['tfe', 'tse']},
    't1': {'required': ['t1'], 'forbidden': ['ce', 'mpr'], 'prefer_order': ['tfe', 'tse']},
    't2fl': {'required': ['flair'], 'forbidden': ['mpr']},
    't2': {'required': ['t2'], 'forbidden': ['mpr'], 'prefer_order': ['axi', 'sag', 'cor']}
}


# --- Custom Exceptions ---
class DicomProcessingError(Exception):
    """Custom exception for DICOM processing errors."""
    pass


# --- Global Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Error Handling Decorator ---
def handle_dicom_error(func):
    """Decorator for consistent DICOM error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


# --- Utility Functions ---
def setup_logging(log_file_path: str):
    """Set up logging to console (INFO) and file (DEBUG)."""
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"File logging configured: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to configure file logging {log_file_path}: {e}")


def normalize_dicom_text(value: Any) -> str:
    """Convert DICOM value to normalized lowercase string."""
    if value is None:
        return ""
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return " ".join(str(v).strip().lower() for v in value if v is not None)
    return str(value).strip().lower()


def safe_float(value: Any, tag_name: str = "value") -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug(f"Cannot convert {tag_name}='{value}' (type: {type(value)}) to float.")
        return None


def get_dicom_value(ds: pydicom.Dataset, tag: Union[tuple, str], default: Any = None) -> Any:
    """Safely extract value from DICOM dataset."""
    try:
        val = ds.get(tag, default)
        if val is default:
            return default
        
        if isinstance(val, pydicom.dataelem.DataElement):
            val = val.value
            if val is None:
                return default
        
        if val is None:
            return default
            
        if isinstance(val, str):
            return val.strip().lower()
        if isinstance(val, (pydicom.multival.MultiValue, list)):
            return [v.strip().lower() if isinstance(v, str) else v for v in val if v is not None]
        return val
    except Exception as e:
        logger.error(f"Exception in get_dicom_value for tag {tag}: {e}")
        return default


@handle_dicom_error
def safe_read_dicom(file_path: str, specific_tags: Optional[List] = None) -> Optional[pydicom.Dataset]:
    """Safely read DICOM file with error handling."""
    return pydicom.dcmread(file_path, stop_before_pixels=True, specific_tags=specific_tags)


def is_dicom_file(file_path: str) -> bool:
    """Check if file is a valid DICOM file."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except pydicom.errors.InvalidDicomError:
        logger.debug(f"File {file_path} is not a valid DICOM.")
        return False
    except Exception as e:
        logger.warning(f"Error checking DICOM file {file_path}: {e}")
        return False


# --- DICOM Scanner Class ---
class DicomScanner:
    """Handles scanning and collecting DICOM files."""
    
    def scan_directory(self, input_dir: str) -> Dict[str, PatientData]:
        """Scan directory and collect DICOM metadata."""
        logger.info("Phase 1: Scanning DICOM files and collecting metadata...")
        
        collected_data = {}
        dicom_file_count = 0
        processed_file_count = 0
        
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                dicom_file_count += 1
                
                if dicom_file_count % 500 == 0:
                    logger.info(f"  Scanned files: {dicom_file_count}...")
                
                if is_dicom_file(file_path):
                    ds = safe_read_dicom(file_path)
                    if ds is None:
                        continue
                        
                    series_info = self._extract_series_info(ds, file_path)
                    if series_info is None:
                        continue
                        
                    self._add_to_collected_data(collected_data, series_info, file_path)
                    processed_file_count += 1
        
        logger.info(f"Phase 1 completed. Total files scanned: {dicom_file_count}. DICOM files processed: {processed_file_count}.")
        return collected_data
    
    def _extract_series_info(self, ds: pydicom.Dataset, file_path: str) -> Optional[Dict]:
        """Extract series information from DICOM dataset."""
        pat_id = get_dicom_value(ds, (0x0010, 0x0020), "UNKNOWN_PATIENT_ID")
        study_uid = get_dicom_value(ds, (0x0020, 0x000D), "UNKNOWN_STUDY_UID")
        series_uid = get_dicom_value(ds, (0x0020, 0x000E), "UNKNOWN_SERIES_UID")
        
        if any(val.startswith("UNKNOWN_") for val in [pat_id, study_uid, series_uid]):
            logger.warning(f"Skipped file {file_path}: missing PatientID, StudyUID or SeriesUID.")
            return None
        
        study_date_str = get_dicom_value(ds, (0x0008, 0x0020), "00000000")
        study_time_str = get_dicom_value(ds, (0x0008, 0x0030), "000000.000000").split('.')[0]
        
        try:
            study_datetime = datetime.strptime(f"{study_date_str}{study_time_str}", "%Y%m%d%H%M%S")
        except ValueError:
            logger.warning(f"Invalid date/time ({study_date_str}/{study_time_str}) for {study_uid}.")
            study_datetime = datetime.min
        
        series_num_val = get_dicom_value(ds, (0x0020, 0x0011))
        try:
            series_number = int(series_num_val) if series_num_val is not None else float('inf')
        except ValueError:
            series_number = float('inf')
            logger.warning(f"SeriesNumber '{series_num_val}' for series {series_uid} is not a number.")
        
        return {
            'pat_id': pat_id,
            'study_uid': study_uid,
            'series_uid': series_uid,
            'study_datetime': study_datetime,
            'series_number': series_number,
            'dataset': ds,
            'protocol_name': normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), "")),
            'series_desc': normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        }
    
    def _add_to_collected_data(self, collected_data: Dict, series_info: Dict, file_path: str):
        """Add series information to collected data structure."""
        pat_id = series_info['pat_id']
        study_uid = series_info['study_uid']
        series_uid = series_info['series_uid']
        
        # Initialize patient data if not exists
        if pat_id not in collected_data:
            collected_data[pat_id] = PatientData(original_id=pat_id, studies={})
        
        # Initialize study data if not exists
        if study_uid not in collected_data[pat_id].studies:
            collected_data[pat_id].studies[study_uid] = StudyInfo(
                uid=study_uid,
                series={},
                study_datetime=series_info['study_datetime']
            )
        
        # Initialize series data if not exists
        if series_uid not in collected_data[pat_id].studies[study_uid].series:
            collected_data[pat_id].studies[study_uid].series[series_uid] = SeriesInfo(
                uid=series_uid,
                files=[],
                series_number=series_info['series_number'],
                study_datetime=series_info['study_datetime'],
                first_dataset=series_info['dataset'],
                protocol_name=series_info['protocol_name'],
                series_desc=series_info['series_desc']
            )
        
        # Add file to series
        collected_data[pat_id].studies[study_uid].series[series_uid].files.append(file_path)


# --- Modality Detector Class ---
class ModalityDetector:
    """Handles modality detection logic."""
    
    def __init__(self):
        self.keywords = MODALITY_KEYWORDS
        self.technical_params = TECHNICAL_PARAMS
        self.legacy_keywords = LEGACY_KEYWORDS_2021_2022
    
    def determine_modality(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Determine modality from DICOM dataset."""
        logger.debug(f"Determining modality for {os.path.basename(file_path)}:")
        
        # Extract basic information
        study_year = self._get_study_year(ds)
        
        # Check for 2021-2022 legacy protocol
        if study_year in [2021, 2022]:
            modality = self._detect_legacy_modality(ds)
            if modality != 'unknown':
                return modality
            logger.debug("File doesn't match 2021-2022 rules - will be ignored.")
            return 'unknown'
        
        # Standard detection workflow
        detectors = [
            self._detect_by_protocol,
            self._detect_by_series_description,
            self._detect_by_technical_params,
            lambda ds, fp: self._detect_by_file_path(fp)
        ]
        
        for detector in detectors:
            modality = detector(ds, file_path)
            if modality != 'unknown':
                return modality
        
        logger.warning(f"Could not determine modality for file: {os.path.basename(file_path)}")
        return 'unknown'
    
    def _get_study_year(self, ds: pydicom.Dataset) -> Optional[int]:
        """Extract study year from DICOM dataset."""
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                return int(study_date_val[:4])
            except ValueError:
                pass
        return None
    
    def _detect_legacy_modality(self, ds: pydicom.Dataset) -> str:
        """Detect modality using 2021-2022 legacy rules."""
        logger.debug("Using 2021-2022 legacy protocol.")
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        
        for modality, rules in self.legacy_keywords.items():
            if self._matches_legacy_keywords(protocol_name, rules):
                logger.debug(f"Legacy detection: selected '{modality}'")
                return modality
        
        return 'unknown'
    
    def _matches_legacy_keywords(self, text: str, rules: Dict) -> bool:
        """Check if text matches legacy keyword rules."""
        # Check forbidden keywords
        if any(fw in text for fw in rules.get('forbidden', [])):
            return False
        
        # Check required keywords
        if not all(rw in text for rw in rules.get('required', [])):
            return False
        
        return True
    
    def _detect_by_protocol(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Detect modality by protocol name and contrast agent."""
        logger.debug("  PRIORITY 1: Analyzing ProtocolName and ContrastAgent")
        
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        
        # Check for contrast
        has_contrast = contrast_agent and contrast_agent not in ["", "none", "no"]
        
        # Check each modality
        for modality, config in self.keywords.items():
            if self._find_keywords_in_text(protocol_name, config['keywords']):
                if modality == 't1c' and (has_contrast or 'contrast' in protocol_name):
                    logger.debug(f"Modality 't1c' detected by ProtocolName: contrast found.")
                    return 't1c'
                elif modality != 't1c' and not has_contrast:
                    logger.debug(f"Modality '{modality}' detected by ProtocolName.")
                    return modality
        
        return 'unknown'
    
    def _detect_by_series_description(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Detect modality by series description."""
        logger.debug("  PRIORITY 2: Analyzing SeriesDescription")
        
        series_desc = normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        has_contrast = contrast_agent and contrast_agent not in ["", "none", "no"]
        
        for modality, config in self.keywords.items():
            if self._find_keywords_in_text(series_desc, config['keywords']):
                if modality == 't1c' and (has_contrast or any(kw in series_desc for kw in ['contrast', 'gad', 'ce'])):
                    logger.debug(f"Modality 't1c' detected by SeriesDescription.")
                    return 't1c'
                elif modality != 't1c' and not has_contrast:
                    logger.debug(f"Modality '{modality}' detected by SeriesDescription.")
                    return modality
        
        return 'unknown'
    
    def _detect_by_technical_params(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Detect modality by technical parameters."""
        logger.debug("  PRIORITY 3: Analyzing technical parameters")
        
        tr_val = safe_float(get_dicom_value(ds, (0x0018, 0x0080)), "TR")
        te_val = safe_float(get_dicom_value(ds, (0x0018, 0x0081)), "TE")
        ti_val = safe_float(get_dicom_value(ds, (0x0018, 0x0082)), "TI")
        
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        has_contrast = contrast_agent and contrast_agent not in ["", "none", "no"]
        
        # FLAIR detection by TI
        if ti_val and ti_val > 1500:
            logger.debug(f"Modality 't2fl' detected by TI={ti_val}.")
            return 't2fl'
        
        # T1C detection by contrast
        if has_contrast and tr_val and te_val and tr_val < 1200 and te_val < 30:
            logger.debug(f"Modality 't1c' detected by contrast agent and TR/TE.")
            return 't1c'
        
        # T1 detection by TR/TE
        if tr_val and te_val and tr_val < 1000 and te_val < 30 and not has_contrast:
            logger.debug(f"Modality 't1' detected by TR={tr_val} and TE={te_val}.")
            return 't1'
        
        # T2 detection by TR/TE
        if tr_val and te_val and tr_val > 2000 and te_val > 70:
            if not ti_val or ti_val < 1500:  # Make sure it's not FLAIR
                logger.debug(f"Modality 't2' detected by TR={tr_val} and TE={te_val}.")
                return 't2'
        
        return 'unknown'
    
    def _detect_by_file_path(self, file_path: str) -> str:
        """Fallback detection by file path."""
        logger.debug("  FALLBACK: Analyzing file path")
        
        path_parts = os.path.normpath(file_path).split(os.sep)
        for part in reversed(path_parts):
            part_lower = part.lower()
            
            for modality, config in self.keywords.items():
                if self._find_keywords_in_text(part_lower, config['keywords']):
                    logger.debug(f"Modality '{modality}' detected by path: '{part_lower}'")
                    return modality
        
        return 'unknown'
    
    def _find_keywords_in_text(self, text: str, keywords: List[str]) -> bool:
        """Check if any keyword is found in text."""
        if not text or not keywords:
            return False
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)


# --- BIDS Organizer Class ---
class BidsOrganizer:
    """Handles BIDS structure creation and file operations."""
    
    def __init__(self, output_dir: str, action_type: str = 'copy'):
        self.output_dir = output_dir
        self.action_type = action_type
        self.detector = ModalityDetector()
    
    def organize_to_bids(self, collected_data: Dict[str, PatientData]):
        """Organize collected data into BIDS structure."""
        logger.info("Phase 2: Creating BIDS structure and copying files...")
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create output directory {self.output_dir}: {e}")
            raise
        
        # Create BIDS patient IDs
        patient_bids_map = self._create_patient_bids_mapping(collected_data)
        
        for patient_data in collected_data.values():
            self._process_patient(patient_data, patient_bids_map)
        
        logger.info("BIDS organization completed!")
    
    def _create_patient_bids_mapping(self, collected_data: Dict) -> Dict[str, str]:
        """Create mapping from original patient IDs to BIDS IDs."""
        sorted_patient_ids = sorted(collected_data.keys())
        return {orig_id: f"sub-{i+1:03d}" for i, orig_id in enumerate(sorted_patient_ids)}
    
    def _process_patient(self, patient_data: PatientData, patient_bids_map: Dict[str, str]):
        """Process a single patient's data."""
        bids_sub_id = patient_bids_map[patient_data.original_id]
        logger.info(f"Processing patient: {patient_data.original_id} -> {bids_sub_id}")
        
        # Create session BIDS mapping
        session_bids_map = self._create_session_bids_mapping(patient_data.studies)
        
        for study_info in patient_data.studies.values():
            self._process_study(study_info, bids_sub_id, session_bids_map)
    
    def _create_session_bids_mapping(self, studies: Dict[str, StudyInfo]) -> Dict[str, str]:
        """Create mapping from original study UIDs to BIDS session IDs."""
        sorted_study_uids = sorted(studies.keys(), key=lambda uid: studies[uid].study_datetime)
        return {orig_id: f"ses-{i+1:03d}" for i, orig_id in enumerate(sorted_study_uids)}
    
    def _process_study(self, study_info: StudyInfo, bids_sub_id: str, session_bids_map: Dict[str, str]):
        """Process a single study/session."""
        bids_ses_id = session_bids_map[study_info.uid]
        logger.info(f"  Processing session: {study_info.uid} -> {bids_ses_id}")
        
        bids_anat_path = os.path.join(self.output_dir, bids_sub_id, bids_ses_id, 'anat')
        
        # Group series by modality
        modality_groups = self._group_series_by_modality(study_info.series)
        
        if not modality_groups:
            logger.warning(f"    No series with known modality for session {bids_ses_id}.")
            return
        
        # Process each modality group
        for modality, series_list in modality_groups.items():
            self._process_modality_group(series_list, modality, bids_anat_path, bids_sub_id, bids_ses_id)
    
    def _group_series_by_modality(self, series_dict: Dict[str, SeriesInfo]) -> Dict[str, List[SeriesInfo]]:
        """Group series by detected modality."""
        modality_groups = defaultdict(list)
        
        for series_info in series_dict.values():
            modality = self.detector.determine_modality(series_info.first_dataset, series_info.files[0])
            
            if modality == 'unknown':
                logger.warning(f"    Skipping series {series_info.uid}: unknown modality.")
                continue
            
            modality_groups[modality].append(series_info)
        
        return dict(modality_groups)
    
    def _process_modality_group(self, series_list: List[SeriesInfo], modality: str, 
                               bids_anat_path: str, bids_sub_id: str, bids_ses_id: str):
        """Process a group of series with the same modality."""
        # Apply priority selection
        selected_series = self._apply_priority_selection(series_list, modality)
        
        # Create modality directory
        bids_modality_dir = os.path.join(bids_anat_path, modality)
        try:
            os.makedirs(bids_modality_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"    Cannot create directory {bids_modality_dir}: {e}")
            return
        
        # Process selected series
        for run_idx, series_info in enumerate(selected_series, 1):
            self._copy_series_files(series_info, modality, bids_modality_dir, 
                                  bids_sub_id, bids_ses_id, run_idx, len(selected_series))
    
    def _apply_priority_selection(self, series_list: List[SeriesInfo], modality: str) -> List[SeriesInfo]:
        """Apply priority selection for series within the same modality."""
        if len(series_list) <= 1:
            return series_list
        
        # Sort by series number by default
        sorted_series = sorted(series_list, key=lambda s: s.series_number)
        
        # Apply priority order if available
        priority_order = MODALITY_KEYWORDS.get(modality, {}).get('priority_order', [])
        if priority_order:
            for preferred in priority_order:
                for series in sorted_series:
                    if preferred in series.protocol_name or preferred in series.series_desc:
                        logger.debug(f"    Selected series by priority '{preferred}' for {modality}")
                        return [series]
            
            logger.warning(f"    No series found matching priority for {modality}. Using first by series number.")
        
        return [sorted_series[0]]
    
    def _copy_series_files(self, series_info: SeriesInfo, modality: str, bids_modality_dir: str,
                          bids_sub_id: str, bids_ses_id: str, run_idx: int, total_runs: int):
        """Copy files for a single series."""
        sorted_files = self._sort_files_by_instance_number(series_info.files)
        
        logger.info(f"    Copying {len(sorted_files)} files for {modality}"
                   f"{f'_run-{run_idx:02d}' if total_runs > 1 else ''} (Series UID: {series_info.uid})")
        
        for slice_idx, src_file_path in enumerate(sorted_files, 1):
            # Generate BIDS filename
            if total_runs > 1:
                bids_filename = f"{bids_sub_id}_{bids_ses_id}_run-{run_idx:02d}_{modality}_{slice_idx:03d}.dcm"
            else:
                bids_filename = f"{bids_sub_id}_{bids_ses_id}_{modality}_{slice_idx:03d}.dcm"
            
            dst_file_path = os.path.join(bids_modality_dir, bids_filename)
            
            # Copy or move file
            try:
                if self.action_type == 'move':
                    shutil.move(src_file_path, dst_file_path)
                else:
                    shutil.copy(src_file_path, dst_file_path)
            except Exception as e:
                logger.error(f"      Failed to {self.action_type} {src_file_path} to {dst_file_path}: {e}")
    
    def _sort_files_by_instance_number(self, dicom_files_paths: List[str]) -> List[str]:
        """Sort DICOM files by InstanceNumber."""
        sorted_files = []
        for f_path in dicom_files_paths:
            try:
                ds_slice = pydicom.dcmread(f_path, stop_before_pixels=True, specific_tags=[(0x0020,0x0013)]) # Читаем только InstanceNumber
                instance_number = get_dicom_value(ds_slice, (0x0020,0x0013))
                if instance_number is not None:
                    try:
                        instance_number = int(instance_number)
                    except ValueError:
                        logger.warning(f"Не удалось конвертировать InstanceNumber '{instance_number}' в int для файла {f_path}. Используем имя файла для сортировки этого элемента.")
                        instance_number = f_path # Fallback для этого файла
                else: # Если InstanceNumber отсутствует
                    logger.warning(f"InstanceNumber отсутствует в файле {f_path}. Используем имя файла для сортировки этого элемента.")
                    instance_number = f_path # Fallback для этого файла
                sorted_files.append((instance_number, f_path))
            except Exception as e:
                logger.error(f"Ошибка чтения InstanceNumber из файла {f_path}: {e}. Файл будет в конце или отсортирован по имени.")
                sorted_files.append((float('inf'), f_path)) # Помещаем файлы с ошибками в конец

        # Сортируем: сначала по числовому InstanceNumber, затем по пути файла (если InstanceNumber был строкой/одинаковый)
        sorted_files.sort(key=lambda x: (isinstance(x[0], str), x[0]))
        return [f_path for _, f_path in sorted_files]
    
# --- CLI Interface ---
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert DICOM files to BIDS format with automatic modality detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dicom/input /path/to/bids/output
  %(prog)s /path/to/dicom/input /path/to/bids/output --action move --log-file conversion.log
  %(prog)s /path/to/dicom/input /path/to/bids/output --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing DICOM files'
    )
    
    parser.add_argument(
        'output_dir', 
        type=str,
        help='Output directory for BIDS structure'
    )
    
    # Optional arguments
    parser.add_argument(
        '--action',
        choices=['copy', 'move'],
        default='copy',
        help='Action to perform on files: copy (default) or move'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='dicom_to_bids.log',
        help='Path to log file (default: dicom_to_bids.log)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging to console'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without copying/moving files'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='DICOM to BIDS Converter v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory.")
        sys.exit(1)
    
    # Validate output directory parent exists
    output_parent = os.path.dirname(os.path.abspath(args.output_dir))
    if not os.path.exists(output_parent):
        print(f"Error: Parent directory of output '{output_parent}' does not exist.")
        sys.exit(1)
    
    # Setup logging
    try:
        setup_logging(args.log_file)
        if args.verbose:
            # Add more verbose console logging
            console_handler = None
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    console_handler = handler
                    break
            if console_handler:
                console_handler.setLevel(logging.DEBUG)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)
    
    # Log startup information
    logger.info("="*60)
    logger.info("DICOM to BIDS Converter Started")
    logger.info("="*60)
    logger.info(f"Input directory: {os.path.abspath(args.input_dir)}")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Log file: {os.path.abspath(args.log_file)}")
    logger.info(f"Dry run: {args.dry_run}")
    
    try:
        # Phase 1: Scan DICOM files
        scanner = DicomScanner()
        collected_data = scanner.scan_directory(args.input_dir)
        
        if not collected_data:
            logger.warning("No valid DICOM files found in input directory.")
            print("Warning: No valid DICOM files found.")
            sys.exit(0)
        
        # Log summary statistics
        total_patients = len(collected_data)
        total_studies = sum(len(patient.studies) for patient in collected_data.values())
        total_series = sum(
            len(study.series) 
            for patient in collected_data.values() 
            for study in patient.studies.values()
        )
        total_files = sum(
            len(series.files)
            for patient in collected_data.values()
            for study in patient.studies.values()
            for series in study.series.values()
        )
        
        logger.info(f"Scan Summary:")
        logger.info(f"  Patients: {total_patients}")
        logger.info(f"  Studies: {total_studies}")
        logger.info(f"  Series: {total_series}")
        logger.info(f"  Files: {total_files}")
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be copied or moved")
            
            # Show what would be processed
            for patient_id, patient_data in collected_data.items():
                logger.info(f"Patient: {patient_id}")
                for study_uid, study_info in patient_data.studies.items():
                    logger.info(f"  Study: {study_uid} ({study_info.study_datetime})")
                    for series_uid, series_info in study_info.series.items():
                        modality = ModalityDetector().determine_modality(
                            series_info.first_dataset, 
                            series_info.files[0]
                        )
                        logger.info(f"    Series: {series_uid} - {modality} ({len(series_info.files)} files)")
        else:
            # Phase 2: Organize to BIDS
            organizer = BidsOrganizer(args.output_dir, args.action)
            organizer.organize_to_bids(collected_data)
        
        logger.info("="*60)
        logger.info("DICOM to BIDS Conversion Completed Successfully")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user.")
        print("\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        print(f"Error: Conversion failed. Check log file for details: {args.log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()