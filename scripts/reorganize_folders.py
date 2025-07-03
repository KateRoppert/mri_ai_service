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
from typing import List, Dict, Optional, Union, Any, Protocol, Tuple
from functools import wraps
from abc import ABC, abstractmethod


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


# --- Protocol Detection Strategy Pattern ---
class ModalityDetectionStrategy(ABC):
    """Abstract base class for modality detection strategies."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this detection strategy."""
        pass
    
    @abstractmethod
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Check if this strategy is applicable to the given dataset."""
        pass
    
    @abstractmethod
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using this strategy. Returns None if not detected."""
        pass
    
    def get_priority(self) -> int:
        """Return priority for this strategy (lower number = higher priority)."""
        return 100
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        return []
    
    def is_exclusive(self) -> bool:
        """Return True if this strategy should prevent fallback to other strategies when applicable."""
        return False


class StandardDetectionStrategy(ModalityDetectionStrategy):
    """Standard modality detection strategy."""
    
    def __init__(self):
        self.keywords = {
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
        
        self.technical_params = {
            't2fl': {'ti_min': 1500, 'tr_min': 4000, 'te_min': 70},
            't1': {'tr_max': 1000, 'te_max': 30},
            't1c': {'tr_max': 1200, 'te_max': 30},
            't2': {'tr_min': 2000, 'te_min': 70}
        }
    
    def get_name(self) -> str:
        return "Standard Detection"
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Standard detection is always applicable as fallback."""
        return True
    
    def get_priority(self) -> int:
        return 1000  # Lowest priority (highest number)
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        return self.keywords.get(modality, {}).get('priority_order', [])
    
    def is_exclusive(self) -> bool:
        """Standard detection is never exclusive - it's the fallback."""
        return False
    
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using standard workflow."""
        logger = logging.getLogger(__name__)
        
        # Try different detection methods in order
        detectors = [
            ("ProtocolName and ContrastAgent", self._detect_by_protocol),
            ("SeriesDescription", self._detect_by_series_description),
            ("Technical Parameters", self._detect_by_technical_params),
            ("File Path", lambda ds, fp: self._detect_by_file_path(fp))
        ]
        
        for detector_name, detector_func in detectors:
            logger.debug(f"  Trying {detector_name}...")
            modality = detector_func(ds, file_path)
            if modality:
                logger.debug(f"  ✓ Detected '{modality}' using {detector_name}")
                return modality
            else:
                logger.debug(f"  ✗ No match using {detector_name}")
        
        return None
    
    def _detect_by_protocol(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality by protocol name and contrast agent."""
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        
        has_contrast = contrast_agent and contrast_agent not in ["", "none", "no"]
        
        for modality, config in self.keywords.items():
            if self._find_keywords_in_text(protocol_name, config['keywords']):
                if modality == 't1c' and (has_contrast or 'contrast' in protocol_name):
                    return 't1c'
                elif modality != 't1c' and not has_contrast:
                    return modality
        
        return None
    
    def _detect_by_series_description(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality by series description."""
        series_desc = normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        has_contrast = contrast_agent and contrast_agent not in ["", "none", "no"]
        
        for modality, config in self.keywords.items():
            if self._find_keywords_in_text(series_desc, config['keywords']):
                if modality == 't1c' and (has_contrast or any(kw in series_desc for kw in ['contrast', 'gad', 'ce'])):
                    return 't1c'
                elif modality != 't1c' and not has_contrast:
                    return modality
        
        return None
    
    def _detect_by_technical_params(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality by technical parameters."""
        tr_val = safe_float(get_dicom_value(ds, (0x0018, 0x0080)), "TR")
        te_val = safe_float(get_dicom_value(ds, (0x0018, 0x0081)), "TE")
        ti_val = safe_float(get_dicom_value(ds, (0x0018, 0x0082)), "TI")
        
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        has_contrast = contrast_agent and contrast_agent not in ["", "none", "no"]
        
        # FLAIR detection by TI
        if ti_val and ti_val > 1500:
            return 't2fl'
        
        # T1C detection by contrast
        if has_contrast and tr_val and te_val and tr_val < 1200 and te_val < 30:
            return 't1c'
        
        # T1 detection by TR/TE
        if tr_val and te_val and tr_val < 1000 and te_val < 30 and not has_contrast:
            return 't1'
        
        # T2 detection by TR/TE
        if tr_val and te_val and tr_val > 2000 and te_val > 70:
            if not ti_val or ti_val < 1500:  # Make sure it's not FLAIR
                return 't2'
        
        return None
    
    def _detect_by_file_path(self, file_path: str) -> Optional[str]:
        """Fallback detection by file path."""
        path_parts = os.path.normpath(file_path).split(os.sep)
        for part in reversed(path_parts):
            part_lower = part.lower()
            
            for modality, config in self.keywords.items():
                if self._find_keywords_in_text(part_lower, config['keywords']):
                    return modality
        
        return None
    
    def _find_keywords_in_text(self, text: str, keywords: List[str]) -> bool:
        """Check if any keyword is found in text."""
        if not text or not keywords:
            return False
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)


class Legacy2021_2022DetectionStrategy(ModalityDetectionStrategy):
    """Detection strategy for 2021-2022 legacy protocol."""
    
    def __init__(self):
        self.keywords = {
            't1c': {'required': ['ce', 't1'], 'forbidden': ['mpr'], 'prefer_order': ['tfe', 'tse']},
            't1': {'required': ['t1'], 'forbidden': ['ce', 'mpr'], 'prefer_order': ['tfe', 'tse']},
            't2fl': {'required': ['flair'], 'forbidden': ['mpr']},
            't2': {'required': ['t2'], 'forbidden': ['mpr'], 'prefer_order': ['axi', 'sag', 'cor']}
        }
    
    def get_name(self) -> str:
        return "Legacy 2021-2022 Protocol"
    
    def get_priority(self) -> int:
        return 10  # Higher priority than standard
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        return self.keywords.get(modality, {}).get('prefer_order', [])
    
    def is_exclusive(self) -> bool:
        """This protocol is exclusive - if year matches, don't fall back to other protocols."""
        return True
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Check if this is a 2021-2022 study."""
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                year = int(study_date_val[:4])
                logger.debug(f"  Study Date: '{year}'")
                return year in [2021, 2022]
            except ValueError:
                pass
        return False
    
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using 2021-2022 legacy rules."""
        logger = logging.getLogger(__name__)
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        
        logger.debug(f"  Checking protocol name: '{protocol_name}'")
        
        for modality, rules in self.keywords.items():
            if self._matches_legacy_keywords(protocol_name, rules):
                logger.debug(f"  ✓ Matched modality '{modality}' with legacy rules")
                return modality
        
        logger.debug("  ✗ No match found with legacy rules")
        return None
    
    def _matches_legacy_keywords(self, text: str, rules: Dict) -> bool:
        """Check if text matches legacy keyword rules."""
        # Check forbidden keywords
        if any(fw in text for fw in rules.get('forbidden', [])):
            return False
        
        # Check required keywords (ALL must be present)
        if not all(rw in text for rw in rules.get('required', [])):
            return False
        
        return True


# Example of how to add a new protocol strategy
class CustomProtocol2018DetectionStrategy(ModalityDetectionStrategy):
    """Example custom protocol for 2018 studies."""
    
    def __init__(self):
        # Define your custom keywords and rules here
        self.keywords = {
            't1c': {'markers': ['ce_t1w'], 'exclude': []},
            't1': {'markers': ['t1w'], 'exclude': ['ce', 'thr']},
            't2fl': {'markers': ['flair']},
            't2': {'markers': ['t2w']}
        }
        # Define custom priority orders
        self.priority_orders = {
            't1c': ['se'],
            't1': ['tse', 'se'],
            't2': ['tse', 'se'],
            't2fl': ['sense', '']
        }
    
    def get_name(self) -> str:
        return "Custom Protocol 2018"
    
    def get_priority(self) -> int:
        return 10  # Higher priority than legacy
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        return self.priority_orders.get(modality, [])
    
    def is_exclusive(self) -> bool:
        """This protocol is exclusive - if year matches, don't fall back to other protocols."""
        return True
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Check if this is a 2018 study with custom protocol."""
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                year = int(study_date_val[:4])
                logger.debug(f"  Study Date: '{year}'")
                return year == 2018
            except ValueError:
                pass
        return False
    
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Implement your custom detection logic here."""
        logger = logging.getLogger(__name__)
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        
        for modality, rules in self.keywords.items():
            markers = rules.get('markers', [])
            exclude = rules.get('exclude', [])
            
            if any(m in protocol_name for m in markers) and not any(e in protocol_name for e in exclude):
                logger.debug(f"  ✓ Detected '{modality}' using custom 2018 rules")
                return modality
        
        return None

# --- Enhanced Modality Detector ---
class EnhancedModalityDetector:
    """Enhanced modality detector supporting multiple detection strategies."""
    
    def __init__(self):
        # Initialize all available strategies
        self.strategies: List[ModalityDetectionStrategy] = [
            Legacy2021_2022DetectionStrategy(),
            CustomProtocol2018DetectionStrategy(),
            StandardDetectionStrategy()  # Always last as fallback
        ]
        # Sort by priority (lower number = higher priority)
        self.strategies.sort(key=lambda s: s.get_priority())
        self.logger = logging.getLogger(__name__)
    
    def add_strategy(self, strategy: ModalityDetectionStrategy):
        """Add a new detection strategy."""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.get_priority())
        self.logger.info(f"Added detection strategy: {strategy.get_name()}")
    
    def determine_modality(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Determine modality using registered strategies."""
        modality, _ = self.determine_modality_with_strategy(ds, file_path)
        return modality
    
    def determine_modality_with_strategy(self, ds: pydicom.Dataset, file_path: str) -> Tuple[str, Optional[ModalityDetectionStrategy]]:
        """Determine modality using registered strategies, also returning which strategy was used."""
        self.logger.debug(f"Determining modality for {os.path.basename(file_path)}:")
        
        # Try each strategy in priority order
        for strategy in self.strategies:
            strategy_name = strategy.get_name()
            
            # Check if strategy is applicable
            if strategy.is_applicable(ds, file_path):
                self.logger.debug(f"→ Trying strategy: {strategy_name}")
                
                # Try to detect modality
                modality = strategy.detect_modality(ds, file_path)
                
                if modality and modality != 'unknown':
                    self.logger.info(f"✓ Modality '{modality}' detected using strategy: {strategy_name}")
                    return modality, strategy
                else:
                    # If this is an exclusive strategy and it didn't match, skip the file entirely
                    if strategy.is_exclusive():
                        self.logger.warning(f"  Exclusive strategy {strategy_name} did not match - skipping file")
                        return 'unknown', None
                    else:
                        self.logger.debug(f"  Strategy {strategy_name} did not detect modality")
            else:
                self.logger.debug(f"  Strategy {strategy_name} is not applicable")
        
        self.logger.warning(f"Could not determine modality for file: {os.path.basename(file_path)}")
        return 'unknown', None


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


# --- BIDS Organizer Class ---
class BidsOrganizer:
    """Handles BIDS structure creation and file operations."""
    
    def __init__(self, output_dir: str, action_type: str = 'copy'):
        self.output_dir = output_dir
        self.action_type = action_type
        self.detector = EnhancedModalityDetector()  # Use enhanced detector
    
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
        
        # Ensure all 4 required modalities have directories
        required_modalities = ['t1', 't1c', 't2fl', 't2']
        
        # Create directories for all required modalities
        for modality in required_modalities:
            modality_dir = os.path.join(bids_anat_path, modality)
            try:
                os.makedirs(modality_dir, exist_ok=True)
                logger.debug(f"    Created directory for {modality}")
            except OSError as e:
                logger.error(f"    Cannot create directory {modality_dir}: {e}")
        
        # Process series that were found
        for modality in required_modalities:
            if modality in modality_groups:
                series_with_strategies = modality_groups[modality]
                self._process_modality_group(series_with_strategies, modality, bids_anat_path, bids_sub_id, bids_ses_id)
            else:
                logger.warning(f"    No series found for required modality: {modality}")
        
        # Log summary
        logger.info(f"    Session {bids_ses_id} processing complete. "
                   f"Found modalities: {', '.join(modality_groups.keys())}")
    
    def _group_series_by_modality(self, series_dict: Dict[str, SeriesInfo]) -> Dict[str, List[Tuple[SeriesInfo, ModalityDetectionStrategy]]]:
        """Group series by detected modality, also tracking which strategy was used."""
        modality_groups = defaultdict(list)
        
        for series_info in series_dict.values():
            modality, strategy = self.detector.determine_modality_with_strategy(series_info.first_dataset, series_info.files[0])
            
            if modality == 'unknown':
                logger.warning(f"    Skipping series {series_info.uid}: unknown modality.")
                continue
            
            modality_groups[modality].append((series_info, strategy))
        
        return dict(modality_groups)
    
    def _process_modality_group(self, series_with_strategies: List[Tuple[SeriesInfo, ModalityDetectionStrategy]], 
                               modality: str, bids_anat_path: str, bids_sub_id: str, bids_ses_id: str):
        """Process a group of series with the same modality."""
        # Extract just the series for selection
        series_list = [series for series, _ in series_with_strategies]
        
        # Get all applicable strategies (might be different strategies for different series)
        strategies = list(set(strategy for _, strategy in series_with_strategies if strategy))
        
        # Apply priority selection - get the best series
        selected_series = self._apply_priority_selection(series_list, modality, strategies)
        
        # Create modality directory (should already exist from _process_study)
        bids_modality_dir = os.path.join(bids_anat_path, modality)
        
        # Process selected series (should be only one - the best match)
        for series_info in selected_series:
            self._copy_series_files(series_info, modality, bids_modality_dir, 
                                  bids_sub_id, bids_ses_id)
    
    def _apply_priority_selection(self, series_list: List[SeriesInfo], modality: str, 
                                 strategies: List[ModalityDetectionStrategy] = None) -> List[SeriesInfo]:
        """Apply priority selection for series within the same modality.
        First tries preference order, then falls back to newest acquisition time."""
        
        if len(series_list) <= 1:
            return series_list
        
        logger.debug(f"    Found {len(series_list)} series for {modality}, applying selection criteria")
        
        # First, try to select by preference order
        # Collect all priority orders from applicable strategies
        all_priority_orders = []
        
        if strategies:
            for strategy in strategies:
                priority_order = strategy.get_priority_order(modality)
                if priority_order:
                    all_priority_orders.extend(priority_order)
        
        # Also check standard keywords as fallback
        standard_keywords = {
            't1c': ['tfe', 'tse'],
            't1': ['tfe', 'tse'],
            't2': ['axi', 'sag', 'cor'],
            't2fl': []
        }
        
        # Combine and deduplicate priority orders (maintain order)
        seen = set()
        priority_order = []
        for item in all_priority_orders + standard_keywords.get(modality, []):
            if item not in seen:
                seen.add(item)
                priority_order.append(item)
        
        # Try to select by preference order
        if priority_order:
            logger.debug(f"    Checking preference order for {modality}: {priority_order}")
            
            for preferred_keyword in priority_order:
                for series in series_list:
                    # Check both protocol name and series description
                    if (preferred_keyword in series.protocol_name.lower() or 
                        preferred_keyword in series.series_desc.lower()):
                        logger.info(f"    Selected series by preference '{preferred_keyword}' for {modality}: "
                                   f"{series.uid} (Protocol: {series.protocol_name})")
                        return [series]
            
            logger.debug(f"    No series matched preference order for {modality}")
        
        # If no preference match found, select by newest acquisition time
        logger.debug(f"    Falling back to newest acquisition time for {modality}")
        
        # Extract acquisition time for each series
        series_with_times = []
        for series in series_list:
            # Get acquisition datetime from the first dataset
            ds = series.first_dataset
            acq_date = get_dicom_value(ds, (0x0008, 0x0022), "")  # AcquisitionDate
            acq_time = get_dicom_value(ds, (0x0008, 0x0032), "").split('.')[0]  # AcquisitionTime
            
            # If no acquisition date/time, fall back to series date/time
            if not acq_date or acq_date == "":
                acq_date = get_dicom_value(ds, (0x0008, 0x0021), "")  # SeriesDate
                acq_time = get_dicom_value(ds, (0x0008, 0x0031), "").split('.')[0]  # SeriesTime
            
            # If still no date/time, use study date/time
            if not acq_date or acq_date == "":
                acq_date = get_dicom_value(ds, (0x0008, 0x0020), "00000000")  # StudyDate
                acq_time = get_dicom_value(ds, (0x0008, 0x0030), "000000").split('.')[0]  # StudyTime
            
            # Parse datetime
            try:
                if len(acq_date) >= 8 and len(acq_time) >= 6:
                    acq_datetime = datetime.strptime(f"{acq_date[:8]}{acq_time[:6]}", "%Y%m%d%H%M%S")
                else:
                    # Fallback to study datetime if parsing fails
                    acq_datetime = series.study_datetime
            except ValueError:
                logger.warning(f"      Could not parse acquisition time for series {series.uid}, using study time")
                acq_datetime = series.study_datetime
            
            series_with_times.append((series, acq_datetime))
            logger.debug(f"      Series {series.uid}: {acq_datetime}, Protocol: {series.protocol_name}")
        
        # Sort by acquisition time (newest first)
        series_with_times.sort(key=lambda x: x[1], reverse=True)
        
        # Select the newest
        selected_series = series_with_times[0][0]
        logger.info(f"    Selected newest series for {modality}: {selected_series.uid} "
                   f"(acquired at {series_with_times[0][1]})")
        
        return [selected_series]
    
    def _copy_series_files(self, series_info: SeriesInfo, modality: str, bids_modality_dir: str,
                          bids_sub_id: str, bids_ses_id: str):
        """Copy files for a single series."""
        sorted_files = self._sort_files_by_instance_number(series_info.files)
        
        logger.info(f"    Copying {len(sorted_files)} files for {modality} (Series UID: {series_info.uid})")
        
        for slice_idx, src_file_path in enumerate(sorted_files, 1):
            # Generate BIDS filename - no run number since we only keep the newest
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
                ds_slice = pydicom.dcmread(f_path, stop_before_pixels=True, specific_tags=[(0x0020,0x0013)])
                instance_number = get_dicom_value(ds_slice, (0x0020,0x0013))
                if instance_number is not None:
                    try:
                        instance_number = int(instance_number)
                    except ValueError:
                        logger.warning(f"Cannot convert InstanceNumber '{instance_number}' to int for {f_path}. Using filename for sorting.")
                        instance_number = f_path
                else:
                    logger.warning(f"InstanceNumber missing in {f_path}. Using filename for sorting.")
                    instance_number = f_path
                sorted_files.append((instance_number, f_path))
            except Exception as e:
                logger.error(f"Error reading InstanceNumber from {f_path}: {e}. File will be sorted at the end.")
                sorted_files.append((float('inf'), f_path))

        # Sort: first by numeric InstanceNumber, then by file path
        sorted_files.sort(key=lambda x: (isinstance(x[0], str), x[0]))
        return [f_path for _, f_path in sorted_files]


# --- Strategy Factory ---
class StrategyFactory:
    """Factory for creating detection strategies based on configuration."""
    
    @staticmethod
    def create_strategies_from_config(config_path: Optional[str] = None) -> List[ModalityDetectionStrategy]:
        """Create strategies from configuration file or return defaults."""
        strategies = [
            Legacy2021_2022DetectionStrategy(),
            CustomProtocol2023DetectionStrategy(),
            StandardDetectionStrategy()
        ]
        
        # If config file is provided, load additional strategies
        if config_path and os.path.exists(config_path):
            # Here you could load JSON/YAML config and create strategies dynamically
            pass
        
        return strategies


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
  %(prog)s /path/to/dicom/input /path/to/bids/output --strategy-config protocols.json
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
        '--strategy-config',
        type=str,
        help='Path to JSON/YAML configuration file for detection strategies'
    )
    
    parser.add_argument(
        '--list-strategies',
        action='store_true',
        help='List all available detection strategies and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='DICOM to BIDS Converter v2.0.0 (Extended Protocol Support)'
    )
    
    args = parser.parse_args()
    
    # Setup logging early
    try:
        setup_logging(args.log_file)
        if args.verbose:
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
    
    # Handle --list-strategies
    if args.list_strategies:
        detector = EnhancedModalityDetector()
        print("\nAvailable Detection Strategies:")
        print("-" * 50)
        for strategy in detector.strategies:
            print(f"- {strategy.get_name()} (priority: {strategy.get_priority()})")
        print("\nStrategies are tried in order of priority (lower number = higher priority)")
        sys.exit(0)
    
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
    
    # Log startup information
    logger.info("="*60)
    logger.info("DICOM to BIDS Converter Started (v2.0.0)")
    logger.info("="*60)
    logger.info(f"Input directory: {os.path.abspath(args.input_dir)}")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Log file: {os.path.abspath(args.log_file)}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # List active strategies
    detector = EnhancedModalityDetector()
    logger.info("Active detection strategies:")
    for strategy in detector.strategies:
        logger.info(f"  - {strategy.get_name()} (priority: {strategy.get_priority()})")
    
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
            
            # Show what would be processed with strategy info
            for patient_id, patient_data in collected_data.items():
                logger.info(f"Patient: {patient_id}")
                for study_uid, study_info in patient_data.studies.items():
                    logger.info(f"  Study: {study_uid} ({study_info.study_datetime})")
                    
                    # Group series by modality to show what would be selected
                    temp_organizer = BidsOrganizer(args.output_dir, args.action)
                    modality_groups = temp_organizer._group_series_by_modality(study_info.series)
                    
                    # Show required modalities
                    required_modalities = ['t1', 't1c', 't2fl', 't2']
                    logger.info(f"    Required modalities: {', '.join(required_modalities)}")
                    
                    for modality in required_modalities:
                        if modality in modality_groups:
                            series_list = modality_groups[modality]
                            if len(series_list) > 1:
                                logger.info(f"    {modality}: {len(series_list)} series found")
                                # Show which would be selected
                                selected = temp_organizer._apply_priority_selection(series_list, modality)
                                logger.info(f"      → Would select: {selected[0].uid} ({len(selected[0].files)} files)")
                            else:
                                logger.info(f"    {modality}: 1 series found - {series_list[0].uid} ({len(series_list[0].files)} files)")
                        else:
                            logger.info(f"    {modality}: NOT FOUND")
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