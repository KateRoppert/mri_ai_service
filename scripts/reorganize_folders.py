import os
import shutil
import pydicom
import pydicom.dataelem
import argparse
import logging
import sys
import json
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


@dataclass
class ModalitySelectionLog:
    """Enhanced logging for modality selection decisions."""
    session_id: str
    modality: str
    selected_protocol: str
    selection_reason: str
    strategy_used: str
    candidates_considered: List[str]
    forbidden_filtered: List[str]
    priority_scores: Dict[str, float]
    year_detected: Optional[int] = None


# --- Enhanced Protocol Detection Strategy Pattern ---
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
    
    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        """Return scoring weights for priority selection."""
        return {}
    
    def is_exclusive(self) -> bool:
        """Return True if this strategy should prevent fallback to other strategies when applicable."""
        return False


class EnhancedStandardDetectionStrategy(ModalityDetectionStrategy):
    """Enhanced standard modality detection strategy based on CSV analysis."""
    
    def __init__(self):
        # Enhanced dictionaries based on CSV analysis
        self.modality_config = {
            't1': {
                'keywords': ['t1', 't1w'],
                'forbidden': ['thr', 'mpr', 'ce', 'pit', 'contrast', 'gad'],
                'priority_sequences': ['tfe', 'tse', 'se'],
                'priority_modifiers': ['3d', 'clear', 'brain'],
                'scoring_weights': {
                    'tfe': 3.0,
                    'tse': 2.0, 
                    'se': 1.0,
                    '3d': 1.5,
                    'clear': 1.2,
                    'brain': 1.1
                }
            },
            't1c': {
                'keywords': ['t1', 'ce'],  # Both must be present
                'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast', '+c', 'post'],
                'forbidden': ['mpr', 'dyn', 'pit', 'spir'],
                'priority_sequences': ['tfe', 'tse', 'se'],
                'priority_modifiers': ['3d', 'brain'],
                'scoring_weights': {
                    'tfe': 3.0,
                    'tse': 2.0,
                    'se': 1.0,
                    '3d': 1.5,
                    'brain': 1.1
                }
            },
            't2': {
                'keywords': ['t2', 't2w'],
                'forbidden': ['ce', 'pit', 'mpr', 'contrast', 'flair'],
                'priority_sequences': ['tse'],
                'priority_modifiers': ['sense', 'brain', 'axi'],
                'scoring_weights': {
                    'tse': 2.0,
                    'sense': 1.3,
                    'brain': 1.1,
                    'axi': 1.0
                }
            },
            't2fl': {
                'keywords': ['flair'],
                'forbidden': ['mpr', 'ce', 'spir', 'contrast'],
                'priority_sequences': [],
                'priority_modifiers': ['3d', 'sense', 'long', 'brain'],
                'scoring_weights': {
                    '3d': 2.0,
                    'sense': 1.3,
                    'long': 1.2,
                    'brain': 1.1
                }
            }
        }
    
    def get_name(self) -> str:
        return "Enhanced Standard Detection"
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Enhanced standard detection is always applicable as fallback."""
        return True
    
    def get_priority(self) -> int:
        return 1000  # Lowest priority (highest number)
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        config = self.modality_config.get(modality, {})
        return config.get('priority_sequences', []) + config.get('priority_modifiers', [])
    
    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        """Return scoring weights for priority selection."""
        config = self.modality_config.get(modality, {})
        return config.get('scoring_weights', {})
    
    def is_exclusive(self) -> bool:
        """Enhanced standard detection is never exclusive - it's the fallback."""
        return False
    
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using enhanced workflow."""
        logger = logging.getLogger(__name__)
        
        # Try different detection methods in order
        detectors = [
            ("Enhanced Protocol Analysis", self._detect_by_enhanced_protocol),
            ("Series Description Analysis", self._detect_by_enhanced_series_description),
            ("Technical Parameters", self._detect_by_technical_params),
            ("File Path Analysis", lambda ds, fp: self._detect_by_file_path(fp))
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
    
    def _detect_by_enhanced_protocol(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Enhanced protocol detection with improved keyword matching."""
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        
        has_contrast = self._check_contrast_presence(ds, protocol_name, contrast_agent)
        
        logger = logging.getLogger(__name__)
        logger.debug(f"    Protocol: '{protocol_name}', Contrast: {has_contrast}")
        
        return self._analyze_text_for_modality(protocol_name, has_contrast)
    
    def _detect_by_enhanced_series_description(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Enhanced series description detection."""
        series_desc = normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        
        has_contrast = self._check_contrast_presence(ds, series_desc, contrast_agent)
        
        logger = logging.getLogger(__name__)
        logger.debug(f"    Series Desc: '{series_desc}', Contrast: {has_contrast}")
        
        return self._analyze_text_for_modality(series_desc, has_contrast)
    
    def _check_contrast_presence(self, ds: pydicom.Dataset, text: str, contrast_agent: str) -> bool:
        """Enhanced contrast detection."""
        # Check contrast agent field
        if contrast_agent and contrast_agent not in ["", "none", "no"]:
            return True
        
        # Check for contrast keywords in text
        contrast_keywords = ['ce', 'contrast', 'gad', 'gadolinium', 'post', '+c', 'enhanced']
        if any(keyword in text.lower() for keyword in contrast_keywords):
            return True
        
        # Check additional DICOM fields
        contrast_bolusagent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1078), ""))
        if contrast_bolusagent and contrast_bolusagent not in ["", "none", "no"]:
            return True
        
        return False
    
    def _analyze_text_for_modality(self, text: str, has_contrast: bool) -> Optional[str]:
        """Analyze text for modality with enhanced logic."""
        if not text:
            return None
        
        logger = logging.getLogger(__name__)
        
        # First check forbidden words for each modality
        candidates = []
        for modality, config in self.modality_config.items():
            # Check forbidden words
            if any(forbidden in text.lower() for forbidden in config.get('forbidden', [])):
                logger.debug(f"      {modality} rejected - forbidden word found")
                continue
            
            # Check required keywords
            if self._check_keywords_match(text, modality, config):
                # Special handling for contrast-dependent modalities
                if modality == 't1c':
                    if has_contrast:
                        candidates.append(modality)
                        logger.debug(f"      {modality} candidate - keywords + contrast")
                elif modality in ['t1', 't2', 't2fl']:
                    # For non-contrast modalities, prefer if no contrast detected
                    if not has_contrast or modality == 't2fl':  # FLAIR can be post-contrast sometimes
                        candidates.append(modality)
                        logger.debug(f"      {modality} candidate - keywords, no contrast conflict")
        
        # If multiple candidates, prioritize based on specificity
        if len(candidates) > 1:
            # FLAIR is most specific
            if 't2fl' in candidates:
                return 't2fl'
            # T1C is more specific than T1
            if 't1c' in candidates and has_contrast:
                return 't1c'
            if 't1' in candidates and not has_contrast:
                return 't1'
            # Return first candidate
            return candidates[0]
        elif len(candidates) == 1:
            return candidates[0]
        
        return None
    
    def _check_keywords_match(self, text: str, modality: str, config: Dict) -> bool:
        """Check if text matches keywords for a modality."""
        text_lower = text.lower()
        
        # Primary keywords (all must be present for t1c)
        keywords = config.get('keywords', [])
        if modality == 't1c':
            # For T1C, require both 't1' and 'ce' (or check alternative keywords)
            primary_match = all(keyword in text_lower for keyword in keywords)
            
            # Check alternative contrast keywords if primary doesn't match
            if not primary_match:
                alt_keywords = config.get('alt_keywords', [])
                if alt_keywords:
                    # Need 't1' plus any contrast indicator
                    has_t1 = 't1' in text_lower
                    has_contrast_kw = any(kw in text_lower for kw in alt_keywords[1:])  # Skip 't1'
                    primary_match = has_t1 and has_contrast_kw
            
            return primary_match
        else:
            # For other modalities, any keyword match is sufficient
            return any(keyword in text_lower for keyword in keywords)
    
    def _detect_by_technical_params(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Enhanced technical parameter detection."""
        tr_val = safe_float(get_dicom_value(ds, (0x0018, 0x0080)), "TR")
        te_val = safe_float(get_dicom_value(ds, (0x0018, 0x0081)), "TE")
        ti_val = safe_float(get_dicom_value(ds, (0x0018, 0x0082)), "TI")
        
        contrast_agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        has_contrast = self._check_contrast_presence(ds, "", contrast_agent)
        
        logger = logging.getLogger(__name__)
        logger.debug(f"    TR={tr_val}, TE={te_val}, TI={ti_val}, Contrast={has_contrast}")
        
        # FLAIR detection by TI (most specific)
        if ti_val and ti_val > 1500:
            logger.debug(f"    Technical params suggest FLAIR (TI={ti_val})")
            return 't2fl'
        
        # T1C detection by contrast + T1 parameters
        if has_contrast and tr_val and te_val and tr_val < 1200 and te_val < 30:
            logger.debug(f"    Technical params suggest T1C (contrast + TR/TE)")
            return 't1c'
        
        # T1 detection by TR/TE without contrast
        if tr_val and te_val and tr_val < 1000 and te_val < 30 and not has_contrast:
            logger.debug(f"    Technical params suggest T1 (TR/TE, no contrast)")
            return 't1'
        
        # T2 detection by TR/TE
        if tr_val and te_val and tr_val > 2000 and te_val > 70:
            if not ti_val or ti_val < 1500:  # Make sure it's not FLAIR
                logger.debug(f"    Technical params suggest T2 (TR/TE, not FLAIR)")
                return 't2'
        
        return None
    
    def _detect_by_file_path(self, file_path: str) -> Optional[str]:
        """Enhanced file path analysis."""
        path_parts = os.path.normpath(file_path).split(os.sep)
        for part in reversed(path_parts):
            part_lower = part.lower()
            
            modality = self._analyze_text_for_modality(part_lower, 'contrast' in part_lower)
            if modality:
                return modality
        
        return None


class YearSpecificDetectionStrategy(ModalityDetectionStrategy):
    """Year-specific detection strategy based on CSV analysis."""
    
    def __init__(self, target_years: List[int], strategy_name: str, keywords_config: Dict):
        self.target_years = target_years
        self.strategy_name = strategy_name
        self.keywords_config = keywords_config
    
    def get_name(self) -> str:
        years_str = ", ".join(map(str, self.target_years))
        return f"{self.strategy_name} ({years_str})"
    
    def get_priority(self) -> int:
        return 10  # Higher priority than standard
    
    def get_priority_order(self, modality: str) -> List[str]:
        """Return the preference order for selecting series within a modality."""
        config = self.keywords_config.get(modality, {})
        return config.get('prefer_order', [])
    
    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        """Return scoring weights for priority selection."""
        config = self.keywords_config.get(modality, {})
        return config.get('scoring_weights', {})
    
    def is_exclusive(self) -> bool:
        """Year-specific protocols are exclusive - if year matches, don't fall back."""
        return True
    
    def is_applicable(self, ds: pydicom.Dataset, file_path: str) -> bool:
        """Check if this is a study from target years."""
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                year = int(study_date_val[:4])
                return year in self.target_years
            except ValueError:
                pass
        return False
    
    def detect_modality(self, ds: pydicom.Dataset, file_path: str) -> Optional[str]:
        """Detect modality using year-specific rules."""
        logger = logging.getLogger(__name__)
        protocol_name = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1030), ""))
        series_desc = normalize_dicom_text(get_dicom_value(ds, (0x0008, 0x103E), ""))
        
        # Combine protocol name and series description for analysis
        combined_text = f"{protocol_name} {series_desc}".strip()
        
        logger.debug(f"  Year-specific analysis: '{combined_text}'")
        
        for modality, rules in self.keywords_config.items():
            if self._matches_year_specific_rules(combined_text, rules):
                logger.debug(f"  ✓ Matched modality '{modality}' with year-specific rules")
                return modality
        
        logger.debug("  ✗ No match found with year-specific rules")
        return None
    
    def _matches_year_specific_rules(self, text: str, rules: Dict) -> bool:
        """Check if text matches year-specific rules."""
        text_lower = text.lower()
        
        # Check forbidden keywords first
        forbidden = rules.get('forbidden', [])
        if any(fw in text_lower for fw in forbidden):
            return False
        
        # Check required keywords
        required = rules.get('required', [])
        if required and not all(rw in text_lower for rw in required):
            return False
        
        # Check marker keywords
        markers = rules.get('markers', [])
        if markers and not any(mk in text_lower for mk in markers):
            return False
        
        return True


# --- Enhanced Modality Detector ---
class EnhancedModalityDetector:
    """Enhanced modality detector with comprehensive year-specific strategies."""
    
    def __init__(self):
        # Initialize all available strategies
        self.strategies: List[ModalityDetectionStrategy] = []
        
        # Add year-specific strategies based on CSV analysis
        self._add_year_specific_strategies()
        
        # Add enhanced standard strategy as fallback
        self.strategies.append(EnhancedStandardDetectionStrategy())
        
        # Sort by priority (lower number = higher priority)
        self.strategies.sort(key=lambda s: s.get_priority())
        self.logger = logging.getLogger(__name__)
        
        # Initialize selection logging
        self.selection_logs: List[ModalitySelectionLog] = []
    
    def _add_year_specific_strategies(self):
        """Add year-specific strategies based on CSV analysis."""
        
        # 2018 Strategy
        strategy_2018 = YearSpecificDetectionStrategy(
            target_years=[2018],
            strategy_name="Protocol 2018",
            keywords_config={
                't1': {
                    'markers': ['t1w'],
                    'forbidden': ['thr', 'ce'],
                    'prefer_order': ['tse', 'se'],
                    'scoring_weights': {'tse': 2.0, 'se': 1.0}
                },
                't1c': {
                    'required': ['ce', 't1w'],
                    'forbidden': [],
                    'prefer_order': ['se'],
                    'scoring_weights': {'se': 2.0}
                },
                't2': {
                    'markers': ['t2w'],
                    'forbidden': [],
                    'prefer_order': ['tse'],
                    'scoring_weights': {'tse': 2.0}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': [],
                    'prefer_order': ['sense', 'long'],
                    'scoring_weights': {'sense': 1.5, 'long': 1.2}
                }
            }
        )
        
        # 2020 Strategy
        strategy_2020 = YearSpecificDetectionStrategy(
            target_years=[2020],
            strategy_name="Protocol 2020",
            keywords_config={
                't1': {
                    'markers': ['t1w', 't1-tse'],
                    'forbidden': ['thr', 'mpr', 'ce'],
                    'prefer_order': ['tse', 'clear', '3d', 'se'],
                    'scoring_weights': {'tse': 2.0, 'clear': 1.5, '3d': 1.3, 'se': 1.0}
                },
                't1c': {
                    'required': ['ce', 't1'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['tse', '3d', 'se'],
                    'scoring_weights': {'tse': 2.0, '3d': 1.5, 'se': 1.0}
                },
                't2': {
                    'markers': ['t2w', 't2-tse'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['tse', 'sense'],
                    'scoring_weights': {'tse': 2.0, 'sense': 1.3}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['3d', 'long'],
                    'scoring_weights': {'3d': 2.0, 'long': 1.2}
                }
            }
        )
        
        # 2021-2022 Strategy
        strategy_2021_2022 = YearSpecificDetectionStrategy(
            target_years=[2021, 2022],
            strategy_name="Protocol 2021-2022",
            keywords_config={
                't1': {
                    'markers': ['t1-tfe', 't1-tse'],
                    'forbidden': ['mpr', 'ce'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },
                't1c': {
                    'required': ['ce', 't1'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },
                't2': {
                    'markers': ['t2-tse'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['tse', 'axi'],
                    'scoring_weights': {'tse': 2.0, 'axi': 1.2}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': ['mpr'],
                    'prefer_order': ['3d'],
                    'scoring_weights': {'3d': 2.0}
                }
            }
        )
        
        # 2023+ Strategy
        strategy_2023_plus = YearSpecificDetectionStrategy(
            target_years=[2023, 2024, 2025],
            strategy_name="Protocol 2023+",
            keywords_config={
                't1': {
                    'markers': ['t1-tfe', 't1-tse'],
                    'forbidden': ['mpr', 'ce'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },
                't1c': {
                    'required': ['ce', 't1'],
                    'forbidden': ['mpr', 'dyn', 'pit'],
                    'prefer_order': ['tfe', 'tse', '3d'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}
                },
                't2': {
                    'markers': ['t2-tse'],
                    'forbidden': ['ce', 'pit', 'mpr'],
                    'prefer_order': ['tse', 'axi'],
                    'scoring_weights': {'tse': 2.0, 'axi': 1.2}
                },
                't2fl': {
                    'markers': ['flair'],
                    'forbidden': ['mpr', 'ce'],
                    'prefer_order': ['3d'],
                    'scoring_weights': {'3d': 2.0}
                }
            }
        )
        
        self.strategies.extend([
            strategy_2018,
            strategy_2020,
            strategy_2021_2022,
            strategy_2023_plus
        ])
    
    def add_strategy(self, strategy: ModalityDetectionStrategy):
        """Add a new detection strategy."""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.get_priority())
        self.logger.info(f"Added detection strategy: {strategy.get_name()}")
    
    def determine_modality_with_details(self, ds: pydicom.Dataset, file_path: str, session_id: str = "") -> Tuple[str, Optional[ModalityDetectionStrategy], Dict]:
        """Determine modality with detailed logging information."""
        self.logger.debug(f"Determining modality for {os.path.basename(file_path)}:")
        
        details = {
            'candidates_considered': [],
            'forbidden_filtered': [],
            'strategy_used': None,
            'year_detected': None
        }
        
        # Extract year for logging
        study_date_val = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date_val, str) and len(study_date_val) >= 4:
            try:
                details['year_detected'] = int(study_date_val[:4])
            except ValueError:
                pass
        
        # Try each strategy in priority order
        for strategy in self.strategies:
            strategy_name = strategy.get_name()
            
            # Check if strategy is applicable
            if strategy.is_applicable(ds, file_path):
                self.logger.debug(f"→ Trying strategy: {strategy_name}")
                details['strategy_used'] = strategy_name
                
                # Try to detect modality
                modality = strategy.detect_modality(ds, file_path)
                
                if modality and modality != 'unknown':
                    self.logger.info(f"✓ Modality '{modality}' detected using strategy: {strategy_name}")
                    return modality, strategy, details
                else:
                    # If this is an exclusive strategy and it didn't match, skip the file entirely
                    if strategy.is_exclusive():
                        self.logger.warning(f"  Exclusive strategy {strategy_name} did not match - skipping file")
                        return 'unknown', None, details
                    else:
                        self.logger.debug(f"  Strategy {strategy_name} did not detect modality")
            else:
                self.logger.debug(f"  Strategy {strategy_name} is not applicable")
        
        self.logger.warning(f"Could not determine modality for file: {os.path.basename(file_path)}")
        return 'unknown', None, details
    
    def determine_modality(self, ds: pydicom.Dataset, file_path: str) -> str:
        """Determine modality using registered strategies."""
        modality, _, _ = self.determine_modality_with_details(ds, file_path)
        return modality


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


# --- Enhanced BIDS Organizer Class ---
class BidsOrganizer:
    """Enhanced BIDS organizer with proper naming conventions and missing modality handling."""
    
    def __init__(self, output_dir: str, action_type: str = 'copy'):
        self.output_dir = output_dir
        self.action_type = action_type
        self.detector = EnhancedModalityDetector()
        self.selection_log = []
        
        # BIDS modality mapping
        self.bids_modality_map = {
            't1': 'T1w',
            't1c': 'T1w-Gd',  # BIDS extension for post-contrast T1
            't2': 'T2w', 
            't2fl': 'FLAIR'
        }
    
    def organize_to_bids(self, collected_data: Dict[str, PatientData]):
        """Organize collected data into BIDS structure with enhanced logging."""
        logger.info("Phase 2: Creating BIDS structure and organizing files...")
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create output directory {self.output_dir}: {e}")
            raise
        
        # Create BIDS patient IDs
        patient_bids_map = self._create_patient_bids_mapping(collected_data)
        
        # Process each patient
        for patient_data in collected_data.values():
            self._process_patient(patient_data, patient_bids_map)
        
        # Generate selection summary
        self._generate_selection_summary()
        
        logger.info("Enhanced BIDS organization completed!")
    
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
        """Process a single study/session with enhanced modality detection."""
        bids_ses_id = session_bids_map[study_info.uid]
        session_id = f"{bids_sub_id}_{bids_ses_id}"
        logger.info(f"  Processing session: {study_info.uid} -> {bids_ses_id}")
        
        # Group series by modality with enhanced detection
        modality_groups, detection_details = self._group_series_by_modality_enhanced(study_info.series, session_id)
        
        # Required modalities
        required_modalities = ['t1', 't1c', 't2', 't2fl']
        found_modalities = list(modality_groups.keys())
        missing_modalities = [m for m in required_modalities if m not in found_modalities]
        
        logger.info(f"    Found modalities: {', '.join(found_modalities) if found_modalities else 'None'}")
        if missing_modalities:
            logger.warning(f"    Missing modalities: {', '.join(missing_modalities)}")
        
        # Only create directories and process found modalities
        if found_modalities:
            bids_anat_path = os.path.join(self.output_dir, bids_sub_id, bids_ses_id, 'anat')
            
            # Process each found modality
            for modality in found_modalities:
                series_with_strategies = modality_groups[modality]
                self._process_modality_group_enhanced(series_with_strategies, modality, bids_anat_path, 
                                                   bids_sub_id, bids_ses_id, session_id)
        else:
            logger.warning(f"    No valid modalities found for session {bids_ses_id} - skipping directory creation")
        
        # Log session summary
        logger.info(f"    Session {bids_ses_id} processing complete. "
                   f"Processed {len(found_modalities)}/{len(required_modalities)} modalities.")
    
    def _group_series_by_modality_enhanced(self, series_dict: Dict[str, SeriesInfo], session_id: str) -> Tuple[Dict, Dict]:
        """Enhanced grouping with detailed logging."""
        modality_groups = defaultdict(list)
        detection_details = {}
        
        for series_info in series_dict.values():
            modality, strategy, details = self.detector.determine_modality_with_details(
                series_info.first_dataset, series_info.files[0], session_id
            )
            
            detection_details[series_info.uid] = {
                'modality': modality,
                'strategy': strategy.get_name() if strategy else None,
                'details': details
            }
            
            if modality == 'unknown':
                logger.warning(f"    Skipping series {series_info.uid}: unknown modality.")
                logger.debug(f"      Protocol: '{series_info.protocol_name}'")
                logger.debug(f"      Series Desc: '{series_info.series_desc}'")
                continue
            
            modality_groups[modality].append((series_info, strategy))
            logger.debug(f"    Series {series_info.uid} -> {modality} (strategy: {strategy.get_name() if strategy else 'None'})")
        
        return dict(modality_groups), detection_details
    
    def _process_modality_group_enhanced(self, series_with_strategies: List[Tuple[SeriesInfo, ModalityDetectionStrategy]], 
                                       modality: str, bids_anat_path: str, bids_sub_id: str, bids_ses_id: str, session_id: str):
        """Enhanced modality group processing with priority scoring."""
        logger.debug(f"    Processing {len(series_with_strategies)} series for modality: {modality}")
        
        if len(series_with_strategies) == 1:
            # Only one series - use it directly
            series_info, strategy = series_with_strategies[0]
            selected_series = [series_info]
            selection_reason = "only_candidate"
            logger.info(f"    Selected only series for {modality}: {series_info.uid}")
        else:
            # Multiple series - apply enhanced selection
            selected_series, selection_reason, scoring_details = self._apply_enhanced_priority_selection(
                series_with_strategies, modality, session_id
            )
        
        # Create modality directory
        bids_modality_dir = os.path.join(bids_anat_path, modality)
        try:
            os.makedirs(bids_modality_dir, exist_ok=True)
            logger.debug(f"    Created directory: {bids_modality_dir}")
        except OSError as e:
            logger.error(f"    Cannot create directory {bids_modality_dir}: {e}")
            return
        
        # Process selected series
        for series_info in selected_series:
            # Log selection decision
            # Find the strategy used for this series
            strategy_used = None
            for series, strategy in series_with_strategies:
                if series.uid == series_info.uid:
                    strategy_used = strategy.get_name() if strategy else "Unknown"
                    break
            
            selection_log_entry = ModalitySelectionLog(
                session_id=session_id,
                modality=modality,
                selected_protocol=series_info.protocol_name,
                selection_reason=selection_reason,
                strategy_used=strategy_used or "Unknown",
                candidates_considered=[series.protocol_name for series, _ in series_with_strategies],
                forbidden_filtered=[],  # Would need to track this in detection
                priority_scores=getattr(self, '_last_scoring_details', {}),
                year_detected=None  # Would extract from study date
            )
            self.selection_log.append(selection_log_entry)
            
            self._copy_series_files_enhanced(series_info, modality, bids_modality_dir, 
                                           bids_sub_id, bids_ses_id)
    
    def _apply_enhanced_priority_selection(self, series_with_strategies: List[Tuple[SeriesInfo, ModalityDetectionStrategy]], 
                                         modality: str, session_id: str) -> Tuple[List[SeriesInfo], str, Dict]:
        """Enhanced priority selection with scoring system."""
        logger.debug(f"    Found {len(series_with_strategies)} series for {modality}, applying enhanced selection")
        
        # Extract series and strategies
        series_list = [series for series, _ in series_with_strategies]
        strategies = [strategy for _, strategy in series_with_strategies if strategy]
        
        # Collect scoring weights from all applicable strategies
        all_scoring_weights = {}
        for strategy in strategies:
            weights = strategy.get_scoring_weights(modality)
            all_scoring_weights.update(weights)
        
        # Calculate scores for each series
        series_scores = {}
        scoring_details = {}
        
        for series_info in series_list:
            score = self._calculate_series_score(series_info, modality, all_scoring_weights)
            series_scores[series_info.uid] = score
            scoring_details[series_info.uid] = {
                'protocol': series_info.protocol_name,
                'series_desc': series_info.series_desc,
                'score': score
            }
            logger.debug(f"      Series {series_info.uid}: score={score:.2f} (Protocol: '{series_info.protocol_name}')")
        
        # Store for logging
        self._last_scoring_details = scoring_details
        
        # Select highest scoring series
        if series_scores:
            best_series_uid = max(series_scores.keys(), key=lambda uid: series_scores[uid])
            best_series = next(s for s in series_list if s.uid == best_series_uid)
            
            logger.info(f"    Selected series by scoring for {modality}: {best_series_uid} "
                       f"(score: {series_scores[best_series_uid]:.2f})")
            return [best_series], "priority_scoring", scoring_details
        
        # Fallback to newest if scoring fails
        logger.debug(f"    Falling back to newest acquisition for {modality}")
        newest_series = self._select_newest_series(series_list)
        return [newest_series], "newest_acquisition", {}
    
    def _calculate_series_score(self, series_info: SeriesInfo, modality: str, scoring_weights: Dict[str, float]) -> float:
        """Calculate priority score for a series."""
        score = 1.0  # Base score
        
        # Combine protocol name and series description for analysis
        combined_text = f"{series_info.protocol_name} {series_info.series_desc}".lower()
        
        # Apply scoring weights
        for keyword, weight in scoring_weights.items():
            if keyword in combined_text:
                score *= weight
                logger.debug(f"        Applied weight {weight} for keyword '{keyword}'")
        
        # Bonus for brain-specific sequences
        if 'brain' in combined_text:
            score *= 1.1
            logger.debug(f"        Applied brain bonus: 1.1")
        
        # Penalty for spine/other anatomy
        anatomy_penalties = {'spine': 0.8, 'cervical': 0.8, 'pit': 0.7, 'pituitary': 0.7}
        for anatomy, penalty in anatomy_penalties.items():
            if anatomy in combined_text:
                score *= penalty
                logger.debug(f"        Applied {anatomy} penalty: {penalty}")
        
        return score
    
    def _select_newest_series(self, series_list: List[SeriesInfo]) -> SeriesInfo:
        """Select the newest series by acquisition time."""
        series_with_times = []
        
        for series in series_list:
            # Get acquisition datetime from the first dataset
            ds = series.first_dataset
            acq_date = get_dicom_value(ds, (0x0008, 0x0022), "")  # AcquisitionDate
            acq_time = get_dicom_value(ds, (0x0008, 0x0032), "").split('.')[0]  # AcquisitionTime
            
            # Fallback hierarchy for datetime
            if not acq_date:
                acq_date = get_dicom_value(ds, (0x0008, 0x0021), "")  # SeriesDate
                acq_time = get_dicom_value(ds, (0x0008, 0x0031), "").split('.')[0]  # SeriesTime
            
            if not acq_date:
                acq_date = get_dicom_value(ds, (0x0008, 0x0020), "00000000")  # StudyDate
                acq_time = get_dicom_value(ds, (0x0008, 0x0030), "000000").split('.')[0]  # StudyTime
            
            # Parse datetime
            try:
                if len(acq_date) >= 8 and len(acq_time) >= 6:
                    acq_datetime = datetime.strptime(f"{acq_date[:8]}{acq_time[:6]}", "%Y%m%d%H%M%S")
                else:
                    acq_datetime = series.study_datetime
            except ValueError:
                acq_datetime = series.study_datetime
            
            series_with_times.append((series, acq_datetime))
        
        # Sort by acquisition time (newest first) and select the first
        series_with_times.sort(key=lambda x: x[1], reverse=True)
        selected_series = series_with_times[0][0]
        
        logger.info(f"    Selected newest series: {selected_series.uid} "
                   f"(acquired at {series_with_times[0][1]})")
        
        return selected_series
    
    def _copy_series_files_enhanced(self, series_info: SeriesInfo, modality: str, bids_modality_dir: str,
                                  bids_sub_id: str, bids_ses_id: str):
        """Copy files with proper BIDS naming conventions."""
        sorted_files = self._sort_files_by_instance_number(series_info.files)
        
        # BIDS suffix mapping
        bids_suffix = self.bids_modality_map.get(modality, modality)
        
        logger.info(f"    Copying {len(sorted_files)} files for {modality} -> {bids_suffix} "
                   f"(Series UID: {series_info.uid})")
        
        for slice_idx, src_file_path in enumerate(sorted_files, 1):
            # Proper BIDS naming: sub-<label>_ses-<label>_<suffix>_<instance>.dcm
            bids_filename = f"{bids_sub_id}_{bids_ses_id}_{bids_suffix}_instance-{slice_idx:03d}.dcm"
            
            dst_file_path = os.path.join(bids_modality_dir, bids_filename)
            
            # Copy or move file
            try:
                if self.action_type == 'move':
                    shutil.move(src_file_path, dst_file_path)
                else:
                    shutil.copy(src_file_path, dst_file_path)
                    
                if slice_idx <= 3:  # Log first few files
                    logger.debug(f"      {self.action_type.capitalize()}d: {os.path.basename(src_file_path)} -> {bids_filename}")
                    
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
    
    def _generate_selection_summary(self):
        """Generate and log selection summary."""
        if not self.selection_log:
            return
        
        logger.info("="*50)
        logger.info("MODALITY SELECTION SUMMARY")
        logger.info("="*50)
        
        # Group by session
        sessions = defaultdict(list)
        for log_entry in self.selection_log:
            sessions[log_entry.session_id].append(log_entry)
        
        for session_id, entries in sessions.items():
            logger.info(f"\nSession: {session_id}")
            for entry in entries:
                logger.info(f"  {entry.modality}: {entry.selected_protocol}")
                logger.info(f"    Strategy: {entry.strategy_used}")
                logger.info(f"    Reason: {entry.selection_reason}")
                if len(entry.candidates_considered) > 1:
                    logger.info(f"    Candidates: {len(entry.candidates_considered)} total")
        
        # Export detailed log to JSON
        log_file = os.path.join(self.output_dir, 'modality_selection_log.json')
        try:
            with open(log_file, 'w') as f:
                json.dump([{
                    'session_id': log.session_id,
                    'modality': log.modality,
                    'selected_protocol': log.selected_protocol,
                    'selection_reason': log.selection_reason,
                    'strategy_used': log.strategy_used,
                    'candidates_considered': log.candidates_considered,
                    'priority_scores': log.priority_scores
                } for log in self.selection_log], f, indent=2)
            logger.info(f"\nDetailed selection log saved to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save selection log: {e}")


# --- CLI Interface ---
def main():
    """Enhanced main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced DICOM to BIDS converter with intelligent modality detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/dicom/input /path/to/bids/output
  %(prog)s /path/to/dicom/input /path/to/bids/output --action move --log-file conversion.log
  %(prog)s /path/to/dicom/input /path/to/bids/output --verbose
  %(prog)s /path/to/dicom/input /path/to/bids/output --dry-run
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
        default='dicom_to_bids_enhanced.log',
        help='Path to log file (default: dicom_to_bids_enhanced.log)'
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
        '--list-strategies',
        action='store_true',
        help='List all available detection strategies and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Enhanced DICOM to BIDS Converter v3.0.0 (CSV-Analysis Based)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
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
        print("\nEnhanced Detection Strategies (CSV-Analysis Based):")
        print("-" * 60)
        for strategy in detector.strategies:
            print(f"- {strategy.get_name()} (priority: {strategy.get_priority()})")
            if hasattr(strategy, 'target_years'):
                print(f"  Target years: {strategy.target_years}")
        print("\nStrategies are tried in order of priority (lower number = higher priority)")
        print("Year-specific strategies are exclusive - they override standard detection")
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
    logger.info("="*70)
    logger.info("Enhanced DICOM to BIDS Converter Started (v3.0.0 - CSV Analysis Based)")
    logger.info("="*70)
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
            
            # Show what would be processed
            temp_organizer = BidsOrganizer(args.output_dir, args.action)
            required_modalities = ['t1', 't1c', 't2', 't2fl']
            
            for patient_id, patient_data in collected_data.items():
                logger.info(f"Patient: {patient_id}")
                
                for study_uid, study_info in patient_data.studies.items():
                    session_id = f"patient_{patient_id}_study_{study_uid[:8]}"
                    logger.info(f"  Study: {study_uid} ({study_info.study_datetime})")
                    
                    # Analyze what would be detected
                    modality_groups, detection_details = temp_organizer._group_series_by_modality_enhanced(
                        study_info.series, session_id
                    )
                    
                    found_modalities = list(modality_groups.keys())
                    missing_modalities = [m for m in required_modalities if m not in found_modalities]
                    
                    logger.info(f"    Would find: {', '.join(found_modalities) if found_modalities else 'None'}")
                    if missing_modalities:
                        logger.info(f"    Would be missing: {', '.join(missing_modalities)}")
                    
                    # Show selection details for found modalities
                    for modality in found_modalities:
                        series_list = [s for s, _ in modality_groups[modality]]
                        if len(series_list) > 1:
                            logger.info(f"    {modality}: {len(series_list)} candidates")
                            for series in series_list:
                                logger.info(f"      - {series.uid}: '{series.protocol_name}'")
                        else:
                            series = series_list[0]
                            logger.info(f"    {modality}: '{series.protocol_name}' ({len(series.files)} files)")
        else:
            # Phase 2: Organize to BIDS with enhanced processing
            organizer = BidsOrganizer(args.output_dir, args.action)
            organizer.organize_to_bids(collected_data)
        
        logger.info("="*70)
        logger.info("Enhanced DICOM to BIDS Conversion Completed Successfully")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user.")
        print("\nConversion interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.exception("Full exception details:")
        print(f"Error: Conversion failed. Check log file for details: {args.log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()