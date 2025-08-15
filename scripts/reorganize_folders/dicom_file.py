"""
Module for handling individual DICOM files with minimal overhead.
Focuses only on tags necessary for modality detection and series organization.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Union, Tuple
from functools import cached_property

import pydicom
from pydicom.dataset import FileDataset

logger = logging.getLogger(__name__)

# Patient/Study identification tags
PATIENT_ID_TAG = (0x0010, 0x0020)
PATIENT_NAME_TAG = (0x0010, 0x0010)
STUDY_INSTANCE_UID_TAG = (0x0020, 0x000D)
STUDY_ID_TAG = (0x0020, 0x0010)

# Series identification tags
SERIES_INSTANCE_UID_TAG = (0x0020, 0x000E)
SERIES_NUMBER_TAG = (0x0020, 0x0011)
INSTANCE_NUMBER_TAG = (0x0020, 0x0013)

# Modality detection tags (primary)
PROTOCOL_NAME_TAG = (0x0018, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)

# Modality detection tags (secondary - MR parameters)
ECHO_TIME_TAG = (0x0018, 0x0081)           # TE
REPETITION_TIME_TAG = (0x0018, 0x0080)     # TR
INVERSION_TIME_TAG = (0x0018, 0x0082)      # TI

# Timing tags for series selection
ACQUISITION_DATE_TAG = (0x0008, 0x0022)
ACQUISITION_TIME_TAG = (0x0008, 0x0032)


class DicomFile:
    """
    Minimal DICOM file handler for pipeline processing.
    
    Features:
    - Lazy loading of DICOM dataset (without pixel data)
    - Access to essential tags only
    - Memory efficient - no pixel data, minimal caching
    """
    
    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize DICOM file handler.
        
        Args:
            filepath: Path to DICOM file
        """
        self.filepath = Path(filepath)
        self._dataset: Optional[FileDataset] = None
        self._tag_cache: dict[Tuple[int, int], Any] = {}
    
    @property
    def dataset(self) -> Optional[FileDataset]:
        """
        Load and cache DICOM dataset (without pixel data).
        
        Returns:
            DICOM dataset or None if file cannot be read
        """
        if self._dataset is None:
            try:
                # Load without pixel data - we never need them at this stage
                self._dataset = pydicom.dcmread(
                    self.filepath, 
                    stop_before_pixels=True,
                    force=False
                )
                logger.debug(f"Successfully loaded DICOM file: {self.filepath}")
            except Exception as e:
                logger.debug(f"Failed to read DICOM file {self.filepath}: {e}")
                return None
        return self._dataset
    
    def is_valid(self) -> bool:
        """
        Check if file is a valid DICOM file.
        
        Returns:
            True if file can be read as DICOM, False otherwise
        """
        return self.dataset is not None
    
    def get_tag(self, tag: Tuple[int, int], default: Any = None) -> Any:
        """
        Get DICOM tag value by numeric tag.
        
        Args:
            tag: Tag as (group, element) tuple
            default: Default value if tag is missing or file is invalid
            
        Returns:
            Tag value or default
        """
        if not self.dataset:
            return default
        
        # Check cache first
        if tag in self._tag_cache:
            return self._tag_cache[tag]
        
        try:
            value = self.dataset.get(tag, default)
            
            # Convert to basic Python types
            if value is not None and value != default:
                if hasattr(value, 'value'):
                    value = value.value
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore').strip()
                elif isinstance(value, str):
                    value = value.strip()
                
                # Cache the processed value
                self._tag_cache[tag] = value
            
            return value
            
        except Exception as e:
            logger.debug(f"Error accessing tag {tag}: {e}")
            return default
    
    # Patient/Study identification
    @cached_property
    def patient_id(self) -> Optional[str]:
        """Get Patient ID."""
        return self.get_tag(PATIENT_ID_TAG)
    
    @cached_property
    def patient_name(self) -> Optional[str]:
        """Get Patient Name."""
        return self.get_tag(PATIENT_NAME_TAG)
    
    @cached_property
    def study_instance_uid(self) -> Optional[str]:
        """Get Study Instance UID."""
        return self.get_tag(STUDY_INSTANCE_UID_TAG)
    
    @cached_property
    def study_id(self) -> Optional[str]:
        """Get Study ID."""
        return self.get_tag(STUDY_ID_TAG)
    
    # Series identification
    @cached_property
    def series_instance_uid(self) -> Optional[str]:
        """Get Series Instance UID."""
        return self.get_tag(SERIES_INSTANCE_UID_TAG)
    
    @cached_property
    def series_number(self) -> Optional[int]:
        """Get Series Number."""
        value = self.get_tag(SERIES_NUMBER_TAG)
        if value is not None:
            try:
                return int(value)
            except (ValueError, TypeError):
                logger.debug(f"Invalid series number value: {value}")
        return None
    
    @cached_property
    def instance_number(self) -> Optional[int]:
        """Get Instance Number."""
        value = self.get_tag(INSTANCE_NUMBER_TAG)
        if value is not None:
            try:
                return int(value)
            except (ValueError, TypeError):
                logger.debug(f"Invalid instance number value: {value}")
        return None
    
    # Primary modality detection
    @cached_property
    def protocol_name(self) -> Optional[str]:
        """Get Protocol Name - primary field for modality detection."""
        return self.get_tag(PROTOCOL_NAME_TAG)
    
    @cached_property
    def series_description(self) -> Optional[str]:
        """Get Series Description - primary field for modality detection."""
        return self.get_tag(SERIES_DESCRIPTION_TAG)
    
    # Timing information
    @cached_property
    def acquisition_date(self) -> Optional[str]:
        """Get Acquisition Date (YYYYMMDD format)."""
        return self.get_tag(ACQUISITION_DATE_TAG)
    
    @cached_property
    def acquisition_time(self) -> Optional[str]:
        """Get Acquisition Time (HHMMSS.FFFFFF format)."""
        return self.get_tag(ACQUISITION_TIME_TAG)
    
    def get_acquisition_datetime(self) -> Optional[str]:
        """
        Get combined acquisition datetime string.
        
        Returns:
            DateTime string in format YYYYMMDDHHMMSS or None
        """
        date = self.acquisition_date
        time = self.acquisition_time
        
        if date and time:
            # Take only HHMMSS part from time, ignore fractional seconds
            time_clean = time.split('.')[0] if '.' in time else time
            return f"{date}{time_clean}"
        elif date:
            return f"{date}000000"
        
        return None
    
    # MR parameters - lazy loading only when needed for modality detection
    def get_echo_time(self) -> Optional[float]:
        """Get Echo Time (TE) in milliseconds. Loaded on demand."""
        value = self.get_tag(ECHO_TIME_TAG)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.debug(f"Invalid echo time value: {value}")
        return None
    
    def get_repetition_time(self) -> Optional[float]:
        """Get Repetition Time (TR) in milliseconds. Loaded on demand."""
        value = self.get_tag(REPETITION_TIME_TAG)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.debug(f"Invalid repetition time value: {value}")
        return None
    
    def get_inversion_time(self) -> Optional[float]:
        """Get Inversion Time (TI) in milliseconds. Loaded on demand."""
        value = self.get_tag(INVERSION_TIME_TAG)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.debug(f"Invalid inversion time value: {value}")
        return None
    
    def __repr__(self) -> str:
        """String representation for developers."""
        return (
            f"DicomFile(file='{self.filepath.name}', "
            f"patient={self.patient_id}, "
            f"series={self.series_instance_uid})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.filepath.name}"
    
    def clear_cache(self):
        """Clear cached data to free memory."""
        self._dataset = None
        self._tag_cache.clear()
        
        # Clear cached properties
        cached_attrs = [
            'patient_id', 'patient_name', 'study_instance_uid', 'study_id',
            'series_instance_uid', 'series_number', 'instance_number',
            'protocol_name', 'series_description',
            'acquisition_date', 'acquisition_time'
        ]
        for attr in cached_attrs:
            self.__dict__.pop(attr, None)