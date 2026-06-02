#!/usr/bin/env python3
"""
Reorganize UPENN-GBM dataset to BIDS format.

This is an updated version that handles nested folder structures where:
- patient/<date-folder>/<series-folder>/*.dcm
- or deeper nesting where DICOMs may be inside the series folder or its subfolders.

Changes:
- scan_patient_series now finds directories that actually contain DICOM files (using rglob)
  and returns series directories together with the nearest ancestor folder that contains a date.
- detect_modality now first attempts to detect modality from folder names (series folder,
  parent, grandparent) and only then reads a DICOM (using rglob to find DICOM files).
- All places that counted or copied DICOM files use rglob to correctly include files in nested dirs.
- Minimal API changes kept local to caller sites.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pydicom
from scripts.performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
import multiprocessing as mp
from functools import partial
import time
import yaml
from scripts.metadata_extractor import MetadataExtractor
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config

# Single source of truth for modalities the pipeline knows how to process.
# Maps internal modality name → BIDS suffix used in output filenames.
# The .keys() of this dict is the input filter: any series whose detected
# modality is not here will be dropped. Adding a new modality (e.g. for
# metastases work) requires only an entry here plus an update to
# configs/lesion_types.yaml.
MODALITY_BIDS_SUFFIX: Dict[str, str] = {
    't1':   'T1w',
    't1c':  'T1wCE',
    't2':   'T2w',
    't2fl': 'FLAIR',
}


@dataclass
class SeriesInfo:
    """Information about a DICOM series."""
    original_path: Path
    patient_id: str
    date: str  # YYYYMMDD format
    modality: Optional[str] = None
    slice_count: int = 0
    series_description: str = ""


@dataclass
class SessionInfo:
    """Information about a session (grouped by date)."""
    date: str  # YYYYMMDD format
    series: Dict[str, SeriesInfo] = field(default_factory=dict)  # modality -> SeriesInfo


# Result of processing one patient. Returned by process_single_patient
# in parallel mode and consumed by run_parallel.
#
# NOTE: this is a plain dict, not a dataclass. A dataclass defined in the
# main script would carry __module__ == '__main__' (or 'main_module'
# depending on how Python loaded the script), which breaks pickling of
# return values from mp.Pool workers — worker processes cannot re-import
# that module by name and pool.imap_unordered silently turns the result
# into a failure. Plain dicts are pickle-safe regardless.
#
# Shape:
#   {
#     "status": "ok" | "skipped" | "failed",
#     "patient_id": str,                 # original_id, always present
#     "new_patient_id": Optional[str],   # set only when status == "ok"
#     "data": Optional[Dict],            # patient_data dict, set only on "ok"
#     "reason": Optional[str],           # human-readable explanation
#   }
PatientResult = Dict[str, object]


class ModalityDetector:
    """Detect modality using DICOM tags: ProtocolName and SeriesDescription.
    
    Enhanced with:
    - Multi-field contrast detection (ContrastBolusAgent + text keywords)
    - Word-boundary-aware 'ce' matching (prevents false positives from 'space', 'sequence', etc.)
    - Technical parameter fallback (TR/TE/TI)
    - INFO-level logging of detection decisions
    """

    # Regex: match 'ce' only as a standalone token, NOT inside words like
    # "space", "sequence", "slice", "balance", "source", "instance"
    _CE_PATTERN = re.compile(r'(?<![a-z])ce(?![a-z])')

    @staticmethod
    def _has_ce_marker(text: str) -> bool:
        """Check if 'ce' appears as a standalone contrast-enhanced marker."""
        return bool(ModalityDetector._CE_PATTERN.search(text))

    # Patterns for each modality (order matters!)
    MODALITY_PATTERNS = {
        't2fl': {  # FLAIR (check first - most specific)
            'keywords': ['flair', 'dark fluid', 't2-flair', 't2 flair'],
            'exclude': ['mpr']
        },
        't1c': {  # T1 with contrast (post)
            'keywords': ['t1'],
            'contrast_keywords': ['post', 'gad', 'contrast', 'c+', 'enhanced', '+c',
                                  'gadolinium', 'postcontrast', 'gd'],
            # Note: 'ce' is checked separately via _has_ce_marker() to avoid
            # false positives from words like "space", "sequence", "slice"
            'exclude': ['mpr', 'dyn', 'pit', 'spir']
        },
        't2': {  # T2
            'keywords': ['t2', 'tse', 'fse', 't2w'],
            # 't1' prevents T1-TSE from matching; 'mpr' prevents derived
            # reconstructions (e.g. MPR CE_T1-TSE) from falling through here
            'exclude': ['flair', 'dark fluid', 't1', 'mpr']
        },
        't1': {  # Plain T1 (non-contrast)
            'keywords': ['t1', 'mprage', 'spgr', 'tfe', 't1w'],
            'exclude': ['post', 'gad', 'contrast', 'c+', 'enhanced', '+c',
                        'gadolinium', 'gd', 'thr', 'pit']
            # Note: 'ce' is checked separately via _has_ce_marker()
        }
    }

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._cache: Dict[Path, Tuple[Optional[str], str]] = {}  # series_path -> (modality, description)

    def detect_modality(self, series_path: Path) -> Tuple[Optional[str], str]:
        """
        Detect modality from DICOM tags: ProtocolName and SeriesDescription.
        
        Enhanced detection cascade:
          1. Pattern matching on ProtocolName + SeriesDescription
          2. Multi-field contrast detection (ContrastBolusAgent, text keywords)
          3. Technical parameter fallback (TR/TE/TI)
        
        Args:
            series_path: Path to series directory
            
        Returns:
            (modality, combined_description) where:
                - modality: 't1', 't1c', 't2', 't2fl', or None
                - combined_description: "ProtocolName | SeriesDescription"
        """
        # Check cache
        if series_path in self._cache:
            return self._cache[series_path]

        # Find first DICOM file (search nested directories)
        dicom_files = sorted([f for f in series_path.rglob("*.dcm") if f.is_file()])
        if not dicom_files:
            self.logger.warning(f"No DICOM files found in {series_path}")
            self._cache[series_path] = (None, "")
            return None, ""

        try:
            # Read DICOM tags (stop before pixels for speed)
            dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
            
            # Get both ProtocolName and SeriesDescription
            protocol_name = str(dcm.get('ProtocolName', '')).lower().strip()
            series_desc = str(dcm.get('SeriesDescription', '')).lower().strip()
            
            # Create combined text for matching
            combined_text = f"{protocol_name} {series_desc}".strip()
            
            if not combined_text:
                self.logger.warning(f"No ProtocolName or SeriesDescription in {dicom_files[0]}")
                self._cache[series_path] = (None, "")
                return None, ""
            
            # Store readable description for logging
            readable_desc = f"{dcm.get('ProtocolName', 'N/A')} | {dcm.get('SeriesDescription', 'N/A')}"
            
            # Detect contrast from multiple DICOM fields
            has_contrast = self._detect_contrast(dcm, combined_text)
            
            # Level 1: Pattern matching on combined text
            modality = self._match_modality(combined_text, has_contrast)
            detection_method = "pattern"
            
            # Level 2: Technical parameter fallback (TR/TE/TI)
            if modality is None:
                modality = self._detect_by_technical_params(dcm, has_contrast)
                if modality:
                    detection_method = "technical_params"
            
            # Cache result
            self._cache[series_path] = (modality, readable_desc)
            
            # Log at INFO level so detection decisions are always visible
            series_name = series_path.name
            if modality:
                self.logger.info(
                    f"    {series_name}: {modality} "
                    f"[{detection_method}, contrast={has_contrast}] "
                    f"({readable_desc})")
            else:
                self.logger.info(
                    f"    {series_name}: NO MATCH "
                    f"[contrast={has_contrast}] "
                    f"({readable_desc})")
            
            return modality, readable_desc

        except Exception as e:
            self.logger.warning(f"Failed to read {dicom_files[0]}: {e}")
            self._cache[series_path] = (None, "")
            return None, ""

    def _detect_contrast(self, dcm: pydicom.Dataset, combined_text: str) -> bool:
        """Detect contrast agent from multiple DICOM fields + text keywords."""
        # Field: ContrastBolusAgent (0018,0010)
        agent = str(dcm.get((0x0018, 0x0010), '')).strip().lower()
        if agent and agent not in ('', 'none', 'no', 'n/a'):
            return True
        
        # Field: ContrastBolusStartTime (0018,1078)
        bolus = str(dcm.get((0x0018, 0x1078), '')).strip().lower()
        if bolus and bolus not in ('', 'none', 'no'):
            return True
        
        # Text keywords (excluding bare 'ce' — handled by _has_ce_marker)
        contrast_kws = [
            'contrast', 'gad', 'gadolinium', 'post',
            '+c', 'c+', 'enhanced', 'postcontrast',
        ]
        if any(kw in combined_text for kw in contrast_kws):
            return True
        
        # 'ce' with word-boundary check
        return self._has_ce_marker(combined_text)

    def _detect_by_technical_params(self, dcm: pydicom.Dataset, has_contrast: bool) -> Optional[str]:
        """Fallback modality detection using TR/TE/TI values."""
        def _float(tag):
            val = dcm.get(tag)
            if val is None:
                return None
            try:
                return float(val.value if hasattr(val, 'value') else val)
            except (ValueError, TypeError):
                return None
        
        tr = _float((0x0018, 0x0080))  # RepetitionTime
        te = _float((0x0018, 0x0081))  # EchoTime
        ti = _float((0x0018, 0x0082))  # InversionTime
        
        # FLAIR: very long TI
        if ti is not None and ti > 1500:
            return 't2fl'
        
        # T1C: contrast + short TR/TE
        if has_contrast and tr is not None and te is not None:
            if tr < 1200 and te < 30:
                return 't1c'
        
        # T1: short TR/TE, no contrast
        if not has_contrast and tr is not None and te is not None:
            if tr < 1000 and te < 30:
                return 't1'
        
        # T2: long TR, long TE, not FLAIR
        if tr is not None and te is not None:
            if tr > 2000 and te > 70:
                if ti is None or ti < 1500:
                    return 't2'
        
        return None

    def _match_modality(self, combined_text: str, has_contrast: bool = False) -> Optional[str]:
        """
        Match combined text (ProtocolName + SeriesDescription) against patterns.
        
        Order matters: check from most specific to most generic.
        
        Args:
            combined_text: Lowercase combined text from DICOM tags
            has_contrast: Whether contrast agent was detected from DICOM fields
            
        Returns:
            Modality name or None
        """
        # 1) Check FLAIR first (most specific)
        pattern = self.MODALITY_PATTERNS['t2fl']
        has_keyword = any(kw in combined_text for kw in pattern['keywords'])
        has_exclusion = any(ex in combined_text for ex in pattern['exclude'])
        
        if has_keyword and not has_exclusion:
            return 't2fl'

        # 2) Check T1 with contrast
        #    'ce' is checked via regex to avoid false positives from "space", "sequence", etc.
        pattern = self.MODALITY_PATTERNS['t1c']
        has_t1 = any(kw in combined_text for kw in pattern['keywords'])
        has_contrast_kw = (
            any(kw in combined_text for kw in pattern['contrast_keywords'])
            or self._has_ce_marker(combined_text)
        )
        has_excl = any(ex in combined_text for ex in pattern['exclude'])
        
        if has_t1 and (has_contrast_kw or has_contrast) and not has_excl:
            return 't1c'

        # 3) Check T2
        pattern = self.MODALITY_PATTERNS['t2']
        has_keyword = any(kw in combined_text for kw in pattern['keywords'])
        has_exclusion = any(ex in combined_text for ex in pattern['exclude'])
        
        if has_keyword and not has_exclusion:
            return 't2'

        # 4) Check T1 (non-contrast) — also reject if standalone 'ce' marker found
        pattern = self.MODALITY_PATTERNS['t1']
        has_keyword = any(kw in combined_text for kw in pattern['keywords'])
        has_exclusion = (
            any(ex in combined_text for ex in pattern['exclude'])
            or self._has_ce_marker(combined_text)
        )
        
        if has_keyword and not has_exclusion:
            return 't1'

        return None


class IDMapper:
    """Maps original patient IDs to new numbered IDs."""

    def __init__(self):
        self._patient_counter = 0
        self._patient_map = {}  # original_id -> new_id

    def get_patient_id(self, original_id: str) -> str:
        """
        Get or create new patient ID.

        Args:
            original_id: Original patient ID (e.g., 'UPENN-GBM-00454')

        Returns:
            New patient ID (e.g., 'sub-001')
        """
        if original_id not in self._patient_map:
            self._patient_counter += 1
            self._patient_map[original_id] = f"sub-{self._patient_counter:03d}"

        return self._patient_map[original_id]

    def get_mapping(self) -> Dict[str, str]:
        """Get complete patient ID mapping."""
        return self._patient_map.copy()


class DatasetScanner:
    """Scan dataset structure (UPENN-GBM, MS archives, etc.)."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def scan_dataset(self, input_dir: Path) -> List[Path]:
        """
        Scan for patient directories.

        Args:
            input_dir: Root directory of the source dataset

        Returns:
            List of patient directory paths
        """
        patient_dirs = []

        for patient_dir in sorted(input_dir.iterdir()):
            # Accept any non-hidden subdirectory as a patient. Different
            # source datasets use different naming (UPENN-GBM-XXX, P000XXX,
            # sub-XXX); the pipeline now supports multiple lesion types.
            if patient_dir.is_dir() and not patient_dir.name.startswith('.'):
                patient_dirs.append(patient_dir)

        self.logger.info(f"Found {len(patient_dirs)} patients")
        return patient_dirs

    def scan_patient_series(self, patient_dir: Path) -> List[Tuple[Path, Optional[str]]]:
        """
        Scan for series directories in patient folder, handling nested directories.

        Returns:
            List of tuples: (series_dir, date_folder_name_or_None)
            - series_dir: directory that directly contains DICOM files (closest ancestor of those files)
            - date_folder_name_or_None: the ancestor folder name that matches date regex (if any)
        """
        # Find all directories that contain at least one DICOM file (search nested)
        series_dirs_set: Set[Path] = set()
        for dcm in patient_dir.rglob("*.dcm"):
            if dcm.is_file():
                series_dirs_set.add(dcm.parent)

        # Convert to sorted list for stable processing
        series_dirs = sorted(series_dirs_set)

        results: List[Tuple[Path, Optional[str]]] = []
        for sdir in series_dirs:
            # Search ancestors up to patient_dir for a folder name with a date
            date_name = None
            current = sdir
            while current != patient_dir and current != current.parent:
                if self.parse_date_from_series_name(current.name):
                    date_name = current.name
                    break
                current = current.parent
            # If not found, also check immediate child of patient_dir (some datasets)
            if not date_name:
                for child in patient_dir.iterdir():
                    if child.is_dir() and child in sdir.parents:
                        if self.parse_date_from_series_name(child.name):
                            date_name = child.name
                            break
            results.append((sdir, date_name))

        return results

    @staticmethod
    def parse_date_from_series_name(series_name: str) -> Optional[str]:
        """
        Extract date from series/session folder name.

        Supports two formats:
          - US-style MM-DD-YYYY (UPENN-GBM):
              '03-08-2012-NA-BrainTumor-13096' -> '20120308'
          - ISO-style YYYY-MM-DD (clinical archives):
              '2023-03-25' -> '20230325'

        Args:
            series_name: Series or session folder name

        Returns:
            Date in YYYYMMDD format or None
        """
        # ISO format YYYY-MM-DD (check first — more specific year prefix)
        m = re.match(r'(\d{4})-(\d{2})-(\d{2})', series_name)
        if m:
            year, month, day = m.groups()
            # Basic sanity: year 1900-2099, valid month/day
            if 1900 <= int(year) <= 2099 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                return f"{year}{month}{day}"

        # US format MM-DD-YYYY (UPENN-GBM)
        m = re.match(r'(\d{2})-(\d{2})-(\d{4})', series_name)
        if m:
            month, day, year = m.groups()
            if 1900 <= int(year) <= 2099 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                return f"{year}{month}{day}"

        return None


class SessionGrouper:
    """Group series by date into sessions."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def group_by_date(self, series_list: List[SeriesInfo]) -> List[SessionInfo]:
        """
        Group series by date.

        Args:
            series_list: List of series information

        Returns:
            List of sessions (one per unique date)
        """
        # Group by date
        date_groups: Dict[str, List[SeriesInfo]] = {}

        for series in series_list:
            if series.date not in date_groups:
                date_groups[series.date] = []
            date_groups[series.date].append(series)

        # Create session objects
        sessions = []
        for date in sorted(date_groups.keys()):
            session = SessionInfo(date=date)

            # Group series by modality within session
            for series in date_groups[date]:
                if series.modality:
                    if series.modality not in session.series:
                        session.series[series.modality] = []

                    if not isinstance(session.series[series.modality], list):
                        session.series[series.modality] = [session.series[series.modality]]

                    session.series[series.modality].append(series)

            sessions.append(session)

        return sessions


class SeriesDeduplicator:
    """Select best series when multiple exist for same modality."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.duplicates_removed = 0

    def deduplicate_session(self, session: SessionInfo) -> SessionInfo:
        """
        Keep only one series per modality (one with most slices).

        Args:
            session: Session with potentially duplicate modalities

        Returns:
            Session with single series per modality
        """
        deduplicated = SessionInfo(date=session.date)

        for modality, series_list in session.series.items():
            if isinstance(series_list, list):
                if len(series_list) > 1:
                    # Multiple series for same modality
                    best_series = self._select_best_series(series_list)
                    deduplicated.series[modality] = best_series

                    removed_count = len(series_list) - 1
                    self.duplicates_removed += removed_count

                    self.logger.info(
                        f"Modality {modality}: {len(series_list)} series found, "
                        f"selected one with {best_series.slice_count} slices"
                    )
                else:
                    deduplicated.series[modality] = series_list[0]
            else:
                deduplicated.series[modality] = series_list

        return deduplicated

    def _select_best_series(self, series_list: List[SeriesInfo]) -> SeriesInfo:
        """Select series with most slices."""
        # Count slices for each series (use rglob to include nested files)
        for series in series_list:
            series.slice_count = len(list(series.original_path.rglob("*.dcm")))

        # Return series with maximum slice count
        return max(series_list, key=lambda s: s.slice_count)


class CompletenessChecker:
    """Check if patients/sessions have all required modalities for the lesion type."""

    DEFAULT_REQUIRED_MODALITIES = {'t1', 't1c', 't2', 't2fl'}  # fallback for unknown types

    def __init__(self, logger: logging.Logger, lesion_type: str = 'glioblastoma'):
        self.logger = logger
        try:
            cfg = load_lesion_type_config(lesion_type)
            self.required_modalities = set(cfg['required_modalities'])
        except KeyError:
            self.logger.warning(
                f"Unknown lesion_type '{lesion_type}', using default modalities"
            )
            self.required_modalities = self.DEFAULT_REQUIRED_MODALITIES
        self.logger.info(
            f"CompletenessChecker initialized for lesion_type={lesion_type}; "
            f"required modalities: {sorted(self.required_modalities)}"
        )

    def check_session(self, session: SessionInfo) -> Tuple[bool, Set[str]]:
        """
        Check if session has all required modalities.

        Returns:
            (is_complete, missing_modalities)
        """
        available = set(session.series.keys())
        missing = self.required_modalities - available
        is_complete = len(missing) == 0

        return is_complete, missing

    def generate_completeness_report(self, mapping_data: Dict) -> Dict:
        """Generate report of incomplete patients/sessions."""
        incomplete_patients = []
        complete_patient_count = 0
        complete_session_count = 0
        total_session_count = 0

        for patient_id, patient_data in mapping_data['patients'].items():
            patient_incomplete_sessions = []

            for session_id, session_data in patient_data['sessions'].items():
                total_session_count += 1
                available = set(session_data['series'].keys())
                missing = self.required_modalities - available

                if missing:
                    patient_incomplete_sessions.append({
                        'session_id': session_id,
                        'date': session_data['original_date'],
                        'missing': sorted(list(missing)),
                        'available': sorted(list(available))
                    })
                else:
                    complete_session_count += 1

            if patient_incomplete_sessions:
                incomplete_patients.append({
                    'patient_id': patient_id,
                    'original_id': patient_data['original_id'],
                    'incomplete_sessions': patient_incomplete_sessions
                })
            else:
                complete_patient_count += 1

        return {
            'incomplete_patients': incomplete_patients,
            'statistics': {
                'total_patients': len(mapping_data['patients']),
                'complete_patients': complete_patient_count,
                'incomplete_patients': len(incomplete_patients),
                'total_sessions': total_session_count,
                'complete_sessions': complete_session_count,
                'incomplete_sessions': total_session_count - complete_session_count
            }
        }


class FileOrganizer:
    """Create BIDS structure and copy files."""

    def __init__(self, output_dir: Path, logger: logging.Logger, dry_run: bool = False,
             metadata_extractor: Optional['MetadataExtractor'] = None,
             metadata_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.logger = logger
        self.dry_run = dry_run
        self.files_copied = 0
        self.files_would_copy = 0
        self.validation_failed = []
        self.metadata_extractor = metadata_extractor
        self.metadata_dir = metadata_dir
        self.metadata_saved = 0

    def create_bids_structure(
        self,
        patient_id: str,
        session_id: str,
        modality: str
    ) -> Path:
        """
        Create BIDS directory structure.

        Structure: output/sub-XXX/ses-XXX/anat/MODALITY/

        Returns:
            Path to modality directory
        """
        modality_dir = (
            self.output_dir /
            patient_id /
            session_id /
            'anat' /
            modality
        )

        if not self.dry_run:
            modality_dir.mkdir(parents=True, exist_ok=True)
        return modality_dir

    def copy_series(
        self,
        source_dir: Path,
        target_dir: Path,
        patient_id: str,
        session_id: str,
        modality: str
    ) -> int:
        """
        Copy DICOM series to target directory with BIDS naming.
        If metadata_extractor is provided, extracts metadata from the first
        file and anonymizes all files by removing configured tags before saving.

        Returns:
            Number of files copied
        """
        source_files = sorted([f for f in source_dir.rglob("*.dcm") if f.is_file()])

        if not source_files:
            self.logger.warning(f"No DICOM files in {source_dir}")
            return 0

        bids_suffix = MODALITY_BIDS_SUFFIX.get(modality, modality.upper())

        if self.dry_run:
            count = len(source_files)
            self.files_would_copy += count
            self.logger.debug(f"    [DRY RUN] Would copy {count} files for {modality}")
            return count

        # Extract and save metadata from first file before anonymization
        if self.metadata_extractor and self.metadata_dir:
            self._extract_and_save_metadata(
                source_files[0], patient_id, session_id, modality
            )

        copied = 0
        for idx, source_file in enumerate(source_files, 1):
            target_name = f"{patient_id}_{session_id}_{bids_suffix}_{idx:04d}.dcm"
            target_path = target_dir / target_name

            try:
                if self.metadata_extractor:
                    # Read → anonymize → save
                    dcm = pydicom.dcmread(str(source_file), force=True)
                    removed = self.metadata_extractor.anonymize_dicom(dcm)
                    dcm.save_as(str(target_path))
                    if idx == 1:  # Log only for first file in series
                        self.logger.info(
                            f"    Anonymized: removed {len(removed)} tags: "
                            f"{', '.join(removed)}"
                        )
                else:
                    # Fallback: plain copy (no config provided)
                    shutil.copy2(source_file, target_path)
                copied += 1
            except Exception as e:
                self.logger.error(f"Failed to process {source_file}: {e}")

        self.files_copied += copied
        return copied

    def _extract_and_save_metadata(
        self,
        dicom_path: Path,
        patient_id: str,
        session_id: str,
        modality: str
    ) -> bool:
        """Extract metadata from first DICOM file and save as JSON."""
        metadata = self.metadata_extractor.extract_metadata(dicom_path)
        if metadata is None:
            self.logger.warning(f"Failed to extract metadata from {dicom_path}")
            return False

        # Reuse save_metadata from MetadataExtractor
        # It expects patient_id without "sub-" prefix and session without "ses-" prefix
        pid = patient_id.replace("sub-", "")
        sid = session_id.replace("ses-", "")
        success = self.metadata_extractor.save_metadata(
            metadata, self.metadata_dir, pid, sid, modality
        )

        if success:
            self.metadata_saved += 1
            self.logger.debug(f"    Metadata saved for {patient_id}/{session_id}/{modality}")
        return success

    def validate_copy(
        self,
        source_dir: Path,
        target_dir: Path,
        patient_id: str,
        session_id: str,
        modality: str
    ) -> bool:
        """
        Validate that slice count matches between source and target.

        Returns:
            True if validation passed
        """

        if self.dry_run:
            # Skip validation in dry run
            return True
        
        source_count = len(list(source_dir.rglob("*.dcm")))
        target_count = len(list(target_dir.glob("*.dcm")))

        if source_count != target_count:
            error_msg = (
                f"{patient_id}/{session_id}/{modality}: "
                f"slice count mismatch (source: {source_count}, target: {target_count})"
            )
            self.logger.error(error_msg)
            self.validation_failed.append(error_msg)
            return False

        return True


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('reorganize_folders')
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers if re-running in same interpreter
    if logger.handlers:
        logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def save_mapping_file(mapping_data: Dict, output_dir: Path):
    """Save dataset mapping to JSON file."""
    mapping_file = output_dir / 'dataset_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)


def save_completeness_report(completeness_data: Dict, output_dir: Path) -> Path:
    """Save completeness report as JSON in incomplete_data/ subfolder."""
    incomplete_dir = output_dir / 'incomplete_data'
    incomplete_dir.mkdir(parents=True, exist_ok=True)

    stats = completeness_data['statistics']
    total_sessions = stats['total_sessions'] or 1
    success_rate = stats['complete_sessions'] / total_sessions * 100

    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stage': '01_reorganize_folders',
        'statistics': {**stats, 'success_rate_percent': round(success_rate, 1)},
        'incomplete_data': completeness_data['incomplete_patients'],
    }

    report_path = incomplete_dir / '01_reorganize_folders_incomplete_data.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


def print_summary(
    mapping_data: Dict,
    completeness_data: Dict,
    deduplicator: SeriesDeduplicator,
    file_organizer: FileOrganizer,
    total_time: float,
    performance_metrics: Optional[Dict] = None,
    dry_run: bool = False,
    lesion_type: str = 'glioblastoma',
):
    """Print final summary report."""
    stats = completeness_data['statistics']

    # Modalities relevant for the active lesion type (same source of truth
    # as CompletenessChecker uses). Fallback to default if lesion_type unknown.
    try:
        _mods = set(load_lesion_type_config(lesion_type)['required_modalities'])
    except KeyError:
        _mods = CompletenessChecker.DEFAULT_REQUIRED_MODALITIES
    relevant_modalities = sorted(_mods)

    print("\n" + "=" * 60)
    print("=== Reorganization Summary ===")
    print("=" * 60)

    print("\n📊 Processing Statistics:")
    print(f"  • Patients processed: {stats['total_patients']}")
    print(f"  • Sessions created: {stats['total_sessions']}")

    # Count series by modality (initialize only modalities relevant for
    # this lesion_type, but accept any others encountered via setdefault).
    modality_counts: Dict[str, int] = {m: 0 for m in relevant_modalities}
    modality_slices: Dict[str, int] = {m: 0 for m in relevant_modalities}

    for patient_data in mapping_data['patients'].values():
        for session_data in patient_data['sessions'].values():
            for modality, series_data in session_data['series'].items():
                modality_counts.setdefault(modality, 0)
                modality_slices.setdefault(modality, 0)
                modality_counts[modality] += 1
                modality_slices[modality] += series_data.get('slice_count', 0)

    print(f"  • Series processed by modality:")
    for modality in relevant_modalities:
        print(f"    - {modality:4s}: {modality_counts.get(modality,0):3d} series "
              f"({modality_slices.get(modality,0):,} slices)")

    print(f"  • Duplicate series removed: {deduplicator.duplicates_removed}")

    if dry_run:
        total_files = getattr(file_organizer, 'files_would_copy', 0)
        print(f"  • Total files that would be copied: {total_files:,}")
    else:
        print(f"  • Total files copied: {file_organizer.files_copied:,}")
        metadata_saved = getattr(file_organizer, 'metadata_saved', 0)
        if metadata_saved:
            print(f"  • Metadata JSON files saved: {metadata_saved:,}")

    print("\n✅ Completeness Analysis:")
    if stats['total_patients']:
        complete_pct = stats['complete_patients'] / stats['total_patients'] * 100
    else:
        complete_pct = 0.0
    print(f"  • Complete patients: {stats['complete_patients']}/{stats['total_patients']} "
          f"({complete_pct:.1f}%)")

    if stats['total_sessions']:
        complete_sess_pct = stats['complete_sessions'] / stats['total_sessions'] * 100
    else:
        complete_sess_pct = 0.0
    print(f"  • Complete sessions: {stats['complete_sessions']}/{stats['total_sessions']} "
          f"({complete_sess_pct:.1f}%)")

    if completeness_data['incomplete_patients']:
        print(f"\n⚠️  Incomplete Data Found:")
        print(f"  • {stats['incomplete_patients']} patients with missing modalities:")

        for patient in completeness_data['incomplete_patients'][:5]:  # Show first 5
            print(f"    - {patient['patient_id']} (original: {patient['original_id']})")
            for session in patient['incomplete_sessions']:
                missing = ', '.join(session['missing'])
                print(f"      └─ {session['session_id']}: missing {missing}")

        if len(completeness_data['incomplete_patients']) > 5:
            remaining = len(completeness_data['incomplete_patients']) - 5
            print(f"    ... and {remaining} more")

        print(f"\n  📄 Detailed report: incomplete_data/01_reorganize_folders_incomplete_data.json")

    print("\n🔍 Validation:")
    if not file_organizer.validation_failed:
        print("  ✓ All copied series passed slice count validation")
    else:
        print(f"  ✗ {len(file_organizer.validation_failed)} series failed validation")
        print("    See log file for details")

    print("\n⚡ Performance:")
    print(f"  • Total time: {total_time:.1f}s")
    
    patients_per_sec = stats['total_patients'] / total_time
    sessions_per_sec = stats['total_sessions'] / total_time
    print(f"  • Throughput: {patients_per_sec:.2f} patients/sec, "
          f"{sessions_per_sec:.2f} sessions/sec")
    
    if performance_metrics:
        if performance_metrics.get('cpu_avg'):
            print(f"  • CPU usage: avg {performance_metrics['cpu_avg']:.1f}%, "
                  f"peak {performance_metrics['cpu_max']:.1f}%")
        if performance_metrics.get('memory_avg_mb'):
            print(f"  • Memory usage: avg {performance_metrics['memory_avg_mb']:.1f}MB, "
                  f"peak {performance_metrics['memory_peak_mb']:.1f}MB")

    print("\n📁 Output:")
    print(f"  • BIDS directory: {mapping_data.get('output_dir', 'N/A')}")
    print(f"  • Mapping file: dataset_mapping.json")
    print(f"  • Incomplete data report: incomplete_data/01_reorganize_folders_incomplete_data.json")

    print("\n" + "=" * 60)

def _process_one_patient_core(
    patient_dir: Path,
    new_patient_id: str,
    modality_detector: 'ModalityDetector',
    scanner: 'DatasetScanner',
    grouper: 'SessionGrouper',
    deduplicator: 'SeriesDeduplicator',
    file_organizer: 'FileOrganizer',
    logger: logging.Logger,
) -> Optional[Dict]:
    """
    Process one patient end-to-end: scan series → detect modalities →
    filter → group into sessions → dedupe → copy files into BIDS layout →
    validate.

    Returns:
        Per-patient dict in the unified format:
            {
                'original_id': str,
                'sessions': {ses-XXX: {...}},
                'duplicates_removed': int,
                'files_copied': int,
                'validation_failed': List[str],
                'metadata_saved': int,
            }
        Returns None if no valid series found (patient should be skipped).

    Caller is responsible for:
        - resolving new_patient_id via IDMapper.get_patient_id();
        - calling check_patient_exists() before this function (to honor
          --force semantics and counting skipped patients).
    """
    original_patient_id = patient_dir.name

    # Scan series (now returns tuples: (series_dir, date_folder_name_or_None))
    series_entries = scanner.scan_patient_series(patient_dir)
    logger.debug(f"  Found {len(series_entries)} series entries (nested-aware)")

    # Process series
    series_list: List[SeriesInfo] = []
    for series_dir, date_folder_name in series_entries:
        # Priority: date from ancestor folder, else try parse from series_dir.name
        date = scanner.parse_date_from_series_name(date_folder_name) if date_folder_name else None
        if not date:
            date = scanner.parse_date_from_series_name(series_dir.name)

        if not date:
            logger.warning(f"  Could not parse date from {series_dir} or its parent folders, skipping")
            continue

        # Detect modality
        modality, series_description = modality_detector.detect_modality(series_dir)

        # Filter: only keep modalities the pipeline supports
        if modality in MODALITY_BIDS_SUFFIX:
            series_info = SeriesInfo(
                original_path=series_dir,
                patient_id=original_patient_id,
                date=date,
                modality=modality,
                series_description=series_description,
            )
            series_info.slice_count = len(list(series_dir.rglob("*.dcm")))
            series_list.append(series_info)
            logger.debug(f"  Series {series_dir}: {modality} ({series_info.slice_count} slices)")
        elif modality:
            logger.debug(f"  Series {series_dir}: {modality} (filtered out)")
        else:
            logger.debug(f"  Series {series_dir}: unknown modality (filtered out)")

    if not series_list:
        logger.warning(f"  No valid series found for {original_patient_id}")
        return None

    # Snapshot duplicates_removed BEFORE this patient's deduplication, so
    # we can attribute per-patient delta correctly when the deduplicator
    # is shared across patients (sequential mode).
    dup_before = deduplicator.duplicates_removed
    files_before = file_organizer.files_copied
    metadata_before = file_organizer.metadata_saved
    val_failed_before = len(file_organizer.validation_failed)

    # Group by date into sessions
    sessions = grouper.group_by_date(series_list)
    logger.debug(f"  Grouped into {len(sessions)} sessions")

    # Deduplicate: keep best series per modality
    sessions = [deduplicator.deduplicate_session(s) for s in sessions]

    patient_data: Dict = {
        'original_id': original_patient_id,
        'sessions': {},
    }

    for session_idx, session in enumerate(sessions, 1):
        new_session_id = f"ses-{session_idx:03d}"
        logger.debug(f"  Session {new_session_id} (date: {session.date})")

        session_data: Dict = {
            'original_date': session.date,
            'series': {},
        }

        for modality, series in session.series.items():
            target_dir = file_organizer.create_bids_structure(
                new_patient_id, new_session_id, modality
            )

            copied = file_organizer.copy_series(
                series.original_path,
                target_dir,
                new_patient_id,
                new_session_id,
                modality,
            )

            file_organizer.validate_copy(
                series.original_path,
                target_dir,
                new_patient_id,
                new_session_id,
                modality,
            )

            session_data['series'][modality] = {
                'original_path': str(series.original_path),
                'slice_count': copied,
                'series_description': series.series_description,
            }

            logger.debug(f"    {modality}: {copied} files copied")

        patient_data['sessions'][new_session_id] = session_data

    # Per-patient deltas (compatible with both shared and per-patient components).
    patient_data['duplicates_removed'] = deduplicator.duplicates_removed - dup_before
    patient_data['files_copied'] = file_organizer.files_copied - files_before
    patient_data['metadata_saved'] = file_organizer.metadata_saved - metadata_before
    patient_data['validation_failed'] = list(file_organizer.validation_failed[val_failed_before:])

    return patient_data


def process_single_patient(
    patient_dir: Path,
    input_dir: Path,
    output_dir: Path,
    patient_mapping: Dict[str, str],  # original_id -> new_id
    log_level: int = logging.INFO,
    force: bool = False,
    dry_run: bool = False,
    tags_config: Optional[Dict] = None,
    metadata_dir: Optional[Path] = None
) -> PatientResult:
    """
    Process a single patient (designed to run in parallel).

    Thin wrapper around _process_one_patient_core: creates per-process
    components, resolves the new patient ID, checks existence, then
    delegates the actual work.

    Returns:
        PatientResult with one of three statuses:
            - "ok":      patient processed; data contains patient_data dict.
            - "skipped": already processed and --force not set.
            - "failed":  no valid series, or an exception occurred.
    """
    original_patient_id = patient_dir.name

    # Setup logging for this process
    logger = logging.getLogger(f'patient_{original_patient_id}')
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)

    try:
        new_patient_id = patient_mapping[original_patient_id]

        # Check if patient already processed
        if check_patient_exists(output_dir, new_patient_id, force, dry_run, logger):
            return {
                "status": "skipped",
                "patient_id": original_patient_id,
                "new_patient_id": None,
                "data": None,
                "reason": "already_exists",
            }

        # Initialize fresh components for this process
        modality_detector = ModalityDetector(logger)
        scanner = DatasetScanner(logger)
        grouper = SessionGrouper(logger)
        deduplicator = SeriesDeduplicator(logger)
        _metadata_extractor = None
        if tags_config:
            _metadata_extractor = MetadataExtractor(tags_config, logger)

        file_organizer = FileOrganizer(
            output_dir, logger, dry_run=dry_run,
            metadata_extractor=_metadata_extractor, metadata_dir=metadata_dir,
        )

        logger.info(f"Processing {original_patient_id} -> {new_patient_id}")

        patient_data = _process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id=new_patient_id,
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
        )

        if patient_data is None:
            return {
                "status": "failed",
                "patient_id": original_patient_id,
                "new_patient_id": None,
                "data": None,
                "reason": "no_valid_series",
            }

        logger.info(f"Completed {original_patient_id}")
        return {
            "status": "ok",
            "patient_id": original_patient_id,
            "new_patient_id": new_patient_id,
            "data": patient_data,
            "reason": None,
        }

    except Exception as e:
        logger.error(f"Failed to process {original_patient_id}: {e}", exc_info=True)
        return {
            "status": "failed",
            "patient_id": original_patient_id,
            "new_patient_id": None,
            "data": None,
            "reason": f"exception: {e}",
        }

def check_patient_exists(output_dir: Path, patient_id: str, force: bool, dry_run: bool, logger: logging.Logger) -> bool:
    """
    Check if patient already processed and handle force flag.
    
    Args:
        output_dir: Output BIDS directory
        patient_id: New patient ID (e.g., 'sub-001')
        force: Force reprocessing flag
        dry_run: Dry run mode flag
        logger: Logger instance
        
    Returns:
        True if should skip processing, False if should process
    """
    if dry_run:
        # In dry run, always process to simulate
        return False
    
    patient_dir = output_dir / patient_id
    
    if not patient_dir.exists():
        # Patient not processed yet
        return False
    
    if force:
        # Force flag set - delete existing and reprocess
        logger.info(f"  Force flag set: deleting existing {patient_id}")
        try:
            shutil.rmtree(patient_dir)
            return False  # Process after deletion
        except Exception as e:
            logger.error(f"  Failed to delete {patient_dir}: {e}")
            return True  # Skip if can't delete
    else:
        # Patient exists and no force flag - skip
        logger.info(f"  Skipping {patient_id}: already processed (use --force to reprocess)")
        return True
    
def load_existing_mapping(output_dir: Path, logger: logging.Logger) -> Dict:
    """
    Load existing mapping file if it exists.
    
    Args:
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Existing mapping data or empty structure
    """
    mapping_file = output_dir / 'dataset_mapping.json'
    
    if not mapping_file.exists():
        logger.info("No existing mapping file found, starting fresh")
        return {
            'patients': {},
            'output_dir': str(output_dir),
            'created_at': datetime.now().isoformat()
        }
    
    try:
        with open(mapping_file, 'r') as f:
            existing_mapping = json.load(f)
        
        logger.info(f"Loaded existing mapping with {len(existing_mapping.get('patients', {}))} patients")
        
        # Update timestamp
        existing_mapping['updated_at'] = datetime.now().isoformat()
        
        return existing_mapping
        
    except Exception as e:
        logger.warning(f"Failed to load existing mapping: {e}, starting fresh")
        return {
            'patients': {},
            'output_dir': str(output_dir),
            'created_at': datetime.now().isoformat()
        }
    
def restore_id_mapper(mapping_data: Dict, logger: logging.Logger) -> IDMapper:
    """
    Restore IDMapper from existing mapping data.
    Continues numbering from last patient ID.
    
    Args:
        mapping_data: Existing mapping data with patients
        logger: Logger instance
        
    Returns:
        IDMapper with restored state
    """
    id_mapper = IDMapper()
    
    if not mapping_data.get('patients'):
        logger.info("No existing patients, starting from sub-001")
        return id_mapper
    
    # Extract all existing patient mappings
    for new_patient_id, patient_data in mapping_data['patients'].items():
        original_id = patient_data['original_id']
        
        # Add to mapper's internal dict
        id_mapper._patient_map[original_id] = new_patient_id
        
        # Extract counter from new_patient_id (e.g., 'sub-042' → 42)
        patient_num = int(new_patient_id.replace('sub-', ''))
        
        # Update counter to max seen
        if patient_num > id_mapper._patient_counter:
            id_mapper._patient_counter = patient_num
    
    logger.info(f"Restored IDMapper: {len(id_mapper._patient_map)} existing patients, "
               f"next will be sub-{id_mapper._patient_counter + 1:03d}")
    
    return id_mapper

def run_sequential(
    input_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    patient_dirs: Optional[List[Path]] = None,
    max_subjects: Optional[int] = None,
    benchmark: bool = False,
    results_dir: Optional[Path] = None,
    mode: str = 'sequential',
    workers: int = 1,
    force: bool = False,
    dry_run: bool = False,
    id_mapper: Optional[IDMapper] = None,
    tags_config: Optional[Dict] = None,
    metadata_dir: Optional[Path] = None,
    lesion_type: str = 'glioblastoma',
) -> Tuple[Dict, Dict, SeriesDeduplicator, 'FileOrganizer', Optional[Dict], Dict[str, int]]:
    """
    Run sequential processing (baseline).

    Returns:
        (mapping_data, completeness_data, deduplicator, file_organizer,
         performance_metrics, run_counters)
        where run_counters = {"successful": N, "skipped": N, "failed": N,
                              "failed_patients": [(orig_id, reason), ...]}.
    """
    # Initialize components
    modality_detector = ModalityDetector(logger)
    # Use provided id_mapper or create new one
    if id_mapper is None:
        id_mapper = IDMapper()
    scanner = DatasetScanner(logger)
    grouper = SessionGrouper(logger)
    deduplicator = SeriesDeduplicator(logger)
    completeness_checker = CompletenessChecker(logger, lesion_type=lesion_type)
    # Create metadata extractor if config provided
    _metadata_extractor = None
    if tags_config:
        _metadata_extractor = MetadataExtractor(tags_config, logger)
    
    file_organizer = FileOrganizer(
        output_dir, logger, dry_run=dry_run,
        metadata_extractor=_metadata_extractor, metadata_dir=metadata_dir
    )

    # Initialize mapping data - load existing or create new
    mapping_data = load_existing_mapping(output_dir, logger)
    
    # Track statistics for this run
    patients_processed_this_run = 0
    patients_skipped_this_run = 0
    patients_failed_this_run = 0
    failed_patients: List[Tuple[str, str]] = []  # (original_id, reason)

    # Start performance monitoring
    monitor = None
    if benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()
        logger.info("Performance monitoring started")

    # Scan dataset
    logger.info("Scanning input directory...")
    
    if patient_dirs is None:
        # Scan all patients if not provided
        patient_dirs = scanner.scan_dataset(input_dir)
        
        if max_subjects and len(patient_dirs) > max_subjects:
            patient_dirs = patient_dirs[:max_subjects]
            logger.info(f"Limited to {max_subjects} patients")
    
    logger.info(f"Processing {len(patient_dirs)} patients")

    # Process each patient
    for patient_idx, patient_dir in enumerate(patient_dirs, 1):
        original_patient_id = patient_dir.name
        new_patient_id = id_mapper.get_patient_id(original_patient_id)

        logger.info(f"Processing patient {patient_idx}/{len(patient_dirs)}: "
                    f"{original_patient_id} -> {new_patient_id}")

        # Check if patient already processed
        if check_patient_exists(output_dir, new_patient_id, force, dry_run, logger):
            patients_skipped_this_run += 1
            continue

        patient_data = _process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id=new_patient_id,
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
        )

        if patient_data is None:
            # No valid series for this patient — already logged inside.
            patients_failed_this_run += 1
            failed_patients.append((original_patient_id, "no_valid_series"))
            continue

        # Drop per-patient stats fields before persisting: aggregate stats
        # live on the shared file_organizer / deduplicator components and
        # are reported separately. This keeps mapping_data['patients'][*]
        # format identical between sequential and parallel modes.
        patient_data.pop('duplicates_removed', None)
        patient_data.pop('files_copied', None)
        patient_data.pop('metadata_saved', None)
        patient_data.pop('validation_failed', None)

        mapping_data['patients'][new_patient_id] = patient_data
        patients_processed_this_run += 1

    # Check completeness
    logger.info("Checking data completeness...")
    completeness_data = completeness_checker.generate_completeness_report(mapping_data)

    # Stop monitoring and collect metrics
    performance_metrics = None
    if monitor:
        monitor.stop()
        performance_metrics = monitor.get_metrics()
        logger.info("Performance monitoring stopped")

    # Log statistics for this run
    total_patients_in_mapping = len(mapping_data['patients'])
    logger.info("="*60)
    logger.info("Run statistics:")
    logger.info(f"  Patients in this run: {len(patient_dirs)}")
    logger.info(f"  New patients processed: {patients_processed_this_run}")
    logger.info(f"  Existing patients skipped: {patients_skipped_this_run}")
    logger.info(f"  Patients failed: {patients_failed_this_run}")
    logger.info(f"  Total patients in mapping: {total_patients_in_mapping}")
    logger.info("="*60)

    run_counters = {
        "successful": patients_processed_this_run,
        "skipped": patients_skipped_this_run,
        "failed": patients_failed_this_run,
        "failed_patients": failed_patients,
    }

    return (
        mapping_data,
        completeness_data,
        deduplicator,
        file_organizer,
        performance_metrics,
        run_counters,
    )

def run_parallel(
    input_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    workers: int = 4,
    patient_dirs: Optional[List[Path]] = None,
    max_subjects: Optional[int] = None,
    benchmark: bool = False,
    results_dir: Optional[Path] = None,
    mode: str = 'parallel',
    force: bool = False,
    dry_run: bool = False,
    id_mapper: Optional[IDMapper] = None,
    tags_config: Optional[Dict] = None,
    metadata_dir: Optional[Path] = None,
    lesion_type: str = 'glioblastoma'
) -> Tuple[Dict, Dict, int, 'FileOrganizer', Optional[Dict], Dict[str, int]]:
    """
    Run parallel processing using multiprocessing.

    Returns:
        (mapping_data, completeness_data, total_duplicates_removed,
         file_organizer_stats, performance_metrics, run_counters)
        where run_counters = {"successful": N, "skipped": N, "failed": N,
                              "failed_patients": [(orig_id, reason), ...]}.
    """
    # Start performance monitoring
    monitor = None
    if benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()
        logger.info("Performance monitoring started")
    
    # Initialize components
    scanner = DatasetScanner(logger)
    # Use provided id_mapper or create new one
    if id_mapper is None:
        id_mapper = IDMapper()
    completeness_checker = CompletenessChecker(logger, lesion_type=lesion_type)
    
    # Scan dataset
    logger.info("Scanning input directory...")
    
    if patient_dirs is None:
        # Scan all patients if not provided
        patient_dirs = scanner.scan_dataset(input_dir)
        
        if max_subjects and len(patient_dirs) > max_subjects:
            patient_dirs = patient_dirs[:max_subjects]
            logger.info(f"Limited to {max_subjects} patients")
    
    logger.info(f"Processing {len(patient_dirs)} patients")
    
    # Pre-create patient ID mapping (must be done before parallel processing)
    logger.info("Creating patient ID mapping...")
    patient_mapping = {}
    for patient_dir in patient_dirs:
        original_id = patient_dir.name
        new_id = id_mapper.get_patient_id(original_id)
        patient_mapping[original_id] = new_id
    
    logger.info(f"Processing {len(patient_dirs)} patients with {workers} workers...")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_patient,
        input_dir=input_dir,
        output_dir=output_dir,
        patient_mapping=patient_mapping,
        log_level=logging.WARNING,
        force=force,
        dry_run=dry_run,
        tags_config=tags_config,
        metadata_dir=metadata_dir
    )
    
    # Map worker status strings to counter keys. "ok" → "successful".
    STATUS_TO_COUNTER = {"ok": "successful", "skipped": "skipped", "failed": "failed"}

    # Process patients in parallel
    results: List[PatientResult] = []
    counters = {"successful": 0, "skipped": 0, "failed": 0}
    failed_patients: List[Tuple[str, str]] = []  # (original_id, reason)

    with mp.Pool(processes=workers) as pool:
        for pr in pool.imap_unordered(process_func, patient_dirs):
            results.append(pr)
            status = pr.get("status", "failed")
            patient_id = pr.get("patient_id", "?")
            reason = pr.get("reason")
            counters[STATUS_TO_COUNTER.get(status, "failed")] += 1
            if status == "failed":
                failed_patients.append((patient_id, reason or "unknown"))
                logger.warning(f"Patient {patient_id} failed: {reason}")
            elif status == "skipped":
                logger.info(f"Patient {patient_id} skipped: {reason}")
            done = len(results)
            logger.info(f"Progress: {done}/{len(patient_dirs)} patients processed")

    logger.info(
        f"Parallel run done: ok={counters['successful']}, "
        f"skipped={counters['skipped']}, failed={counters['failed']}"
    )

    # Aggregate results
    logger.info("Aggregating results...")
    mapping_data = load_existing_mapping(output_dir, logger)

    total_duplicates_removed = 0
    total_files_copied = 0
    total_metadata_saved = 0
    all_validation_failed: List[str] = []

    for pr in results:
        if pr.get("status") != "ok":
            continue
        patient_data = pr.get("data")
        new_patient_id = pr.get("new_patient_id")
        if patient_data is None or new_patient_id is None:
            continue
        # Extract and remove processing stats
        duplicates = patient_data.pop('duplicates_removed', 0)
        files_copied = patient_data.pop('files_copied', 0)
        metadata_saved = patient_data.pop('metadata_saved', 0)
        validation_failed = patient_data.pop('validation_failed', [])

        total_duplicates_removed += duplicates
        total_files_copied += files_copied
        total_metadata_saved += metadata_saved
        all_validation_failed.extend(validation_failed)

        mapping_data['patients'][new_patient_id] = patient_data

    # Create a mock FileOrganizer for stats (parallel doesn't use single organizer)
    class FileOrganizerStats:
        def __init__(self, files_copied, validation_failed, metadata_saved):
            self.files_copied = files_copied
            self.validation_failed = validation_failed
            self.metadata_saved = metadata_saved

    file_organizer = FileOrganizerStats(
        total_files_copied, all_validation_failed, total_metadata_saved
    )

    run_counters = {**counters, "failed_patients": failed_patients}
    
    # Check completeness
    logger.info("Checking data completeness...")
    completeness_data = completeness_checker.generate_completeness_report(mapping_data)
    
    # Stop monitoring and collect metrics
    performance_metrics = None
    if monitor:
        monitor.stop()
        performance_metrics = monitor.get_metrics()
        logger.info("Performance monitoring stopped")

    return (
        mapping_data,
        completeness_data,
        total_duplicates_removed,
        file_organizer,
        performance_metrics,
        run_counters,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize source dataset (UPENN-GBM, MS-clinical, etc.) to BIDS format'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Input UPENN-GBM directory'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output BIDS directory'
    )
    parser.add_argument(
        '--mode',
        choices=['sequential', 'parallel'],
        default='sequential',
        help='Processing mode (default: sequential)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--log_file',
        type=Path,
        default=None,
        help='Path to log file'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Simulate without copying files'
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing: delete existing patient directories and reprocess'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Enable performance benchmarking'
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results'),
        help='Directory for benchmark results (default: results)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to DICOM tags YAML config for metadata extraction and anonymization'
    )
    parser.add_argument(
        '--metadata-dir',
        type=Path,
        default=None,
        help='Output directory for extracted metadata JSON files'
    )
    parser.add_argument(
        '--lesion-type',
        type=str,
        default='glioblastoma',
        choices=['glioblastoma', 'multiple_sclerosis'],
        help='Lesion type — affects which modalities are considered required '
             '(glio: T1+T1c+T2+FLAIR; MS: T1+T2+FLAIR).'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    if args.log_file is None:
        args.log_file = args.output_dir / 'reorganize.log'

    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Starting dataset reorganization")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log file: {args.log_file}")

    # Load tags config for anonymization
    tags_config = None
    metadata_dir = None
    if args.config:
        if not args.config.exists():
            logger.error(f"Tags config not found: {args.config}")
            sys.exit(1)
        with open(args.config, 'r') as f:
            tags_config = yaml.safe_load(f)
        logger.info(f"Tags config loaded: {args.config}")

        metadata_dir = args.metadata_dir if args.metadata_dir else (args.output_dir.parent / 'metadata')
        metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Metadata output: {metadata_dir}")
        logger.info("Anonymization ENABLED")
    else:
        logger.info("Anonymization DISABLED (no --config provided)")

    if args.dry_run:
        logger.warning("DRY RUN MODE - No files will be copied")
        print("\n⚠️  DRY RUN MODE - Simulation only, no files will be copied\n")
        # Don't return - continue with dry run processing

    # Start processing
    start_time = datetime.now()

    # Scan dataset to get all patient directories
    logger.info("Scanning dataset for patients...")
    scanner = DatasetScanner(logger)
    all_patient_dirs = scanner.scan_dataset(args.input_dir)

    if args.max_subjects:
        all_patient_dirs = all_patient_dirs[:args.max_subjects]
        logger.info(f"Limited to {args.max_subjects} patients")

    logger.info(f"Total patients to process: {len(all_patient_dirs)}")
    logger.info("=" * 60)

    # Initialize or restore IDMapper from any previously saved mapping
    existing_mapping = load_existing_mapping(args.output_dir, logger)
    id_mapper = restore_id_mapper(existing_mapping, logger)

    if args.mode == 'sequential':
        (
            mapping_data,
            completeness_data,
            deduplicator,
            file_organizer,
            performance_metrics,
            run_counters,
        ) = run_sequential(
            args.input_dir,
            args.output_dir,
            logger,
            patient_dirs=all_patient_dirs,
            max_subjects=None,
            benchmark=args.benchmark,
            results_dir=args.results_dir,
            mode=args.mode,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
            id_mapper=id_mapper,
            tags_config=tags_config,
            metadata_dir=metadata_dir,
            lesion_type=args.lesion_type,
        )
        dedup_stats = deduplicator
        file_organizer_stats = file_organizer

    else:
        (
            mapping_data,
            completeness_data,
            total_duplicates_removed,
            file_organizer,
            performance_metrics,
            run_counters,
        ) = run_parallel(
            args.input_dir,
            args.output_dir,
            logger,
            workers=args.workers,
            patient_dirs=all_patient_dirs,
            max_subjects=None,
            benchmark=args.benchmark,
            results_dir=args.results_dir,
            mode=args.mode,
            force=args.force,
            dry_run=args.dry_run,
            id_mapper=id_mapper,
            tags_config=tags_config,
            metadata_dir=metadata_dir,
            lesion_type=args.lesion_type,
        )

        class _DeduplicatorStats:
            def __init__(self, n):
                self.duplicates_removed = n

        dedup_stats = _DeduplicatorStats(total_duplicates_removed)
        file_organizer_stats = file_organizer

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETED")
    logger.info("=" * 60)

    # Save outputs
    if args.dry_run:
        logger.info("[DRY RUN] Would save mapping file...")
        logger.info("[DRY RUN] Would save completeness report...")
        print("\n📄 [DRY RUN] Reports not saved (use without --dry_run to save)")
    else:
        logger.info("Saving mapping file...")
        save_mapping_file(mapping_data, args.output_dir)

        logger.info("Saving completeness report...")
        save_completeness_report(completeness_data, args.output_dir)

    # Save benchmark metrics
    if args.benchmark and performance_metrics:
        logger.info("Saving benchmark metrics...")

        args.results_dir.mkdir(parents=True, exist_ok=True)

        stats = completeness_data['statistics']
        total_patients = stats['total_patients']
        total_sessions = stats['total_sessions']

        total_series = sum(
            len(patient_data['sessions'][session_id]['series'])
            for patient_data in mapping_data['patients'].values()
            for session_id in patient_data['sessions']
        )

        time_per_patient = total_time / total_patients if total_patients > 0 else 0
        throughput_patients = total_patients / total_time if total_time > 0 else 0

        benchmark_logger = BenchmarkLogger(args.results_dir)
        baseline_time = benchmark_logger.get_baseline_time()

        speedup = None
        efficiency = None
        if args.mode == 'sequential' and args.workers == 1:
            speedup = 1.0
            efficiency = 1.0
        elif baseline_time:
            speedup = baseline_time / time_per_patient
            efficiency = speedup / args.workers if args.workers > 0 else None

        experiment_id = f"{args.mode}_w{args.workers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        attempted = run_counters["successful"] + run_counters["failed"]
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            mode=args.mode if not args.dry_run else f"{args.mode}_dryrun",
            workers=args.workers,
            total_series=attempted,
            successful=run_counters["successful"],
            failed=run_counters["failed"],
            skipped=run_counters["skipped"],
            total_time=total_time,
            time_per_series=time_per_patient,
            throughput=throughput_patients,
            cpu_avg=performance_metrics.get('cpu_avg'),
            cpu_max=performance_metrics.get('cpu_max'),
            memory_avg_mb=performance_metrics.get('memory_avg_mb'),
            memory_peak_mb=performance_metrics.get('memory_peak_mb'),
            speedup=speedup,
            efficiency=efficiency,
        )

        benchmark_logger.log_metrics(metrics)

        logger.info(f"Benchmark metrics saved to {args.results_dir / 'metrics.csv'}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Patients: {total_patients} ({time_per_patient:.2f}s each)")
        logger.info(f"  Sessions: {total_sessions}")
        logger.info(f"  Series: {total_series}")
        logger.info(f"  Throughput: {throughput_patients:.2f} patients/sec")
        if performance_metrics.get('cpu_avg'):
            logger.info(f"  CPU: avg {performance_metrics['cpu_avg']:.1f}%, "
                        f"peak {performance_metrics['cpu_max']:.1f}%")
        if performance_metrics.get('memory_avg_mb'):
            logger.info(f"  Memory: avg {performance_metrics['memory_avg_mb']:.1f}MB, "
                        f"peak {performance_metrics['memory_peak_mb']:.1f}MB")
        if speedup:
            logger.info(f"  Speedup: {speedup:.2f}x")
        if efficiency:
            logger.info(f"  Efficiency: {efficiency:.1f}%")

    # Print summary
    print_summary(
        mapping_data,
        completeness_data,
        dedup_stats,
        file_organizer_stats,
        total_time,
        performance_metrics,
        dry_run=args.dry_run,
        lesion_type=args.lesion_type,
    )

    # Per-status run summary
    logger.info(
        "Run counters: ok=%d, skipped=%d, failed=%d",
        run_counters["successful"],
        run_counters["skipped"],
        run_counters["failed"],
    )
    if run_counters["failed_patients"]:
        logger.warning("Failed patients (%d):", len(run_counters["failed_patients"]))
        for orig_id, reason in run_counters["failed_patients"]:
            logger.warning("  - %s: %s", orig_id, reason)

    logger.info("=" * 60)
    logger.info("Dataset reorganization completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()