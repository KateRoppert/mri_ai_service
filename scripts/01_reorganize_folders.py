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
            'exclude': []
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
            'exclude': ['flair', 'dark fluid']
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
    """Scan UPENN-GBM dataset structure."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def scan_dataset(self, input_dir: Path) -> List[Path]:
        """
        Scan for patient directories.

        Args:
            input_dir: Root directory of UPENN-GBM dataset

        Returns:
            List of patient directory paths
        """
        patient_dirs = []

        for patient_dir in sorted(input_dir.iterdir()):
            if patient_dir.is_dir() and patient_dir.name.startswith('UPENN-GBM-'):
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
        Extract date from series folder name.

        Example: '03-08-2012-NA-BrainTumor-13096' -> '20120308'

        Args:
            series_name: Series folder name

        Returns:
            Date in YYYYMMDD format or None
        """
        match = re.match(r'(\d{2})-(\d{2})-(\d{4})', series_name)
        if match:
            month, day, year = match.groups()
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
    """Check if patients/sessions have all 4 modalities."""

    REQUIRED_MODALITIES = {'t1', 't1c', 't2', 't2fl'}

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def check_session(self, session: SessionInfo) -> Tuple[bool, Set[str]]:
        """
        Check if session has all required modalities.

        Returns:
            (is_complete, missing_modalities)
        """
        available = set(session.series.keys())
        missing = self.REQUIRED_MODALITIES - available
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
                missing = self.REQUIRED_MODALITIES - available

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

    def __init__(self, output_dir: Path, logger: logging.Logger, dry_run: bool = False):
        self.output_dir = output_dir
        self.logger = logger
        self.dry_run = dry_run
        self.files_copied = 0
        self.files_would_copy = 0
        self.validation_failed = []

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

        Returns:
            Number of files copied
        """
        # Use rglob to collect all DICOM files in nested structure
        source_files = sorted([f for f in source_dir.rglob("*.dcm") if f.is_file()])

        if not source_files:
            self.logger.warning(f"No DICOM files in {source_dir}")
            return 0

        # BIDS naming: sub-XXX_ses-XXX_MODALITY_NNNN.dcm
        # Map modality folder name to BIDS suffix
        bids_suffix_map = {
            't1': 'T1w',
            't1c': 'T1wCE',
            't2': 'T2w',
            't2fl': 'FLAIR'
        }
        bids_suffix = bids_suffix_map.get(modality, modality.upper())

        if self.dry_run:
            # Dry run: just count files
            count = len(source_files)
            self.files_would_copy += count
            self.logger.debug(f"    [DRY RUN] Would copy {count} files for {modality}")
            return count

        copied = 0
        for idx, source_file in enumerate(source_files, 1):
            target_name = f"{patient_id}_{session_id}_{bids_suffix}_{idx:04d}.dcm"
            target_path = target_dir / target_name

            try:
                shutil.copy2(source_file, target_path)
                copied += 1
            except Exception as e:
                self.logger.error(f"Failed to copy {source_file}: {e}")

        self.files_copied += copied
        return copied

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


def save_completeness_report(completeness_data: Dict, output_dir: Path):
    """Save completeness report to text file."""
    report_file = output_dir / 'incomplete_data.txt'

    with open(report_file, 'w') as f:
        f.write("=== Incomplete Data Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        stats = completeness_data['statistics']
        total_patients = stats['total_patients'] or 1
        total_sessions = stats['total_sessions'] or 1

        f.write(f"Total patients: {stats['total_patients']}\n")
        f.write(f"Complete patients: {stats['complete_patients']} "
                f"({stats['complete_patients']/total_patients*100:.1f}%)\n")
        f.write(f"Incomplete patients: {stats['incomplete_patients']} "
                f"({stats['incomplete_patients']/total_patients*100:.1f}%)\n\n")

        f.write(f"Total sessions: {stats['total_sessions']}\n")
        f.write(f"Complete sessions: {stats['complete_sessions']} "
                f"({stats['complete_sessions']/total_sessions*100:.1f}%)\n")
        f.write(f"Incomplete sessions: {stats['incomplete_sessions']} "
                f"({stats['incomplete_sessions']/total_sessions*100:.1f}%)\n\n")

        if completeness_data['incomplete_patients']:
            f.write("=== Details ===\n\n")

            for patient in completeness_data['incomplete_patients']:
                f.write(f"Patient: {patient['patient_id']} ({patient['original_id']})\n")

                for session in patient['incomplete_sessions']:
                    f.write(f"  Session: {session['session_id']} ({session['date']})\n")
                    f.write(f"    Missing: {', '.join(session['missing'])}\n")
                    f.write(f"    Available: {', '.join(session['available'])}\n")

                f.write("\n")


def print_summary(
    mapping_data: Dict,
    completeness_data: Dict,
    deduplicator: SeriesDeduplicator,
    file_organizer: FileOrganizer,
    total_time: float,
    performance_metrics: Optional[Dict] = None,
    dry_run: bool = False
):
    """Print final summary report."""
    stats = completeness_data['statistics']

    print("\n" + "=" * 60)
    print("=== Reorganization Summary ===")
    print("=" * 60)

    print("\n📊 Processing Statistics:")
    print(f"  • Patients processed: {stats['total_patients']}")
    print(f"  • Sessions created: {stats['total_sessions']}")

    # Count series by modality
    modality_counts = {'t1': 0, 't1c': 0, 't2': 0, 't2fl': 0}
    modality_slices = {'t1': 0, 't1c': 0, 't2': 0, 't2fl': 0}

    for patient_data in mapping_data['patients'].values():
        for session_data in patient_data['sessions'].values():
            for modality, series_data in session_data['series'].items():
                modality_counts.setdefault(modality, 0)
                modality_slices.setdefault(modality, 0)
                modality_counts[modality] += 1
                modality_slices[modality] += series_data.get('slice_count', 0)

    print(f"  • Series processed by modality:")
    for modality in ['t1', 't1c', 't2', 't2fl']:
        print(f"    - {modality:4s}: {modality_counts.get(modality,0):3d} series "
              f"({modality_slices.get(modality,0):,} slices)")

    print(f"  • Duplicate series removed: {deduplicator.duplicates_removed}")

    if dry_run:
        total_files = getattr(file_organizer, 'files_would_copy', 0)
        print(f"  • Total files that would be copied: {total_files:,}")
    else:
        print(f"  • Total files copied: {file_organizer.files_copied:,}")

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

        print(f"\n  📄 Detailed report: incomplete_data.txt")

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
    print(f"  • Incomplete data report: incomplete_data.txt")

    print("\n" + "=" * 60)

def process_single_patient(
    patient_dir: Path,
    input_dir: Path,
    output_dir: Path,
    patient_mapping: Dict[str, str],  # original_id -> new_id
    log_level: int = logging.INFO,
    force: bool = False,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    Process a single patient (designed to run in parallel).
    
    Args:
        patient_dir: Patient directory path
        input_dir: Root input directory
        output_dir: Root output directory
        patient_mapping: Mapping of original to new patient IDs
        log_level: Logging level
        
    Returns:
        Dictionary with patient data or None if failed
    """
    # Setup logging for this process
    logger = logging.getLogger(f'patient_{patient_dir.name}')
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    
    try:
        # Check if patient already processed
        original_patient_id = patient_dir.name
        new_patient_id = patient_mapping[original_patient_id]
        if check_patient_exists(output_dir, new_patient_id, force, dry_run, logger):
            return None  # Skip this patient
        
        # Initialize components for this process
        modality_detector = ModalityDetector(logger)
        scanner = DatasetScanner(logger)
        grouper = SessionGrouper(logger)
        deduplicator = SeriesDeduplicator(logger)
        file_organizer = FileOrganizer(output_dir, logger, dry_run=dry_run)
        
        original_patient_id = patient_dir.name
        new_patient_id = patient_mapping[original_patient_id]
        
        logger.info(f"Processing {original_patient_id} -> {new_patient_id}")
        
        # Scan series
        series_entries = scanner.scan_patient_series(patient_dir)
        logger.debug(f"  Found {len(series_entries)} series entries")
        
        # Process series
        series_list: List[SeriesInfo] = []
        for series_dir, date_folder_name in series_entries:
            # Parse date
            date = scanner.parse_date_from_series_name(date_folder_name) if date_folder_name else None
            if not date:
                date = scanner.parse_date_from_series_name(series_dir.name)
            
            if not date:
                logger.warning(f"  Could not parse date from {series_dir}, skipping")
                continue
            
            # Detect modality
            modality, series_description = modality_detector.detect_modality(series_dir)
            
            # Filter: only keep 4 target modalities
            if modality in {'t1', 't1c', 't2', 't2fl'}:
                series_info = SeriesInfo(
                    original_path=series_dir,
                    patient_id=original_patient_id,
                    date=date,
                    modality=modality,
                    series_description=series_description
                )
                series_info.slice_count = len(list(series_dir.rglob("*.dcm")))
                series_list.append(series_info)
                logger.debug(f"  Series {series_dir.name}: {modality} ({series_info.slice_count} slices)")
            elif modality:
                logger.debug(f"  Series {series_dir.name}: {modality} (filtered out)")
            else:
                logger.debug(f"  Series {series_dir.name}: unknown modality (filtered out)")
        
        if not series_list:
            logger.warning(f"  No valid series found for {original_patient_id}")
            return None
        
        # Group by date into sessions
        sessions = grouper.group_by_date(series_list)
        logger.debug(f"  Grouped into {len(sessions)} sessions")
        
        # Deduplicate
        sessions = [deduplicator.deduplicate_session(s) for s in sessions]
        
        # Process each session
        patient_data = {
            'original_id': original_patient_id,
            'sessions': {},
            'duplicates_removed': deduplicator.duplicates_removed,
            'files_copied': 0,
            'validation_failed': []
        }
        
        for session_idx, session in enumerate(sessions, 1):
            new_session_id = f"ses-{session_idx:03d}"
            
            logger.debug(f"  Session {new_session_id} (date: {session.date})")
            
            session_data = {
                'original_date': session.date,
                'series': {}
            }
            
            # Process each modality in session
            for modality, series in session.series.items():
                # Create BIDS structure
                target_dir = file_organizer.create_bids_structure(
                    new_patient_id, new_session_id, modality
                )
                
                # Copy files
                copied = file_organizer.copy_series(
                    series.original_path,
                    target_dir,
                    new_patient_id,
                    new_session_id,
                    modality
                )
                
                # Validate
                file_organizer.validate_copy(
                    series.original_path,
                    target_dir,
                    new_patient_id,
                    new_session_id,
                    modality
                )
                
                # Update mapping
                session_data['series'][modality] = {
                    'original_path': str(series.original_path),
                    'slice_count': copied,
                    'series_description': series.series_description
                }
                
                logger.debug(f"    {modality}: {copied} files copied")
            
            patient_data['sessions'][new_session_id] = session_data
        
        # Store stats from this patient's processing
        patient_data['files_copied'] = file_organizer.files_copied
        patient_data['validation_failed'] = file_organizer.validation_failed
        
        logger.info(f"Completed {original_patient_id}")
        
        return {new_patient_id: patient_data}
        
    except Exception as e:
        logger.error(f"Failed to process {patient_dir.name}: {e}", exc_info=True)
        return None

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
    
def create_batches(items: List, batch_size: int) -> List[List]:
    """
    Split list into batches.
    
    Args:
        items: List of items to split
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    if batch_size is None or batch_size <= 0:
        return [items]  # Single batch with all items
    
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches

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
    id_mapper: Optional[IDMapper] = None 
) -> Tuple[Dict, Dict, SeriesDeduplicator, FileOrganizer]:
    """
    Run sequential processing (baseline).

    Returns:
        (mapping_data, completeness_data, deduplicator, file_organizer)
    """
    # Initialize components
    modality_detector = ModalityDetector(logger)
    # Use provided id_mapper or create new one
    if id_mapper is None:
        id_mapper = IDMapper()
    scanner = DatasetScanner(logger)
    grouper = SessionGrouper(logger)
    deduplicator = SeriesDeduplicator(logger)
    completeness_checker = CompletenessChecker(logger)
    file_organizer = FileOrganizer(output_dir, logger, dry_run=dry_run)

    # Initialize mapping data - load existing or create new
    mapping_data = load_existing_mapping(output_dir, logger)
    
    # Track statistics for this run
    patients_processed_this_run = 0
    patients_skipped_this_run = 0

    # Start performance monitoring
    monitor = None
    if benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()
        logger.info("Performance monitoring started")

    # Scan dataset
    logger.info("Scanning input directory...")
    scanner = DatasetScanner(logger)
    
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

        # Scan series (now returns tuples: (series_dir, date_folder_name_or_None))
        series_entries = scanner.scan_patient_series(patient_dir)
        logger.debug(f"  Found {len(series_entries)} series entries (nested-aware)")

        # Process series
        series_list: List[SeriesInfo] = []
        for series_dir, date_folder_name in series_entries:
            # Priority: date from ancestor folder (date_folder_name), else try parse from series_dir.name
            date = scanner.parse_date_from_series_name(date_folder_name) if date_folder_name else None
            if not date:
                date = scanner.parse_date_from_series_name(series_dir.name)

            if not date:
                logger.warning(f"  Could not parse date from {series_dir} or its parent folders, skipping")
                continue

            # Detect modality (returns modality and series_description)
            modality, series_description = modality_detector.detect_modality(series_dir)

            # Filter: only keep 4 target modalities
            if modality in {'t1', 't1c', 't2', 't2fl'}:
                series_info = SeriesInfo(
                    original_path=series_dir,
                    patient_id=original_patient_id,
                    date=date,
                    modality=modality,
                    series_description=series_description
                )
                # slice_count can be computed later in deduplicator, but store current count for info
                series_info.slice_count = len(list(series_dir.rglob("*.dcm")))
                series_list.append(series_info)
                logger.debug(f"  Series {series_dir}: {modality} ({series_info.slice_count} slices)")
            elif modality:
                logger.debug(f"  Series {series_dir}: {modality} (filtered out)")
            else:
                logger.debug(f"  Series {series_dir}: unknown modality (filtered out)")

        if not series_list:
            logger.warning(f"  No valid series found for {original_patient_id}")
            continue

        # Group by date into sessions
        sessions = grouper.group_by_date(series_list)
        logger.debug(f"  Grouped into {len(sessions)} sessions")

        # Deduplicate: keep best series per modality
        sessions = [deduplicator.deduplicate_session(s) for s in sessions]

        # Process each session
        patient_data = {
            'original_id': original_patient_id,
            'sessions': {}
        }

        for session_idx, session in enumerate(sessions, 1):
            new_session_id = f"ses-{session_idx:03d}"

            logger.debug(f"  Session {new_session_id} (date: {session.date})")

            session_data = {
                'original_date': session.date,
                'series': {}
            }

            # Process each modality in session
            for modality, series in session.series.items():
                # Create BIDS structure
                target_dir = file_organizer.create_bids_structure(
                    new_patient_id, new_session_id, modality
                )

                # Copy files (source_dir may have nested DICOMs)
                copied = file_organizer.copy_series(
                    series.original_path,
                    target_dir,
                    new_patient_id,
                    new_session_id,
                    modality
                )

                # Validate
                file_organizer.validate_copy(
                    series.original_path,
                    target_dir,
                    new_patient_id,
                    new_session_id,
                    modality
                )

                # Update mapping
                session_data['series'][modality] = {
                    'original_path': str(series.original_path),
                    'slice_count': copied,
                    'series_description': series.series_description
                }

                logger.debug(f"    {modality}: {copied} files copied")

            patient_data['sessions'][new_session_id] = session_data

        mapping_data['patients'][new_patient_id] = patient_data
        patients_processed_this_run += 1

    # Check completeness
    logger.info("Checking data completeness...")
    completeness_data = completeness_checker.generate_completeness_report(mapping_data)

    # Stop monitoring and collect metrics
    performance_metrics = None
    if monitor:
        performance_metrics = monitor.get_metrics()
        logger.info("Performance monitoring stopped")

    # Log statistics for this run
    total_patients_in_mapping = len(mapping_data['patients'])
    logger.info("="*60)
    logger.info("Run statistics:")
    logger.info(f"  Patients in this batch: {len(patient_dirs)}")
    logger.info(f"  New patients processed: {patients_processed_this_run}")
    logger.info(f"  Existing patients skipped: {patients_skipped_this_run}")
    logger.info(f"  Total patients in mapping: {total_patients_in_mapping}")
    logger.info("="*60)
    
    return mapping_data, completeness_data, deduplicator, file_organizer, performance_metrics

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
    id_mapper: Optional[IDMapper] = None 
) -> Tuple[Dict, Dict, int, FileOrganizer, Optional[Dict]]:
    """
    Run parallel processing using multiprocessing.
    
    Returns:
        (mapping_data, completeness_data, total_duplicates_removed, file_organizer_stats, performance_metrics)
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
    completeness_checker = CompletenessChecker(logger)
    
    # Scan dataset
    logger.info("Scanning input directory...")
    scanner = DatasetScanner(logger)
    
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
        dry_run=dry_run  # Reduce noise in parallel mode
    )
    
    # Process patients in parallel
    results = []
    with mp.Pool(processes=workers) as pool:
        # Use imap_unordered for better performance
        for result in pool.imap_unordered(process_func, patient_dirs):
            if result:
                results.append(result)
                logger.info(f"Progress: {len(results)}/{len(patient_dirs)} patients completed")
    
    logger.info(f"Completed processing {len(results)} patients")
    
    # Aggregate results
    logger.info("Aggregating results...")
    # Initialize mapping data - load existing or create new
    mapping_data = load_existing_mapping(output_dir, logger)
    
    total_duplicates_removed = 0
    total_files_copied = 0
    all_validation_failed = []
    
    for result in results:
        for patient_id, patient_data in result.items():
            # Extract and remove processing stats
            duplicates = patient_data.pop('duplicates_removed', 0)
            files_copied = patient_data.pop('files_copied', 0)
            validation_failed = patient_data.pop('validation_failed', [])
            
            total_duplicates_removed += duplicates
            total_files_copied += files_copied
            all_validation_failed.extend(validation_failed)
            
            # Store patient data
            mapping_data['patients'][patient_id] = patient_data
    
    # Create a mock FileOrganizer for stats (parallel version doesn't use single organizer)
    class FileOrganizerStats:
        def __init__(self, files_copied, validation_failed):
            self.files_copied = files_copied
            self.validation_failed = validation_failed
    
    file_organizer = FileOrganizerStats(total_files_copied, all_validation_failed)
    
    # Check completeness
    logger.info("Checking data completeness...")
    completeness_data = completeness_checker.generate_completeness_report(mapping_data)
    
    # Stop monitoring and collect metrics
    performance_metrics = None
    if monitor:
        monitor.stop()
        performance_metrics = monitor.get_metrics()
        logger.info("Performance monitoring stopped")
    
    return mapping_data, completeness_data, total_duplicates_removed, file_organizer, performance_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize UPENN-GBM dataset to BIDS format (nested folders aware)'
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
        '--batch-size',
        type=int,
        default=None,
        help='Process patients in batches of N (e.g., 10). Useful for incremental processing.'
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
    
    total_patients = len(all_patient_dirs)
    logger.info(f"Total patients to process: {total_patients}")
    
    # Create batches
    batch_size = args.batch_size if args.batch_size else total_patients
    batches = create_batches(all_patient_dirs, batch_size)
    
    logger.info(f"Processing in {len(batches)} batch(es) of up to {batch_size} patients each")
    logger.info("="*60)
    
    # Initialize or restore IDMapper ONCE for all batches
    existing_mapping = load_existing_mapping(args.output_dir, logger)
    id_mapper = restore_id_mapper(existing_mapping, logger)
    
    # Initialize aggregation lists
    all_deduplicators = []
    all_file_organizers = []
    all_performance_metrics = []
    
    # Initialize result variables (updated in batch loop)
    mapping_data = existing_mapping
    completeness_data = {}
    duplicates_removed = 0
    
    # Process each batch
    all_deduplicators = []
    all_file_organizers = []
    all_performance_metrics = []
    
    for batch_idx, batch_patient_dirs in enumerate(batches, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {batch_idx}/{len(batches)}: Processing {len(batch_patient_dirs)} patients")
        logger.info(f"{'='*60}\n")
        
        batch_start_time = datetime.now()
        
        if args.mode == 'sequential':
            # Process batch sequentially
            # Note: patient_dirs is already filtered in run_sequential
            mapping_data, completeness_data, deduplicator, file_organizer, performance_metrics = run_sequential(
                args.input_dir,
                args.output_dir,
                logger,
                patient_dirs=batch_patient_dirs,
                max_subjects=None,  # We already filtered
                benchmark=args.benchmark,
                results_dir=args.results_dir,
                mode=args.mode,
                workers=args.workers,
                force=args.force,
                dry_run=args.dry_run,
                id_mapper=id_mapper
            )
            all_deduplicators.append(deduplicator)
            all_file_organizers.append(file_organizer)
            
        else:
            # Process batch in parallel
            mapping_data, completeness_data, duplicates_removed, file_organizer, performance_metrics = run_parallel(
                args.input_dir,
                args.output_dir,
                logger,
                workers=args.workers,
                patient_dirs=batch_patient_dirs,
                max_subjects=None,  # We already filtered
                benchmark=args.benchmark,
                results_dir=args.results_dir,
                mode=args.mode,
                force=args.force,
                dry_run=args.dry_run,
                id_mapper=id_mapper
            )
            # Create mock deduplicator for consistency
            class DeduplicatorStats:
                def __init__(self, duplicates_removed):
                    self.duplicates_removed = duplicates_removed
            all_deduplicators.append(DeduplicatorStats(duplicates_removed))
            all_file_organizers.append(file_organizer)
        
        if performance_metrics:
            all_performance_metrics.append(performance_metrics)
        
        batch_end_time = datetime.now()
        batch_time = (batch_end_time - batch_start_time).total_seconds()
        
        logger.info(f"\nBatch {batch_idx} completed in {batch_time:.1f}s")
        
        # Save intermediate mapping after each batch
        logger.info("Saving intermediate mapping...")
        save_mapping_file(mapping_data, args.output_dir)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*60)
    logger.info("ALL BATCHES COMPLETED")
    logger.info("="*60)
    
    # Aggregate statistics from all batches
    total_duplicates_removed = sum(d.duplicates_removed for d in all_deduplicators)
    total_files_copied = sum(fo.files_copied for fo in all_file_organizers)
    all_validation_failed = []
    for fo in all_file_organizers:
        all_validation_failed.extend(fo.validation_failed)
    
    # Create aggregated objects for summary
    class AggregatedDeduplicator:
        def __init__(self, duplicates_removed):
            self.duplicates_removed = duplicates_removed
    
    class AggregatedFileOrganizer:
        def __init__(self, files_copied, validation_failed):
            self.files_copied = files_copied
            self.validation_failed = validation_failed
    
    dedup_stats = AggregatedDeduplicator(total_duplicates_removed)
    file_organizer_stats = AggregatedFileOrganizer(total_files_copied, all_validation_failed)
    
    # Aggregate performance metrics (average)
    performance_metrics = None
    if all_performance_metrics:
        # Filter valid metrics for averaging
        cpu_avg_values = [m.get('cpu_avg', 0) for m in all_performance_metrics if m.get('cpu_avg')]
        memory_avg_values = [m.get('memory_avg_mb', 0) for m in all_performance_metrics if m.get('memory_avg_mb')]
        cpu_max_values = [m.get('cpu_max', 0) for m in all_performance_metrics if m.get('cpu_max')]
        memory_peak_values = [m.get('memory_peak_mb', 0) for m in all_performance_metrics if m.get('memory_peak_mb')]
        
        performance_metrics = {
            'cpu_avg': sum(cpu_avg_values) / len(cpu_avg_values) if cpu_avg_values else 0.0,
            'cpu_max': max(cpu_max_values) if cpu_max_values else 0.0,
            'memory_avg_mb': sum(memory_avg_values) / len(memory_avg_values) if memory_avg_values else 0.0,
            'memory_peak_mb': max(memory_peak_values) if memory_peak_values else 0.0,
            'duration': total_time
        }

    # Save outputs
    logger.info("Saving mapping file...")
    save_mapping_file(mapping_data, args.output_dir)

    logger.info("Saving completeness report...")
    save_completeness_report(completeness_data, args.output_dir)

    # Save benchmark metrics
    if args.benchmark and performance_metrics:
        logger.info("Saving benchmark metrics...")
        
        # Create results directory
        args.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate derived metrics
        stats = completeness_data['statistics']
        total_patients = stats['total_patients']
        total_sessions = stats['total_sessions']
        
        # Count total series
        total_series = sum(
            len(patient_data['sessions'][session_id]['series'])
            for patient_data in mapping_data['patients'].values()
            for session_id in patient_data['sessions']
        )
        
        # Calculate basic timing metrics
        time_per_patient = total_time / total_patients if total_patients > 0 else 0
        throughput_patients = total_patients / total_time if total_time > 0 else 0
        
        # Get baseline time for speedup calculation
        benchmark_logger = BenchmarkLogger(args.results_dir)
        baseline_time = benchmark_logger.get_baseline_time()
        
        speedup = None
        efficiency = None
        if args.mode == 'sequential' and args.workers == 1:
            # This IS the baseline
            speedup = 1.0
            efficiency = 1.0
        elif baseline_time:
            # Compare to baseline
            speedup = baseline_time / time_per_patient
            efficiency = speedup / args.workers if args.workers > 0 else None
        
        # Create ExperimentMetrics object
        experiment_id = f"{args.mode}_w{args.workers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Count successful/failed/skipped (we don't track these separately now, so use totals)
        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            mode=args.mode if not args.dry_run else f"{args.mode}_dryrun",
            workers=args.workers,
            total_series=total_patients,  # Using patients as "series" for this script
            successful=total_patients,
            failed=0,
            skipped=0,
            total_time=total_time,
            time_per_series=time_per_patient,
            throughput=throughput_patients,
            cpu_avg=performance_metrics.get('cpu_avg'),
            cpu_max=performance_metrics.get('cpu_max'),
            memory_avg_mb=performance_metrics.get('memory_avg_mb'),
            memory_peak_mb=performance_metrics.get('memory_peak_mb'),
            speedup=speedup,
            efficiency=efficiency
        )
        
        # Save metrics
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


    # Create mock deduplicator for print_summary
    class DeduplicatorStats:
        def __init__(self, duplicates_removed):
            self.duplicates_removed = duplicates_removed
    
    dedup_stats = DeduplicatorStats(duplicates_removed)

    # Generate final completeness report for ALL patients in mapping
    logger.info("Generating final completeness report for all patients...")
    completeness_checker = CompletenessChecker(logger)
    completeness_data = completeness_checker.generate_completeness_report(mapping_data)
    
    # Save final reports
    if args.dry_run:
        logger.info("[DRY RUN] Would save mapping file...")
        logger.info("[DRY RUN] Would save completeness report...")
        print("\n📄 [DRY RUN] Reports not saved (use without --dry_run to save)")
    else:
        logger.info("Saving final mapping file...")
        save_mapping_file(mapping_data, args.output_dir)
        
        logger.info("Saving final completeness report...")
        save_completeness_report(completeness_data, args.output_dir)

    # Print summary
    print_summary(
        mapping_data,
        completeness_data,
        dedup_stats,
        file_organizer,
        total_time,
        performance_metrics,
        dry_run=args.dry_run
    )

    logger.info("=" * 60)
    logger.info("Dataset reorganization completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()