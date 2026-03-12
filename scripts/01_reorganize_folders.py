#!/usr/bin/env python3
"""
Stage 1: Reorganize DICOM dataset to BIDS format.

Enhanced version combining advanced multi-level modality detection with
robust pipeline infrastructure: incremental processing, validation,
batch processing, and comprehensive completeness reporting.

Supports:
  - UPENN-GBM dataset (folder-based date extraction)
  - Generic DICOM datasets (header-based date extraction)
  - Multi-level modality detection: protocol → description → technical params
  - Year-specific detection strategies for heterogeneous multi-center data
  - Scoring-based series deduplication
  - Incremental processing with resume capability
  - Parallel and sequential processing modes

Output:
  - BIDS directory structure: sub-XXX/ses-XXX/anat/MODALITY/*.dcm
  - dataset_mapping.json    — full ID and session mapping
  - incomplete_data.txt     — human-readable completeness report
  - modality_selection.json — (optional) detailed detection decisions
"""

import argparse
import json
import logging
import mmap
import re
import shutil
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import multiprocessing as mp

import pydicom
import pydicom.dataelem

from scripts.performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics


# ════════════════════════════════════════════════════════════════
# Data Classes
# ════════════════════════════════════════════════════════════════

@dataclass
class SeriesInfo:
    """Information about a single DICOM series."""
    original_path: Path
    patient_id: str
    date: str                                   # YYYYMMDD
    modality: Optional[str] = None
    slice_count: int = 0
    series_description: str = ""
    protocol_name: str = ""
    detection_score: float = 0.0
    detection_strategy: str = ""


@dataclass
class SessionInfo:
    """A session groups series acquired on the same date."""
    date: str                                   # YYYYMMDD
    series: Dict[str, 'SeriesInfo'] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
# DICOM Utility Functions
# ════════════════════════════════════════════════════════════════

# Tags we always need when reading headers
_HEADER_TAGS = [
    (0x0008, 0x0060),  # Modality
    (0x0008, 0x103E),  # SeriesDescription
    (0x0018, 0x1030),  # ProtocolName
    (0x0018, 0x0010),  # ContrastBolusAgent
    (0x0018, 0x1078),  # ContrastBolusStartTime
    (0x0008, 0x0020),  # StudyDate
    (0x0008, 0x0030),  # StudyTime
    (0x0020, 0x000D),  # StudyInstanceUID
    (0x0020, 0x000E),  # SeriesInstanceUID
    (0x0020, 0x0011),  # SeriesNumber
    (0x0020, 0x0013),  # InstanceNumber
    (0x0018, 0x0080),  # RepetitionTime
    (0x0018, 0x0081),  # EchoTime
    (0x0018, 0x0082),  # InversionTime
    (0x0010, 0x0020),  # PatientID
    (0x0008, 0x0008),  # ImageType
]


def normalize_dicom_text(value: Any) -> str:
    """Convert any DICOM value to a normalised lowercase string."""
    if value is None:
        return ""
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return " ".join(str(v).strip().lower() for v in value if v is not None)
    return str(value).strip().lower()


def safe_float(value: Any) -> Optional[float]:
    """Safely convert to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def get_dicom_value(ds: pydicom.Dataset, tag: Union[tuple, str],
                    default: Any = None) -> Any:
    """Safely extract a value from a DICOM dataset."""
    try:
        val = ds.get(tag, default)
        if val is default:
            return default
        if isinstance(val, pydicom.dataelem.DataElement):
            val = val.value
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace').strip()
        if isinstance(val, pydicom.uid.UID):
            val = str(val)
        return val
    except Exception:
        return default


@lru_cache(maxsize=4096)
def read_dicom_header(file_path: str) -> Optional[pydicom.Dataset]:
    """Read DICOM header with mmap + caching for speed."""
    try:
        with open(file_path, 'rb') as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                return pydicom.filereader.dcmread(
                    mm, stop_before_pixels=True, specific_tags=_HEADER_TAGS)
            finally:
                mm.close()
    except (ValueError, OSError):
        # Fallback for empty files or unsupported filesystems
        try:
            return pydicom.dcmread(
                file_path, stop_before_pixels=True,
                specific_tags=_HEADER_TAGS, force=True)
        except Exception:
            return None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# Modality Detection — Strategy Pattern
# ════════════════════════════════════════════════════════════════

class ModalityDetectionStrategy(ABC):
    """Base class for pluggable modality detection strategies."""

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def detect(self, text: str, has_contrast: bool,
               ds: Optional[pydicom.Dataset] = None) -> Optional[str]: ...

    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        return {}

    def get_priority(self) -> int:
        """Lower number = tried first."""
        return 100

    def is_applicable(self, study_year: Optional[int]) -> bool:
        return True

    def is_exclusive(self) -> bool:
        """If True and applicable, skip remaining strategies on miss."""
        return False


class StandardDetectionStrategy(ModalityDetectionStrategy):
    """
    Multi-level text-based detection (protocol + series description).

    Checks modalities from most specific (FLAIR) to least specific (T1).
    Uses forbidden-word filtering and separate contrast logic for T1/T1C.
    """

    CONFIGS: Dict[str, dict] = {
        't2fl': {
            'keywords':  ['flair', 'dark fluid', 't2-flair', 't2 flair'],
            'forbidden': ['mpr'],
            'scoring':   {'3d': 2.0, 'sense': 1.3, 'long': 1.2,
                          'brain': 1.1, 'view': 0.1, 'mpr': 0.1},
        },
        't1c': {
            'keywords':  ['t1'],
            'contrast_required': True,
            'contrast_keywords': [
                'post', 'gad', 'gadolinium', 'contrast', 'c+', '+c',
                'ce', 'enhanced', 'postcontrast', '_c', 'gd', 'c+t1',
                't1+c', 't1c+',
            ],
            'forbidden': ['mpr', 'dyn', 'pit', 'spir'],
            'scoring':   {'tfe': 3.0, 'tse': 2.0, '3d': 1.5, 'se': 1.0,
                          'brain': 1.1, 'axi': 1.3, 'sag': 1.2, 'cor': 1.1,
                          'mpr': 0.1},
        },
        't2': {
            'keywords':  ['t2', 't2w', 't2-tse', 't2_tse'],
            'forbidden': ['flair', 'dark fluid', 'mpr', 'pit'],
            'scoring':   {'tse': 2.0, 'sense': 1.3, 'axi': 1.3,
                          'brain': 1.1, 'sag': 1.2, 'cor': 1.1, 'mpr': 0.1},
        },
        't1': {
            'keywords':  ['t1', 't1w', 'mprage', 'spgr', 'tfe',
                          't1-tfe', 't1-tse'],
            'forbidden': ['post', 'gad', 'contrast', 'c+', '+c', 'ce',
                          'enhanced', 'mpr', 'thr', 'pit'],
            'scoring':   {'tfe': 3.0, 'tse': 2.0, '3d': 1.5, 'clear': 1.2,
                          'brain': 1.1, 'axi': 1.3, 'sag': 1.2, 'cor': 1.1,
                          'mpr': 0.1},
        },
    }

    # Checked in this order (most specific → least specific)
    _ORDER = ['t2fl', 't1c', 't2', 't1']

    def get_name(self) -> str:
        return "Standard"

    def get_priority(self) -> int:
        return 1000

    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        cfg = self.CONFIGS.get(modality, {})
        return cfg.get('scoring', {})

    def detect(self, text: str, has_contrast: bool,
               ds: Optional[pydicom.Dataset] = None) -> Optional[str]:
        if not text:
            return None
        text_lower = text.lower()

        for modality in self._ORDER:
            cfg = self.CONFIGS[modality]

            # Check forbidden words
            if any(fw in text_lower for fw in cfg.get('forbidden', [])):
                continue

            # Check required keywords
            has_kw = any(kw in text_lower for kw in cfg['keywords'])
            if not has_kw:
                continue

            # Contrast logic for T1C
            if cfg.get('contrast_required'):
                contrast_kws = cfg.get('contrast_keywords', [])
                text_has_contrast = any(ck in text_lower for ck in contrast_kws)
                if has_contrast or text_has_contrast:
                    return 't1c'
                continue  # keyword matched T1 but no contrast → skip t1c

            return modality

        return None


class TechnicalParamStrategy(ModalityDetectionStrategy):
    """
    Detect modality from TR / TE / TI when text-based methods fail.

    Thresholds are standard MRI physics ranges:
      FLAIR  — TI > 1500 ms
      T1     — TR < 1000, TE < 30 (no contrast)
      T1C    — TR < 1200, TE < 30 (contrast present)
      T2     — TR > 2000, TE > 70, TI absent or < 1500
    """

    def get_name(self) -> str:
        return "TechnicalParams"

    def get_priority(self) -> int:
        return 2000

    def detect(self, text: str, has_contrast: bool,
               ds: Optional[pydicom.Dataset] = None) -> Optional[str]:
        if ds is None:
            return None

        tr = safe_float(get_dicom_value(ds, (0x0018, 0x0080)))
        te = safe_float(get_dicom_value(ds, (0x0018, 0x0081)))
        ti = safe_float(get_dicom_value(ds, (0x0018, 0x0082)))

        # FLAIR: very long TI
        if ti and ti > 1500:
            return 't2fl'

        # T1C: contrast + short TR/TE
        if has_contrast and tr and te and tr < 1200 and te < 30:
            return 't1c'

        # T1: short TR/TE, no contrast
        if tr and te and tr < 1000 and te < 30 and not has_contrast:
            return 't1'

        # T2: long TR, long TE, not FLAIR
        if tr and te and tr > 2000 and te > 70:
            if not ti or ti < 1500:
                return 't2'

        return None


class YearSpecificStrategy(ModalityDetectionStrategy):
    """
    Year-specific keyword configs for multi-center datasets
    where naming conventions changed between acquisition years.
    """

    def __init__(self, years: Tuple[int, ...], name: str,
                 configs: Dict[str, dict]):
        self._years = years
        self._name = name
        self._configs = configs  # modality → {required, markers, alt_keywords, forbidden}

    def get_name(self) -> str:
        return self._name

    def get_priority(self) -> int:
        return 10

    def is_exclusive(self) -> bool:
        return True

    def is_applicable(self, study_year: Optional[int]) -> bool:
        return study_year is not None and study_year in self._years

    def get_scoring_weights(self, modality: str) -> Dict[str, float]:
        return self._configs.get(modality, {}).get('scoring_weights', {})

    def detect(self, text: str, has_contrast: bool,
               ds: Optional[pydicom.Dataset] = None) -> Optional[str]:
        if not text:
            return None
        text_lower = text.lower()

        # Check modalities in specificity order
        for modality in ['t2fl', 't1c', 't2', 't1']:
            rules = self._configs.get(modality)
            if not rules:
                continue
            if self._matches(text_lower, rules, has_contrast):
                return modality
        return None

    def _matches(self, text: str, rules: dict, has_contrast: bool) -> bool:
        # Forbidden
        if any(fw in text for fw in rules.get('forbidden', [])):
            return False

        # Required keywords (all must match)
        required = rules.get('required', [])
        if required:
            if not all(rw in text for rw in required):
                # Try alt_keywords
                alt = rules.get('alt_keywords', [])
                if alt:
                    base_kw = required[0] if required else None
                    if base_kw and base_kw not in text:
                        return False
                    contrast_alts = [k for k in alt if k != base_kw]
                    if not any(k in text for k in contrast_alts):
                        return False
                else:
                    return False

        # Markers (any must match)
        markers = rules.get('markers', [])
        if markers and not any(mk in text for mk in markers):
            return False

        return True


# ── Built-in year-specific configs (MS60 dataset) ──────────────

_YEAR_STRATEGY_DEFS = [
    {
        'years': (2018,),
        'name': 'Protocol-2018',
        'configs': {
            't1c': {'required': ['ce', 't1'], 'forbidden': [],
                    'scoring_weights': {'se': 2.0}},
            't1':  {'markers': ['t1w', 't1'], 'forbidden': ['thr', 'ce'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, 'se': 1.0}},
            't2':  {'markers': ['t2w', 't2'], 'forbidden': ['flair'],
                    'scoring_weights': {'tse': 2.0, 'tra': 1.5}},
            't2fl': {'markers': ['flair'], 'forbidden': [],
                     'scoring_weights': {'sense': 1.5, 'long': 1.2}},
        },
    },
    {
        'years': (2020,),
        'name': 'Protocol-2020',
        'configs': {
            't1c': {'required': ['ce', 't1'],
                    'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast',
                                     '+c', 'post', 'c+', '_c', 'gd'],
                    'forbidden': ['mpr'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5,
                                        'se': 1.0, 'clear': 1.5}},
            't1':  {'markers': ['t1w', 't1-tse', 't1'], 'forbidden': ['thr', 'mpr', 'ce'],
                    'scoring_weights': {'tfe': 2.5, 'tse': 2.0, 'clear': 1.5,
                                        '3d': 1.3}},
            't2':  {'markers': ['t2w', 't2-tse', 't2'], 'forbidden': ['mpr', 'flair'],
                    'scoring_weights': {'tse': 2.0, 'clear': 1.4, 'sense': 1.3}},
            't2fl': {'markers': ['flair'], 'forbidden': ['mpr'],
                     'scoring_weights': {'3d': 2.0, 'long': 1.2, 'sense': 1.1}},
        },
    },
    {
        'years': (2021, 2022),
        'name': 'Protocol-2021-2022',
        'configs': {
            't1c': {'required': ['ce', 't1'],
                    'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast',
                                     '+c', 'post', 'c+', '_c', 'gd'],
                    'forbidden': ['mpr'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}},
            't1':  {'markers': ['t1-tfe', 't1-tse', 't1w', 't1'],
                    'forbidden': ['mpr', 'ce', 'gd'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}},
            't2':  {'markers': ['t2-tse', 't2_tse', 't2'], 'forbidden': ['mpr', 'flair'],
                    'scoring_weights': {'tse': 2.0, 'axi': 1.2}},
            't2fl': {'markers': ['flair'], 'forbidden': ['mpr'],
                     'scoring_weights': {'3d': 2.0}},
        },
    },
    {
        'years': (2023, 2024, 2025, 2026),
        'name': 'Protocol-2023+',
        'configs': {
            't1c': {'required': ['ce', 't1'],
                    'alt_keywords': ['t1', 'contrast', 'gad', 'postcontrast',
                                     '+c', 'post', 'c+', '_c', 'gd'],
                    'forbidden': ['mpr', 'dyn', 'pit'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}},
            't1':  {'markers': ['t1-tfe', 't1-tse', 't1'],
                    'forbidden': ['mpr', 'ce'],
                    'scoring_weights': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5}},
            't2':  {'markers': ['t2-tse', 't2_tse', 't2'],
                    'forbidden': ['pit', 'mpr', 'flair'],
                    'scoring_weights': {'tse': 2.0, 'axi': 1.2}},
            't2fl': {'markers': ['flair'], 'forbidden': ['mpr'],
                     'scoring_weights': {'3d': 2.0}},
        },
    },
]


def _build_year_strategies() -> List[YearSpecificStrategy]:
    return [
        YearSpecificStrategy(
            years=tuple(d['years']),
            name=d['name'],
            configs=d['configs'],
        )
        for d in _YEAR_STRATEGY_DEFS
    ]


# ════════════════════════════════════════════════════════════════
# ModalityDetector — orchestrator
# ════════════════════════════════════════════════════════════════

# Anatomy penalties applied to scoring
_ANATOMY_PENALTIES = {
    'spine': 0.8, 'cervical': 0.8, 'pit': 0.7, 'pituitary': 0.7,
}


class ModalityDetector:
    """
    Enhanced modality detector with multi-level detection and scoring.

    Detection cascade:
      1. Year-specific strategy (if study year matches)
      2. Standard text-based detection (protocol + description keywords)
      3. Technical parameter detection (TR / TE / TI)

    Each detection also calculates a quality score used for deduplication.
    """

    def __init__(self, logger: logging.Logger, *,
                 enable_year_strategies: bool = True):
        self.logger = logger
        self._cache: Dict[Path, Tuple[Optional[str], str, float, str]] = {}

        # Build strategy chain ordered by priority
        self._strategies: List[ModalityDetectionStrategy] = []
        if enable_year_strategies:
            self._strategies.extend(_build_year_strategies())
        self._strategies.append(StandardDetectionStrategy())
        self._strategies.append(TechnicalParamStrategy())
        self._strategies.sort(key=lambda s: s.get_priority())

    @property
    def strategies(self) -> List[ModalityDetectionStrategy]:
        return list(self._strategies)

    def detect_modality(self, series_path: Path
                        ) -> Tuple[Optional[str], str, float, str]:
        """
        Detect modality for a series directory.

        Returns:
            (modality, description, score, strategy_name)
        """
        if series_path in self._cache:
            return self._cache[series_path]

        result = self._detect(series_path)
        self._cache[series_path] = result
        return result

    # ── internals ──────────────────────────────────────────────

    def _detect(self, series_path: Path
                ) -> Tuple[Optional[str], str, float, str]:
        # Find first DICOM
        dicom_files = sorted(
            f for f in series_path.rglob("*.dcm") if f.is_file())
        if not dicom_files:
            self.logger.warning(f"No DICOM files in {series_path}")
            return None, "", 0.0, ""

        ds = read_dicom_header(str(dicom_files[0]))
        if ds is None:
            # Fallback: read without mmap cache
            try:
                ds = pydicom.dcmread(
                    str(dicom_files[0]), stop_before_pixels=True, force=True)
            except Exception as exc:
                self.logger.warning(f"Cannot read {dicom_files[0]}: {exc}")
                return None, "", 0.0, ""

        # Extract text fields
        protocol_name = normalize_dicom_text(
            get_dicom_value(ds, (0x0018, 0x1030), ""))
        series_desc = normalize_dicom_text(
            get_dicom_value(ds, (0x0008, 0x103E), ""))
        combined = f"{protocol_name} {series_desc}".strip()

        readable_desc = (
            f"{ds.get('ProtocolName', 'N/A')} | "
            f"{ds.get('SeriesDescription', 'N/A')}")

        if not combined:
            self.logger.debug(f"Empty protocol/description in {dicom_files[0]}")
            return None, readable_desc, 0.0, ""

        # Determine contrast presence from multiple DICOM fields
        has_contrast = self._check_contrast(ds, combined)

        # Extract study year for year-specific strategies
        study_year = self._extract_study_year(ds)

        # Run strategy chain
        modality: Optional[str] = None
        strategy_name = ""

        for strategy in self._strategies:
            if not strategy.is_applicable(study_year):
                continue

            detected = strategy.detect(combined, has_contrast, ds)
            if detected:
                modality = detected
                strategy_name = strategy.get_name()
                self.logger.debug(
                    f"  {strategy_name} → {modality} for: {readable_desc}")
                break

            if strategy.is_exclusive():
                # Exclusive strategy was applicable but didn't match → stop
                self.logger.debug(
                    f"  Exclusive strategy {strategy.get_name()} "
                    f"did not match — skipping remaining")
                break

        if modality is None:
            self.logger.debug(f"  No modality matched for: {readable_desc}")
            return None, readable_desc, 0.0, ""

        # Calculate score
        score = self._calculate_score(combined, modality, strategy_name)

        return modality, readable_desc, score, strategy_name

    def _check_contrast(self, ds: pydicom.Dataset, text: str) -> bool:
        """Multi-field contrast detection."""
        # Field: ContrastBolusAgent (0018,0010)
        agent = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x0010), ""))
        if agent and agent not in ("", "none", "no"):
            return True

        # Field: ContrastBolusStartTime (0018,1078)
        bolus = normalize_dicom_text(get_dicom_value(ds, (0x0018, 0x1078), ""))
        if bolus and bolus not in ("", "none", "no"):
            return True

        # Text keywords
        contrast_kws = [
            'ce', 'contrast', 'gad', 'gadolinium', 'post',
            '+c', 'c+', 'enhanced', 'postcontrast',
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in contrast_kws)

    @staticmethod
    def _extract_study_year(ds: pydicom.Dataset) -> Optional[int]:
        study_date = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date, str) and len(study_date) >= 4:
            try:
                return int(study_date[:4])
            except ValueError:
                pass
        return None

    def _calculate_score(self, text: str, modality: str,
                         strategy_name: str) -> float:
        """Multiplicative scoring from strategy weights + anatomy bonuses."""
        score = 1.0
        text_lower = text.lower()

        # Collect weights from the matching strategy
        for strategy in self._strategies:
            if strategy.get_name() == strategy_name:
                weights = strategy.get_scoring_weights(modality)
                for keyword, weight in weights.items():
                    if keyword in text_lower:
                        score *= weight
                break

        # Brain bonus
        if 'brain' in text_lower:
            score *= 1.1

        # Anatomy penalties
        for anatomy, penalty in _ANATOMY_PENALTIES.items():
            if anatomy in text_lower:
                score *= penalty

        return round(score, 4)


# ════════════════════════════════════════════════════════════════
# IDMapper
# ════════════════════════════════════════════════════════════════

class IDMapper:
    """Maps original patient IDs to sequential BIDS IDs."""

    def __init__(self):
        self._counter = 0
        self._map: Dict[str, str] = {}

    def get_patient_id(self, original_id: str) -> str:
        if original_id not in self._map:
            self._counter += 1
            self._map[original_id] = f"sub-{self._counter:03d}"
        return self._map[original_id]

    def get_mapping(self) -> Dict[str, str]:
        return self._map.copy()


# ════════════════════════════════════════════════════════════════
# DatasetScanner
# ════════════════════════════════════════════════════════════════

class DatasetScanner:
    """Scan dataset for patient directories and series."""

    _SKIP_DIRS = frozenset([
        'logs', 'derivatives', 'sourcedata', 'code', '__pycache__',
    ])

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def scan_dataset(self, input_dir: Path,
                     patient_prefix: Optional[str] = None) -> List[Path]:
        """
        Find patient directories.

        If *patient_prefix* is given, only dirs starting with that prefix
        are returned.  Otherwise the prefix is auto-detected from the
        directory contents (e.g. ``UPENN-GBM-``).
        """
        if patient_prefix is None:
            patient_prefix = self._detect_prefix(input_dir)

        patient_dirs = []
        for entry in sorted(input_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith('.'):
                continue
            if entry.name in self._SKIP_DIRS:
                continue
            if patient_prefix and not entry.name.startswith(patient_prefix):
                continue
            patient_dirs.append(entry)

        self.logger.info(f"Found {len(patient_dirs)} patient directories"
                         + (f" (prefix: {patient_prefix!r})"
                            if patient_prefix else ""))
        return patient_dirs

    def scan_patient_series(
        self, patient_dir: Path
    ) -> List[Tuple[Path, Optional[str]]]:
        """
        Find series directories inside a patient folder.

        Returns list of ``(series_dir, date_string_or_None)`` where
        *date_string* is in ``YYYYMMDD`` format (extracted from ancestor
        folder names or, as fallback, from DICOM headers).
        """
        # Find every directory that directly contains ≥1 DICOM file
        series_dirs: Set[Path] = set()
        for dcm in patient_dir.rglob("*.dcm"):
            if dcm.is_file():
                series_dirs.add(dcm.parent)

        results: List[Tuple[Path, Optional[str]]] = []
        for sdir in sorted(series_dirs):
            # Try folder-name date (UPENN-GBM style: MM-DD-YYYY-…)
            date = self._find_date_in_ancestors(sdir, patient_dir)
            # Fallback: read StudyDate from first DICOM header
            if not date:
                date = self._read_date_from_dicom(sdir)
            results.append((sdir, date))

        return results

    # ── helpers ─────────────────────────────────────────────────

    def _find_date_in_ancestors(self, sdir: Path,
                                patient_dir: Path) -> Optional[str]:
        """Walk up from *sdir* to *patient_dir* looking for a date folder."""
        current = sdir
        while current != patient_dir and current != current.parent:
            d = self.parse_date_from_folder_name(current.name)
            if d:
                return d
            current = current.parent
        # Also check immediate children of patient_dir
        for child in patient_dir.iterdir():
            if child.is_dir() and child in sdir.parents:
                d = self.parse_date_from_folder_name(child.name)
                if d:
                    return d
        return None

    def _read_date_from_dicom(self, series_dir: Path) -> Optional[str]:
        """Read StudyDate from the first DICOM file in *series_dir*."""
        dcm_files = sorted(series_dir.rglob("*.dcm"))
        if not dcm_files:
            return None
        ds = read_dicom_header(str(dcm_files[0]))
        if ds is None:
            return None
        study_date = get_dicom_value(ds, (0x0008, 0x0020), "")
        if isinstance(study_date, str) and len(study_date) >= 8:
            return study_date[:8]
        return None

    @staticmethod
    def parse_date_from_folder_name(name: str) -> Optional[str]:
        """
        Extract date from a folder name.

        Supported formats:
          ``MM-DD-YYYY-…``  → ``YYYYMMDD``   (UPENN-GBM style)
          ``YYYYMMDD…``     → ``YYYYMMDD``
        """
        # MM-DD-YYYY (UPENN-GBM)
        m = re.match(r'(\d{2})-(\d{2})-(\d{4})', name)
        if m:
            month, day, year = m.groups()
            return f"{year}{month}{day}"
        # YYYYMMDD
        m = re.match(r'(\d{8})', name)
        if m:
            candidate = m.group(1)
            try:
                datetime.strptime(candidate, "%Y%m%d")
                return candidate
            except ValueError:
                pass
        return None

    @staticmethod
    def _detect_prefix(input_dir: Path) -> Optional[str]:
        dirs = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.'))
        if not dirs:
            return None
        known = ['UPENN-GBM-', 'sub-', 'patient-', 'PATIENT-', 'pat-']
        for prefix in known:
            if sum(d.name.startswith(prefix) for d in dirs) > len(dirs) * 0.5:
                return prefix
        return None


# ════════════════════════════════════════════════════════════════
# SessionGrouper
# ════════════════════════════════════════════════════════════════

class SessionGrouper:
    """Group series by acquisition date into sessions."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def group_by_date(self, series_list: List[SeriesInfo]) -> List[SessionInfo]:
        groups: Dict[str, List[SeriesInfo]] = defaultdict(list)
        for s in series_list:
            groups[s.date].append(s)

        sessions = []
        for date in sorted(groups):
            session = SessionInfo(date=date)
            for series in groups[date]:
                if series.modality:
                    if series.modality not in session.series:
                        session.series[series.modality] = []
                    lst = session.series[series.modality]
                    if not isinstance(lst, list):
                        session.series[series.modality] = [lst]
                    session.series[series.modality].append(series)
            sessions.append(session)

        return sessions


# ════════════════════════════════════════════════════════════════
# SeriesDeduplicator (enhanced: score → slice_count → newest)
# ════════════════════════════════════════════════════════════════

class SeriesDeduplicator:
    """Select best series when multiple exist for the same modality."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.duplicates_removed = 0

    def deduplicate_session(self, session: SessionInfo) -> SessionInfo:
        deduped = SessionInfo(date=session.date)

        for modality, series_list in session.series.items():
            if isinstance(series_list, list) and len(series_list) > 1:
                best = self._select_best(series_list)
                deduped.series[modality] = best
                removed = len(series_list) - 1
                self.duplicates_removed += removed
                self.logger.info(
                    f"  {modality}: {len(series_list)} candidates → "
                    f"selected {best.original_path.name} "
                    f"(score={best.detection_score:.2f}, "
                    f"slices={best.slice_count})")
            elif isinstance(series_list, list):
                deduped.series[modality] = series_list[0]
            else:
                deduped.series[modality] = series_list

        return deduped

    @staticmethod
    def _select_best(series_list: List[SeriesInfo]) -> SeriesInfo:
        """Primary: highest detection score.  Secondary: most slices."""
        for s in series_list:
            if s.slice_count == 0:
                s.slice_count = len(list(s.original_path.rglob("*.dcm")))
        return max(series_list, key=lambda s: (s.detection_score, s.slice_count))


# ════════════════════════════════════════════════════════════════
# CompletenessChecker
# ════════════════════════════════════════════════════════════════

REQUIRED_MODALITIES = {'t1', 't1c', 't2', 't2fl'}


class CompletenessChecker:
    """Check whether each session contains all four target modalities."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def check_session(self, session: SessionInfo) -> Tuple[bool, Set[str]]:
        available = set(session.series.keys())
        missing = REQUIRED_MODALITIES - available
        return len(missing) == 0, missing

    def generate_completeness_report(self, mapping_data: Dict) -> Dict:
        incomplete_patients: List[dict] = []
        complete_patients = complete_sessions = total_sessions = 0

        for patient_id, pdata in mapping_data['patients'].items():
            patient_issues = []
            for session_id, sdata in pdata['sessions'].items():
                total_sessions += 1
                available = set(sdata['series'].keys())
                missing = REQUIRED_MODALITIES - available
                if missing:
                    patient_issues.append({
                        'session_id': session_id,
                        'date': sdata['original_date'],
                        'missing': sorted(missing),
                        'available': sorted(available),
                    })
                else:
                    complete_sessions += 1
            if patient_issues:
                incomplete_patients.append({
                    'patient_id': patient_id,
                    'original_id': pdata['original_id'],
                    'incomplete_sessions': patient_issues,
                })
            else:
                complete_patients += 1

        return {
            'incomplete_patients': incomplete_patients,
            'statistics': {
                'total_patients': len(mapping_data['patients']),
                'complete_patients': complete_patients,
                'incomplete_patients': len(incomplete_patients),
                'total_sessions': total_sessions,
                'complete_sessions': complete_sessions,
                'incomplete_sessions': total_sessions - complete_sessions,
            },
        }


# ════════════════════════════════════════════════════════════════
# FileOrganizer
# ════════════════════════════════════════════════════════════════

_BIDS_SUFFIX = {'t1': 'T1w', 't1c': 'T1wCE', 't2': 'T2w', 't2fl': 'FLAIR'}


class FileOrganizer:
    """Create BIDS directories and copy/validate files."""

    def __init__(self, output_dir: Path, logger: logging.Logger,
                 dry_run: bool = False):
        self.output_dir = output_dir
        self.logger = logger
        self.dry_run = dry_run
        self.files_copied = 0
        self.files_would_copy = 0
        self.validation_failed: List[str] = []

    def create_bids_structure(self, patient_id: str, session_id: str,
                              modality: str) -> Path:
        target = self.output_dir / patient_id / session_id / 'anat' / modality
        if not self.dry_run:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def copy_series(self, source_dir: Path, target_dir: Path,
                    patient_id: str, session_id: str,
                    modality: str) -> int:
        source_files = sorted(
            f for f in source_dir.rglob("*.dcm") if f.is_file())
        if not source_files:
            self.logger.warning(f"No DICOM files in {source_dir}")
            return 0

        suffix = _BIDS_SUFFIX.get(modality, modality.upper())

        if self.dry_run:
            count = len(source_files)
            self.files_would_copy += count
            return count

        copied = 0
        for idx, src in enumerate(source_files, 1):
            dst = target_dir / f"{patient_id}_{session_id}_{suffix}_{idx:04d}.dcm"
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as exc:
                self.logger.error(f"Failed to copy {src}: {exc}")

        self.files_copied += copied
        return copied

    def validate_copy(self, source_dir: Path, target_dir: Path,
                      patient_id: str, session_id: str,
                      modality: str) -> bool:
        if self.dry_run:
            return True
        src_count = len(list(source_dir.rglob("*.dcm")))
        tgt_count = len(list(target_dir.glob("*.dcm")))
        if src_count != tgt_count:
            msg = (f"{patient_id}/{session_id}/{modality}: "
                   f"slice mismatch (src={src_count}, tgt={tgt_count})")
            self.logger.error(msg)
            self.validation_failed.append(msg)
            return False
        return True


# ════════════════════════════════════════════════════════════════
# Logging Setup
# ════════════════════════════════════════════════════════════════

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger('reorganize_folders')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    return logger


# ════════════════════════════════════════════════════════════════
# Persistence Helpers
# ════════════════════════════════════════════════════════════════

def save_mapping_file(mapping_data: Dict, output_dir: Path):
    with open(output_dir / 'dataset_mapping.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)


def save_completeness_report(data: Dict, output_dir: Path):
    path = output_dir / 'incomplete_data.txt'
    stats = data['statistics']

    with open(path, 'w') as f:
        f.write("=== Incomplete Data Report ===\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        tp = stats['total_patients'] or 1
        ts = stats['total_sessions'] or 1

        f.write(f"Total patients: {stats['total_patients']}\n")
        f.write(f"Complete patients: {stats['complete_patients']} "
                f"({stats['complete_patients']/tp*100:.1f}%)\n")
        f.write(f"Incomplete patients: {stats['incomplete_patients']} "
                f"({stats['incomplete_patients']/tp*100:.1f}%)\n\n")

        f.write(f"Total sessions: {stats['total_sessions']}\n")
        f.write(f"Complete sessions: {stats['complete_sessions']} "
                f"({stats['complete_sessions']/ts*100:.1f}%)\n")
        f.write(f"Incomplete sessions: {stats['incomplete_sessions']} "
                f"({stats['incomplete_sessions']/ts*100:.1f}%)\n\n")

        if data['incomplete_patients']:
            f.write("=== Details ===\n\n")
            for patient in data['incomplete_patients']:
                f.write(f"Patient: {patient['patient_id']} "
                        f"({patient['original_id']})\n")
                for sess in patient['incomplete_sessions']:
                    f.write(f"  Session: {sess['session_id']} "
                            f"({sess['date']})\n")
                    f.write(f"    Missing:   {', '.join(sess['missing'])}\n")
                    f.write(f"    Available: {', '.join(sess['available'])}\n")
                f.write("\n")


def save_selection_log(log_entries: List[dict], output_dir: Path):
    """Save detailed modality-selection decisions to JSON."""
    path = output_dir / 'modality_selection.json'
    with open(path, 'w') as f:
        json.dump(log_entries, f, indent=2)


def load_existing_mapping(output_dir: Path,
                          logger: logging.Logger) -> Dict:
    mapping_file = output_dir / 'dataset_mapping.json'
    if not mapping_file.exists():
        logger.info("No existing mapping — starting fresh")
        return {
            'patients': {},
            'output_dir': str(output_dir),
            'created_at': datetime.now().isoformat(),
        }
    try:
        with open(mapping_file) as f:
            data = json.load(f)
        logger.info(f"Loaded existing mapping with "
                    f"{len(data.get('patients', {}))} patients")
        data['updated_at'] = datetime.now().isoformat()
        return data
    except Exception as exc:
        logger.warning(f"Failed to load mapping ({exc}), starting fresh")
        return {
            'patients': {},
            'output_dir': str(output_dir),
            'created_at': datetime.now().isoformat(),
        }


def restore_id_mapper(mapping_data: Dict,
                      logger: logging.Logger) -> IDMapper:
    mapper = IDMapper()
    if not mapping_data.get('patients'):
        logger.info("No existing patients, starting from sub-001")
        return mapper

    for new_id, pdata in mapping_data['patients'].items():
        mapper._map[pdata['original_id']] = new_id
        num = int(new_id.replace('sub-', ''))
        if num > mapper._counter:
            mapper._counter = num

    logger.info(f"Restored IDMapper: {len(mapper._map)} patients, "
                f"next → sub-{mapper._counter + 1:03d}")
    return mapper


def check_patient_exists(output_dir: Path, patient_id: str,
                         force: bool, dry_run: bool,
                         logger: logging.Logger) -> bool:
    if dry_run:
        return False
    pdir = output_dir / patient_id
    if not pdir.exists():
        return False
    if force:
        logger.info(f"  --force: removing existing {patient_id}")
        try:
            shutil.rmtree(pdir)
            return False
        except Exception as exc:
            logger.error(f"  Cannot delete {pdir}: {exc}")
            return True
    logger.info(f"  Skipping {patient_id}: already exists "
                f"(use --force to reprocess)")
    return True


def create_batches(items: list, batch_size: Optional[int]) -> List[list]:
    if not batch_size or batch_size <= 0:
        return [items]
    return [items[i:i + batch_size]
            for i in range(0, len(items), batch_size)]


# ════════════════════════════════════════════════════════════════
# Summary Printer
# ════════════════════════════════════════════════════════════════

def print_summary(mapping_data, completeness_data, dedup_stats,
                  file_organizer, total_time,
                  performance_metrics=None, dry_run=False):
    stats = completeness_data['statistics']

    print("\n" + "=" * 60)
    print("=== Reorganization Summary ===")
    print("=" * 60)

    print(f"\n  Patients processed: {stats['total_patients']}")
    print(f"  Sessions created:   {stats['total_sessions']}")

    mc = defaultdict(int)
    ms = defaultdict(int)
    for pdata in mapping_data['patients'].values():
        for sdata in pdata['sessions'].values():
            for mod, sinfo in sdata['series'].items():
                mc[mod] += 1
                ms[mod] += sinfo.get('slice_count', 0)

    print("  Series by modality:")
    for mod in ['t1', 't1c', 't2', 't2fl']:
        print(f"    {mod:4s}: {mc.get(mod,0):3d} series "
              f"({ms.get(mod,0):,} slices)")

    print(f"  Duplicates removed: {dedup_stats.duplicates_removed}")

    if dry_run:
        total_f = getattr(file_organizer, 'files_would_copy', 0)
        print(f"  Files (would copy): {total_f:,}")
    else:
        print(f"  Files copied:       {file_organizer.files_copied:,}")

    tp = stats['total_patients'] or 1
    ts = stats['total_sessions'] or 1
    print(f"\n  Complete patients: "
          f"{stats['complete_patients']}/{stats['total_patients']} "
          f"({stats['complete_patients']/tp*100:.1f}%)")
    print(f"  Complete sessions: "
          f"{stats['complete_sessions']}/{stats['total_sessions']} "
          f"({stats['complete_sessions']/ts*100:.1f}%)")

    if completeness_data['incomplete_patients']:
        print(f"\n  Incomplete: {stats['incomplete_patients']} patients")
        for pat in completeness_data['incomplete_patients'][:5]:
            print(f"    - {pat['patient_id']} ({pat['original_id']})")
            for sess in pat['incomplete_sessions']:
                print(f"        {sess['session_id']}: "
                      f"missing {', '.join(sess['missing'])}")
        rem = len(completeness_data['incomplete_patients']) - 5
        if rem > 0:
            print(f"    ... and {rem} more")

    if not file_organizer.validation_failed:
        print("\n  Validation: all series passed")
    else:
        print(f"\n  Validation: {len(file_organizer.validation_failed)} "
              f"series FAILED — see log")

    print(f"\n  Total time: {total_time:.1f}s")
    if stats['total_patients'] and total_time > 0:
        print(f"  Throughput: {stats['total_patients']/total_time:.2f} "
              f"patients/sec")
    if performance_metrics:
        if performance_metrics.get('cpu_avg'):
            print(f"  CPU:    avg {performance_metrics['cpu_avg']:.1f}%, "
                  f"peak {performance_metrics['cpu_max']:.1f}%")
        if performance_metrics.get('memory_avg_mb'):
            print(f"  Memory: avg {performance_metrics['memory_avg_mb']:.1f}MB, "
                  f"peak {performance_metrics['memory_peak_mb']:.1f}MB")

    print(f"\n  Output:  {mapping_data.get('output_dir', 'N/A')}")
    print(f"  Reports: dataset_mapping.json, incomplete_data.txt")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
# Per-Patient Processing (for parallel mode)
# ════════════════════════════════════════════════════════════════

def process_single_patient(
    patient_dir: Path,
    input_dir: Path,
    output_dir: Path,
    patient_mapping: Dict[str, str],
    log_level: int = logging.INFO,
    force: bool = False,
    dry_run: bool = False,
    enable_year_strategies: bool = True,
) -> Optional[Dict]:
    """Process one patient — designed to run in a worker process."""
    logger = logging.getLogger(f'patient_{patient_dir.name}')
    logger.setLevel(log_level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(h)

    try:
        original_id = patient_dir.name
        new_id = patient_mapping[original_id]
        if check_patient_exists(output_dir, new_id, force, dry_run, logger):
            return None

        detector = ModalityDetector(logger,
                                    enable_year_strategies=enable_year_strategies)
        scanner = DatasetScanner(logger)
        grouper = SessionGrouper(logger)
        deduplicator = SeriesDeduplicator(logger)
        organizer = FileOrganizer(output_dir, logger, dry_run=dry_run)

        return _process_patient_core(
            patient_dir, original_id, new_id,
            detector, scanner, grouper, deduplicator, organizer, logger)

    except Exception as exc:
        logger.error(f"Failed to process {patient_dir.name}: {exc}",
                     exc_info=True)
        return None


def _process_patient_core(
    patient_dir, original_id, new_id,
    detector, scanner, grouper, deduplicator, organizer, logger,
) -> Optional[Dict]:
    """Shared core logic used by both sequential and parallel paths."""
    logger.info(f"Processing {original_id} -> {new_id}")

    series_entries = scanner.scan_patient_series(patient_dir)
    logger.debug(f"  Found {len(series_entries)} series entries")

    series_list: List[SeriesInfo] = []
    selection_log: List[dict] = []

    for series_dir, date_str in series_entries:
        if not date_str:
            logger.warning(f"  No date for {series_dir} — skipping")
            continue

        modality, desc, score, strategy = detector.detect_modality(series_dir)

        if modality in REQUIRED_MODALITIES:
            info = SeriesInfo(
                original_path=series_dir,
                patient_id=original_id,
                date=date_str,
                modality=modality,
                series_description=desc,
                detection_score=score,
                detection_strategy=strategy,
            )
            info.slice_count = len(list(series_dir.rglob("*.dcm")))
            series_list.append(info)
            logger.debug(f"  {series_dir.name}: {modality} "
                         f"(score={score:.2f}, slices={info.slice_count})")

            selection_log.append({
                'patient': new_id,
                'series_dir': str(series_dir.name),
                'modality': modality,
                'description': desc,
                'score': score,
                'strategy': strategy,
            })
        elif modality:
            logger.debug(f"  {series_dir.name}: {modality} (filtered)")
        else:
            logger.debug(f"  {series_dir.name}: unknown modality")

    if not series_list:
        logger.warning(f"  No valid series for {original_id}")
        return None

    sessions = grouper.group_by_date(series_list)
    sessions = [deduplicator.deduplicate_session(s) for s in sessions]

    patient_data: Dict = {
        'original_id': original_id,
        'sessions': {},
        'duplicates_removed': deduplicator.duplicates_removed,
        'files_copied': 0,
        'validation_failed': [],
        'selection_log': selection_log,
    }

    for idx, session in enumerate(sessions, 1):
        ses_id = f"ses-{idx:03d}"
        ses_data: Dict = {'original_date': session.date, 'series': {}}

        for modality, series in session.series.items():
            target = organizer.create_bids_structure(new_id, ses_id, modality)
            copied = organizer.copy_series(
                series.original_path, target, new_id, ses_id, modality)
            organizer.validate_copy(
                series.original_path, target, new_id, ses_id, modality)

            ses_data['series'][modality] = {
                'original_path': str(series.original_path),
                'slice_count': copied,
                'series_description': series.series_description,
                'detection_score': series.detection_score,
                'detection_strategy': series.detection_strategy,
            }

        patient_data['sessions'][ses_id] = ses_data

    patient_data['files_copied'] = organizer.files_copied
    patient_data['validation_failed'] = organizer.validation_failed
    logger.info(f"Completed {original_id}")
    return {new_id: patient_data}


# ════════════════════════════════════════════════════════════════
# Sequential Processing
# ════════════════════════════════════════════════════════════════

def run_sequential(
    input_dir, output_dir, logger, *,
    patient_dirs=None, max_subjects=None, benchmark=False,
    results_dir=None, mode='sequential', workers=1,
    force=False, dry_run=False, id_mapper=None,
    enable_year_strategies=True,
):
    detector = ModalityDetector(logger,
                                enable_year_strategies=enable_year_strategies)
    if id_mapper is None:
        id_mapper = IDMapper()
    scanner = DatasetScanner(logger)
    grouper = SessionGrouper(logger)
    deduplicator = SeriesDeduplicator(logger)
    completeness_checker = CompletenessChecker(logger)
    file_organizer = FileOrganizer(output_dir, logger, dry_run=dry_run)

    mapping_data = load_existing_mapping(output_dir, logger)

    monitor = None
    if benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()

    if patient_dirs is None:
        patient_dirs = scanner.scan_dataset(input_dir)
        if max_subjects:
            patient_dirs = patient_dirs[:max_subjects]

    processed = skipped = 0

    for idx, patient_dir in enumerate(patient_dirs, 1):
        original_id = patient_dir.name
        new_id = id_mapper.get_patient_id(original_id)
        logger.info(f"[{idx}/{len(patient_dirs)}] "
                    f"{original_id} -> {new_id}")

        if check_patient_exists(output_dir, new_id, force, dry_run, logger):
            skipped += 1
            continue

        result = _process_patient_core(
            patient_dir, original_id, new_id,
            detector, scanner, grouper, deduplicator, file_organizer, logger)

        if result:
            for pid, pdata in result.items():
                # Remove transient keys before storing
                pdata.pop('duplicates_removed', None)
                pdata.pop('files_copied', None)
                pdata.pop('validation_failed', None)
                sel_log = pdata.pop('selection_log', [])
                mapping_data['patients'][pid] = pdata
            processed += 1

    completeness_data = completeness_checker.generate_completeness_report(
        mapping_data)

    perf = None
    if monitor:
        perf = monitor.get_metrics()

    logger.info(f"Run stats: {processed} processed, {skipped} skipped, "
                f"{len(mapping_data['patients'])} total in mapping")

    return mapping_data, completeness_data, deduplicator, file_organizer, perf


# ════════════════════════════════════════════════════════════════
# Parallel Processing
# ════════════════════════════════════════════════════════════════

def run_parallel(
    input_dir, output_dir, logger, *,
    workers=4, patient_dirs=None, max_subjects=None,
    benchmark=False, results_dir=None, mode='parallel',
    force=False, dry_run=False, id_mapper=None,
    enable_year_strategies=True,
):
    monitor = None
    if benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()

    scanner = DatasetScanner(logger)
    if id_mapper is None:
        id_mapper = IDMapper()
    completeness_checker = CompletenessChecker(logger)

    if patient_dirs is None:
        patient_dirs = scanner.scan_dataset(input_dir)
        if max_subjects:
            patient_dirs = patient_dirs[:max_subjects]

    # Pre-create mapping (must be before fork)
    patient_mapping = {
        pd.name: id_mapper.get_patient_id(pd.name) for pd in patient_dirs}

    logger.info(f"Parallel: {len(patient_dirs)} patients, {workers} workers")

    func = partial(
        process_single_patient,
        input_dir=input_dir,
        output_dir=output_dir,
        patient_mapping=patient_mapping,
        log_level=logging.WARNING,
        force=force,
        dry_run=dry_run,
        enable_year_strategies=enable_year_strategies,
    )

    results = []
    with mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(func, patient_dirs):
            if result:
                results.append(result)
                logger.info(f"  Progress: {len(results)}/{len(patient_dirs)}")

    # Aggregate
    mapping_data = load_existing_mapping(output_dir, logger)
    total_dup = total_copied = 0
    all_vfail: List[str] = []

    for result in results:
        for pid, pdata in result.items():
            total_dup += pdata.pop('duplicates_removed', 0)
            total_copied += pdata.pop('files_copied', 0)
            all_vfail.extend(pdata.pop('validation_failed', []))
            pdata.pop('selection_log', None)
            mapping_data['patients'][pid] = pdata

    class _Stats:
        def __init__(self, dup, copied, vfail):
            self.duplicates_removed = dup
            self.files_copied = copied
            self.validation_failed = vfail

    file_stats = _Stats(total_dup, total_copied, all_vfail)
    completeness_data = completeness_checker.generate_completeness_report(
        mapping_data)

    perf = None
    if monitor:
        monitor.stop()
        perf = monitor.get_metrics()

    return (mapping_data, completeness_data, total_dup,
            file_stats, perf)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Reorganize DICOM dataset to BIDS format '
                    '(enhanced modality detection)')
    parser.add_argument('input_dir', type=Path,
                        help='Input DICOM directory')
    parser.add_argument('output_dir', type=Path,
                        help='Output BIDS directory')
    parser.add_argument('--mode', choices=['sequential', 'parallel'],
                        default='sequential',
                        help='Processing mode (default: sequential)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--log_file', type=Path, default=None,
                        help='Log file path')
    parser.add_argument('--dry_run', action='store_true',
                        help='Simulate without copying files')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Process at most N patients (for testing)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Process in batches of N')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess existing patients')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enable performance benchmarking')
    parser.add_argument('--results_dir', type=Path, default=Path('results'),
                        help='Benchmark results directory')
    parser.add_argument('--patient-prefix', type=str, default=None,
                        help='Patient directory prefix filter '
                             '(auto-detected if omitted)')
    parser.add_argument('--no-year-strategies', action='store_true',
                        help='Disable year-specific detection strategies')
    parser.add_argument('--list-strategies', action='store_true',
                        help='List detection strategies and exit')

    args = parser.parse_args()

    # ── List strategies ────────────────────────────────────────
    if args.list_strategies:
        print("\nModality Detection Strategies:")
        print("-" * 50)
        strategies = _build_year_strategies()
        strategies.append(StandardDetectionStrategy())
        strategies.append(TechnicalParamStrategy())
        strategies.sort(key=lambda s: s.get_priority())
        for s in strategies:
            extra = ""
            if isinstance(s, YearSpecificStrategy):
                extra = f"  years={s._years}, exclusive"
            print(f"  [{s.get_priority():>5d}] {s.get_name()}{extra}")
        print("\nStrategies are tried in priority order (lower = first).")
        sys.exit(0)

    # ── Validate ───────────────────────────────────────────────
    if not args.input_dir.exists():
        print(f"Error: input directory does not exist: {args.input_dir}")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file is None:
        args.log_file = args.output_dir / 'reorganize.log'
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Stage 1: DICOM → BIDS Reorganization")
    logger.info("=" * 60)
    logger.info(f"Input:   {args.input_dir}")
    logger.info(f"Output:  {args.output_dir}")
    logger.info(f"Mode:    {args.mode} (workers={args.workers})")
    logger.info(f"Dry run: {args.dry_run}")

    if args.dry_run:
        logger.warning("DRY RUN — no files will be copied")

    enable_year = not args.no_year_strategies

    # ── Scan ───────────────────────────────────────────────────
    start_time = datetime.now()
    scanner = DatasetScanner(logger)
    all_patient_dirs = scanner.scan_dataset(args.input_dir,
                                            args.patient_prefix)
    if args.max_subjects:
        all_patient_dirs = all_patient_dirs[:args.max_subjects]

    batches = create_batches(all_patient_dirs,
                             args.batch_size or len(all_patient_dirs))
    logger.info(f"{len(all_patient_dirs)} patients in "
                f"{len(batches)} batch(es)")

    existing_mapping = load_existing_mapping(args.output_dir, logger)
    id_mapper = restore_id_mapper(existing_mapping, logger)

    # ── Process batches ────────────────────────────────────────
    all_dedup = []
    all_fo = []
    all_perf = []
    mapping_data = existing_mapping
    completeness_data: Dict = {}
    duplicates_removed = 0

    for bi, batch_dirs in enumerate(batches, 1):
        logger.info(f"\nBatch {bi}/{len(batches)}: "
                    f"{len(batch_dirs)} patients")
        batch_t0 = datetime.now()

        if args.mode == 'sequential':
            (mapping_data, completeness_data,
             dedup, fo, perf) = run_sequential(
                args.input_dir, args.output_dir, logger,
                patient_dirs=batch_dirs, benchmark=args.benchmark,
                results_dir=args.results_dir, mode=args.mode,
                force=args.force, dry_run=args.dry_run,
                id_mapper=id_mapper,
                enable_year_strategies=enable_year,
            )
            all_dedup.append(dedup)
            all_fo.append(fo)
        else:
            (mapping_data, completeness_data,
             dup_rem, fo, perf) = run_parallel(
                args.input_dir, args.output_dir, logger,
                workers=args.workers, patient_dirs=batch_dirs,
                benchmark=args.benchmark, results_dir=args.results_dir,
                mode=args.mode, force=args.force, dry_run=args.dry_run,
                id_mapper=id_mapper,
                enable_year_strategies=enable_year,
            )

            class _DedupStats:
                def __init__(self, n):
                    self.duplicates_removed = n
            all_dedup.append(_DedupStats(dup_rem))
            all_fo.append(fo)

        if perf:
            all_perf.append(perf)

        batch_dt = (datetime.now() - batch_t0).total_seconds()
        logger.info(f"Batch {bi} done in {batch_dt:.1f}s")

        # Intermediate save
        save_mapping_file(mapping_data, args.output_dir)

    # ── Final reports ──────────────────────────────────────────
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    total_dup = sum(d.duplicates_removed for d in all_dedup)
    total_copied = sum(fo.files_copied for fo in all_fo)
    all_vfail = [msg for fo in all_fo for msg in fo.validation_failed]

    class _AggDedup:
        def __init__(self, n):
            self.duplicates_removed = n

    class _AggFO:
        def __init__(self, c, v):
            self.files_copied = c
            self.files_would_copy = c
            self.validation_failed = v

    dedup_stats = _AggDedup(total_dup)
    fo_stats = _AggFO(total_copied, all_vfail)

    # Regenerate completeness from final mapping
    checker = CompletenessChecker(logger)
    completeness_data = checker.generate_completeness_report(mapping_data)

    # Aggregate performance
    performance_metrics = None
    if all_perf:
        performance_metrics = {
            'cpu_avg': sum(p.get('cpu_avg', 0) for p in all_perf) / len(all_perf),
            'cpu_max': max((p.get('cpu_max', 0) for p in all_perf), default=0),
            'memory_avg_mb': sum(p.get('memory_avg_mb', 0) for p in all_perf) / len(all_perf),
            'memory_peak_mb': max((p.get('memory_peak_mb', 0) for p in all_perf), default=0),
            'duration': total_time,
        }

    if not args.dry_run:
        save_mapping_file(mapping_data, args.output_dir)
        save_completeness_report(completeness_data, args.output_dir)
    else:
        logger.info("[DRY RUN] Reports not saved")

    # ── Benchmark metrics ──────────────────────────────────────
    if args.benchmark and performance_metrics:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        stats = completeness_data['statistics']
        tp = stats['total_patients'] or 1
        ts = stats['total_sessions']
        tpp = total_time / tp
        throughput = tp / total_time if total_time > 0 else 0

        bl = BenchmarkLogger(args.results_dir)
        baseline = bl.get_baseline_time()
        speedup = efficiency = None
        if args.mode == 'sequential' and args.workers == 1:
            speedup = efficiency = 1.0
        elif baseline:
            speedup = baseline / tpp
            efficiency = speedup / args.workers if args.workers else None

        metrics = ExperimentMetrics(
            experiment_id=(f"{args.mode}_w{args.workers}_"
                           f"{datetime.now():%Y%m%d_%H%M%S}"),
            timestamp=datetime.now().isoformat(),
            mode=f"{args.mode}{'_dryrun' if args.dry_run else ''}",
            workers=args.workers,
            total_series=tp,
            successful=tp, failed=0, skipped=0,
            total_time=total_time,
            time_per_series=tpp,
            throughput=throughput,
            cpu_avg=performance_metrics.get('cpu_avg'),
            cpu_max=performance_metrics.get('cpu_max'),
            memory_avg_mb=performance_metrics.get('memory_avg_mb'),
            memory_peak_mb=performance_metrics.get('memory_peak_mb'),
            speedup=speedup, efficiency=efficiency,
        )
        bl.log_metrics(metrics)
        logger.info(f"Benchmark saved to {args.results_dir / 'metrics.csv'}")

    print_summary(mapping_data, completeness_data, dedup_stats,
                  fo_stats, total_time, performance_metrics,
                  dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Stage 1 completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()