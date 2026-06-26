import logging
import sys
import importlib.util
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
DATASET_DIR = PROJ_ROOT / "data" / "MS_5" / "P000915"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_integration")

from utils.config_loader import load_series_scoring_config

pytestmark = pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="data/MS_5/P000915 fixture dataset not present in this checkout",
)


def _build_sessions():
    logger = logging.getLogger("test_ms5_integration")
    scoring_config = load_series_scoring_config()
    scanner = reorganize_mod.DatasetScanner(logger)
    detector = reorganize_mod.ModalityDetector(logger, scoring_config=scoring_config)
    grouper = reorganize_mod.SessionGrouper(logger)
    dedup = reorganize_mod.SeriesDeduplicator(logger, scoring_config=scoring_config)

    series_list = []
    for series_dir, date_folder_name in scanner.scan_patient_series(DATASET_DIR):
        date = scanner.parse_date_from_series_name(date_folder_name) if date_folder_name else None
        if not date:
            date = scanner.parse_date_from_series_name(series_dir.name)
        if not date:
            continue
        modality, desc, tech_meta = detector.detect_modality(series_dir)
        if modality not in reorganize_mod.MODALITY_BIDS_SUFFIX:
            continue
        series_list.append(reorganize_mod.SeriesInfo(
            original_path=series_dir, patient_id=DATASET_DIR.name, date=date,
            modality=modality, series_description=desc,
            slice_thickness_mm=tech_meta.get('slice_thickness_mm'),
            ti_ms=tech_meta.get('ti_ms'),
            has_contrast=tech_meta.get('has_contrast', False),
        ))

    sessions = grouper.group_by_date(series_list)
    return [dedup.deduplicate_session(s) for s in sessions]


class TestMS5RealDataIntegration:
    def test_two_sessions_found(self):
        sessions = _build_sessions()
        assert len(sessions) == 2

    def test_t1c_selects_3d_not_mpr(self):
        # series_dir.name is just the bare numeric folder ID (e.g. "1101");
        # the actual protocol text lives in series_description
        # ("ProtocolName | SeriesDescription"), read from DICOM tags.
        sessions = _build_sessions()
        for session in sessions:
            t1c = session.series.get('t1c')
            assert t1c is not None, f"session {session.date} lost t1c entirely"
            assert 'mpr' not in t1c.series_description.lower()

    def test_t1_selects_3d_not_mpr(self):
        sessions = _build_sessions()
        for session in sessions:
            t1 = session.series.get('t1')
            assert t1 is not None, f"session {session.date} lost t1 entirely"
            assert 'mpr' not in t1.series_description.lower()

    def test_flair_selects_3d_not_mpr(self):
        sessions = _build_sessions()
        for session in sessions:
            flair = session.series.get('t2fl')
            assert flair is not None, f"session {session.date} lost t2fl entirely"
            assert 'mpr' not in flair.series_description.lower()

    def test_no_spine_series_in_any_modality_group(self):
        sessions = _build_sessions()
        for session in sessions:
            for modality, series in session.series.items():
                desc_lower = series.series_description.lower()
                assert 'spine' not in desc_lower
                assert 'cervical' not in desc_lower

    def test_t2_selects_brain_series_not_cspine(self):
        sessions = _build_sessions()
        for session in sessions:
            t2 = session.series.get('t2')
            assert t2 is not None, f"session {session.date} lost t2 entirely"
            assert 'spine' not in t2.series_description.lower()
