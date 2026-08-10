# tests/stage01/test_excluded_series_in_patient_data.py
import logging
import sys
import importlib.util
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_excluded_patientdata")


class TestExcludedSeriesInPatientData:
    def test_incomplete_session_has_status_and_unrecognized_excluded_series(
        self, make_dicom_series, tmp_path
    ):
        patient_dir = tmp_path / "PAT001"
        make_dicom_series(
            patient_dir / "2023-01-01" / "t1_series",
            protocol_name="t1_mprage_sag", series_description="t1_mprage_sag",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "t2_series",
            protocol_name="t2_tse_tra", series_description="t2_tse_tra",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "flair_series",
            protocol_name="t2_flair", series_description="t2_flair",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "weird_series",
            protocol_name="xyz_47_unknown", series_description="xyz_47_unknown",
            image_type=['ORIGINAL', 'PRIMARY'],
        )

        logger = logging.getLogger("test_excluded_patientdata")
        modality_detector = reorganize_mod.ModalityDetector(logger)
        scanner = reorganize_mod.DatasetScanner(logger)
        grouper = reorganize_mod.SessionGrouper(logger)
        deduplicator = reorganize_mod.SeriesDeduplicator(logger)
        completeness_checker = reorganize_mod.CompletenessChecker(logger, lesion_type="glioblastoma")
        file_organizer = reorganize_mod.FileOrganizer(tmp_path / "output", logger)

        patient_data = reorganize_mod._process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id="sub-001",
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
            lesion_type="glioblastoma",
            completeness_checker=completeness_checker,
        )

        session = patient_data['sessions']['ses-001']
        assert session['status'] == 'incomplete'
        assert len(session['excluded_series']) == 1
        entry = session['excluded_series'][0]
        assert entry['series_description'] == "xyz_47_unknown | xyz_47_unknown"
        assert entry['detected_modality'] is None
        assert entry['reason'] == 'unrecognized'
        assert 'original_path' in entry
        assert 'slice_count' in entry

    def test_dedup_loser_appears_in_excluded_series_with_modality_and_reason(
        self, make_dicom_series, tmp_path
    ):
        patient_dir = tmp_path / "PAT002"
        # Two plain T1 candidates for the same session -> one wins dedup, one is retained as excluded
        make_dicom_series(
            patient_dir / "2023-01-01" / "t1_a",
            protocol_name="t1_mprage_sag", series_description="t1_mprage_sag",
            image_type=['ORIGINAL', 'PRIMARY'],
        )
        make_dicom_series(
            patient_dir / "2023-01-01" / "t1_b",
            protocol_name="t1_mprage_sag_repeat", series_description="t1_mprage_sag_repeat",
            image_type=['ORIGINAL', 'PRIMARY'],
        )

        logger = logging.getLogger("test_excluded_dedup")
        modality_detector = reorganize_mod.ModalityDetector(logger)
        scanner = reorganize_mod.DatasetScanner(logger)
        grouper = reorganize_mod.SessionGrouper(logger)
        deduplicator = reorganize_mod.SeriesDeduplicator(logger)
        completeness_checker = reorganize_mod.CompletenessChecker(logger, lesion_type="glioblastoma")
        file_organizer = reorganize_mod.FileOrganizer(tmp_path / "output2", logger)

        patient_data = reorganize_mod._process_one_patient_core(
            patient_dir=patient_dir,
            new_patient_id="sub-002",
            modality_detector=modality_detector,
            scanner=scanner,
            grouper=grouper,
            deduplicator=deduplicator,
            file_organizer=file_organizer,
            logger=logger,
            lesion_type="glioblastoma",
            completeness_checker=completeness_checker,
        )

        session = patient_data['sessions']['ses-001']
        assert len(session['excluded_series']) == 1
        entry = session['excluded_series'][0]
        assert entry['detected_modality'] == 't1'
        assert entry['reason'] == 'lost_deduplication'
        # the winning t1 is still the one actually present in series
        assert 't1' in session['series']
