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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_routing")


def _run_one_patient(tmp_path, make_dicom_series, series_specs, include_incomplete=False):
    patient_dir = tmp_path / "PAT001"
    for name, protocol in series_specs:
        make_dicom_series(
            patient_dir / "2023-01-01" / name,
            protocol_name=protocol, series_description=protocol,
            image_type=['ORIGINAL', 'PRIMARY'],
        )
    output_dir = tmp_path / "output"
    logger = logging.getLogger(f"test_routing_{include_incomplete}")
    modality_detector = reorganize_mod.ModalityDetector(logger)
    scanner = reorganize_mod.DatasetScanner(logger)
    grouper = reorganize_mod.SessionGrouper(logger)
    deduplicator = reorganize_mod.SeriesDeduplicator(logger)
    completeness_checker = reorganize_mod.CompletenessChecker(logger, lesion_type="glioblastoma")
    file_organizer = reorganize_mod.FileOrganizer(output_dir, logger)

    reorganize_mod._process_one_patient_core(
        patient_dir=patient_dir,
        new_patient_id="sub-001",
        modality_detector=modality_detector,
        scanner=scanner,
        grouper=grouper,
        deduplicator=deduplicator,
        file_organizer=file_organizer,
        logger=logger,
        completeness_checker=completeness_checker,
        lesion_type="glioblastoma",
        include_incomplete=include_incomplete,
    )
    return output_dir


class TestIncompleteSessionRouting:
    def test_incomplete_session_goes_to_incomplete_subdir_by_default(self, make_dicom_series, tmp_path):
        output_dir = _run_one_patient(
            tmp_path, make_dicom_series,
            [("t1", "t1_mprage_sag"), ("t2", "t2_tse_tra")],  # missing t1c, t2fl
        )
        assert not (output_dir / "sub-001" / "ses-001").exists()
        assert (output_dir / "_incomplete" / "sub-001" / "ses-001" / "anat" / "t1").exists()

    def test_complete_session_goes_to_main_tree(self, make_dicom_series, tmp_path):
        output_dir = _run_one_patient(
            tmp_path, make_dicom_series,
            [
                ("t1", "t1_mprage_sag"), ("t1c", "t1_mprage_sag_KM"),
                ("t2", "t2_tse_tra"), ("t2fl", "t2_flair"),
            ],
        )
        assert (output_dir / "sub-001" / "ses-001" / "anat" / "t1").exists()
        assert not (output_dir / "_incomplete").exists()

    def test_include_incomplete_flag_forces_main_tree(self, make_dicom_series, tmp_path):
        output_dir = _run_one_patient(
            tmp_path, make_dicom_series,
            [("t1", "t1_mprage_sag"), ("t2", "t2_tse_tra")],  # missing t1c, t2fl
            include_incomplete=True,
        )
        assert (output_dir / "sub-001" / "ses-001" / "anat" / "t1").exists()
        assert not (output_dir / "_incomplete").exists()


class TestCheckPatientExists:
    """
    Regression coverage for check_patient_exists() also recognizing patients
    who are still routed under output_dir/_incomplete/<patient_id>/... after a
    manual relabel — previously it only looked at output_dir/<patient_id>,
    which caused requeue to silently re-scan raw DICOM and discard whatever
    relabel_series() had already written into _incomplete/.
    """

    def _logger(self, name):
        return logging.getLogger(name)

    def test_patient_only_in_incomplete_dir_no_force_skips_and_preserves(self, tmp_path):
        output_dir = tmp_path / "output"
        incomplete_dir = output_dir / "_incomplete" / "sub-001"
        (incomplete_dir / "ses-001" / "anat").mkdir(parents=True)
        (incomplete_dir / "ses-001" / "anat" / "marker.txt").write_text("relabel work")

        result = reorganize_mod.check_patient_exists(
            output_dir, "sub-001", force=False, dry_run=False, logger=self._logger("t1")
        )

        assert result is True
        assert incomplete_dir.exists()
        assert (incomplete_dir / "ses-001" / "anat" / "marker.txt").exists()

    def test_patient_only_in_incomplete_dir_with_force_deletes_and_processes(self, tmp_path):
        output_dir = tmp_path / "output"
        incomplete_dir = output_dir / "_incomplete" / "sub-001"
        (incomplete_dir / "ses-001" / "anat").mkdir(parents=True)
        (incomplete_dir / "ses-001" / "anat" / "marker.txt").write_text("relabel work")

        result = reorganize_mod.check_patient_exists(
            output_dir, "sub-001", force=True, dry_run=False, logger=self._logger("t2")
        )

        assert result is False
        assert not incomplete_dir.exists()

    def test_patient_in_neither_location_processes(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        result = reorganize_mod.check_patient_exists(
            output_dir, "sub-001", force=False, dry_run=False, logger=self._logger("t3")
        )

        assert result is False

    def test_patient_in_main_dir_no_force_skips_and_preserves(self, tmp_path):
        output_dir = tmp_path / "output"
        complete_dir = output_dir / "sub-001"
        (complete_dir / "ses-001" / "anat").mkdir(parents=True)
        (complete_dir / "ses-001" / "anat" / "marker.txt").write_text("complete data")

        result = reorganize_mod.check_patient_exists(
            output_dir, "sub-001", force=False, dry_run=False, logger=self._logger("t4")
        )

        assert result is True
        assert complete_dir.exists()
        assert (complete_dir / "ses-001" / "anat" / "marker.txt").exists()
        assert not (output_dir / "_incomplete").exists()
