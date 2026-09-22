"""
Bug: DatasetScanner.scan_dataset() unconditionally treated every direct
subdirectory of input_dir as a separate patient, with no validation. This is
correct when input_dir is a batch root (children really are patients), but
wrong when input_dir is pointed directly at one patient's own folder — its
subdirectories (session-date folders, or modality folders like the
dropbox_33/117-152 batch's KA117/{t1ce,t2w,t1w,flair}) then get misdetected
as separate patients.

Fix: if input_dir has 2+ direct subdirectories and DICOM files under every
one of them report the same patient identity (PatientID, falling back to
PatientName), input_dir itself is treated as the single patient instead of
splitting its children. Verified against real data: dropbox_33/117-152/KA117
(single visit, 4 modality subfolders, same PatientID) and
SibBMS/P000067 (6 genuinely different real visits spanning 2010-2021, same
PatientID) — both keep a stable identity across the folders this heuristic
compares.
"""
import sys
import importlib.util
import logging
from pathlib import Path

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_scan_dataset")
DatasetScanner = reorganize_mod.DatasetScanner


def _write_fake_dicom(path: Path, patient_id: str = "", patient_name: str = ""):
    """Write one minimal-but-real DICOM file at the exact given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    if patient_id:
        ds.PatientID = patient_id
    if patient_name:
        ds.PatientName = patient_name
    ds.save_as(str(path), write_like_original=False)


def _scanner():
    return DatasetScanner(logging.getLogger("test_scan_dataset"))


class TestScanDatasetSinglePatientDetection:
    def test_batch_with_different_patient_ids_stays_split(self, tmp_path):
        """Unchanged behavior: real multi-patient batch, each child its own patient."""
        batch = tmp_path / "117-152"
        _write_fake_dicom(batch / "KA117" / "t1w" / "IM-0001", patient_id="P117")
        _write_fake_dicom(batch / "KA118" / "t1w" / "IM-0001", patient_id="P118")

        result = _scanner().scan_dataset(batch)

        assert sorted(p.name for p in result) == ["KA117", "KA118"]

    def test_single_patient_folder_with_matching_modality_subfolders_not_split(self, tmp_path):
        """The exact KA117 case: pointing input_dir at one patient's own
        folder, whose subfolders are modality directories, not other
        patients — all share the same PatientID."""
        patient_dir = tmp_path / "KA117"
        for modality in ["t1ce", "t2w", "t1w", "flair"]:
            _write_fake_dicom(patient_dir / modality / "IM-0001", patient_id="2026_5_20_14_32_41_951")

        result = _scanner().scan_dataset(patient_dir)

        assert result == [patient_dir]

    def test_single_patient_folder_with_matching_session_subfolders_not_split(self, tmp_path):
        """The general multi-session case (SibBMS/P000067-style): several
        genuinely different visit-date folders, same PatientID throughout."""
        patient_dir = tmp_path / "P000067"
        for date in ["2010-12-03", "2014-04-26", "2015-04-20"]:
            _write_fake_dicom(patient_dir / date / "IM-0001", patient_id="P000067")

        result = _scanner().scan_dataset(patient_dir)

        assert result == [patient_dir]

    def test_single_child_directory_unaffected(self, tmp_path):
        """Only one subdirectory: nothing to compare, existing behavior
        (treat that one child as the patient) is already correct."""
        batch = tmp_path / "batch"
        _write_fake_dicom(batch / "KA117" / "t1w" / "IM-0001", patient_id="P117")

        result = _scanner().scan_dataset(batch)

        assert result == [batch / "KA117"]

    def test_patient_folder_with_one_study_subfolder_is_the_patient(self, tmp_path):
        """Kate's private_363 layout: <patient>/<study-description>/<series>.

        Pointing input_dir at KA01, whose single child is the study folder
        "Nr Gruppe Mrnc3", used to return that child as the patient — so
        original_id became the study description. That description is
        IDENTICAL across patients (it is StudyDescription, not an identity),
        so KA01 and KA02 both collapsed onto one original_id and therefore
        onto one sub-NNN. The DICOM identity names the patient, and here it
        matches the input folder, not the child.
        """
        patient_dir = tmp_path / "KA01"
        study_dir = patient_dir / "Nr Gruppe Mrnc3"
        for series in ["t1_mprage_sag_p2 - 16001", "t2_space_flair_fs - 35001"]:
            _write_fake_dicom(study_dir / series / "IM-0001", patient_id="KA01")

        result = _scanner().scan_dataset(patient_dir)

        assert result == [patient_dir]

    def test_cohort_with_one_patient_still_returns_that_patient(self, tmp_path):
        """The mirror case that must NOT change: input_dir is a batch that
        happens to hold exactly one patient folder, and the identity matches
        the CHILD's name — the child is the patient, as before."""
        batch = tmp_path / "KA"
        _write_fake_dicom(batch / "KA01" / "t1w" / "IM-0001", patient_id="KA01")

        result = _scanner().scan_dataset(batch)

        assert result == [batch / "KA01"]

    def test_one_child_and_identity_matches_neither_name_keeps_old_behavior(self, tmp_path):
        """Folder names that follow no relation to PatientID (the existing
        test_single_child_directory_unaffected case) must keep today's
        behavior rather than guess."""
        batch = tmp_path / "batch"
        _write_fake_dicom(batch / "KA117" / "t1w" / "IM-0001", patient_id="P117")

        result = _scanner().scan_dataset(batch)

        assert result == [batch / "KA117"]

    def test_patient_folder_match_is_case_insensitive(self, tmp_path):
        """Real data differs in case: the DICOM here reads "KA01" while the
        folder could be "ka01" — the same patient either way."""
        patient_dir = tmp_path / "ka01"
        _write_fake_dicom(patient_dir / "NR Gruppe MRNC3" / "s1" / "IM-0001",
                          patient_id="KA01")

        result = _scanner().scan_dataset(patient_dir)

        assert result == [patient_dir]

    def test_missing_patient_id_fails_open_to_existing_behavior(self, tmp_path):
        """One child has no readable identity (no DICOM files) — don't guess,
        fall back to treating children as separate patients."""
        batch = tmp_path / "batch"
        _write_fake_dicom(batch / "KA117" / "t1w" / "IM-0001", patient_id="P117")
        (batch / "empty_dir").mkdir(parents=True)

        result = _scanner().scan_dataset(batch)

        assert sorted(p.name for p in result) == ["KA117", "empty_dir"]

    def test_patient_name_used_when_patient_id_blank(self, tmp_path):
        """Fallback identity signal when PatientID isn't populated."""
        patient_dir = tmp_path / "KA200"
        for modality in ["t1w", "t2w"]:
            _write_fake_dicom(patient_dir / modality / "IM-0001", patient_name="Anon_KA200")

        result = _scanner().scan_dataset(patient_dir)

        assert result == [patient_dir]
