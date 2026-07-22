"""
Regression tests for two Stage 01 gaps found while onboarding data/dropbox_33
(a real clinical dataset with no dates in its folder names and some patients'
DICOM files having no extension):

1. Date resolution (root cause of "no_valid_series" for all 19 patients):
   DatasetScanner only extracted the session date from folder NAMES (two
   hardcoded regex formats). dropbox_33 names folders by protocol/series
   description ("Nr Gruppe Mrnc3", "t1_mprage_fs_sag_1mm_iso - 32001") —
   no folder anywhere in the hierarchy carries a date — even though the
   DICOM tags themselves (StudyDate etc.) always do for real clinical data.
   Fixed by DatasetScanner.parse_date_from_dicom(): a third fallback tier
   that reads the date from DICOM tags when folder-name parsing (the fast,
   unchanged path for datasets that already embed dates in folder names)
   yields nothing.

2. DICOM discovery (KI-029): every file-discovery call site in this script
   used `rglob("*.dcm")`, silently dropping series whose files have no
   extension (a real pattern in 3/19 dropbox_33 patients — confirmed via a
   Part-10 DICM magic marker at byte offset 128, same as any other DICOM
   file). Fixed by find_dicom_files()/is_dicom_file(): recognizes any
   extension via the DICM magic marker, with a fast path that trusts
   *.dcm directly (no extra I/O for datasets that already use it).
"""
import sys
import importlib.util
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_dicom_discovery")
DatasetScanner = reorganize_mod.DatasetScanner
is_dicom_file = reorganize_mod.is_dicom_file
find_dicom_files = reorganize_mod.find_dicom_files


def _write_fake_dicom(path: Path, study_date: str = "", protocol_name: str = "t1"):
    """Write one minimal-but-real DICOM file at the exact given path (any name/extension)."""
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
    ds.ProtocolName = protocol_name
    ds.SeriesDescription = protocol_name
    if study_date:
        ds.StudyDate = study_date
    ds.save_as(str(path), write_like_original=False)


# --- is_dicom_file / find_dicom_files (KI-029) -------------------------------

def test_dcm_extension_is_trusted_without_reading_bytes(tmp_path):
    """The .dcm fast path must accept the file even if it weren't valid DICOM
    (proves it doesn't fall through to a magic-byte check for this case)."""
    f = tmp_path / "not_really_dicom.dcm"
    f.write_bytes(b"this is not a dicom file")
    assert is_dicom_file(f) is True


def test_extensionless_dicom_file_is_recognized(tmp_path):
    """Real-world case: dropbox_33's KA117/KA119/KA120 store DICOM files with
    no extension at all (e.g. "23831328"). Must be recognized via the DICM
    magic marker at byte offset 128."""
    f = tmp_path / "23831328"
    _write_fake_dicom(f)
    assert is_dicom_file(f) is True


def test_non_dicom_extensionless_file_is_rejected(tmp_path):
    """A random extensionless file (no DICM magic) must not be misidentified."""
    f = tmp_path / "readme"
    f.write_bytes(b"just some text, not a dicom file at all, padded out" + b"x" * 200)
    assert is_dicom_file(f) is False


def test_short_file_is_rejected_not_crashed(tmp_path):
    """A file shorter than 132 bytes must return False, not raise."""
    f = tmp_path / "tiny"
    f.write_bytes(b"short")
    assert is_dicom_file(f) is False


def test_find_dicom_files_mixed_extensions(tmp_path):
    """A series directory with a mix of .dcm and extensionless DICOM files,
    plus an unrelated non-DICOM file, must return only the real DICOM ones."""
    series_dir = tmp_path / "series"
    _write_fake_dicom(series_dir / "a.dcm")
    _write_fake_dicom(series_dir / "23831328")  # extensionless, like KA117
    (series_dir / "DICOMDIR.txt").write_bytes(b"not dicom" + b"x" * 200)

    found = find_dicom_files(series_dir)

    assert {f.name for f in found} == {"a.dcm", "23831328"}


# --- DatasetScanner.parse_date_from_dicom (date fallback) -------------------

def test_parse_date_from_dicom_reads_study_date(tmp_path):
    series_dir = tmp_path / "t1_mprage_fs_sag_1mm_iso - 32001"
    _write_fake_dicom(series_dir / "IM-0001", study_date="20230427")

    assert DatasetScanner.parse_date_from_dicom(series_dir) == "20230427"


def test_parse_date_from_dicom_none_when_tag_absent(tmp_path):
    series_dir = tmp_path / "series_no_date"
    _write_fake_dicom(series_dir / "IM-0001", study_date="")

    assert DatasetScanner.parse_date_from_dicom(series_dir) is None


def test_parse_date_from_dicom_none_when_no_dicom_files(tmp_path):
    series_dir = tmp_path / "empty_series"
    series_dir.mkdir()

    assert DatasetScanner.parse_date_from_dicom(series_dir) is None


# --- End-to-end: the exact dropbox_33 folder shape --------------------------

def test_patient_with_no_date_in_any_folder_name_is_still_processed(tmp_path):
    """Reproduces the original bug end-to-end: patient/study-name/series-name
    folders that carry no date anywhere, but whose DICOM tags do. Before the
    fix, DatasetScanner found the series but every one was skipped for lack
    of a parseable date, and the patient was dropped entirely
    ("no_valid_series", exactly as seen for all 19 dropbox_33 patients)."""
    patient_dir = tmp_path / "KA01"
    study_dir = patient_dir / "Nr Gruppe Mrnc3"
    _write_fake_dicom(
        study_dir / "t1_mprage_fs_sag_1mm_iso - 32001" / "IM-0001",
        study_date="20230427", protocol_name="t1_mprage_fs_sag_1mm_iso",
    )

    logger = __import__("logging").getLogger("test")
    scanner = DatasetScanner(logger)
    entries = scanner.scan_patient_series(patient_dir)

    assert len(entries) == 1
    series_dir, date_folder_name = entries[0]
    assert date_folder_name is None  # no ancestor folder name carries a date

    # The full 3-tier fallback chain _process_one_patient_core uses:
    date = scanner.parse_date_from_series_name(date_folder_name) if date_folder_name else None
    if not date:
        date = scanner.parse_date_from_series_name(series_dir.name)
    if not date:
        date = scanner.parse_date_from_dicom(series_dir)

    assert date == "20230427"


def test_extensionless_series_files_are_all_counted(tmp_path):
    """A series where every file lacks a .dcm extension must still report the
    correct file count (used for slice_count / dedup tie-breaking) — before
    the fix this silently counted 0 files for such series."""
    series_dir = tmp_path / "t1ce"
    for i in range(3):
        _write_fake_dicom(series_dir / f"{20000000 + i}")  # numeric, no extension

    assert len(find_dicom_files(series_dir)) == 3
