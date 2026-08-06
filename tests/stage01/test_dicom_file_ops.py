import logging
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from utils.dicom_file_ops import (
    is_dicom_file, find_dicom_files, MODALITY_BIDS_SUFFIX, copy_and_anonymize_series,
)

LOGGER = logging.getLogger("test_dicom_file_ops")


class TestModalityBidsSuffix:
    def test_known_modalities(self):
        assert MODALITY_BIDS_SUFFIX == {
            't1': 'T1w', 't1c': 'T1wCE', 't2': 'T2w', 't2fl': 'FLAIR',
        }


class TestIsDicomFile:
    def test_dcm_extension_trusted_without_reading(self, tmp_path):
        f = tmp_path / "not_really_dicom.dcm"
        f.write_bytes(b"garbage")
        assert is_dicom_file(f) is True

    def test_extensionless_file_checked_by_magic_bytes(self, tmp_path):
        f = tmp_path / "23831328"
        f.write_bytes(b"\x00" * 128 + b"DICM" + b"restofdata")
        assert is_dicom_file(f) is True

    def test_non_dicom_extensionless_file_rejected(self, tmp_path):
        f = tmp_path / "README"
        f.write_bytes(b"just some text, not a dicom file at all")
        assert is_dicom_file(f) is False


class TestFindDicomFiles:
    def test_finds_dcm_files_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.dcm").write_bytes(b"x")
        (tmp_path / "sub" / "b.dcm").write_bytes(b"x")
        (tmp_path / "readme.txt").write_bytes(b"not dicom")
        found = find_dicom_files(tmp_path)
        assert len(found) == 2
        assert all(f.suffix == ".dcm" for f in found)


class TestCopyAndAnonymizeSeries:
    def test_plain_copy_without_metadata_extractor(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "IM-0001.dcm").write_bytes(b"fake dicom content")
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        source_files = find_dicom_files(source_dir)
        copied = copy_and_anonymize_series(
            source_files, target_dir, "sub-001", "ses-001", "t1c",
        )

        assert copied == 1
        assert (target_dir / "sub-001_ses-001_T1wCE_0001.dcm").exists()

    def test_returns_zero_and_logs_on_write_failure(self, tmp_path, caplog):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "IM-0001.dcm").write_bytes(b"fake dicom content")
        # target_dir intentionally NOT created -> shutil.copy2 raises FileNotFoundError
        target_dir = tmp_path / "does_not_exist"

        source_files = find_dicom_files(source_dir)
        copied = copy_and_anonymize_series(
            source_files, target_dir, "sub-001", "ses-001", "t1",
        )
        assert copied == 0
