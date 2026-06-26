"""
Regression test: ModalityDetector._detect_contrast must treat a DICOM
ContrastBolusAgent (0018,0010) / ContrastBolusStartTime (0018,1078) tag
that is PRESENT but EMPTY the same as an ABSENT tag — i.e. no contrast.

Bug: pydicom.Dataset.get(tag, default) only returns `default` when the
tag is absent. When the tag is present with an empty value, .get()
returns the DataElement object itself. str(DataElement) produces a
non-empty repr (e.g. "(0018,0010) Contrast/Bolus Agent    LO: ''"),
which is truthy and slips past the ('', 'none', 'no', 'n/a') exclusion
list, so _detect_contrast incorrectly returns True.

This was caught by tests/stage01/test_ms5_integration.py against the
real dataset data/MS_5/P000915, where a plain non-contrast 3D T1 series
("T1-TFE (3D brain)") has an empty-but-present ContrastBolusAgent tag
and was being misclassified as t1c instead of t1.
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_detect_contrast")
ModalityDetector = reorganize_mod.ModalityDetector


def _dataset_with_empty_tag(tag):
    """Build a minimal pydicom Dataset with `tag` explicitly present but empty.

    Note: tests/stage01/conftest.py's make_dicom_series fixture only sets
    ContrastBolusAgent when contrast_bolus_agent is truthy (`if
    contrast_bolus_agent: ds.ContrastBolusAgent = ...`), so it cannot
    produce a present-but-empty tag. We build the Dataset directly here
    to reproduce the exact real-world condition (tag present, value '').
    """
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.ProtocolName = "T1-TFE (3D brain)"
    ds.SeriesDescription = "T1-TFE (3D brain)"
    # Explicitly add the tag with an empty LO/value — present, not absent.
    ds.add_new(tag, 'LO', '')
    return ds


class TestDetectContrastEmptyTag:
    def test_empty_contrast_bolus_agent_tag_is_not_contrast(self):
        """Present-but-empty ContrastBolusAgent (0018,0010) must not be
        treated as a contrast marker."""
        detector = ModalityDetector(logger=__import__("logging").getLogger("test"))
        dcm = _dataset_with_empty_tag((0x0018, 0x0010))
        combined_text = "t1-tfe (3d brain) t1-tfe (3d brain)"
        assert detector._detect_contrast(dcm, combined_text) is False

    def test_empty_contrast_bolus_start_time_tag_is_not_contrast(self):
        """Present-but-empty ContrastBolusStartTime (0018,1078) must not
        be treated as a contrast marker."""
        detector = ModalityDetector(logger=__import__("logging").getLogger("test"))
        dcm = _dataset_with_empty_tag((0x0018, 0x1078))
        combined_text = "t1-tfe (3d brain) t1-tfe (3d brain)"
        assert detector._detect_contrast(dcm, combined_text) is False

    def test_absent_tag_still_not_contrast(self):
        """Sanity check: a Dataset with neither tag at all still returns
        False (the previously-working "absent tag" case must not regress)."""
        detector = ModalityDetector(logger=__import__("logging").getLogger("test"))
        ds = Dataset()
        combined_text = "t1-tfe (3d brain) t1-tfe (3d brain)"
        assert detector._detect_contrast(ds, combined_text) is False

    def test_genuinely_present_contrast_agent_still_detected(self):
        """Sanity check: a real, non-empty ContrastBolusAgent value must
        still be detected as contrast (no regression on the true-positive
        path)."""
        detector = ModalityDetector(logger=__import__("logging").getLogger("test"))
        ds = Dataset()
        ds.add_new((0x0018, 0x0010), 'LO', 'Gadovist')
        combined_text = "ce_t1-tfe (3d brain) ce_t1-tfe (3d brain)"
        assert detector._detect_contrast(ds, combined_text) is True
