"""
BO (data/clinical_dicom/BO) marks contrast series with a bare, standalone
"c"/"C" token instead of a word like "MDC"/"KM"/"CE" — e.g.
"t1_space_sag_iso c", "t1_mprage_Sag_C", "t1_space_sag_iso+C",
"t1_space_sag_iso + c". ContrastBolusAgent is empty for these series (same
situation as KI-048/MDC), so with no text marker recognized, the contrast
T1 collided with the sibling plain-T1 series on modality "t1", lost the
scoring/dedup tie-break, and was silently discarded — t1c ended up missing
for 16+ BO patients even though it's required for glioblastoma
segmentation.

Real-world impact confirmed via a full-corpus scan (76 series across BO,
0 across BA/dropbox_33): every "c"/"C" occurrence in this dataset is a
delimiter-separated suffix (preceded by a space, "+", or "_" — never glued
onto a longer word) meaning "with contrast", consistently for both T1 and
T2 series.

A single ASCII letter is a much higher collision risk than "km"/"mdc"/"ce"
(2-3 chars), so unlike those patterns, the word-boundary here also excludes
digits — not just letters — on both sides. Without that, "c1"/"c2"
(e.g. vertebra levels, coil channel labels) would false-positive; those
series are excluded before this ever runs anyway (anatomy_exclude), but
the digit boundary keeps the marker itself as narrow as the KM/MDC
precedent.
"""
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_bare_c")
ModalityDetector = reorganize_mod.ModalityDetector


class TestBareCContrastMarker:
    def test_bare_c_marker_alone_is_recognized(self):
        assert ModalityDetector._has_c_marker("t1_space_sag_iso c") is True

    def test_underscore_c_marker_is_recognized(self):
        assert ModalityDetector._has_c_marker("t1_mprage_sag_c") is True

    def test_plus_c_marker_is_recognized(self):
        assert ModalityDetector._has_c_marker("t1_space_sag_iso+c") is True

    def test_plus_space_c_marker_is_recognized(self):
        assert ModalityDetector._has_c_marker("t1_space_sag_iso + c") is True

    def test_c_as_substring_inside_another_word_is_not_a_false_positive(self):
        assert ModalityDetector._has_c_marker("coronal t1") is False
        assert ModalityDetector._has_c_marker("calibration scan") is False

    def test_c_followed_by_digit_is_not_a_false_positive(self):
        """Vertebra level / coil channel style tokens ("c1", "c2") must not match."""
        assert ModalityDetector._has_c_marker("t1 mprage c1 level") is False
        assert ModalityDetector._has_c_marker("some scan c2") is False

    def test_no_c_marker_present(self):
        assert ModalityDetector._has_c_marker("t1_mprage_sag_p2") is False

    def test_real_bo126_bare_c_series_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Reproduces the exact real-world series name from data/clinical_dicom/BO/BO-126."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_space_sag_iso c",
            series_description="t1_space_sag_iso c",
            image_type=['ORIGINAL', 'SECONDARY', 'M', 'ND', 'NORM'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_bare_c"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"
        assert tech_meta["has_contrast"] is True

    def test_real_bo149_underscore_c_series_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Reproduces data/clinical_dicom/BO/BO-149's "t1_mprage_Sag_C"."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mprage_Sag_C",
            series_description="t1_mprage_Sag_C",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_bare_c"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"

    def test_real_bo17_plus_space_c_series_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Reproduces data/clinical_dicom/BO/BO-17's "t1_space_sag_iso + c" (space around "+")."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_space_sag_iso + c",
            series_description="t1_space_sag_iso + c",
            image_type=['ORIGINAL', 'SECONDARY', 'M', 'ND', 'NORM'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_bare_c"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"

    def test_plain_t1_without_c_marker_still_classified_as_t1(self, make_dicom_series, tmp_path):
        """The sibling plain-T1 series from the same patient must be unaffected."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_space_sag_iso",
            series_description="t1_space_sag_iso",
            image_type=['ORIGINAL', 'SECONDARY', 'M', 'ND', 'NORM'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_bare_c"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1"
        assert tech_meta["has_contrast"] is False
