"""
Italian clinical archives mark contrast series with "MDC" (Mezzo Di
Contrasto), e.g. "sT1W_3D_MDC", "Ax T1 3D MDC" — the same kind of
site-specific contrast marker as the German "KM" (KI-048), but for a
different language, and previously not recognized at all.

Real-world impact (data/clinical_dicom/BA): BA2 and BA4 both name their
contrast series "sT1W_3D_MDC" and — unlike BA3, whose scanner populated
ContrastBolusAgent ('gADOVIST') — have ContrastBolusAgent empty. With
neither the DICOM tag nor a text marker available, BA2/BA4's contrast T1
was classified as plain t1, collided with the sibling plain-T1 series, and
was discarded by the deduplicator — t1c ended up missing for both patients
even though it's required for glioblastoma segmentation.

Mirrors _has_km_marker()/_CE_PATTERN exactly: word-boundary regex, not a
bare substring check, so "mdc" only matches as a standalone token.
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_mdc")
ModalityDetector = reorganize_mod.ModalityDetector


class TestMdcContrastMarker:
    def test_mdc_marker_alone_is_recognized(self):
        assert ModalityDetector._has_mdc_marker("st1w_3d_mdc") is True

    def test_mdc_as_substring_inside_another_word_is_not_a_false_positive(self):
        # Guard against accidental substring matches — same concern 'ce'/'km' guard against.
        assert ModalityDetector._has_mdc_marker("amdcx protocol") is False

    def test_no_mdc_marker_present(self):
        assert ModalityDetector._has_mdc_marker("t1_mprage_sag_p2") is False

    def test_real_ba2_series_name_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Reproduces the exact real-world series name from data/clinical_dicom/BA/BA2."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="sT1W_3D_MDC",
            series_description="sT1W_3D_MDC",
            image_type=['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_mdc"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"
        assert tech_meta["has_contrast"] is True

    def test_real_ba3_series_name_with_space_separated_mdc_classified_as_t1c(self, make_dicom_series, tmp_path):
        """BA3's series name — MDC as its own word — must also work."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="ENCEFALO brupa",
            series_description="Ax T1 3D MDC",
            image_type=['ORIGINAL', 'PRIMARY', 'OTHER'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_mdc"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"

    def test_plain_t1_without_mdc_still_classified_as_t1(self, make_dicom_series, tmp_path):
        """The sibling plain-T1 series from the same patient must be unaffected."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1 se",
            series_description="T1 se",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_mdc"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1"
        assert tech_meta["has_contrast"] is False
