"""
KI-048: German contrast marker "KM" (Kontrastmittel) was not recognized as a
contrast indicator, so a post-contrast T1 series named e.g.
"T1W_FFE 5mm tra KM" was classified as plain t1 instead of t1c.

Real-world impact (data/dropbox_33/KA53): two series both resolved to "t1"
(the true plain T1 and the mislabeled contrast T1), the deduplicator picked
one by slice-count tie-break, and the true contrast series was silently
discarded — t1c ended up entirely missing for that patient, even though
t1c is a required modality for glioblastoma.

Mirrors the existing 'ce' marker handling exactly: word-boundary regex
(not a bare substring check), so "km" only matches as a standalone token —
guards against a word that happens to contain "km" as a substring.
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_ki048")
ModalityDetector = reorganize_mod.ModalityDetector


class TestKi048KmContrastMarker:
    def test_km_marker_alone_is_recognized(self):
        assert ModalityDetector._has_km_marker("t1w_ffe 5mm tra km") is True

    def test_km_as_substring_inside_another_word_is_not_a_false_positive(self):
        # Guard against accidental substring matches, same concern the 'ce'
        # pattern was built to avoid (e.g. "space", "sequence", "slice").
        assert ModalityDetector._has_km_marker("skmr protocol") is False

    def test_no_km_marker_present(self):
        assert ModalityDetector._has_km_marker("t1_mprage_sag_p2") is False

    def test_real_ka53_series_name_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Reproduces the exact real-world series name from data/dropbox_33/KA53."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1W_FFE 5mm tra KM",
            series_description="T1W_FFE 5mm tra KM",
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_ki048"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"
        assert tech_meta["has_contrast"] is True

    def test_plain_t1_without_km_still_classified_as_t1(self, make_dicom_series, tmp_path):
        """The sibling plain-T1 series from the same patient must be unaffected."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1W_FFE",
            series_description="T1W_FFE",
        )
        import logging
        detector = ModalityDetector(logging.getLogger("test_ki048"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1"
        assert tech_meta["has_contrast"] is False
