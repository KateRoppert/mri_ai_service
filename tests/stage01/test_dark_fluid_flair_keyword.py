"""
Bug: MODALITY_PATTERNS['t2fl']['keywords'] listed 'dark fluid' (space), but
real clinical series names use a hyphen: "t2_tirm_tra_dark-fluid FS",
"t2_space_dark-fluid_sag_fs_iso". A substring check on 'dark fluid' never
matches text containing 'dark-fluid', so these series fell through to t2
(the 't2' keyword matches trivially) instead of being recognized as FLAIR.

Real-world impact (data/dropbox_33/1-116): KA45, KA68, KA08 all name their
FLAIR series "t2_tirm_tra_dark-fluid FS" — none contain the word "flair" at
all, so this was the only detection path available and it was silently
misclassifying every one of them as t2.
"""
import sys
import importlib.util
import logging
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_darkfluid")
ModalityDetector = reorganize_mod.ModalityDetector


class TestDarkFluidFlairKeyword:
    def test_hyphenated_dark_fluid_classified_as_t2fl(self, make_dicom_series, tmp_path):
        """Regression for KA45/KA68/KA08 (data/dropbox_33/1-116)."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_tirm_tra_dark-fluid FS",
            series_description="t2_tirm_tra_dark-fluid FS",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_darkfluid"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t2fl"

    def test_space_variant_still_classified_as_t2fl(self, make_dicom_series, tmp_path):
        """Unchanged behavior: the original space-separated form must keep working."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_space dark fluid sag",
            series_description="t2_space dark fluid sag",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_darkfluid"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t2fl"

    def test_hyphenated_dark_fluid_not_misclassified_as_t2(self, make_dicom_series, tmp_path):
        """Direct regression check: this used to resolve to 't2', not None/other."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_tse_dark-fluid_tra",
            series_description="t2_tse_dark-fluid_tra",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_darkfluid"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality != "t2"
        assert modality == "t2fl"

    def test_plain_t2_without_dark_fluid_still_classified_as_t2(self, make_dicom_series, tmp_path):
        """Unchanged behavior: a real plain T2 series must not be affected."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_tse_tra_4mm",
            series_description="t2_tse_tra_4mm",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_darkfluid"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t2"
