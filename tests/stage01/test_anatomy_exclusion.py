import logging
import sys
import importlib.util
from pathlib import Path

import pytest

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_anatomy")
ModalityDetector = reorganize_mod.ModalityDetector

ANATOMY_CONFIG = {'anatomy_exclude': {'keywords': ['spine', 'cervical', 'c-spine', 'pituitary', 'orbit', 'neck']}}


@pytest.fixture
def detector():
    return ModalityDetector(logging.getLogger("test_anatomy"), scoring_config=ANATOMY_CONFIG)


@pytest.fixture
def detector_no_config():
    return ModalityDetector(logging.getLogger("test_anatomy_noconfig"))


class TestIsExcludedAnatomy:
    def test_c_spine_text_excluded(self, detector):
        assert detector._is_excluded_anatomy("ce_t1-tse (sag c-spine)") == 'c-spine'

    def test_cervical_text_excluded(self, detector):
        assert detector._is_excluded_anatomy("t2-tse cervical spine sagittal") is not None

    def test_pituitary_text_excluded(self, detector):
        assert detector._is_excluded_anatomy("t1 pituitary dynamic") == 'pituitary'

    def test_normal_brain_text_not_excluded(self, detector):
        assert detector._is_excluded_anatomy("t1-tfe (3d brain)") is None

    def test_no_scoring_config_excludes_nothing(self, detector_no_config):
        # Backward-compatible default: without an explicit scoring_config,
        # no anatomy keywords are configured, so nothing is excluded.
        assert detector_no_config._is_excluded_anatomy("ce_t1-tse (sag c-spine)") is None


class TestDetectModalityExcludesAnatomy:
    def test_c_spine_series_returns_none_modality(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="CE_T1-TSE (sag C-spine)",
            series_description="CE_T1-TSE (sag C-spine)",
        )
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality is None
        assert tech_meta == {}

    def test_normal_brain_series_not_excluded(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1-TFE (3D brain)",
            series_description="T1-TFE (3D brain)",
        )
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t1"
