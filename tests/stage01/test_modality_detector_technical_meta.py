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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_techmeta")
ModalityDetector = reorganize_mod.ModalityDetector


@pytest.fixture
def detector():
    return ModalityDetector(logging.getLogger("test_techmeta"))


class TestDetectModalityTechnicalMeta:
    def test_returns_three_tuple(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1-TFE (3D brain)",
            series_description="T1-TFE (3D brain)",
        )
        result = detector.detect_modality(series_dir)
        assert len(result) == 3

    def test_slice_thickness_extracted(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1-TFE (3D brain)",
            series_description="T1-TFE (3D brain)",
            slice_thickness=1.1,
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['slice_thickness_mm'] == 1.1

    def test_ti_extracted_for_flair(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="FLAIR (3D brain)",
            series_description="FLAIR (3D brain)",
            inversion_time=1660.0,
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['ti_ms'] == 1660.0

    def test_has_contrast_true_when_bolus_agent_present(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="CE_T1-TFE (3D brain)",
            series_description="CE_T1-TFE (3D brain)",
            contrast_bolus_agent="Gadovist",
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['has_contrast'] is True

    def test_missing_tags_default_to_none(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T2-TSE (axi brain)",
            series_description="T2-TSE (axi brain)",
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['slice_thickness_mm'] is None
        assert tech_meta['ti_ms'] is None
