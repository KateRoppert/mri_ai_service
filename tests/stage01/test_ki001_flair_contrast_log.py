import io
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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_ki001")
ModalityDetector = reorganize_mod.ModalityDetector


def _detect_with_captured_log(series_dir):
    logger = logging.getLogger("test_ki001")
    logger.setLevel(logging.INFO)
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        detector = ModalityDetector(logger)
        result = detector.detect_modality(series_dir)
    finally:
        logger.removeHandler(handler)
    return result, log_capture.getvalue()


class TestKi001FlairContrastLog:
    def test_flair_with_contrast_log_is_informational_not_alarming(self, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="FLAIR (3D brain) post-contrast",
            series_description="FLAIR (3D brain) post-contrast",
            contrast_bolus_agent="Gadovist",
        )
        (modality, _, _), log_output = _detect_with_captured_log(series_dir)
        assert modality == "t2fl"
        assert "contrast=True" not in log_output
        assert "informational" in log_output

    def test_t1c_with_contrast_log_unchanged(self, make_dicom_series, tmp_path):
        """Non-FLAIR contrast logging keeps its original, unmodified wording."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="CE_T1-TFE (3D brain)",
            series_description="CE_T1-TFE (3D brain)",
            contrast_bolus_agent="Gadovist",
        )
        (modality, _, _), log_output = _detect_with_captured_log(series_dir)
        assert modality == "t1c"
        assert "contrast=True" in log_output
