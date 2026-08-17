"""
KI-052: the pipeline timeout was a flat 7200s constant for the whole
multi-stage run regardless of dataset size — too short for a full
~175-patient clinical dataset (stage 5 alone measured 56 min / 161
patients on real BO data). It must scale with how many patients are
actually being processed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from pipeline_manager import PipelineManager


def test_estimate_pipeline_timeout_scales_with_patient_count(tmp_path):
    for i in range(5):
        (tmp_path / f"patient-{i}").mkdir()
    (tmp_path / ".hidden").mkdir()  # must not be counted as a patient
    (tmp_path / "a_file.txt").write_text("x")  # not a directory, must not be counted

    pm = PipelineManager()
    timeout = pm.estimate_pipeline_timeout(str(tmp_path))

    expected = settings.pipeline_timeout_base_seconds + settings.pipeline_timeout_per_patient_seconds * 5
    assert timeout == expected


def test_estimate_pipeline_timeout_returns_base_for_missing_path(tmp_path):
    pm = PipelineManager()
    timeout = pm.estimate_pipeline_timeout(str(tmp_path / "does-not-exist"))
    assert timeout == settings.pipeline_timeout_base_seconds


def test_estimate_pipeline_timeout_base_for_empty_directory(tmp_path):
    pm = PipelineManager()
    timeout = pm.estimate_pipeline_timeout(str(tmp_path))
    assert timeout == settings.pipeline_timeout_base_seconds
