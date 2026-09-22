"""
Tests for PipelineMonitor._create_kappa_uploader threading the run's fixed
kappa_dataset_id (backend/numbering.py, Task 4) into KappaUploader, so upload
cannot resolve a different dataset than the one Stage 01 numbered subjects for.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pipeline_monitor as pm


def test_uploader_receives_the_runs_dataset_id():
    fake_run = SimpleNamespace(kappa_dataset_id=337)
    fake_session = {"kappa_token": "tok", "user_id": 26, "user_type_id": 3}

    with patch("database.SessionLocal", return_value=SimpleNamespace(close=lambda: None)), \
         patch("database.get_pipeline_run", return_value=fake_run), \
         patch("kappa_auth.get_session", return_value=fake_session), \
         patch("kappa_uploader.KappaUploader") as MockUploader:
        MockUploader.return_value = "uploader-instance"

        result = pm.pipeline_monitor._create_kappa_uploader(
            "run-1", "/out", "session-1", "glioblastoma"
        )

    assert result == "uploader-instance"
    _, kwargs = MockUploader.call_args
    assert kwargs["dataset_id"] == 337


def test_uploader_gets_none_when_run_has_no_dataset():
    """A run started before this existed, or a CLI-adjacent path, must still
    produce an uploader — it falls back to resolving at upload."""
    fake_run = SimpleNamespace(kappa_dataset_id=None)
    fake_session = {"kappa_token": "tok", "user_id": 26, "user_type_id": 3}

    with patch("database.SessionLocal", return_value=SimpleNamespace(close=lambda: None)), \
         patch("database.get_pipeline_run", return_value=fake_run), \
         patch("kappa_auth.get_session", return_value=fake_session), \
         patch("kappa_uploader.KappaUploader") as MockUploader:
        MockUploader.return_value = "uploader-instance"

        pm.pipeline_monitor._create_kappa_uploader(
            "run-1", "/out", "session-1", "glioblastoma"
        )

    _, kwargs = MockUploader.call_args
    assert kwargs["dataset_id"] is None


def test_returns_none_when_run_is_missing():
    """The run row could be gone (race with cleanup); must not raise."""
    fake_session = {"kappa_token": "tok", "user_id": 26, "user_type_id": 3}

    with patch("database.SessionLocal", return_value=SimpleNamespace(close=lambda: None)), \
         patch("database.get_pipeline_run", return_value=None), \
         patch("kappa_auth.get_session", return_value=fake_session), \
         patch("kappa_uploader.KappaUploader") as MockUploader:
        MockUploader.return_value = "uploader-instance"

        result = pm.pipeline_monitor._create_kappa_uploader(
            "run-1", "/out", "session-1", "glioblastoma"
        )

    assert result == "uploader-instance"
    _, kwargs = MockUploader.call_args
    assert kwargs["dataset_id"] is None
