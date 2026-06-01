"""
Tests for KI-015: warning when an empty Kappa dataset exists for the same
lesion_type before auto-creating a new one.

All Kappa API calls are mocked — no network required.
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# backend/ is the working dir for these tests; add it to path
sys.path.insert(0, str(Path(__file__).parent))


def _make_uploader(lesion_type: str = "glioblastoma"):
    """Build a KappaUploader with dummy credentials (no network calls)."""
    # Patch preprocessing_version so we don't need a real config file
    with patch("kappa_uploader.compute_preprocessing_id", return_value="abc123"):
        from kappa_uploader import KappaUploader
        uploader = KappaUploader(
            run_id="test_run",
            output_path="/tmp/fake_output",
            token="fake_token",
            user_id=1,
            user_type_id=1,
            lesion_type=lesion_type,
            preprocessing_config_path="/tmp/fake_config.yaml",
        )
    return uploader


class TestWarnIfEmptyDatasetExists:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_warning_logged_when_empty_dataset_with_same_lesion_type(self, caplog):
        uploader = _make_uploader("glioblastoma")

        datasets = [{"datasetId": 42, "datasetName": "glioblastoma_abc12345"}]
        entities_empty = []

        with patch("kappa_uploader.list_user_datasets", new=AsyncMock(return_value=datasets)), \
             patch("kappa_uploader.get_dataset_entities", new=AsyncMock(return_value=entities_empty)):

            with caplog.at_level(logging.WARNING, logger="kappa_uploader"):
                self._run(uploader._warn_if_empty_dataset_exists())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "42" in warnings[0].message or "42" in str(warnings[0].args)
        assert "glioblastoma" in warnings[0].message or "glioblastoma" in str(warnings[0].args)

    def test_no_warning_when_dataset_has_entities(self, caplog):
        uploader = _make_uploader("glioblastoma")

        datasets = [{"datasetId": 42, "datasetName": "glioblastoma_abc12345"}]
        entities_non_empty = [{"entityId": "e1"}, {"entityId": "e2"}]

        with patch("kappa_uploader.list_user_datasets", new=AsyncMock(return_value=datasets)), \
             patch("kappa_uploader.get_dataset_entities", new=AsyncMock(return_value=entities_non_empty)):

            with caplog.at_level(logging.WARNING, logger="kappa_uploader"):
                self._run(uploader._warn_if_empty_dataset_exists())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0

    def test_no_warning_for_different_lesion_type(self, caplog):
        uploader = _make_uploader("glioblastoma")

        datasets = [{"datasetId": 99, "datasetName": "multiple_sclerosis_xyz"}]
        entities_empty = []

        with patch("kappa_uploader.list_user_datasets", new=AsyncMock(return_value=datasets)), \
             patch("kappa_uploader.get_dataset_entities", new=AsyncMock(return_value=entities_empty)):

            with caplog.at_level(logging.WARNING, logger="kappa_uploader"):
                self._run(uploader._warn_if_empty_dataset_exists())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0

    def test_no_warning_when_no_datasets_exist(self, caplog):
        uploader = _make_uploader("glioblastoma")

        with patch("kappa_uploader.list_user_datasets", new=AsyncMock(return_value=[])), \
             patch("kappa_uploader.get_dataset_entities", new=AsyncMock(return_value=[])):

            with caplog.at_level(logging.WARNING, logger="kappa_uploader"):
                self._run(uploader._warn_if_empty_dataset_exists())

        assert not any(r.levelno == logging.WARNING for r in caplog.records)

    def test_multiple_empty_datasets_produce_multiple_warnings(self, caplog):
        uploader = _make_uploader("multiple_sclerosis")

        datasets = [
            {"datasetId": 10, "datasetName": "multiple_sclerosis_aaa"},
            {"datasetId": 11, "datasetName": "multiple_sclerosis_bbb"},
        ]

        with patch("kappa_uploader.list_user_datasets", new=AsyncMock(return_value=datasets)), \
             patch("kappa_uploader.get_dataset_entities", new=AsyncMock(return_value=[])):

            with caplog.at_level(logging.WARNING, logger="kappa_uploader"):
                self._run(uploader._warn_if_empty_dataset_exists())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2

    def test_network_error_does_not_raise(self):
        uploader = _make_uploader("glioblastoma")

        with patch("kappa_uploader.list_user_datasets",
                   new=AsyncMock(side_effect=Exception("network error"))):
            # Must not raise
            self._run(uploader._warn_if_empty_dataset_exists())
