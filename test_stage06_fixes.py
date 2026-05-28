"""
Tests for bugs fixed in stage 06 (06_segmentation.py).

Covered cases:
- BIDSScanner.scan: max_subjects limits by subject count, not session count
- _scan_output_structure: finds masks in lesion_type/ subfolder (rglob fix)
- _save_benchmark_metrics: uses stats.skipped not hardcoded 0
- _process_sessions_async: does NOT call stats.log_summary() (only run() does)
"""

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"

sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# aiohttp is not installed in the test environment
if "aiohttp" not in sys.modules:
    sys.modules["aiohttp"] = MagicMock()


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


seg_mod = _load_module("06_segmentation.py", "seg06")

BIDSScanner = seg_mod.BIDSScanner
ProcessingStats = seg_mod.ProcessingStats
SegmentationRunner = seg_mod.SegmentationRunner
SubjectSession = seg_mod.SubjectSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(output_dir: Path = None) -> SegmentationRunner:
    mock_config = MagicMock()
    mock_config.get_model_name.return_value = "mock_model"
    mock_registry = MagicMock()
    mock_registry.get_url_for.return_value = "http://mock"
    mock_registry.get_service_id.return_value = "mock-seg"
    mock_args = MagicMock()
    mock_args.lesion_type = "gbm"
    mock_args.max_concurrent = 1
    mock_args.benchmark = False
    mock_args.server_name = "test_server"
    mock_args.gpu_count = None
    mock_args.gpu_ids = None
    if output_dir is not None:
        mock_args.output_dir = output_dir
    runner = SegmentationRunner(mock_config, mock_registry, mock_args)
    return runner


# ---------------------------------------------------------------------------
# BIDSScanner.scan — max_subjects counts subjects, not sessions
# ---------------------------------------------------------------------------

class TestBIDSScannerMaxSubjects:
    def test_max_subjects_limits_by_subject_count_not_session_count(self, tmp_path):
        # 4 subjects, each with 3 sessions
        for i in range(1, 5):
            for j in range(1, 4):
                (tmp_path / f"sub-{i:03d}" / f"ses-{j:02d}" / "anat").mkdir(parents=True)

        modality_map = {"t1": "T1w", "t1c": "ce_T1w", "t2": "T2w", "t2fl": "FLAIR"}
        scanner = BIDSScanner(tmp_path, tmp_path / "out", modality_map)

        mock_session = MagicMock(spec=SubjectSession)
        scanner._scan_subject = MagicMock(return_value=[mock_session] * 3)

        result = scanner.scan(max_subjects=2)

        assert scanner._scan_subject.call_count == 2, "Should scan exactly 2 subjects"
        assert len(result) == 6, "2 subjects × 3 sessions = 6"

    def test_without_limit_processes_all_subjects(self, tmp_path):
        for i in range(1, 5):
            (tmp_path / f"sub-{i:03d}" / "anat").mkdir(parents=True)

        modality_map = {"t1": "T1w", "t1c": "ce_T1w", "t2": "T2w", "t2fl": "FLAIR"}
        scanner = BIDSScanner(tmp_path, tmp_path / "out", modality_map)
        scanner._scan_subject = MagicMock(return_value=[])

        scanner.scan(max_subjects=None)

        assert scanner._scan_subject.call_count == 4

    def test_multi_session_subject_counts_as_one_subject(self, tmp_path):
        # 1 subject with 5 sessions — max_subjects=1 should return all 5 sessions
        for j in range(1, 6):
            (tmp_path / "sub-001" / f"ses-{j:02d}" / "anat").mkdir(parents=True)

        modality_map = {"t1": "T1w", "t1c": "ce_T1w", "t2": "T2w", "t2fl": "FLAIR"}
        scanner = BIDSScanner(tmp_path, tmp_path / "out", modality_map)
        mock_session = MagicMock(spec=SubjectSession)
        scanner._scan_subject = MagicMock(return_value=[mock_session] * 5)

        result = scanner.scan(max_subjects=1)

        assert scanner._scan_subject.call_count == 1
        assert len(result) == 5, "All 5 sessions of the single subject must be included"

    def test_limit_larger_than_subject_count_processes_all(self, tmp_path):
        for i in range(1, 3):
            (tmp_path / f"sub-{i:03d}" / "anat").mkdir(parents=True)

        modality_map = {"t1": "T1w", "t1c": "ce_T1w", "t2": "T2w", "t2fl": "FLAIR"}
        scanner = BIDSScanner(tmp_path, tmp_path / "out", modality_map)
        scanner._scan_subject = MagicMock(return_value=[])

        scanner.scan(max_subjects=10)

        assert scanner._scan_subject.call_count == 2


# ---------------------------------------------------------------------------
# _scan_output_structure — rglob finds masks in lesion_type/ subfolders
# ---------------------------------------------------------------------------

class TestScanOutputStructure:
    def test_finds_mask_in_lesion_type_subfolder(self, tmp_path):
        # New convention: anat/glioma/sub-001_ses-001_T1w_segmask.nii.gz
        mask_dir = tmp_path / "sub-001" / "ses-001" / "anat" / "glioma"
        mask_dir.mkdir(parents=True)
        (mask_dir / "sub-001_ses-001_T1w_segmask.nii.gz").write_bytes(b"")

        runner = _make_runner(output_dir=tmp_path)
        structure = runner._scan_output_structure()

        assert "001" in structure
        assert "001" in structure["001"]
        assert "segmask" in structure["001"]["001"]

    def test_also_finds_mask_directly_in_anat(self, tmp_path):
        # Backward-compat: mask at anat/ level (no lesion subfolder)
        anat_dir = tmp_path / "sub-002" / "ses-001" / "anat"
        anat_dir.mkdir(parents=True)
        (anat_dir / "sub-002_ses-001_T1w_segmask.nii.gz").write_bytes(b"")

        runner = _make_runner(output_dir=tmp_path)
        structure = runner._scan_output_structure()

        assert "002" in structure
        assert "segmask" in structure["002"]["001"]

    def test_no_mask_returns_empty_structure(self, tmp_path):
        anat_dir = tmp_path / "sub-003" / "ses-001" / "anat"
        anat_dir.mkdir(parents=True)
        # No mask file present

        runner = _make_runner(output_dir=tmp_path)
        structure = runner._scan_output_structure()

        assert structure == {}

    def test_multiple_subjects_multiple_sessions(self, tmp_path):
        for sub in ("001", "002"):
            for ses in ("001", "002"):
                d = tmp_path / f"sub-{sub}" / f"ses-{ses}" / "anat" / "ms"
                d.mkdir(parents=True)
                (d / f"sub-{sub}_ses-{ses}_T1w_segmask.nii.gz").write_bytes(b"")

        runner = _make_runner(output_dir=tmp_path)
        structure = runner._scan_output_structure()

        assert set(structure.keys()) == {"001", "002"}
        for pat in ("001", "002"):
            assert set(structure[pat].keys()) == {"001", "002"}


# ---------------------------------------------------------------------------
# _save_benchmark_metrics — skipped reflects stats.skipped, not 0
# ---------------------------------------------------------------------------

class TestSaveBenchmarkMetricsSkipped:
    def _runner_for_benchmark(self) -> SegmentationRunner:
        runner = _make_runner()
        runner.pipeline_start_time = time.time() - 10
        runner.gpu_metrics_collected = []
        runner.monitor = MagicMock()
        runner.monitor.get_metrics.return_value = {}
        runner.benchmark_logger = MagicMock()
        runner.benchmark_logger.get_baseline_time.return_value = None
        return runner

    def test_skipped_value_taken_from_stats(self):
        runner = self._runner_for_benchmark()
        runner.stats.skipped = 7
        runner.stats.total = 10
        runner.stats.successful = 3
        runner.stats.failed = 0

        runner._save_benchmark_metrics([])

        call_args = runner.benchmark_logger.log_metrics.call_args
        metrics = call_args[0][0]
        assert metrics.skipped == 7

    def test_skipped_zero_when_no_skips_occurred(self):
        runner = self._runner_for_benchmark()
        runner.stats.skipped = 0
        runner.stats.total = 5
        runner.stats.successful = 5
        runner.stats.failed = 0

        runner._save_benchmark_metrics([])

        call_args = runner.benchmark_logger.log_metrics.call_args
        metrics = call_args[0][0]
        assert metrics.skipped == 0

    def test_skipped_nonzero_when_skip_existing_used(self):
        runner = self._runner_for_benchmark()
        runner.stats.skipped = 3
        runner.stats.total = 8
        runner.stats.successful = 5
        runner.stats.failed = 0

        runner._save_benchmark_metrics([])

        call_args = runner.benchmark_logger.log_metrics.call_args
        metrics = call_args[0][0]
        assert metrics.skipped == 3


# ---------------------------------------------------------------------------
# _process_sessions_async — log_summary NOT called; only run() calls it
# ---------------------------------------------------------------------------

class TestProcessSessionsAsyncNoLogSummary:
    def test_async_processing_does_not_call_log_summary(self):
        runner = _make_runner()

        # Inject module-level 'args' global required by the method (Bug 5 workaround)
        seg_mod.args = runner.args

        with patch.object(seg_mod, "AsyncSegmentationClient", MagicMock()):
            with patch.object(runner.stats, "log_summary") as mock_summary:
                asyncio.run(runner._process_sessions_async([]))
                mock_summary.assert_not_called()

    def test_async_processing_returns_true_when_no_failures(self):
        runner = _make_runner()
        seg_mod.args = runner.args

        with patch.object(seg_mod, "AsyncSegmentationClient", MagicMock()):
            result = asyncio.run(runner._process_sessions_async([]))
            assert result is True  # stats.failed == 0
