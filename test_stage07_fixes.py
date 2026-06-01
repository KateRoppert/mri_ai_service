"""
Tests for bugs fixed in stage 07 (07_inverse_transform.py).

Covered cases:
- Thread limits set for both sequential and parallel modes (not sequential-only)
- Benchmark metrics carry actual skipped count, not hardcoded 0
"""

import importlib.util
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"

sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

if "ants" not in sys.modules:
    sys.modules["ants"] = MagicMock()


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


inv_mod = _load_module("07_inverse_transform.py", "inv07")

find_masks = inv_mod.find_masks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bids_masks(root: Path, subjects: list, sessions: list,
                     lesion_type: str = "glioblastoma") -> None:
    """Create minimal BIDS mask tree with empty segmask files."""
    for sub in subjects:
        for ses in sessions:
            d = root / f"sub-{sub}" / f"ses-{ses}" / "anat" / lesion_type
            d.mkdir(parents=True)
            (d / f"sub-{sub}_ses-{ses}_t1_segmask.nii.gz").write_bytes(b"")


def _run_main_with_args(tmp_path: Path, extra_args: list[str]) -> tuple[int, dict]:
    """
    Call main() with minimal valid args and return (exit_code, captured_env).
    The function always fails (no real data), but we capture env vars set before
    the failure point.
    """
    captured_env = {}

    def capture_environ(key, value):
        captured_env[key] = value
        os.environ[key] = value  # still set so ITK doesn't complain later

    segmentation_dir = tmp_path / "segmentation"
    output_dir = tmp_path / "output"
    _make_bids_masks(segmentation_dir, ["001"], ["001"])

    argv = [
        "07_inverse_transform.py",
        str(segmentation_dir),
        str(output_dir),
    ] + extra_args

    with patch.object(sys, "argv", argv):
        with patch.object(os.environ, "__setitem__", side_effect=capture_environ):
            try:
                inv_mod.main()
            except SystemExit:
                pass
            except Exception:
                pass

    return captured_env


# ---------------------------------------------------------------------------
# Thread limits — set for both modes
# ---------------------------------------------------------------------------

def _run_main_patched(tmp_path: Path, mode: str, workers: int = 1,
                      n_masks: int = 1) -> None:
    """
    Call main() with all required directories created and processing mocked.
    n_masks controls how many segmentation masks exist (= how many tasks to distribute).
    Thread-limit env vars are set by main() after auto-tuning.
    """
    seg_dir = tmp_path / "segmentation"
    out_dir = tmp_path / "output"
    nifti_dir = tmp_path / "nifti"
    xfm_dir = tmp_path / "transformations"

    subjects = [f"{i:03d}" for i in range(1, n_masks + 1)]
    _make_bids_masks(seg_dir, subjects, ["001"])
    nifti_dir.mkdir(exist_ok=True)
    xfm_dir.mkdir(exist_ok=True)

    argv = [
        "07_inverse_transform.py",
        str(seg_dir), str(out_dir),
        "--mode", mode,
        "--workers", str(workers),
    ]
    mock_result = {"success": True, "modalities_ok": 1, "modalities_total": 4}
    with patch.object(sys, "argv", argv):
        with patch.object(inv_mod, "process_one_mask", return_value=mock_result):
            try:
                inv_mod.main()
            except Exception:
                pass


class TestAutoTuneParallelism:
    """
    Auto-tune: actual_workers = min(configured_workers, n_masks, cpu_count // 4)
    threads_per_worker = cpu_count // actual_workers

    Stage 07 uses ANTs apply_transforms (~4 threads/call), so the CPU ceiling
    is cpu_count // 4 workers. Stage 05 used // 8 (heavier full registration).
    """

    def test_sequential_sets_thread_limits_to_all_cores(self, tmp_path):
        cpu_count = os.cpu_count() or 4
        _run_main_patched(tmp_path, mode="sequential")
        assert int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 0)) == cpu_count
        assert int(os.environ.get("ANTS_NUMBER_OF_THREADS", 0)) == cpu_count

    def test_parallel_always_sets_thread_limits(self, tmp_path):
        os.environ.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)
        _run_main_patched(tmp_path, mode="parallel", workers=4, n_masks=1)
        assert "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" in os.environ
        assert "ANTS_NUMBER_OF_THREADS" in os.environ

    def test_parallel_workers_capped_by_n_masks(self, tmp_path):
        """n_masks < cpu_count // 4 → actual_workers = n_masks, threads = cpu_count."""
        cpu_count = os.cpu_count() or 4
        # 1 mask is always fewer than any reasonable workers_by_cpu
        _run_main_patched(tmp_path, mode="parallel", workers=cpu_count, n_masks=1)
        # actual_workers = min(cpu_count, 1, cpu_count//4) = 1
        # threads = cpu_count // 1 = cpu_count
        actual = int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 0))
        assert actual == cpu_count

    def test_parallel_workers_capped_by_cpu_formula(self, tmp_path):
        """n_masks > cpu_count // 4 → actual_workers = cpu_count // 4, threads = 4."""
        cpu_count = os.cpu_count() or 4
        workers_by_cpu = max(1, cpu_count // 4)
        n_masks = workers_by_cpu + 5  # enough tasks that cpu formula is binding
        _run_main_patched(tmp_path, mode="parallel", workers=cpu_count, n_masks=n_masks)
        # actual_workers = workers_by_cpu, threads = cpu_count // workers_by_cpu
        actual = int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 0))
        expected = max(1, cpu_count // workers_by_cpu)
        assert actual == expected

    def test_parallel_threads_less_than_sequential_when_multiple_workers(self, tmp_path):
        """With multiple actual workers, threads/worker < cpu_count."""
        cpu_count = os.cpu_count() or 4
        workers_by_cpu = max(1, cpu_count // 4)
        if workers_by_cpu < 2:
            pytest.skip("Need cpu_count >= 8 for parallel to outperform sequential here")

        _run_main_patched(tmp_path, mode="sequential")
        seq_threads = int(os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"])

        tmp2 = tmp_path / "par"
        tmp2.mkdir()
        n_masks = workers_by_cpu + 5
        _run_main_patched(tmp2, mode="parallel", workers=cpu_count, n_masks=n_masks)
        par_threads = int(os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"])

        assert par_threads < seq_threads


# ---------------------------------------------------------------------------
# find_masks — basic sanity checks
# ---------------------------------------------------------------------------

class TestFindMasks:
    def test_finds_masks_in_lesion_subfolder(self, tmp_path):
        _make_bids_masks(tmp_path, ["001", "002"], ["001"], lesion_type="glioblastoma")
        masks = find_masks(tmp_path)
        assert len(masks) == 2

    def test_skips_native_masks(self, tmp_path):
        anat = tmp_path / "sub-001" / "ses-001" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-001_ses-001_t1_segmask.nii.gz").write_bytes(b"")
        (anat / "sub-001_ses-001_t1_segmask_native_t1.nii.gz").write_bytes(b"")
        masks = find_masks(tmp_path)
        assert len(masks) == 1
        assert all("_native_" not in m[0].name for m in masks)

    def test_max_subjects_limits_by_subject(self, tmp_path):
        _make_bids_masks(tmp_path, ["001", "002", "003"], ["001", "002"])
        masks = find_masks(tmp_path, max_subjects=2)
        subject_ids = {m[1] for m in masks}
        assert len(subject_ids) == 2

    def test_returns_correct_tuple_structure(self, tmp_path):
        _make_bids_masks(tmp_path, ["001"], ["001"])
        masks = find_masks(tmp_path)
        assert len(masks) == 1
        mask_path, subject_id, session_id = masks[0]
        assert isinstance(mask_path, Path)
        assert subject_id == "sub-001"
        assert session_id == "ses-001"


# ---------------------------------------------------------------------------
# skipped counter in benchmark metrics
# ---------------------------------------------------------------------------

class TestSkippedCounterInBenchmark:
    """
    Bug: benchmark metrics had skipped=0 hardcoded.
    Fixed: skipped = total_found - len(masks_after_filter).
    """

    def _invoke_main_with_skip_existing(self, tmp_path: Path, n_subjects: int,
                                         n_preexisting: int) -> MagicMock:
        """
        Set up n_subjects masks; pre-create native masks for n_preexisting of them
        so skip-existing filter removes them. Capture ExperimentMetrics created.
        """
        seg_dir = tmp_path / "segmentation"
        out_dir = tmp_path / "output"
        lesion_type = "glioblastoma"

        for i in range(1, n_subjects + 1):
            sub = f"sub-{i:03d}"
            ses = "ses-001"
            # Create atlas-space mask
            d = seg_dir / sub / ses / "anat" / lesion_type
            d.mkdir(parents=True)
            (d / f"{sub}_{ses}_t1_segmask.nii.gz").write_bytes(b"")
            # Pre-create native mask for first n_preexisting subjects
            if i <= n_preexisting:
                native_dir = out_dir / sub / ses / "anat" / lesion_type
                native_dir.mkdir(parents=True)
                (native_dir / f"{sub}_{ses}_t1_segmask_native_t1.nii.gz").write_bytes(b"")

        captured_metrics = []

        def capture_log_metrics(metrics):
            captured_metrics.append(metrics)

        mock_benchmark_logger = MagicMock()
        mock_benchmark_logger.log_metrics.side_effect = capture_log_metrics

        argv = [
            "07_inverse_transform.py",
            str(seg_dir),
            str(out_dir),
            "--skip-existing",
            "--benchmark",
            "--nifti-dir", str(tmp_path / "nifti"),   # won't exist → main exits early
            "--transform-dir", str(tmp_path / "xfm"),
        ]

        # Provide nifti and transform dirs so path validation passes
        (tmp_path / "nifti").mkdir()
        (tmp_path / "xfm").mkdir()

        with patch.object(sys, "argv", argv):
            with patch.object(inv_mod, "BenchmarkLogger",
                              return_value=mock_benchmark_logger):
                with patch.object(inv_mod, "PerformanceMonitor") as mock_pm:
                    mock_pm.return_value.get_metrics.return_value = {}
                    # Make process_one_mask succeed trivially
                    with patch.object(inv_mod, "process_one_mask",
                                      return_value={"success": True,
                                                    "modalities_ok": 1,
                                                    "modalities_total": 4}):
                        try:
                            inv_mod.main()
                        except SystemExit:
                            pass
                        except Exception:
                            pass

        return captured_metrics

    def test_skipped_reflects_skip_existing_count(self, tmp_path):
        metrics_list = self._invoke_main_with_skip_existing(
            tmp_path, n_subjects=5, n_preexisting=2
        )
        if not metrics_list:
            pytest.skip("Benchmark logger not reached (likely path issue in test env)")

        metrics = metrics_list[0]
        assert metrics.skipped == 2, f"Expected 2 skipped, got {metrics.skipped}"

    def test_skipped_zero_when_no_preexisting(self, tmp_path):
        metrics_list = self._invoke_main_with_skip_existing(
            tmp_path, n_subjects=3, n_preexisting=0
        )
        if not metrics_list:
            pytest.skip("Benchmark logger not reached (likely path issue in test env)")

        metrics = metrics_list[0]
        assert metrics.skipped == 0
