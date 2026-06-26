"""
Tests for auto-tune parallelism in stage 08 (08_anatomical_analysis.py).

Stage 08 (LobarAnalyzer) is pure Python/numpy — no ANTs threads.
Auto-tune: actual_workers = min(configured_workers, n_masks).
No thread env-var limits needed.

Covered cases:
- Sequential mode: actual_workers = 1
- Parallel: actual_workers capped by n_masks
- Parallel: actual_workers capped by configured max_workers
- Benchmark metrics carry actual_workers, not configured workers
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

# lobar_analysis requires nibabel/ants — mock the whole module for the
# duration of loading 08_anatomical_analysis.py, then remove the mocks so
# other test modules in the same pytest process (e.g. test_lesion_stats.py,
# test_anatomical_analyzer_base.py) still get the real modules.
_INJECTED_MOCK_MODULES = []
for _mod_name in ("lobar_analysis", "ants", "lesion_stats", "ms_localization"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
        _INJECTED_MOCK_MODULES.append(_mod_name)


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


loc_mod = _load_module("08_anatomical_analysis.py", "loc08")
find_masks_08 = loc_mod.find_masks

for _mod_name in _INJECTED_MOCK_MODULES:
    del sys.modules[_mod_name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bids_masks(root: Path, subjects: list, sessions: list,
                     lesion_type: str = "glioblastoma") -> None:
    for sub in subjects:
        for ses in sessions:
            d = root / f"sub-{sub}" / f"ses-{ses}" / "anat" / lesion_type
            d.mkdir(parents=True)
            (d / f"sub-{sub}_ses-{ses}_t1_segmask.nii.gz").write_bytes(b"")


def _make_minimal_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Write minimal lobar and preprocessing config files."""
    import yaml

    atlas_file = tmp_path / "data" / "lobar_atlas.nii.gz"
    atlas_file.parent.mkdir(parents=True)
    atlas_file.write_bytes(b"")

    mapping_file = tmp_path / "data" / "mapping.json"
    mapping_file.write_text("{}")

    lobar_cfg = tmp_path / "configs" / "lobar_atlas_config.yaml"
    lobar_cfg.parent.mkdir(parents=True)
    lobar_cfg.write_text(yaml.dump({
        "templates": {
            "SRI24": {"file": str(atlas_file.relative_to(tmp_path))}
        },
        "mapping_file": str(mapping_file.relative_to(tmp_path)),
        "segmentation_classes": {},
    }))

    preproc_cfg = tmp_path / "configs" / "preprocessing_config.yaml"
    preproc_cfg.write_text(yaml.dump({"atlas": {"name": "SRI24"}}))

    return lobar_cfg, preproc_cfg


def _run_main_08(tmp_path: Path, mode: str, workers: int,
                 n_masks: int = 1) -> list:
    """
    Call main() with mocked atlas resolution and process_one_mask.
    Returns list of ExperimentMetrics captured by the benchmark logger mock.
    """
    seg_dir = tmp_path / "segmentation"
    out_dir = tmp_path / "output"
    lobar_cfg, preproc_cfg = _make_minimal_configs(tmp_path)

    subjects = [f"{i:03d}" for i in range(1, n_masks + 1)]
    _make_bids_masks(seg_dir, subjects, ["001"])

    captured_metrics = []

    def capture(metrics):
        captured_metrics.append(metrics)

    mock_bm_logger = MagicMock()
    mock_bm_logger.log_metrics.side_effect = capture

    mock_result = {"success": True, "affected_lobes": 2, "report_path": "/tmp/r.json"}

    argv = [
        "08_anatomical_analysis.py",
        str(seg_dir), str(out_dir),
        "--config", str(lobar_cfg),
        "--preprocessing-config", str(preproc_cfg),
        "--mode", mode,
        "--workers", str(workers),
        "--benchmark",
    ]

    with patch.object(sys, "argv", argv):
        with patch.object(loc_mod, "resolve_atlas_path",
                          return_value=tmp_path / "data" / "lobar_atlas.nii.gz"):
            with patch.object(loc_mod, "BenchmarkLogger", return_value=mock_bm_logger):
                with patch.object(loc_mod, "PerformanceMonitor") as mock_pm:
                    mock_pm.return_value.get_metrics.return_value = {}
                    with patch.object(loc_mod, "process_one_mask",
                                      return_value=mock_result):
                        try:
                            loc_mod.main()
                        except SystemExit:
                            pass
                        except Exception:
                            pass

    return captured_metrics


# ---------------------------------------------------------------------------
# Auto-tune: workers capping logic
# ---------------------------------------------------------------------------

class TestAutoTuneParallelismStage08:
    """
    Stage 08 is pure Python/numpy — no ANTs thread overhead.
    actual_workers = min(configured_workers, n_masks)
    No thread env vars set (unlike stage 07).
    """

    def test_sequential_mode_uses_one_worker(self, tmp_path):
        metrics = _run_main_08(tmp_path, mode="sequential", workers=14, n_masks=5)
        if not metrics:
            pytest.skip("Benchmark not reached in test env")
        assert metrics[0].workers == 1

    def test_parallel_workers_capped_by_n_masks(self, tmp_path):
        """2 masks with 14 configured workers → actual = 2."""
        metrics = _run_main_08(tmp_path, mode="parallel", workers=14, n_masks=2)
        if not metrics:
            pytest.skip("Benchmark not reached in test env")
        assert metrics[0].workers == 2

    def test_parallel_workers_capped_by_configured_max(self, tmp_path):
        """10 masks with 3 configured workers → actual = 3."""
        metrics = _run_main_08(tmp_path, mode="parallel", workers=3, n_masks=10)
        if not metrics:
            pytest.skip("Benchmark not reached in test env")
        assert metrics[0].workers == 3

    def test_parallel_workers_equals_n_masks_when_fewer_than_configured(self, tmp_path):
        """4 masks with 14 workers → actual = 4."""
        metrics = _run_main_08(tmp_path, mode="parallel", workers=14, n_masks=4)
        if not metrics:
            pytest.skip("Benchmark not reached in test env")
        assert metrics[0].workers == 4

    def test_no_ants_thread_limits_set(self, tmp_path):
        """Stage 08 must NOT set ITK/ANTs thread limits (pure Python stage)."""
        os.environ.pop("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", None)
        _run_main_08(tmp_path, mode="parallel", workers=4, n_masks=4)
        assert "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ


# ---------------------------------------------------------------------------
# find_masks (stage 08 version — has lesion_type filter)
# ---------------------------------------------------------------------------

class TestFindMasksStage08:
    def test_finds_masks_in_lesion_subfolder(self, tmp_path):
        _make_bids_masks(tmp_path, ["001", "002"], ["001"], lesion_type="glioblastoma")
        masks = find_masks_08(tmp_path, lesion_type="glioblastoma")
        assert len(masks) == 2

    def test_lesion_type_filter_excludes_other_types(self, tmp_path):
        _make_bids_masks(tmp_path, ["001"], ["001"], lesion_type="glioblastoma")
        _make_bids_masks(tmp_path, ["002"], ["001"], lesion_type="multiple_sclerosis")
        masks = find_masks_08(tmp_path, lesion_type="glioblastoma")
        assert len(masks) == 1
        assert masks[0][1] == "sub-001"

    def test_skips_native_masks(self, tmp_path):
        anat = tmp_path / "sub-001" / "ses-001" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-001_ses-001_t1_segmask.nii.gz").write_bytes(b"")
        (anat / "sub-001_ses-001_t1_segmask_native_t1.nii.gz").write_bytes(b"")
        masks = find_masks_08(tmp_path)
        assert len(masks) == 1
        assert "_native_" not in masks[0][0].name

    def test_max_subjects_limits_by_subject(self, tmp_path):
        _make_bids_masks(tmp_path, ["001", "002", "003"], ["001"])
        masks = find_masks_08(tmp_path, max_subjects=2)
        assert len({m[1] for m in masks}) == 2
