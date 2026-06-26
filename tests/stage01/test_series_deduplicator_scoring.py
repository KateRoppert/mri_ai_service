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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_dedup")
SeriesInfo = reorganize_mod.SeriesInfo
SeriesDeduplicator = reorganize_mod.SeriesDeduplicator

sys.path.insert(0, str(PROJ_ROOT))
from utils.config_loader import load_series_scoring_config


def _series(tmp_path, name, modality, n_files=1):
    series_dir = tmp_path / name
    series_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake")
    return SeriesInfo(
        original_path=series_dir, patient_id="P1", date="20230101",
        modality=modality, series_description=name,
    )


@pytest.fixture
def deduplicator():
    return SeriesDeduplicator(
        logging.getLogger("test_dedup"),
        scoring_config=load_series_scoring_config(),
    )


class TestSelectBestSeries:
    def test_picks_3d_over_mpr_reformat(self, deduplicator, tmp_path):
        threed = _series(tmp_path, "CE_T1-TFE (3D brain)", "t1c")
        mpr = _series(tmp_path, "MPR CE_T1-TFE 2.5mm (cor brain)", "t1c")
        best = deduplicator._select_best_series([threed, mpr], "t1c")
        assert best is threed

    def test_single_candidate_returned_unchanged(self, deduplicator, tmp_path):
        only = _series(tmp_path, "CE_T1-TFE (3D brain)", "t1c")
        best = deduplicator._select_best_series([only], "t1c")
        assert best is only

    def test_falls_back_to_slice_count_when_scores_tie(self, deduplicator, tmp_path):
        a = _series(tmp_path, "unrelated text a", "t1c", n_files=1)
        b = _series(tmp_path, "unrelated text b", "t1c", n_files=2)
        best = deduplicator._select_best_series([a, b], "t1c")
        assert best is b


class TestSeriesDeduplicatorConfigDefault:
    def test_no_scoring_config_does_not_crash(self, tmp_path):
        """A deduplicator built without scoring_config (e.g. ad-hoc script
        usage) must still work, falling back to slice_count for everything."""
        dedup = SeriesDeduplicator(logging.getLogger("test_dedup_noconfig"))
        a = _series(tmp_path, "series a", "t1c", n_files=1)
        b = _series(tmp_path, "series b", "t1c", n_files=3)
        best = dedup._select_best_series([a, b], "t1c")
        assert best is b
