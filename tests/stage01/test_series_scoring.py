import logging
import sys
import importlib.util
from pathlib import Path

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_scoring")
SeriesInfo = reorganize_mod.SeriesInfo
score_series = reorganize_mod.score_series

SCORING_CONFIG = {
    'failure_markers': {'keywords': ['failed', 'repeat', 'motion'], 'penalty': 0.5},
    'derived_reconstruction_penalty': 0.1,
    'text_weights': {
        't1c': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5},
        't2fl': {'3d': 2.0},
    },
    'resolution_scoring': {
        'reference_slice_thickness_mm': 1.0, 'min_factor': 0.5, 'max_factor': 2.0,
    },
    'flair_ti_bonus': {'threshold_ms': 1500, 'bonus': 1.2},
}

LOGGER = logging.getLogger("test_scoring")


def _series(desc, modality="t1c", slice_thickness=None, ti=None, is_derived=False):
    return SeriesInfo(
        original_path=Path(f"/fake/{desc}"),
        patient_id="P1", date="20230101",
        modality=modality, series_description=desc,
        slice_thickness_mm=slice_thickness, ti_ms=ti,
        is_derived_reconstruction=is_derived,
    )


class TestScoreSeries:
    def test_primary_scores_higher_than_derived_reconstruction(self):
        primary = _series("CE_T1-TFE (3D brain)")
        derived = _series("MPR CE_T1-TFE 2.5mm (cor brain)", is_derived=True)
        assert score_series(primary, "t1c", SCORING_CONFIG, LOGGER) > score_series(derived, "t1c", SCORING_CONFIG, LOGGER)

    def test_derived_reconstruction_still_scores_positive(self):
        """A derived reconstruction must still get a usable (nonzero) score
        — it needs to be selectable as a last resort when it's the only
        candidate for a modality, not effectively excluded."""
        derived = _series("MPR CE_T1-TFE 2.5mm (cor brain)", is_derived=True)
        assert score_series(derived, "t1c", SCORING_CONFIG, LOGGER) > 0

    def test_tfe_scores_higher_than_tse(self):
        tfe = _series("CE_T1-TFE (3D brain)")
        tse = _series("CE_T1-TSE (3D brain)")
        assert score_series(tfe, "t1c", SCORING_CONFIG, LOGGER) > score_series(tse, "t1c", SCORING_CONFIG, LOGGER)

    def test_failure_marker_reduces_score(self):
        clean = _series("CE_T1-TFE (3D brain)")
        repeated = _series("CE_T1-TFE (3D brain) repeat motion")
        assert score_series(repeated, "t1c", SCORING_CONFIG, LOGGER) < score_series(clean, "t1c", SCORING_CONFIG, LOGGER)

    def test_thinner_slice_scores_higher(self):
        thin = _series("CE_T1-TFE (3D brain)", slice_thickness=1.0)
        thick = _series("CE_T1-TFE (3D brain)", slice_thickness=2.5)
        assert score_series(thin, "t1c", SCORING_CONFIG, LOGGER) > score_series(thick, "t1c", SCORING_CONFIG, LOGGER)

    def test_flair_long_ti_gets_bonus(self):
        long_ti = _series("FLAIR (3D brain)", modality="t2fl", ti=1800.0)
        short_ti = _series("FLAIR (3D brain)", modality="t2fl", ti=1000.0)
        assert score_series(long_ti, "t2fl", SCORING_CONFIG, LOGGER) > score_series(short_ti, "t2fl", SCORING_CONFIG, LOGGER)

    def test_no_signal_returns_neutral_score(self):
        plain = _series("some unrelated text")
        assert score_series(plain, "t1c", SCORING_CONFIG, LOGGER) == 1.0

    def test_missing_resolution_tag_does_not_crash(self):
        no_thickness = _series("CE_T1-TFE (3D brain)", slice_thickness=None)
        assert score_series(no_thickness, "t1c", SCORING_CONFIG, LOGGER) > 0
