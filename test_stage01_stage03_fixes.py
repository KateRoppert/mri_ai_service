"""
Tests for bugs fixed in stage 01 (01_reorganize_folders.py) and
stage 03 (03_convert_to_nifti.py).

Covered cases:
- ModalityDetector._match_modality: correct exclusions for T1-TSE, MPR FLAIR, MPR T1-TFE
- save_completeness_report: JSON in incomplete_data/ subfolder, correct structure
- NiftiConverter._process_parallel: correct aggregation from worker stats
"""

import json
import logging
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"

sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders")
convert_mod = _load_module("03_convert_to_nifti.py", "convert_nifti")

ModalityDetector = reorganize_mod.ModalityDetector
save_completeness_report = reorganize_mod.save_completeness_report
NiftiConverter = convert_mod.NiftiConverter


# ---------------------------------------------------------------------------
# ModalityDetector._match_modality
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return ModalityDetector(logging.getLogger("test_modality"))


class TestMatchModality:
    def test_t1_tse_not_classified_as_t2(self, detector):
        # t1-tse has 'tse' (a t2 keyword) but 't1' is in t2's exclude list
        result = detector._match_modality("t1-tse 2.5mm cor brain")
        assert result != "t2"

    def test_t1_tse_classified_as_t1(self, detector):
        # plain T1-TSE should resolve to t1
        result = detector._match_modality("t1-tse 2.5mm cor brain")
        assert result == "t1"

    def test_mpr_ce_t1_tse_returns_none(self, detector):
        # mpr excludes t2fl and t1c; 'ce' marker excludes t1 → no match
        result = detector._match_modality("mpr ce_t1-tse 2.5mm cor brain")
        assert result is None

    def test_mpr_flair_not_classified_as_t2fl(self, detector):
        # 'mpr' is in t2fl exclude list
        result = detector._match_modality("mpr flair brain")
        assert result != "t2fl"

    def test_mpr_flair_returns_none(self, detector):
        # 'mpr' excludes t2fl; no other modality keyword matches → None
        result = detector._match_modality("mpr flair brain")
        assert result is None

    def test_normal_t2_tse_classified_as_t2(self, detector):
        # plain T2 TSE without any exclusion keywords → t2
        result = detector._match_modality("t2 tse brain")
        assert result == "t2"

    def test_flair_without_mpr_classified_as_t2fl(self, detector):
        # FLAIR without 'mpr' → t2fl
        result = detector._match_modality("flair 3d brain")
        assert result == "t2fl"

    def test_mpr_t1_tfe_classified_as_t1(self, detector):
        # 'mpr' is NOT in t1 exclude list → MPR T1-TFE must remain t1
        result = detector._match_modality("mpr t1-tfe 2.5mm axi brain")
        assert result == "t1"


# ---------------------------------------------------------------------------
# save_completeness_report
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_completeness_data():
    return {
        'incomplete_patients': [
            {
                'patient_id': 'sub-001',
                'original_id': 'P000915',
                'incomplete_sessions': [
                    {
                        'session_id': 'ses-001',
                        'date': '20220118',
                        'missing': ['t2'],
                        'available': ['t1', 't2fl'],
                    }
                ],
            }
        ],
        'statistics': {
            'total_patients': 1,
            'complete_patients': 0,
            'incomplete_patients': 1,
            'total_sessions': 2,
            'complete_sessions': 1,
            'incomplete_sessions': 1,
        },
    }


class TestSaveCompletenessReport:
    def _report_path(self, tmp_path: Path) -> Path:
        return tmp_path / 'incomplete_data' / '01_reorganize_folders_incomplete_data.json'

    def test_creates_json_in_subfolder(self, tmp_path, sample_completeness_data):
        save_completeness_report(sample_completeness_data, tmp_path)
        assert self._report_path(tmp_path).exists()

    def test_does_not_create_txt_file(self, tmp_path, sample_completeness_data):
        save_completeness_report(sample_completeness_data, tmp_path)
        assert not (tmp_path / 'incomplete_data.txt').exists()

    def test_json_has_required_top_level_keys(self, tmp_path, sample_completeness_data):
        save_completeness_report(sample_completeness_data, tmp_path)
        data = json.loads(self._report_path(tmp_path).read_text())
        assert {'timestamp', 'stage', 'statistics', 'incomplete_data'} <= data.keys()

    def test_stage_value_is_correct(self, tmp_path, sample_completeness_data):
        save_completeness_report(sample_completeness_data, tmp_path)
        data = json.loads(self._report_path(tmp_path).read_text())
        assert data['stage'] == '01_reorganize_folders'

    def test_statistics_contains_success_rate(self, tmp_path, sample_completeness_data):
        save_completeness_report(sample_completeness_data, tmp_path)
        data = json.loads(self._report_path(tmp_path).read_text())
        assert 'success_rate_percent' in data['statistics']
        # 1 complete out of 2 sessions → 50.0%
        assert data['statistics']['success_rate_percent'] == 50.0

    def test_incomplete_data_contains_patient_records(self, tmp_path, sample_completeness_data):
        save_completeness_report(sample_completeness_data, tmp_path)
        data = json.loads(self._report_path(tmp_path).read_text())
        assert len(data['incomplete_data']) == 1
        assert data['incomplete_data'][0]['patient_id'] == 'sub-001'

    def test_all_complete_patients_produces_empty_incomplete_data(self, tmp_path):
        completeness_data = {
            'incomplete_patients': [],
            'statistics': {
                'total_patients': 2,
                'complete_patients': 2,
                'incomplete_patients': 0,
                'total_sessions': 4,
                'complete_sessions': 4,
                'incomplete_sessions': 0,
            },
        }
        save_completeness_report(completeness_data, tmp_path)
        data = json.loads(self._report_path(tmp_path).read_text())
        assert data['incomplete_data'] == []
        assert data['statistics']['success_rate_percent'] == 100.0


# ---------------------------------------------------------------------------
# NiftiConverter._process_parallel
# ---------------------------------------------------------------------------

@pytest.fixture
def converter():
    logger = logging.getLogger("test_converter")
    logger.setLevel(logging.CRITICAL)
    # NiftiConverter.__init__ calls _check_dcm2niix which runs the external binary.
    # Patch it to a no-op so tests work without dcm2niix installed.
    with patch.object(NiftiConverter, '_check_dcm2niix', return_value=None):
        return NiftiConverter(logger)


def _make_worker_results(specs):
    """Build fake Pool.map results from (total, successful, failed, skipped) tuples."""
    return [
        (successful > 0, {
            'total_series': total,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
        })
        for total, successful, failed, skipped in specs
    ]


def _run_parallel_with_mock(converter, mock_results, workers=2):
    """Call _process_parallel with Pool.map replaced by a mock."""
    mock_pool = MagicMock()
    mock_pool.__enter__ = MagicMock(return_value=mock_pool)
    mock_pool.__exit__ = MagicMock(return_value=False)
    mock_pool.map = MagicMock(return_value=mock_results)

    fake_series = [(Path('/tmp/series'), 'sub-001', 'ses-001', 't1')]
    with patch.object(convert_mod, 'Pool', MagicMock(return_value=mock_pool)):
        converter._process_parallel(fake_series, Path('/tmp/output'), workers)


class TestProcessParallel:
    def test_all_successful_aggregated_correctly(self, converter):
        results = _make_worker_results([(2, 2, 0, 0), (3, 3, 0, 0)])
        _run_parallel_with_mock(converter, results)
        assert converter.stats == {'total_series': 5, 'successful': 5, 'failed': 0, 'skipped': 0}

    def test_failed_series_counted_correctly(self, converter):
        results = _make_worker_results([(3, 2, 1, 0), (2, 0, 2, 0)])
        _run_parallel_with_mock(converter, results)
        assert converter.stats['total_series'] == 5
        assert converter.stats['successful'] == 2
        assert converter.stats['failed'] == 3

    def test_skipped_not_counted_as_successful(self, converter):
        results = _make_worker_results([(3, 1, 0, 2)])
        _run_parallel_with_mock(converter, results, workers=1)
        assert converter.stats['successful'] == 1
        assert converter.stats['skipped'] == 2
        # skipped must not inflate successful
        assert converter.stats['successful'] + converter.stats['skipped'] <= converter.stats['total_series']

    def test_total_series_matches_sum_of_all_workers(self, converter):
        results = _make_worker_results([(4, 3, 1, 0), (6, 4, 1, 1), (2, 2, 0, 0)])
        _run_parallel_with_mock(converter, results, workers=3)
        assert converter.stats['total_series'] == 12
        assert converter.stats['successful'] == 9
        assert converter.stats['failed'] == 2
        assert converter.stats['skipped'] == 1
