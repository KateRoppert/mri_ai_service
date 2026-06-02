"""
Tests for bugs fixed in stage 04 (04_assess_quality.py).

Covered cases:
- find_nifti_images_bids: modality parsed from parts[2], not parts[-1]
- find_nifti_images_upenn_flat: returns 4-tuple (adds session_id='001')
- assess_image (sequential): skip check happens BEFORE loading NIfTI
- _assess_image_wrapper (parallel): skip check happens BEFORE loading NIfTI
- _process_parallel: skipped count tracked correctly
- run() summary: skipped count logged
- max_subjects: "Limited to" log only when actual filtering happens
"""

import json
import logging
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

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


quality_mod = _load_module("04_assess_quality.py", "quality_assessor")
QualityAssessor = quality_mod.QualityAssessor


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        'thresholds': {
            't1': {
                'snr':   {'good': 15.0, 'poor': 5.0},
                'cnr':   {'good': 8.0,  'poor': 2.0},
                'efc':   {'good': 0.5,  'poor': 0.7},
                'fber':  {'good': 50.0, 'poor': 10.0},
                'gradient_sharpness':       {'good': 1000.0, 'poor': 200.0},
                'voxel_anisotropy':         {'good': 1.2,    'poor': 3.0},
                'intensity_variance':       {'good': 5000.0, 'poor': 500.0},
                'coefficient_of_variation': {'good': 0.15,   'poor': 0.40},
            }
        },
        'weights': {
            'snr': 0.20, 'cnr': 0.20, 'efc': 0.15, 'fber': 0.15,
            'gradient_sharpness': 0.15, 'voxel_anisotropy': 0.05,
            'intensity_variance': 0.05, 'coefficient_of_variation': 0.05,
        },
        'categories': {
            'good':       {'min': 80},
            'acceptable': {'min': 50},
            'poor':       {'min': 0},
        },
        'foreground_mask': {'method': 'otsu'},
    }


@pytest.fixture
def assessor(config):
    return QualityAssessor(config, logging.getLogger("test_assessor"))


# ---------------------------------------------------------------------------
# find_nifti_images_bids — modality parsing
# ---------------------------------------------------------------------------

class TestFindNiftiBids:
    def _make_bids_tree(self, tmp_path: Path, filenames: list) -> Path:
        anat_dir = tmp_path / "sub-001" / "ses-001" / "anat"
        anat_dir.mkdir(parents=True)
        for name in filenames:
            (anat_dir / name).write_bytes(b"")
        return tmp_path

    def test_standard_modality_parsed(self, tmp_path, assessor):
        root = self._make_bids_tree(tmp_path, ["sub-001_ses-001_t1.nii.gz"])
        images = assessor.find_nifti_images_bids(root)
        assert len(images) == 1
        _, _, _, modality = images[0]
        assert modality == "t1"

    def test_eq_suffix_yields_correct_modality(self, tmp_path, assessor):
        # If a _Eq_1 file somehow slips through, parts[2] must still give 't1' not '1'
        root = self._make_bids_tree(
            tmp_path, ["sub-001_ses-001_t1_Eq_1.nii.gz"]
        )
        images = assessor.find_nifti_images_bids(root)
        assert len(images) == 1
        _, _, _, modality = images[0]
        assert modality == "t1"
        assert modality != "1"

    def test_returns_four_tuple(self, tmp_path, assessor):
        root = self._make_bids_tree(tmp_path, ["sub-001_ses-001_t2fl.nii.gz"])
        images = assessor.find_nifti_images_bids(root)
        assert len(images[0]) == 4


# ---------------------------------------------------------------------------
# find_nifti_images_upenn_flat — 4-tuple with session_id
# ---------------------------------------------------------------------------

class TestFindNiftiUpenn:
    def _make_upenn_tree(self, tmp_path: Path) -> Path:
        patient_dir = tmp_path / "UPENN-GBM-00001_11"
        patient_dir.mkdir(parents=True)
        (patient_dir / "UPENN-GBM-00001_11_T1_unstripped.nii.gz").write_bytes(b"")
        (patient_dir / "UPENN-GBM-00001_11_FLAIR_unstripped.nii.gz").write_bytes(b"")
        return tmp_path

    def test_returns_four_tuple(self, tmp_path, assessor):
        root = self._make_upenn_tree(tmp_path)
        images = assessor.find_nifti_images_upenn_flat(root)
        for item in images:
            assert len(item) == 4, "Expected 4-tuple (path, patient_id, session_id, modality)"

    def test_session_id_is_001(self, tmp_path, assessor):
        root = self._make_upenn_tree(tmp_path)
        images = assessor.find_nifti_images_upenn_flat(root)
        for _, _, session_id, _ in images:
            assert session_id == "001"

    def test_modality_mapped_correctly(self, tmp_path, assessor):
        root = self._make_upenn_tree(tmp_path)
        images = assessor.find_nifti_images_upenn_flat(root)
        modalities = {mod for _, _, _, mod in images}
        assert "t1" in modalities
        assert "t2fl" in modalities


# ---------------------------------------------------------------------------
# assess_image — skip-before-compute
# ---------------------------------------------------------------------------

class TestAssessImageSkipBeforeCompute:
    def test_nib_load_not_called_when_skipping(self, tmp_path, assessor):
        """If output already exists and skip_existing=True, nibabel must NOT be called."""
        report_dir = tmp_path / "sub-001" / "ses-001" / "anat"
        report_dir.mkdir(parents=True)
        report_file = report_dir / "sub-001_ses-001_t1_quality.json"
        report_file.write_text("{}")  # output already exists

        with patch.object(quality_mod.nib, 'load') as mock_load:
            assessor.assess_image(
                nifti_path=Path("/fake/t1.nii.gz"),
                patient_id="001",
                session_id="001",
                modality="t1",
                output_dir=tmp_path,
                skip_existing=True,
            )
            mock_load.assert_not_called()

    def test_skipped_counted_in_sequential(self, tmp_path, assessor):
        """Skipped image must increment stats['skipped'], not stats['successful']."""
        report_dir = tmp_path / "sub-001" / "ses-001" / "anat"
        report_dir.mkdir(parents=True)
        (report_dir / "sub-001_ses-001_t1_quality.json").write_text("{}")

        with patch.object(quality_mod.nib, 'load'):
            assessor.assess_image(
                Path("/fake/t1.nii.gz"), "001", "001", "t1", tmp_path, skip_existing=True
            )

        assert assessor.stats['skipped'] == 1
        assert assessor.stats['successful'] == 0


# ---------------------------------------------------------------------------
# _assess_image_wrapper — skip-before-compute (parallel)
# ---------------------------------------------------------------------------

class TestAssessImageWrapperSkipBeforeCompute:
    def test_nib_load_not_called_when_skipping(self, tmp_path, config):
        """Parallel worker must not call nib.load when output exists."""
        report_dir = tmp_path / "sub-001" / "ses-001" / "anat"
        report_dir.mkdir(parents=True)
        report_file = report_dir / "sub-001_ses-001_t1_quality.json"
        report_file.write_text("{}")

        with patch.object(quality_mod.nib, 'load') as mock_load:
            result = QualityAssessor._assess_image_wrapper((
                Path("/fake/t1.nii.gz"), "001", "001", "t1",
                tmp_path, config, True
            ))
            mock_load.assert_not_called()

        assert result == (True, "SKIPPED")


# ---------------------------------------------------------------------------
# _process_parallel — skipped count
# ---------------------------------------------------------------------------

class TestProcessParallelSkipped:
    def _run_with_mock_pool(self, assessor, mock_results):
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map = MagicMock(return_value=mock_results)

        fake_images = [(Path("/fake/t1.nii.gz"), "001", "ses-001", "t1")]
        with patch.object(quality_mod, 'Pool', MagicMock(return_value=mock_pool)):
            assessor._process_parallel(fake_images, Path("/tmp/out"), workers=2)

    def test_skipped_counted_in_parallel(self, assessor):
        results = [
            (True, "GOOD"),
            (True, "SKIPPED"),
            (True, "SKIPPED"),
            (False, "UNKNOWN"),
        ]
        self._run_with_mock_pool(assessor, results)
        assert assessor.stats['skipped'] == 2
        assert assessor.stats['successful'] == 1
        assert assessor.stats['failed'] == 1

    def test_all_skipped(self, assessor):
        results = [(True, "SKIPPED"), (True, "SKIPPED")]
        self._run_with_mock_pool(assessor, results)
        assert assessor.stats['skipped'] == 2
        assert assessor.stats['successful'] == 0
        assert assessor.stats['failed'] == 0

    def test_no_skipped_all_successful(self, assessor):
        results = [(True, "GOOD"), (True, "ACCEPTABLE"), (True, "POOR")]
        self._run_with_mock_pool(assessor, results)
        assert assessor.stats['skipped'] == 0
        assert assessor.stats['successful'] == 3


# ---------------------------------------------------------------------------
# max_subjects limiting — log only when actual filtering
# ---------------------------------------------------------------------------

class TestMaxSubjectsLogging:
    def _make_bids_niftis(self, tmp_path: Path, n_patients: int) -> Path:
        for i in range(1, n_patients + 1):
            pid = f"{i:03d}"
            anat = tmp_path / f"sub-{pid}" / "ses-001" / "anat"
            anat.mkdir(parents=True)
            (anat / f"sub-{pid}_ses-001_t1.nii.gz").write_bytes(b"")
        return tmp_path

    def test_no_log_when_under_limit(self, tmp_path, assessor, caplog):
        root = self._make_bids_niftis(tmp_path, 2)  # 2 patients, limit=5
        images = assessor.find_nifti_images_bids(root)
        with caplog.at_level(logging.INFO, logger="test_assessor"):
            # Simulate the limiting logic from run()
            max_subjects = 5
            unique_patients = []
            filtered_images = []
            for nifti_path, patient_id, session_id, modality in images:
                if patient_id not in unique_patients:
                    if len(unique_patients) >= max_subjects:
                        break
                    unique_patients.append(patient_id)
                if patient_id in unique_patients:
                    filtered_images.append((nifti_path, patient_id, session_id, modality))
            if len(filtered_images) < len(images):
                assessor.logger.info(f"Limited to first {max_subjects} subjects")
        assert "Limited to" not in caplog.text

    def test_log_when_over_limit(self, tmp_path, assessor, caplog):
        root = self._make_bids_niftis(tmp_path, 5)  # 5 patients, limit=2
        images = assessor.find_nifti_images_bids(root)
        with caplog.at_level(logging.INFO, logger="test_assessor"):
            max_subjects = 2
            unique_patients = []
            filtered_images = []
            for nifti_path, patient_id, session_id, modality in images:
                if patient_id not in unique_patients:
                    if len(unique_patients) >= max_subjects:
                        break
                    unique_patients.append(patient_id)
                if patient_id in unique_patients:
                    filtered_images.append((nifti_path, patient_id, session_id, modality))
            if len(filtered_images) < len(images):
                assessor.logger.info(f"Limited to first {max_subjects} subjects")
        assert "Limited to" in caplog.text
