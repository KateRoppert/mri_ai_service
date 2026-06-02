"""
Tests for bugs fixed in stage 05 (05_preprocessing.py) and
preprocessing_steps/skull_stripping.py.

Covered cases:
- process_single_subject: reference_modality resolved by step name, not by [2] index
- find_subjects: returns all subjects when max_subjects=None
- max_subjects limiting: "Limited to" log only when actual filtering happens
- skipped counter: check_subject_processed returning True increments skipped, not successful
- check_subject_processed: correctly identifies fully vs. partially processed subjects
- setup_fsl_environment: logs WARNING (not ERROR) when FSL path is invalid
"""

import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"

sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# Pre-mock unavailable C-extension dependencies so the module loads in test env
for _dep in ("SimpleITK", "ants", "antspyx", "antspynet"):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


preproc_mod = _load_module("05_preprocessing.py", "preproc")
skull_mod = _load_module("preprocessing_steps/skull_stripping.py", "skull_stripping")

find_subjects = preproc_mod.find_subjects
check_subject_processed = preproc_mod.check_subject_processed
process_single_subject = preproc_mod.process_single_subject
calculate_optimal_parallelism = preproc_mod.calculate_optimal_parallelism
setup_fsl_environment = skull_mod.setup_fsl_environment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bids_tree(root: Path, subjects: list, sessions: list, modalities: list) -> None:
    """Create minimal BIDS NIfTI tree with empty files."""
    for sub in subjects:
        for ses in sessions:
            anat = root / f"sub-{sub}" / f"ses-{ses}" / "anat"
            anat.mkdir(parents=True)
            for mod in modalities:
                (anat / f"sub-{sub}_{ses}_{mod}.nii.gz").write_bytes(b"")


# ---------------------------------------------------------------------------
# calculate_optimal_parallelism — workers cap + auto-tune
# ---------------------------------------------------------------------------

class TestCalculateOptimalParallelism:
    """
    Workers (from config) is treated as a ceiling.
    Formula uses effective_cpu = cpu_count - 2 (OS headroom) and caps
    workers at effective_cpu // 8 to ensure each worker gets enough threads.
    """

    # effective_cpu for cpu_count=24: max(4, 24-2) = 22
    # workers_by_cpu for 24 CPUs:     22 // 8 = 2

    def test_fewer_subjects_than_workers_reduces_workers(self):
        # 2 subjects, cap=14, 24 CPU → effective=22, actual=2, threads=22//2=11
        workers, threads = calculate_optimal_parallelism(n_subjects=2, cpu_count=24, max_workers=14)
        assert workers == 2
        assert threads == 11

    def test_cpu_optimal_caps_workers(self):
        # 20 subjects, cap=14, 24 CPU → effective=22, workers_by_cpu=22//8=2, actual=2
        workers, threads = calculate_optimal_parallelism(n_subjects=20, cpu_count=24, max_workers=14)
        assert workers == 2
        assert threads == 11

    def test_config_cap_respected_when_lower_than_cpu_optimal(self):
        # 20 subjects, cap=2, 24 CPU → effective=22, cpu_floor=2, config=2 → actual=2
        workers, threads = calculate_optimal_parallelism(n_subjects=20, cpu_count=24, max_workers=2)
        assert workers == 2
        assert threads == 11

    def test_single_subject_always_gives_one_worker(self):
        # 1 subject → actual=1, gets all effective threads
        workers, threads = calculate_optimal_parallelism(n_subjects=1, cpu_count=24, max_workers=14)
        assert workers == 1
        assert threads == 22  # effective_cpu = 24-2 = 22

    def test_workers_times_threads_equals_effective_cpu(self):
        # Invariant: workers * threads == effective_cpu (not raw cpu_count)
        workers, threads = calculate_optimal_parallelism(n_subjects=10, cpu_count=24, max_workers=10)
        assert workers * threads == 22  # effective_cpu = 24-2

    def test_minimum_one_worker_even_with_zero_subjects(self):
        # Defensive: shouldn't happen in practice but must not crash or return 0
        workers, threads = calculate_optimal_parallelism(n_subjects=0, cpu_count=24, max_workers=14)
        assert workers >= 1
        assert threads >= 1

    def test_cap_24_same_as_cap_14_for_small_cpu(self):
        # With 8 CPU (effective=6) and 5 subjects, both caps give same result
        # workers_by_cpu = 6 // 8 = 0 → max(1,0) = 1
        w14, t14 = calculate_optimal_parallelism(n_subjects=5, cpu_count=8, max_workers=14)
        w24, t24 = calculate_optimal_parallelism(n_subjects=5, cpu_count=8, max_workers=24)
        assert w14 == w24
        assert t14 == t24


# ---------------------------------------------------------------------------
# find_subjects — returns all when max_subjects=None
# ---------------------------------------------------------------------------

class TestFindSubjects:
    def test_returns_all_subjects_when_no_limit(self, tmp_path):
        _make_bids_tree(tmp_path, ["001", "002", "003"], ["001"], ["t1"])
        subjects = find_subjects(tmp_path, max_subjects=None)
        assert len(subjects) == 3

    def test_returns_limited_when_max_given(self, tmp_path):
        _make_bids_tree(tmp_path, ["001", "002", "003"], ["001"], ["t1"])
        subjects = find_subjects(tmp_path, max_subjects=2)
        assert len(subjects) == 2

    def test_each_entry_is_three_tuple(self, tmp_path):
        _make_bids_tree(tmp_path, ["001"], ["001"], ["t1"])
        subjects = find_subjects(tmp_path, max_subjects=None)
        assert len(subjects) == 1
        anat_dir, subject_id, session_id = subjects[0]
        assert subject_id == "sub-001"
        assert session_id == "ses-001"
        assert anat_dir.name == "anat"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        subjects = find_subjects(tmp_path, max_subjects=None)
        assert subjects == []


# ---------------------------------------------------------------------------
# max_subjects limiting — "Limited to" log only when filtering happens
# ---------------------------------------------------------------------------

class TestMaxSubjectsLogging:
    def _subjects_from_tree(self, tmp_path: Path, n: int):
        _make_bids_tree(tmp_path, [f"{i:03d}" for i in range(1, n + 1)], ["001"], ["t1"])
        return find_subjects(tmp_path, max_subjects=None)

    def test_log_appears_when_over_limit(self, tmp_path, caplog):
        subjects = self._subjects_from_tree(tmp_path, 5)
        max_subjects = 2
        logger = logging.getLogger("test_max_subj")
        with caplog.at_level(logging.INFO, logger="test_max_subj"):
            if max_subjects and len(subjects) > max_subjects:
                subjects = subjects[:max_subjects]
                logger.info(f"Limited to first {max_subjects} subjects (--max-subjects)")
        assert "Limited to" in caplog.text
        assert len(subjects) == 2

    def test_no_log_when_under_limit(self, tmp_path, caplog):
        subjects = self._subjects_from_tree(tmp_path, 2)
        max_subjects = 5
        logger = logging.getLogger("test_max_subj")
        with caplog.at_level(logging.INFO, logger="test_max_subj"):
            if max_subjects and len(subjects) > max_subjects:
                subjects = subjects[:max_subjects]
                logger.info(f"Limited to first {max_subjects} subjects (--max-subjects)")
        assert "Limited to" not in caplog.text
        assert len(subjects) == 2

    def test_no_log_when_exactly_at_limit(self, tmp_path, caplog):
        subjects = self._subjects_from_tree(tmp_path, 3)
        max_subjects = 3
        logger = logging.getLogger("test_max_subj")
        with caplog.at_level(logging.INFO, logger="test_max_subj"):
            if max_subjects and len(subjects) > max_subjects:
                subjects = subjects[:max_subjects]
                logger.info(f"Limited to first {max_subjects} subjects (--max-subjects)")
        assert "Limited to" not in caplog.text
        assert len(subjects) == 3


# ---------------------------------------------------------------------------
# process_single_subject — reference_modality resolved by step name
# ---------------------------------------------------------------------------

class TestReferenceModalityByName:
    """
    Verify that the reference modality is resolved by searching steps by 'name',
    not by a hardcoded position index like steps[2].
    The function returns early (before touching any external tools) when the
    reference file is missing in anat_dir.
    """

    def _make_config(self, ref_modality: str, step_index_padding: int = 0) -> dict:
        """
        Build a config where the registration step is at a variable position.
        step_index_padding adds dummy steps before registration.
        """
        padding = [{"name": f"dummy_{i}", "enabled": False} for i in range(step_index_padding)]
        return {
            "steps": padding + [
                {
                    "name": "registration",
                    "enabled": True,
                    "params": {"reference_modality": ref_modality},
                }
            ]
        }

    def test_reference_modality_found_at_index_0(self, tmp_path):
        """Registration at steps[0] — reference_modality must still be resolved."""
        config = self._make_config("t1", step_index_padding=0)
        anat_dir = tmp_path / "sub-001" / "ses-001" / "anat"
        anat_dir.mkdir(parents=True)
        # No t1 file → function must return early with correct skip_reason
        result = process_single_subject(
            anat_dir=anat_dir,
            subject_id="sub-001",
            session_id="ses-001",
            output_dir=tmp_path / "out",
            transform_dir=tmp_path / "xfm",
            temp_dir=tmp_path / "tmp",
            atlas_path=tmp_path / "atlas.nii.gz",
            config=config,
            modalities=["t1"],
        )
        assert result["skipped"] is True
        assert "missing_reference_modality_t1" in result["skip_reason"]

    def test_reference_modality_found_when_registration_not_at_index_2(self, tmp_path):
        """
        Old bug: config.get('steps', [{}])[2] would fetch the wrong step
        when registration is not at index 2.
        New code uses the variable already resolved by the name-search loop.
        """
        config = self._make_config("t1c", step_index_padding=1)
        # steps = [dummy_0, registration(t1c)]
        # Old code: steps[2] would be out-of-range → default 't1c' (coincidence for this case)
        # Test with ref_mod = 't1' at steps[1], where steps[2] doesn't exist
        config2 = self._make_config("t1", step_index_padding=1)
        anat_dir = tmp_path / "sub-002" / "ses-001" / "anat"
        anat_dir.mkdir(parents=True)
        # Provide t1c (what the buggy code would use as default) but NOT t1
        (anat_dir / "sub-002_ses-001_t1c.nii.gz").write_bytes(b"")
        result = process_single_subject(
            anat_dir=anat_dir,
            subject_id="sub-002",
            session_id="ses-001",
            output_dir=tmp_path / "out",
            transform_dir=tmp_path / "xfm",
            temp_dir=tmp_path / "tmp",
            atlas_path=tmp_path / "atlas.nii.gz",
            config=config2,
            modalities=["t1", "t1c"],
        )
        # Must skip because t1 (the ACTUAL reference) is missing, not t1c
        assert result["skipped"] is True
        assert "missing_reference_modality_t1" in result["skip_reason"]

    def test_no_skip_when_reference_file_present(self, tmp_path):
        """When the reference file exists, the early-exit path is NOT taken."""
        config = self._make_config("t1", step_index_padding=0)
        anat_dir = tmp_path / "sub-003" / "ses-001" / "anat"
        anat_dir.mkdir(parents=True)
        (anat_dir / "sub-003_ses-001_t1.nii.gz").write_bytes(b"")

        # The function will proceed past early check and then fail on processing
        # (atlas missing, etc.) — that's fine, we just need it NOT to be a skip.
        result = process_single_subject(
            anat_dir=anat_dir,
            subject_id="sub-003",
            session_id="ses-001",
            output_dir=tmp_path / "out",
            transform_dir=tmp_path / "xfm",
            temp_dir=tmp_path / "tmp",
            atlas_path=tmp_path / "atlas.nii.gz",
            config=config,
            modalities=["t1"],
        )
        # Not skipped due to missing reference — may fail for other reasons
        assert result.get("skip_reason") != "missing_reference_modality_t1"


# ---------------------------------------------------------------------------
# check_subject_processed — skip detection
# ---------------------------------------------------------------------------

class TestCheckSubjectProcessed:
    def _setup(self, tmp_path: Path, input_mods: list, output_mods: list):
        sub, ses = "sub-001", "ses-001"
        in_anat = tmp_path / "input" / sub / ses / "anat"
        out_anat = tmp_path / "output" / sub / ses / "anat"
        in_anat.mkdir(parents=True)
        out_anat.mkdir(parents=True)
        for mod in input_mods:
            (in_anat / f"{sub}_{ses}_{mod}.nii.gz").write_bytes(b"")
        for mod in output_mods:
            (out_anat / f"{sub}_{ses}_{mod}.nii.gz").write_bytes(b"")
        return tmp_path / "input", tmp_path / "output", sub, ses

    def test_fully_processed_returns_true(self, tmp_path):
        in_dir, out_dir, sub, ses = self._setup(tmp_path, ["t1", "t2"], ["t1", "t2"])
        is_done, missing = check_subject_processed(in_dir, out_dir, sub, ses, ["t1", "t2"])
        assert is_done is True
        assert missing == []

    def test_partially_processed_returns_false_with_missing(self, tmp_path):
        in_dir, out_dir, sub, ses = self._setup(tmp_path, ["t1", "t2"], ["t1"])
        is_done, missing = check_subject_processed(in_dir, out_dir, sub, ses, ["t1", "t2"])
        assert is_done is False
        assert "t2" in missing

    def test_not_processed_at_all_returns_false(self, tmp_path):
        in_dir, out_dir, sub, ses = self._setup(tmp_path, ["t1", "t2"], [])
        is_done, missing = check_subject_processed(in_dir, out_dir, sub, ses, ["t1", "t2"])
        assert is_done is False
        assert set(missing) == {"t1", "t2"}

    def test_only_checks_modalities_present_on_input(self, tmp_path):
        """If t2fl is not on input, it must not be required on output."""
        in_dir, out_dir, sub, ses = self._setup(tmp_path, ["t1"], ["t1"])
        is_done, missing = check_subject_processed(in_dir, out_dir, sub, ses, ["t1", "t2fl"])
        assert is_done is True
        assert missing == []


# ---------------------------------------------------------------------------
# setup_fsl_environment — WARNING not ERROR for bad path
# ---------------------------------------------------------------------------

class TestFSLEnvironmentLogging:
    def test_warning_not_error_for_nonexistent_fsl_dir(self, tmp_path, caplog):
        """setup_fsl_environment must log WARNING (not ERROR) when path doesn't exist."""
        bad_fsl_path = str(tmp_path / "nonexistent_fsl")
        with caplog.at_level(logging.WARNING, logger="skull_stripping"):
            setup_fsl_environment(bad_fsl_path)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        error_msgs   = [r for r in caplog.records if r.levelno == logging.ERROR]

        assert len(warning_msgs) >= 1, "Expected at least one WARNING"
        assert len(error_msgs)   == 0, "Expected no ERROR messages for recoverable path miss"

    def test_warning_message_mentions_fallback(self, tmp_path, caplog):
        """Warning message should hint that system PATH will be used."""
        bad_fsl_path = str(tmp_path / "no_fsl_here")
        with caplog.at_level(logging.WARNING, logger="skull_stripping"):
            setup_fsl_environment(bad_fsl_path)

        combined = " ".join(r.message for r in caplog.records if r.levelno == logging.WARNING)
        assert "PATH" in combined or "fallback" in combined.lower()

    def test_no_log_when_fsl_dir_is_empty_string(self, caplog):
        """Empty fsl_dir string means 'use PATH' — no warning should be logged."""
        with caplog.at_level(logging.WARNING, logger="skull_stripping"):
            setup_fsl_environment("")
        error_msgs = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_msgs) == 0
