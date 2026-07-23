"""
Bug: a dual-echo series (e.g. "pd+t2_tse_tra", PD=echo1, T2=echo2) makes
dcm2niix write a suffixed filename (e.g. "..._t2_e2.nii.gz") because the
EchoNumbers tag differs from the default. convert_series() only checked the
exact expected filename, reported "output file was not created" as a
failure, and left the real (correctly converted) data orphaned under the
suffixed name.

Real-world impact: data/dropbox_33/117-152/KA130 (sub-028) lost its t2
series this way and was the only one of 33 patients in that batch not
segmented (segmentation requires all 4 modalities).

Fix: when the exact name is missing but exactly one suffixed file exists,
rename it (and its .json sidecar, if present) to the canonical name and
report success. Zero suffixed files is the original failure. More than one
suffixed file is real ambiguity (e.g. two distinct echoes) and must fail
with a descriptive reason rather than guess.
"""
import sys
import logging
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock

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


def _make_converter(conv_mod):
    """NiftiConverter with the __init__ dcm2niix-availability check stubbed out."""
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        return conv_mod.NiftiConverter(logging.getLogger("test_multiecho"))


def _run_creating_files(*filenames_and_contents):
    """
    subprocess.run stub for the actual conversion call: creates the given
    (name, content) files under the '-o' output directory, ignores the '-h'
    health-check call (no '-o' present then), and returns returncode=0.
    """
    def _run(cmd, **kwargs):
        if '-o' in cmd:
            out_dir = Path(cmd[cmd.index('-o') + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, content in filenames_and_contents:
                (out_dir / name).write_text(content)
        return MagicMock(returncode=0, stdout="", stderr="")
    return _run


class TestMultiEchoConversionRecovery:
    def test_single_suffixed_output_is_renamed_to_canonical(self, tmp_path):
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_1")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t2_e2.nii.gz", "fake-nifti"),
            ("sub-028_ses-001_t2_e2.json", "{}"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is True
        assert error is None
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert (anat_dir / "sub-028_ses-001_t2.nii.gz").exists()
        assert (anat_dir / "sub-028_ses-001_t2.json").exists()
        assert not (anat_dir / "sub-028_ses-001_t2_e2.nii.gz").exists()
        assert not (anat_dir / "sub-028_ses-001_t2_e2.json").exists()

    def test_single_suffixed_output_without_json_sidecar(self, tmp_path):
        """The .json sidecar is optional (this pipeline runs dcm2niix with -b n)."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_2")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t2_e2.nii.gz", "fake-nifti"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is True
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert (anat_dir / "sub-028_ses-001_t2.nii.gz").exists()

    def test_multiple_suffixed_outputs_fail_with_descriptive_reason(self, tmp_path):
        """Real ambiguity (two distinct echoes) must not be silently resolved."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_3")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t2_e1.nii.gz", "fake-nifti-1"),
            ("sub-028_ses-001_t2_e2.nii.gz", "fake-nifti-2"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is False
        assert "sub-028_ses-001_t2_e1.nii.gz" in error
        assert "sub-028_ses-001_t2_e2.nii.gz" in error
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert not (anat_dir / "sub-028_ses-001_t2.nii.gz").exists()
        assert (anat_dir / "sub-028_ses-001_t2_e1.nii.gz").exists()
        assert (anat_dir / "sub-028_ses-001_t2_e2.nii.gz").exists()

    def test_no_output_at_all_still_fails_as_before(self, tmp_path):
        """Unchanged behavior: dcm2niix returns 0 but produces nothing."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_4")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files()):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is False
        assert error == "dcm2niix finished successfully but output file was not created"

    def test_exact_output_with_extra_suffixed_artifact_still_cleans_up_as_before(self, tmp_path):
        """Unchanged behavior: when the exact file exists, extra suffixed
        artifacts (e.g. MPR duplicates) are deleted, not renamed over it."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_5")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t1"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t1.nii.gz", "canonical"),
            ("sub-028_ses-001_t1_Eq_1.nii.gz", "duplicate"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t1", output_dir, "001"
            )

        assert success is True
        assert error is None
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert (anat_dir / "sub-028_ses-001_t1.nii.gz").read_text() == "canonical"
        assert not (anat_dir / "sub-028_ses-001_t1_Eq_1.nii.gz").exists()
