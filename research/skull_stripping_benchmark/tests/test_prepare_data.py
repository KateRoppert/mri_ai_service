"""Tests for prepare_data.py — the benchmark's input builder.

The benchmark compares strippers on identical inputs, so what matters here is
that the config it hands Stage 05 really disables skull stripping and really
lands in atlas space. A mistake in either turns every downstream number into
a comparison of different things.

The Stage 05 subprocess itself is integration territory (needs FSL/ANTs and
real volumes); only the pure config/command building is unit-tested.
"""
from pathlib import Path

import prepare_data


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

def test_skull_stripping_is_disabled():
    """The whole point: benchmark inputs must still have the skull on."""
    cfg = prepare_data.build_benchmark_config()
    steps = {s["name"]: s for s in cfg["steps"]}

    assert steps["skull_stripping"]["enabled"] is False


def test_atlas_is_mni152():
    """Masks are compared in one space; mni_mask only makes sense in MNI."""
    cfg = prepare_data.build_benchmark_config()

    assert "MNI152" in cfg["atlas"]["name"]


def test_bias_correction_is_registration_only():
    """Bias correction must not alter the volume the tools see.

    use_for_registration_only keeps it as a registration aid, so the benchmark
    input stays the original intensities.
    """
    cfg = prepare_data.build_benchmark_config()
    bc = next(s for s in cfg["steps"] if s["name"] == "bias_correction")

    assert bc["params"]["use_for_registration_only"] is True


def test_registration_enabled_at_1mm():
    cfg = prepare_data.build_benchmark_config()
    reg = next(s for s in cfg["steps"] if s["name"] == "registration")

    assert reg["enabled"] is True
    assert reg["params"]["output_resolution"] == [1.0, 1.0, 1.0]


def test_config_is_derived_from_the_live_one(tmp_path):
    """Built from the production config, not hand-written.

    A hand-written config drifts: the benchmark would silently stop matching
    the pipeline it is supposed to characterise.
    """
    cfg = prepare_data.build_benchmark_config()

    # Keys the production config carries that a from-scratch dict would miss.
    assert "fsl" in cfg
    assert cfg["atlas"].get("cache_dir")


# ---------------------------------------------------------------------------
# Stage 05 invocation
# ---------------------------------------------------------------------------

def test_stage05_command_uses_positional_dirs(tmp_path):
    """Stage 05 takes input/output positionally — not as --input-dir flags.

    The June draft used flags; that CLI does not exist and the run would die
    on argument parsing.
    """
    cmd = prepare_data.build_stage05_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        config_path=tmp_path / "cfg.yaml",
        lesion_type="glioblastoma",
        max_subjects=None,
    )

    assert "--input-dir" not in cmd
    assert "--output-dir" not in cmd
    assert str(tmp_path / "in") in cmd
    assert str(tmp_path / "out") in cmd


def test_stage05_command_passes_lesion_type(tmp_path):
    """Modalities come from lesion_types.yaml via this flag, not the config."""
    cmd = prepare_data.build_stage05_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        config_path=tmp_path / "cfg.yaml",
        lesion_type="multiple_sclerosis",
        max_subjects=None,
    )

    assert "--lesion-type" in cmd
    assert "multiple_sclerosis" in cmd


def test_stage05_command_has_no_transform_dir_flag(tmp_path):
    """Stage 05 derives it as output_dir.parent/transformations; no flag exists."""
    cmd = prepare_data.build_stage05_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        config_path=tmp_path / "cfg.yaml",
        lesion_type="glioblastoma",
        max_subjects=None,
    )

    assert "--transform-dir" not in cmd


def test_stage05_command_includes_max_subjects_when_given(tmp_path):
    cmd = prepare_data.build_stage05_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        config_path=tmp_path / "cfg.yaml",
        lesion_type="glioblastoma",
        max_subjects=3,
    )

    assert "--max-subjects" in cmd
    assert "3" in cmd


# ---------------------------------------------------------------------------
# Skip-existing
# ---------------------------------------------------------------------------

def test_prepared_subject_detected_as_done(tmp_path):
    import nibabel as nib
    import numpy as np

    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    for mod in ("t1", "t2"):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                 str(anat / f"sub-001_ses-001_{mod}.nii.gz"))

    assert prepare_data.subject_is_prepared(tmp_path, "sub-001", "ses-001", ["t1", "t2"])


def test_truncated_output_is_not_treated_as_done(tmp_path):
    """A half-written volume must be redone, not counted as prepared.

    exists() would accept it; the benchmark would then measure strippers on a
    corrupt input.
    """
    import nibabel as nib
    import numpy as np

    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    good = anat / "sub-001_ses-001_t1.nii.gz"
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32), np.eye(4)), str(good))
    blob = good.read_bytes()
    good.write_bytes(blob[: len(blob) // 2])

    assert not prepare_data.subject_is_prepared(tmp_path, "sub-001", "ses-001", ["t1"])


def test_stage05_command_uses_absolute_paths(tmp_path, monkeypatch):
    """Paths must survive the subprocess's working directory.

    run_stage05 executes Stage 05 with cwd=scripts/, so a relative path from
    the caller resolves against the wrong directory and Stage 05 dies with
    "Input directory not found" — caught on the first real smoke run.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()

    cmd = prepare_data.build_stage05_command(
        input_dir=Path("in"),
        output_dir=Path("out"),
        config_path=Path("cfg.yaml"),
        lesion_type="glioblastoma",
        max_subjects=None,
    )

    for arg in cmd[2:5]:
        if arg.startswith("-"):
            continue
        assert Path(arg).is_absolute(), f"{arg!r} must be absolute"


def test_stage05_runs_from_the_project_root(monkeypatch):
    """Stage 05 must run with the same cwd production gives it.

    Production runs stages from the project root (pipeline_manager sets
    cwd=pipeline_root and the orchestrator inherits it), and the config's
    relative paths — `atlas.cache_dir: data/templates` — only resolve there.
    Running from scripts/ made ANTs fail to open the atlas.
    """
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(prepare_data.subprocess, "run", fake_run)
    prepare_data.run_stage05(["python", "noop"])

    assert Path(seen["cwd"]) == prepare_data.PROJECT_ROOT
