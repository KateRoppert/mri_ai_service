"""
Bug: apply_transform() built an interp_map to numeric ITK interpolator
codes but never used it — instead passed the raw `interpolation` string
argument (default "Linear", capitalized) straight to
ants.apply_transforms()'s `interpolator` kwarg. ANTsPy only accepts a
specific set of lowercase-camelCase interpolator names (e.g. "linear",
not "Linear"), so the default value always failed:
"interpolator not supported - see {'linear', 'nearestNeighbor', ...}".

Real-world impact: registration.py calls apply_transform() with the
default interpolation (no explicit override) specifically to replace the
bias-corrected reference-modality (t1) output with a version derived from
the original, non-bias-corrected image — matching how every other
modality is finalized (bias correction is meant to be registration-only,
per preprocessing_config.yaml's `use_for_registration_only: true`).
Because this call always failed, the final t1 output silently stayed
bias-corrected instead, inconsistent with t1c/t2/t2fl, on every
preprocessing run to date.
"""
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, SCRIPTS_DIR / "preprocessing_steps" / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


registration = _load_module("registration.py", "registration_interp_fix")


def _run_apply_transform(tmp_path, **kwargs):
    captured = {}

    def fake_apply_transforms(fixed, moving, transformlist, interpolator, whichtoinvert):
        captured["interpolator"] = interpolator
        return MagicMock()

    with patch.object(registration, "load_ants_image", return_value=MagicMock()), \
         patch.object(registration, "save_ants_image"), \
         patch.object(registration.ants, "apply_transforms", side_effect=fake_apply_transforms):
        result = registration.apply_transform(
            moving_path=tmp_path / "moving.nii.gz",
            fixed_path=tmp_path / "fixed.nii.gz",
            transform_path=tmp_path / "transform.mat",
            output_path=tmp_path / "out.nii.gz",
            **kwargs,
        )
    return result, captured.get("interpolator")


class TestApplyTransformInterpolator:
    def test_default_interpolation_maps_to_valid_antspy_string(self, tmp_path):
        """Default call (no explicit interpolation=) must pass a lowercase
        ANTsPy-valid string, not the raw capitalized default "Linear"."""
        result, interpolator = _run_apply_transform(tmp_path)
        assert result["success"] is True
        assert interpolator == "linear"

    def test_capitalized_bspline_maps_correctly(self, tmp_path):
        result, interpolator = _run_apply_transform(tmp_path, interpolation="BSpline")
        assert result["success"] is True
        assert interpolator == "bSpline"

    def test_capitalized_nearest_neighbor_maps_correctly(self, tmp_path):
        result, interpolator = _run_apply_transform(tmp_path, interpolation="NearestNeighbor")
        assert result["success"] is True
        assert interpolator == "nearestNeighbor"

    def test_already_antspy_style_string_passes_through_unchanged(self, tmp_path):
        """Regression: register_ms_zone_atlases.py already calls this with
        the ANTsPy-correct lowercase-camelCase string directly (not one of
        the capitalized preset keys) — the fix must not break that call site."""
        result, interpolator = _run_apply_transform(tmp_path, interpolation="nearestNeighbor")
        assert result["success"] is True
        assert interpolator == "nearestNeighbor"
