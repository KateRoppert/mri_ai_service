"""
KI-053: nib.aff2axcodes(affine) returns None for an axis when the affine
matrix is degenerate/non-orthogonal — reorient.py did ''.join(axcodes)
straight afterward, crashing with a cryptic "sequence item 2: expected
str instance, NoneType found" instead of a clear, actionable message.
Reproduced twice on real data: BO-214's t1c series (2026-08-14 and
2026-08-17 runs) — t1/t2/t2fl of the same patient reoriented fine, only
t1c's affine was degenerate.
"""
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


reorient = _load_module("reorient.py", "reorient_orientation_guard")


class TestReorientToStandardOrientationGuard:
    def test_returns_clear_error_when_axcodes_contains_none(self, tmp_path):
        fake_img = MagicMock()
        fake_img.affine = MagicMock()
        with patch.object(reorient.nib, "load", return_value=fake_img), \
             patch.object(reorient.nib, "aff2axcodes", return_value=(None, 'P', 'S')):
            result = reorient.reorient_to_standard(
                tmp_path / "sub-214_ses-001_t1c.nii.gz",
                tmp_path / "out.nii.gz",
            )

        assert result["success"] is False
        assert "degenerate" in result["error"].lower()
        assert "sub-214_ses-001_t1c.nii.gz" in result["error"]

    def test_still_succeeds_for_a_normal_orientation(self, tmp_path):
        fake_img = MagicMock()
        fake_img.affine = MagicMock()
        fake_img.header = MagicMock()
        fake_reoriented = MagicMock()
        fake_reoriented.affine = fake_img.affine
        fake_reoriented.get_fdata.return_value = MagicMock()
        with patch.object(reorient.nib, "load", return_value=fake_img), \
             patch.object(reorient.nib, "aff2axcodes", return_value=('L', 'P', 'S')), \
             patch.object(reorient.nib, "save"):
            result = reorient.reorient_to_standard(
                tmp_path / "sub-001_ses-001_t1.nii.gz",
                tmp_path / "out.nii.gz",
                target_orientation="LPS",
            )

        assert result["success"] is True
        assert result["original_orientation"] == "LPS"
        assert result["transformation_applied"] is False


class TestCheckOrientationGuard:
    def test_raises_clear_error_when_axcodes_contains_none(self, tmp_path):
        fake_img = MagicMock()
        fake_img.affine = MagicMock()
        with patch.object(reorient.nib, "load", return_value=fake_img), \
             patch.object(reorient.nib, "aff2axcodes", return_value=(None, 'P', 'S')):
            with pytest.raises(ValueError, match="degenerate"):
                reorient.check_orientation(tmp_path / "sub-214_ses-001_t1c.nii.gz")
