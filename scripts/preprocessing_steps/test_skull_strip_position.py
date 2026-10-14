"""
Experiment switch: skull stripping before or after atlas registration (KI-062).

Today stage 05 registers the skull-on t1 to MNI152 and strips afterwards.
That registration lands wrong and differs between runs (up to 63 mm on 2D
sessions). Registering the stripped brain to the brain template was the
most even fix in a side experiment; `position: before_registration` lets
stage 05 do that; it is the default since 2026-10-14, and
`position: after_registration` keeps the old order.
"""
import sys
from pathlib import Path

import ants
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing_steps import registration  # noqa: E402
from preprocessing_steps import skull_stripping  # noqa: E402
from preprocessing_steps.registration import (  # noqa: E402
    register_to_atlas,
    resolve_brain_template,
    warp_mask_to_atlas,
)
from preprocessing_steps.skull_stripping import (  # noqa: E402
    create_native_brain_mask,
    skull_strip_position,
)


# --- the switch ---------------------------------------------------------------

def test_default_position_is_before_registration():
    """Default since 2026-10-14 (KI-062): better and repeatable registration."""
    assert skull_strip_position({}) == "before_registration"
    assert skull_strip_position(None) == "before_registration"


def test_after_registration_is_still_accepted():
    assert skull_strip_position({"position": "after_registration"}) == "after_registration"


def test_before_registration_is_accepted():
    assert skull_strip_position({"position": "before_registration"}) == "before_registration"


def test_unknown_position_is_an_error():
    with pytest.raises(ValueError, match="position"):
        skull_strip_position({"position": "sideways"})


# --- brain template -----------------------------------------------------------

def test_brain_template_next_to_the_atlas(tmp_path):
    atlas = tmp_path / "MNI152_T1_1mm.nii.gz"
    atlas.write_text("x")
    brain = tmp_path / "MNI152_T1_1mm_brain.nii.gz"
    brain.write_text("x")
    assert resolve_brain_template(atlas) == brain


def test_brain_template_from_fsl(tmp_path, monkeypatch):
    atlas = tmp_path / "templates" / "MNI152_T1_1mm.nii.gz"
    atlas.parent.mkdir()
    atlas.write_text("x")
    fsl = tmp_path / "fsl"
    (fsl / "data" / "standard").mkdir(parents=True)
    brain = fsl / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"
    brain.write_text("x")
    monkeypatch.setenv("FSLDIR", str(fsl))
    assert resolve_brain_template(atlas) == brain


def test_brain_template_missing_is_a_clear_error(tmp_path, monkeypatch):
    atlas = tmp_path / "sri24_t1.nii.gz"
    atlas.write_text("x")
    monkeypatch.setenv("FSLDIR", str(tmp_path / "nofsl"))
    with pytest.raises(FileNotFoundError, match="sri24_t1_brain"):
        resolve_brain_template(atlas)


# --- mask into atlas space -----------------------------------------------------

def _identity_transform(path):
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    ants.write_transform(tx, str(path))
    return path


def test_native_mask_lands_on_the_atlas_grid_and_stays_binary(tmp_path):
    native = np.zeros((40, 40, 20), dtype="float32")
    native[10:30, 10:30, 5:15] = 1
    native_img = ants.from_numpy(native, spacing=(1.0, 1.0, 2.0))
    atlas = ants.from_numpy(np.zeros((50, 50, 50), dtype="float32"), spacing=(1.0, 1.0, 1.0))
    ants.image_write(native_img, str(tmp_path / "mask.nii.gz"))
    ants.image_write(atlas, str(tmp_path / "atlas.nii.gz"))

    out = warp_mask_to_atlas(tmp_path / "mask.nii.gz", tmp_path / "atlas.nii.gz",
                             _identity_transform(tmp_path / "id.mat"), tmp_path / "warped.nii.gz")

    warped = ants.image_read(str(out)).numpy()
    assert warped.shape == (50, 50, 50)
    assert set(np.unique(warped)) <= {0.0, 1.0}
    # 20 x 20 x (10 slices x 2 mm) = 8000 mm3 of brain, at 1 mm3 per voxel
    assert abs(warped.sum() - 8000) < 800


# --- native brain mask ----------------------------------------------------------

class _FakeStripper:
    name = "fake"
    uses_gpu = False

    def __init__(self):
        self.seen_spacing = None

    def strip(self, input_path, output_path, mask_path, params=None):
        img = ants.image_read(str(input_path))
        self.seen_spacing = img.spacing
        mask = img.new_image_like((img.numpy() > 0).astype("float32"))
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(mask, str(mask_path))
        ants.image_write(img * mask, str(output_path))
        return {"success": True, "output_path": str(output_path), "mask_path": str(mask_path)}


def test_native_mask_is_made_on_a_1mm_copy(tmp_path, monkeypatch):
    fine = np.zeros((60, 60, 60), dtype="float32")
    fine[15:45, 15:45, 15:45] = 100
    ref = tmp_path / "sub-001_ses-001_t1.nii.gz"
    ants.image_write(ants.from_numpy(fine, spacing=(0.5, 0.5, 0.5)), str(ref))
    stripper = _FakeStripper()
    monkeypatch.setattr(skull_stripping, "get_stripper", lambda params: stripper)

    result = create_native_brain_mask(ref, tmp_path / "native", {"method": "fake"})

    assert result["success"] is True
    assert stripper.seen_spacing == pytest.approx((1.0, 1.0, 1.0))
    assert Path(result["mask_path"]).exists()
    assert Path(result["brain_path"]).exists()
    assert result["method"] == "fake"


def test_native_mask_failure_is_reported(tmp_path, monkeypatch):
    ref = tmp_path / "sub-001_ses-001_t1.nii.gz"
    ants.image_write(ants.from_numpy(np.ones((10, 10, 10), dtype="float32")), str(ref))

    class Broken(_FakeStripper):
        def strip(self, **kw):
            return {"success": False, "error": "boom"}

    monkeypatch.setattr(skull_stripping, "get_stripper", lambda params: Broken())
    result = create_native_brain_mask(ref, tmp_path / "native", {"method": "broken"})
    assert result["success"] is False
    assert "boom" in result["error"]


# --- registration overrides --------------------------------------------------

@pytest.fixture
def spy_registration(monkeypatch):
    calls = []
    real = ants.registration

    def spy(fixed, moving, **kwargs):
        calls.append({"fixed_shape": fixed.shape, "moving_shape": moving.shape,
                      "random_seed": kwargs.get("random_seed", "absent")})
        return real(fixed=fixed, moving=moving, **kwargs)

    monkeypatch.setattr(registration.ants, "registration", spy)
    return calls


def _vol(tmp_path, name, shape):
    rng = np.random.default_rng(0)
    vol = rng.random(shape).astype("float32")
    path = tmp_path / name
    ants.image_write(ants.from_numpy(vol), str(path))
    return path


def test_default_registration_call_is_unchanged(tmp_path, spy_registration):
    atlas = _vol(tmp_path, "atlas.nii.gz", (20, 20, 20))
    moving = _vol(tmp_path, "t1.nii.gz", (20, 20, 20))
    register_to_atlas(moving, atlas, tmp_path / "out.nii.gz", tmp_path / "x.mat",
                      number_of_iterations=5)
    assert spy_registration[0]["random_seed"] == "absent"


def test_overrides_register_brain_to_brain_template_with_a_seed(tmp_path, spy_registration):
    atlas = _vol(tmp_path, "atlas.nii.gz", (20, 20, 20))
    atlas_brain = _vol(tmp_path, "atlas_brain.nii.gz", (18, 18, 18))
    moving = _vol(tmp_path, "t1.nii.gz", (20, 20, 20))
    brain = _vol(tmp_path, "t1_brain.nii.gz", (16, 16, 16))

    result = register_to_atlas(moving, atlas, tmp_path / "out.nii.gz", tmp_path / "x.mat",
                               number_of_iterations=5, moving_override=brain,
                               atlas_override=atlas_brain, random_seed=42)

    assert result["success"] is True
    assert spy_registration[0]["fixed_shape"] == (18, 18, 18)
    assert spy_registration[0]["moving_shape"] == (16, 16, 16)
    assert spy_registration[0]["random_seed"] == 42
