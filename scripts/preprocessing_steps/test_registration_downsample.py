"""
Intra-session registration is computed on <=1 mm copies.

SibBMS P000067 ses-005/006 are 3D 0.35 mm acquisitions (t1 231 Mvox, t2fl
171-186 Mvox). Registering t2fl to t1 at native resolution needed >9.2 GB and
the worker was OOM-killed on every run, losing both sessions. A rigid
transform lives in mm, and every stage-05 output lands on the 1 mm atlas
grid, so the transform can be found on 1 mm copies and applied to the
originals.
"""
import sys
from pathlib import Path

import ants
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as nd_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing_steps import registration  # noqa: E402
from preprocessing_steps.registration import (  # noqa: E402
    register_modalities,
    registration_spacing,
)


@pytest.mark.parametrize("native, expected", [
    ((0.347, 0.347, 0.55), (1.0, 1.0, 1.0)),   # ses-005 t1: everything coarsened
    ((0.47, 0.47, 5.0), (1.0, 1.0, 5.0)),      # 2D t2: thick slices kept
    ((1.2, 1.2, 6.0), (1.2, 1.2, 6.0)),        # coarse on every axis: untouched
])
def test_registration_spacing_never_upsamples(native, expected):
    assert registration_spacing(native, 1.0) == pytest.approx(expected)


def _blob_volume(spacing, shift_mm=(0.0, 0.0, 0.0), size_mm=64.0):
    """A textured 'head' sampled at `spacing` (mm, a multiple of 0.25).

    Texture is smoothed noise (fixed seed, ~1.5 mm grain) inside an
    ellipsoid, built once on a 0.25 mm master grid, shifted there, then
    subsampled — so fixed and moving share one texture. Smooth or
    hard-edged blobs gave ANTs' default 20% random metric sampling too
    little to hold on to (2.3-4.0 mm recovered for a 3 mm shift); real MRI
    and this texture give 2.96-3.01 mm.
    """
    master = 0.25
    rng = np.random.default_rng(0)
    n = int(size_mm / master)
    tex = gaussian_filter(rng.standard_normal((n, n, n)).astype("float32"), sigma=6)
    ax = (np.arange(n) + 0.5) * master - size_mm / 2
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    head = (((x / 24) ** 2 + (y / 20) ** 2 + (z / 16) ** 2) <= 1).astype("float32")
    vol = nd_shift(head * (100 + 400 * tex), [s / master for s in shift_mm], order=1)
    steps = [int(round(sp / master)) for sp in spacing]
    vol = vol[::steps[0], ::steps[1], ::steps[2]]
    return ants.from_numpy(np.ascontiguousarray(vol), spacing=tuple(float(sp) for sp in spacing))


@pytest.fixture
def spy_registration(monkeypatch):
    seen = []
    real = ants.registration

    def spy(fixed, moving, **kwargs):
        seen.append({"fixed": fixed.spacing, "moving": moving.spacing,
                     "fixed_shape": fixed.shape})
        return real(fixed=fixed, moving=moving, **kwargs)

    monkeypatch.setattr(registration.ants, "registration", spy)
    return seen


def test_fine_images_are_registered_on_1mm_copies(tmp_path, spy_registration):
    fixed = _blob_volume((0.5, 0.5, 0.5))
    moving = _blob_volume((0.5, 0.5, 0.5), shift_mm=(3.0, 0.0, 0.0))
    ants.image_write(fixed, str(tmp_path / "t1.nii.gz"))
    ants.image_write(moving, str(tmp_path / "t2fl.nii.gz"))

    result = register_modalities(
        reference_path=tmp_path / "t1.nii.gz",
        moving_path=tmp_path / "t2fl.nii.gz",
        output_path=tmp_path / "out" / "t2fl_in_t1.nii.gz",
        transform_path=tmp_path / "xfm" / "t2fl_to_t1.mat",
    )

    assert result["success"] is True
    assert (tmp_path / "xfm" / "t2fl_to_t1.mat").exists()
    assert spy_registration[0]["fixed"] == pytest.approx((1.0, 1.0, 1.0))
    assert spy_registration[0]["moving"] == pytest.approx((1.0, 1.0, 1.0))

    # The transform found on the copies maps fixed-space points to moving
    # space: the moving blob sits +3 mm along x.
    tx = ants.read_transform(str(tmp_path / "xfm" / "t2fl_to_t1.mat"))
    mapped = tx.apply_to_point((0.0, 0.0, 0.0))
    assert mapped[0] == pytest.approx(3.0, abs=0.3)
    assert abs(mapped[1]) < 0.3 and abs(mapped[2]) < 0.3


def test_coarse_images_are_registered_as_before(tmp_path, spy_registration):
    fixed = _blob_volume((1.25, 1.25, 1.25))
    moving = _blob_volume((1.25, 1.25, 1.25), shift_mm=(2.0, 0.0, 0.0))
    ants.image_write(fixed, str(tmp_path / "t1.nii.gz"))
    ants.image_write(moving, str(tmp_path / "t2.nii.gz"))

    register_modalities(
        reference_path=tmp_path / "t1.nii.gz",
        moving_path=tmp_path / "t2.nii.gz",
        output_path=tmp_path / "out.nii.gz",
        transform_path=tmp_path / "t2_to_t1.mat",
    )

    assert spy_registration[0]["fixed"] == pytest.approx((1.25, 1.25, 1.25))
    assert spy_registration[0]["fixed_shape"] == fixed.shape


def test_threshold_is_configurable(tmp_path, spy_registration):
    fixed = _blob_volume((0.5, 0.5, 0.5))
    ants.image_write(fixed, str(tmp_path / "t1.nii.gz"))
    ants.image_write(fixed, str(tmp_path / "t2fl.nii.gz"))

    register_modalities(
        reference_path=tmp_path / "t1.nii.gz",
        moving_path=tmp_path / "t2fl.nii.gz",
        output_path=tmp_path / "out.nii.gz",
        transform_path=tmp_path / "x.mat",
        max_registration_spacing_mm=2.0,
    )

    assert spy_registration[0]["fixed"] == pytest.approx((2.0, 2.0, 2.0))
