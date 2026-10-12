"""
N4 runs on a <=1 mm copy when its output only feeds registration.

SibBMS P000067 ses-005 (3D 0.35 mm, t1 231 Mvox) needed ~7 GB for N4 at
native resolution, the largest stage-05 peak. The corrected image is only used
to compute transforms, which registration computes on <=1 mm copies anyway.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing_steps.bias_correction import (  # noqa: E402
    apply_n4_bias_correction,
    process_subject_bias_correction,
    working_spacing,
)


@pytest.mark.parametrize("native, expected", [
    ((0.347, 0.347, 0.55), (1.0, 1.0, 1.0)),
    ((0.47, 0.47, 5.0), (1.0, 1.0, 5.0)),
    ((1.2, 1.2, 6.0), (1.2, 1.2, 6.0)),
])
def test_working_spacing_never_upsamples(native, expected):
    assert working_spacing(native, 1.0) == pytest.approx(expected)


def test_working_spacing_none_keeps_native():
    assert working_spacing((0.35, 0.35, 0.35), None) == pytest.approx((0.35,) * 3)


def _phantom(path, spacing, size_mm=48.0, bias=True):
    """A bright ellipsoid 'head' with a smooth multiplicative bias, oblique
    direction and non-zero origin so geometry mistakes show up."""
    n = [int(round(size_mm / s)) for s in spacing]
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, k) for k in n[::-1]], indexing="ij")
    head = ((x / 0.8) ** 2 + (y / 0.7) ** 2 + (z / 0.75) ** 2) < 1
    field = 1.0 + (0.4 * x if bias else 0.0)
    rng = np.random.default_rng(0)
    data = (head * (100.0 + rng.normal(0, 2, head.shape)) * field).astype(np.float32)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(spacing)
    img.SetOrigin((12.0, -30.0, 7.5))
    c, s_ = np.cos(0.2), np.sin(0.2)
    img.SetDirection((c, -s_, 0.0, s_, c, 0.0, 0.0, 0.0, -1.0))
    sitk.WriteImage(img, str(path))
    return img


def _bounds(img):
    """Physical positions of the first and last voxel centres."""
    last = [k - 1 for k in img.GetSize()]
    return (np.array(img.TransformIndexToPhysicalPoint([0, 0, 0])),
            np.array(img.TransformContinuousIndexToPhysicalPoint(last)))


def test_fine_image_corrected_on_coarse_copy_same_physical_space(tmp_path):
    src = _phantom(tmp_path / "t1.nii.gz", (0.4, 0.4, 0.5))
    out = tmp_path / "out" / "t1.nii.gz"

    res = apply_n4_bias_correction(tmp_path / "t1.nii.gz", out,
                                   shrink_factor=2, n_iterations=[20, 20],
                                   max_spacing_mm=1.0)

    assert res["success"], res
    got = sitk.ReadImage(str(out))
    assert got.GetSpacing() == pytest.approx((1.0, 1.0, 1.0), abs=1e-4)
    assert got.GetOrigin() == pytest.approx(src.GetOrigin(), abs=1e-4)
    assert got.GetDirection() == pytest.approx(src.GetDirection(), abs=1e-5)
    # Same physical extent within one coarse voxel.
    s0, s1 = _bounds(src)
    g0, g1 = _bounds(got)
    assert np.allclose(g0, s0, atol=1e-3)
    assert np.allclose(g1, s1, atol=1.0)


def test_thick_axes_are_not_touched(tmp_path):
    _phantom(tmp_path / "t2.nii.gz", (0.5, 0.5, 4.0), size_mm=48.0)
    out = tmp_path / "out" / "t2.nii.gz"

    res = apply_n4_bias_correction(tmp_path / "t2.nii.gz", out, shrink_factor=1,
                                   n_iterations=[10], max_spacing_mm=1.0)

    assert res["success"], res
    assert sitk.ReadImage(str(out)).GetSpacing() == pytest.approx((1.0, 1.0, 4.0), abs=1e-4)


def test_bias_is_reduced_on_the_copy(tmp_path):
    _phantom(tmp_path / "t1.nii.gz", (0.5, 0.5, 0.5))
    out = tmp_path / "out" / "t1.nii.gz"

    apply_n4_bias_correction(tmp_path / "t1.nii.gz", out, shrink_factor=2,
                             n_iterations=[50, 50], max_spacing_mm=1.0)

    before = sitk.GetArrayFromImage(_phantom(tmp_path / "ref.nii.gz", (1.0, 1.0, 1.0)))
    after = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
    inside = before > 50

    def cv(a):
        return a[inside].std() / a[inside].mean()

    assert cv(after) < 0.7 * cv(before)


def _subject(tmp_path, spacing):
    anat = tmp_path / "in" / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    _phantom(anat / "sub-001_ses-001_t1.nii.gz", spacing)
    return anat


def test_registration_only_output_is_downsampled(tmp_path):
    anat = _subject(tmp_path, (0.4, 0.4, 0.4))
    res = process_subject_bias_correction(
        anat, tmp_path / "final", ["t1"],
        {"use_for_registration_only": True, "shrink_factor": 2, "n_iterations": [10]},
        temp_dir=tmp_path / "tmp", max_spacing_mm=1.0)

    out = Path(res["t1"]["output_path"])
    assert out.parent == tmp_path / "tmp" / "sub-001" / "ses-001" / "anat"
    assert sitk.ReadImage(str(out)).GetSpacing() == pytest.approx((1.0,) * 3, abs=1e-4)


def test_final_output_keeps_native_resolution(tmp_path):
    anat = _subject(tmp_path, (0.4, 0.4, 0.4))
    res = process_subject_bias_correction(
        anat, tmp_path / "final", ["t1"],
        {"use_for_registration_only": False, "shrink_factor": 2, "n_iterations": [10]},
        temp_dir=tmp_path / "tmp", max_spacing_mm=1.0)

    out = Path(res["t1"]["output_path"])
    assert out.parent == tmp_path / "final" / "sub-001" / "ses-001" / "anat"
    assert sitk.ReadImage(str(out)).GetSpacing() == pytest.approx((0.4,) * 3, abs=1e-4)
