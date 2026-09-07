import numpy as np
import nibabel as nib
from pathlib import Path

from preprocessing_steps.skull_stripping.mni_mask import MniMaskStripper
from preprocessing_steps.skull_stripping import dispatcher


def _write(tmp, name, arr):
    p = tmp / name
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), p)
    return p


def test_registered_in_strippers():
    assert "mni_mask" in dispatcher.STRIPPERS
    assert dispatcher.STRIPPERS["mni_mask"]().name == "mni_mask"


def test_uses_gpu_is_false():
    assert MniMaskStripper.uses_gpu is False


def test_dilation_loose_is_superset_of_strict():
    base = np.zeros((10, 10, 10))
    base[4:6, 4:6, 4:6] = 1
    strict = MniMaskStripper._dilate(base, radius_vox=0)
    loose = MniMaskStripper._dilate(base, radius_vox=2)
    assert loose.sum() > strict.sum()
    assert np.all(loose[strict > 0] > 0)


def test_strip_produces_mask_and_output(tmp_path):
    img = np.random.rand(10, 10, 10).astype(np.float32)
    in_path = _write(tmp_path, "in.nii.gz", img)
    mask_template = np.zeros((10, 10, 10))
    mask_template[3:7, 3:7, 3:7] = 1
    tmpl_path = _write(tmp_path, "mni_mask.nii.gz", mask_template)
    s = MniMaskStripper()
    out = tmp_path / "out.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    r = s.strip(
        in_path,
        out,
        mask,
        {"variant": "strict", "mask_template": str(tmpl_path)},
    )
    assert r["success"] is True
    assert out.exists() and mask.exists()
    applied = nib.load(str(out)).get_fdata()
    assert np.allclose(applied[mask_template == 0], 0)


def test_loose_mask_covers_more_than_strict(tmp_path):
    img = np.ones((10, 10, 10), dtype=np.float32)
    in_path = _write(tmp_path, "in.nii.gz", img)
    mask_template = np.zeros((10, 10, 10))
    mask_template[4:6, 4:6, 4:6] = 1
    tmpl_path = _write(tmp_path, "mni_mask.nii.gz", mask_template)
    s = MniMaskStripper()
    strict_mask = tmp_path / "strict.nii.gz"
    loose_mask = tmp_path / "loose.nii.gz"
    s.strip(in_path, tmp_path / "s.nii.gz", strict_mask,
            {"variant": "strict", "mask_template": str(tmpl_path)})
    s.strip(in_path, tmp_path / "l.nii.gz", loose_mask,
            {"variant": "loose", "dilation_vox": 2, "mask_template": str(tmpl_path)})
    strict = nib.load(str(strict_mask)).get_fdata() > 0
    loose = nib.load(str(loose_mask)).get_fdata() > 0
    assert loose.sum() > strict.sum()
    assert np.all(loose[strict])


def test_strip_fills_internal_holes_in_template(tmp_path):
    img = np.ones((12, 12, 12), dtype=np.float32)
    in_path = _write(tmp_path, "in.nii.gz", img)
    mask_template = np.zeros((12, 12, 12), dtype=np.uint8)
    mask_template[2:10, 2:10, 2:10] = 1
    mask_template[5:7, 5:7, 5:7] = 0  # enclosed cavity
    tmpl_path = _write(tmp_path, "mni_mask.nii.gz", mask_template)
    s = MniMaskStripper()
    mask = tmp_path / "mask.nii.gz"
    r = s.strip(
        in_path,
        tmp_path / "out.nii.gz",
        mask,
        {"variant": "strict", "mask_template": str(tmpl_path)},
    )
    assert r["success"] is True
    out_mask = nib.load(str(mask)).get_fdata() > 0
    assert out_mask[6, 6, 6]
    assert out_mask[2:10, 2:10, 2:10].all()
    applied = nib.load(str(tmp_path / "out.nii.gz")).get_fdata()
    assert applied[6, 6, 6] == 1.0


def test_repo_template_strip_has_no_enclosed_holes(tmp_path):
    from preprocessing_steps.skull_stripping.mni_mask import resolve_mask_template
    from preprocessing_steps.skull_stripping.validation import mask_metrics

    tmpl = resolve_mask_template()
    if tmpl is None:
        return
    tmpl_img = nib.load(str(tmpl))
    in_path = tmp_path / "in.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.ones(tmpl_img.shape, dtype=np.float32),
            tmpl_img.affine,
            tmpl_img.header,
        ),
        str(in_path),
    )
    mask = tmp_path / "mask.nii.gz"
    r = MniMaskStripper().strip(
        in_path, tmp_path / "out.nii.gz", mask, {"variant": "strict"}
    )
    assert r["success"] is True
    metrics = mask_metrics(mask)
    assert metrics["hole_volume_ml"] < 0.5
    assert metrics["n_holes"] == 0
