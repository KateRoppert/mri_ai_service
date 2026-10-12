"""
Transforms are applied in single precision.

ants.apply_transforms in double cost +5.6 GB on a 3D 0.35 mm original
(231 Mvox). Float halves that; intensities change by ~1e-5 of the range, and
label masks taken back to native space must stay exact labels.
"""
import sys
from pathlib import Path

import ants
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing_steps.registration import (  # noqa: E402
    apply_transform,
    inverse_transform_mask,
)


def _rigid(path, angle=0.1, shift=(2.0, -3.0, 1.5)):
    tx = ants.create_ants_transform(transform_type="Euler3DTransform",
                                    parameters=(angle, 0.0, -angle / 2, *shift))
    ants.write_transform(tx, str(path))
    return path


def _image(spacing, values):
    return ants.from_numpy(values.astype(np.float32), spacing=spacing)


def test_inverse_mask_keeps_exact_labels(tmp_path):
    labels = np.zeros((40, 40, 40))
    labels[10:20, 10:25, 12:30] = 1
    labels[22:30, 15:20, 5:15] = 2
    ants.image_write(_image((1.0, 1.0, 1.0), labels), str(tmp_path / "mask.nii.gz"))
    native = _image((0.6, 0.6, 0.8), np.ones((70, 70, 50)))
    ants.image_write(native, str(tmp_path / "native.nii.gz"))

    res = inverse_transform_mask(tmp_path / "mask.nii.gz", tmp_path / "native.nii.gz",
                                 [_rigid(tmp_path / "t.mat")], tmp_path / "out.nii.gz")

    assert res["success"], res
    out = ants.image_read(str(tmp_path / "out.nii.gz")).numpy()
    assert set(np.unique(out)) <= {0.0, 1.0, 2.0}
    assert (out == 1).sum() > 0 and (out == 2).sum() > 0


def test_apply_transform_matches_double_precision(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.normal(1000, 200, (48, 48, 40))
    moving = _image((0.5, 0.5, 0.6), vol)
    fixed = _image((1.0, 1.0, 1.0), np.zeros((30, 30, 30)))
    ants.image_write(moving, str(tmp_path / "m.nii.gz"))
    ants.image_write(fixed, str(tmp_path / "f.nii.gz"))
    tx = _rigid(tmp_path / "t.mat")

    res = apply_transform(tmp_path / "m.nii.gz", tmp_path / "f.nii.gz", tx, tmp_path / "o.nii.gz")

    assert res["success"], res
    got = ants.image_read(str(tmp_path / "o.nii.gz")).numpy()
    ref = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[str(tx)],
                                interpolator="linear").numpy()
    assert np.abs(got - ref).max() < 1e-3 * np.abs(ref).max()
