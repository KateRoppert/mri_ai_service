import numpy as np
import nibabel as nib
from preprocessing_steps.skull_stripping.validation import (
    cascade_decision_message,
    format_mask_check_line,
    mask_metrics,
    validate_mask,
)


def _write_mask(tmp, arr, zooms=(1.0, 1.0, 1.0)):
    p = tmp / "mask.nii.gz"
    img = nib.Nifti1Image(arr.astype(np.uint8), np.eye(4))
    hdr = img.header
    hdr.set_zooms(zooms)
    nib.save(img, p)
    return p


def test_valid_single_blob_passes(tmp_path):
    arr = np.zeros((40, 40, 40))
    arr[10:30, 10:30, 10:30] = 1  # ~8000 vox = 8 ml @1mm
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=2500)
    assert res["valid"] is True


def test_empty_mask_fails(tmp_path):
    arr = np.zeros((40, 40, 40))
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=2500)
    assert res["valid"] is False and "volume" in res["reason"]


def test_too_large_fails(tmp_path):
    arr = np.ones((40, 40, 40))  # 64000 vox = 64 ml; set max below it
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=10)
    assert res["valid"] is False


def test_fragmented_mask_fails(tmp_path):
    arr = np.zeros((40, 40, 40))
    arr[2:6, 2:6, 2:6] = 1  # blob A
    arr[30:38, 30:38, 30:38] = 1  # blob B (larger, disconnected)
    res = validate_mask(
        _write_mask(tmp_path, arr),
        min_ml=0.01,
        max_ml=2500,
        min_dominant_fraction=0.9,
        speckle_ml=0.0,
    )
    assert res["valid"] is False and "component" in res["reason"]


def test_speckles_below_one_ml_do_not_fail_lcc(tmp_path):
    """GBM HD-BET often leaves <1 ml islands; those must not trip cascade."""
    arr = np.zeros((80, 80, 80))
    arr[10:70, 10:70, 10:70] = 1  # 216 ml blob
    arr[2, 2, 2] = 1  # 0.001 ml speckle
    res = validate_mask(
        _write_mask(tmp_path, arr),
        min_ml=5,
        max_ml=2500,
        min_dominant_fraction=0.95,
        speckle_ml=1.0,
    )
    assert res["valid"] is True
    assert res["dominant_fraction"] >= 0.95


def test_missing_mask_file_fails(tmp_path):
    res = validate_mask(tmp_path / "nope.nii.gz")
    assert res["valid"] is False
    assert "volume" in res["reason"] or "missing" in res["reason"]


def test_review_flags_do_not_fail_when_fail_closed_false(tmp_path):
    arr = np.zeros((40, 40, 40))
    arr[10:30, 10:30, 10:30] = 1  # 8 ml — review-small, not catastrophic
    res = validate_mask(
        _write_mask(tmp_path, arr),
        min_ml=300,
        max_ml=2500,
        fail_closed=False,
        review_min_ml=1000,
        review_max_ml=1800,
    )
    assert res["valid"] is True
    flags = res["review_flags"]
    assert any("VOLUME" in f for f in flags)


def test_empty_still_fails_when_fail_closed_false(tmp_path):
    arr = np.zeros((40, 40, 40))
    res = validate_mask(_write_mask(tmp_path, arr), fail_closed=False)
    assert res["valid"] is False


def test_internal_holes_fail_even_when_volume_and_lcc_pass(tmp_path):
    """Swiss-cheese mask: one connected blob, adult volume, enclosed cavities.

    LCC of the foreground stays ~1.0 — that is why the old gate accepted
    MNI-mask outputs with black holes throughout the brain.
    """
    arr = np.zeros((80, 80, 80), dtype=np.uint8)
    arr[10:70, 10:70, 10:70] = 1  # 216 ml solid cube @1mm
    arr[25:45, 25:45, 25:45] = 0  # 8 ml enclosed hole
    arr[50:58, 50:58, 50:58] = 0  # 0.5 ml enclosed hole
    res = validate_mask(
        _write_mask(tmp_path, arr),
        min_ml=50,
        max_ml=2500,
        min_dominant_fraction=0.70,
        max_hole_ml=5.0,
    )
    assert res["valid"] is False
    assert "hole" in res["reason"]
    assert res["hole_volume_ml"] > 5.0


def test_solid_mask_reports_zero_hole_volume(tmp_path):
    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[5:35, 5:35, 5:35] = 1
    metrics = mask_metrics(_write_mask(tmp_path, arr))
    assert metrics["hole_volume_ml"] == 0.0
    assert metrics["n_holes"] == 0
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=2500)
    assert res["valid"] is True


def test_holes_become_review_flags_when_fail_closed_false(tmp_path):
    arr = np.zeros((80, 80, 80), dtype=np.uint8)
    arr[10:70, 10:70, 10:70] = 1
    arr[25:45, 25:45, 25:45] = 0
    res = validate_mask(
        _write_mask(tmp_path, arr),
        min_ml=50,
        max_ml=2500,
        max_hole_ml=5.0,
        fail_closed=False,
    )
    assert res["valid"] is True
    assert any("HOLE" in f for f in res["review_flags"])


def test_mask_metrics_reports_volume_and_edge_touch(tmp_path):
    arr = np.zeros((10, 10, 10))
    arr[0:3, 0:3, 0:3] = 1  # touches FOV corner
    metrics = mask_metrics(_write_mask(tmp_path, arr), speckle_ml=0.0)
    assert metrics["mask_volume_ml"] > 0
    assert metrics["edge_touch_ratio"] > 0
    assert metrics["n_components"] == 1
    assert 0 < metrics["lcc_ratio"] <= 1


def test_format_mask_check_line_includes_volume_and_flags():
    line = format_mask_check_line({
        "volume_ml": 1342.4,
        "dominant_fraction": 0.991,
        "edge_touch_ratio": 0.012,
        "n_components": 3,
        "n_components_kept": 1,
        "valid": True,
        "reason": "ok",
        "review_flags": ["MASK_VOLUME_TOO_SMALL"],
    })
    assert "volume=1342.4 ml" in line
    assert "LCC=0.991" in line
    assert "holes=0.0 ml" in line
    assert "review_flags=MASK_VOLUME_TOO_SMALL" in line


def test_cascade_decision_accept_does_not_retry():
    msg = cascade_decision_message(
        name="bet",
        accepted=True,
        remaining=["hdbet"],
        check={"valid": True, "reason": "ok", "review_flags": []},
    )
    assert "ACCEPT 'bet'" in msg
    assert "not trying remaining ['hdbet']" in msg


def test_cascade_decision_reject_retries_next():
    msg = cascade_decision_message(
        name="hdbet",
        accepted=False,
        remaining=["bet"],
        check={"valid": False, "reason": "volume 80ml out of [300,2500]",
               "review_flags": []},
    )
    assert "REJECT 'hdbet'" in msg
    assert "trying next: bet" in msg


def test_cascade_decision_review_flags_still_accept():
    msg = cascade_decision_message(
        name="hdbet",
        accepted=True,
        remaining=["bet"],
        check={"valid": True, "reason": "ok",
               "review_flags": ["MASK_VOLUME_TOO_SMALL"]},
    )
    assert "ACCEPT 'hdbet'" in msg
    assert "review_flags=MASK_VOLUME_TOO_SMALL" in msg
    assert "not a retry" in msg


def test_hole_gate_rejects_swiss_cheese_seen_in_production(tmp_path):
    """Calibrated on real runs, not guessed.

    Every correct mask measured on this project's data has 0.00 ml of
    enclosed holes; the defective MNI mask (KA126/sub-024) had 10.71 ml
    across 11 cavities and passed the old 20 ml gate. Cavities inside a
    brain mask are swiss cheese — ventricles belong to the mask as 1s — so
    there is no legitimate middle ground to protect.
    """
    import numpy as np
    import nibabel as nib
    from preprocessing_steps.skull_stripping.validation import validate_mask

    arr = np.zeros((30, 30, 30), dtype=np.uint8)
    arr[3:27, 3:27, 3:27] = 1                     # ~1728 ml at 5 mm voxels
    arr[12:15, 12:15, 12:15] = 0                  # enclosed cavity ~3.4 ml
    path = tmp_path / "holed.nii.gz"
    img = nib.Nifti1Image(arr, np.eye(4))
    img.header.set_zooms((5.0, 5.0, 5.0))
    nib.save(img, str(path))

    check = validate_mask(path)

    assert check["hole_volume_ml"] > 0
    assert check["valid"] is False, "an enclosed cavity must fail the hard gate"
    assert "holes" in check["reason"]


# ---------------------------------------------------------------------------
# Intensity-based leakage (needs the skull-on image, not just the mask)
# ---------------------------------------------------------------------------

def _skull_on_phantom(tmp_path, with_fat_blob: bool):
    """A crude head: dim background, mid-intensity 'brain', bright 'fat' rim.

    Mirrors what separates the classes on real data — the brightest voxels in
    a head T1 are fat (scalp, orbital), and brain has essentially none.
    """
    import numpy as np
    import nibabel as nib

    img = np.zeros((40, 40, 40), dtype=np.float32)
    img[5:35, 5:35, 5:35] = 100.0          # head/tissue
    img[12:28, 12:28, 12:28] = 300.0       # brain
    if with_fat_blob:
        img[8:11, 8:11, 8:11] = 3000.0     # fat blob (orbit/scalp)
    path = tmp_path / "head.nii.gz"
    head = nib.Nifti1Image(img, np.eye(4))
    head.header.set_zooms((5.0, 5.0, 5.0))
    nib.save(head, str(path))
    return path


def _mask_file(tmp_path, arr, name="mask.nii.gz"):
    """Write a mask with 5 mm voxels so its volume lands inside the hard gates.

    At 1 mm these phantoms are a few ml and fail the volume catastrophe check,
    which returns before any review flag is computed.
    """
    import nibabel as nib
    p = tmp_path / name
    img = nib.Nifti1Image(arr.astype("uint8"), np.eye(4))
    img.header.set_zooms((5.0, 5.0, 5.0))
    nib.save(img, str(p))
    return p


def test_leak_fraction_zero_for_a_brain_only_mask(tmp_path):
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import mask_metrics

    img = _skull_on_phantom(tmp_path, with_fat_blob=True)
    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[12:28, 12:28, 12:28] = 1           # exactly the brain
    m = mask_metrics(_mask_file(tmp_path, arr), image_path=img)

    assert m["leak_fraction"] == 0.0


def test_leak_fraction_fires_when_the_mask_swallows_fat(tmp_path):
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import mask_metrics

    img = _skull_on_phantom(tmp_path, with_fat_blob=True)
    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[12:28, 12:28, 12:28] = 1
    arr[8:11, 8:11, 8:11] = 1              # the fat blob too
    m = mask_metrics(_mask_file(tmp_path, arr), image_path=img)

    assert m["leak_fraction"] > 0
    assert m["leak_ml"] > 0


def test_leak_metrics_absent_without_an_image(tmp_path):
    """Geometry-only callers must keep working — the image is optional."""
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import mask_metrics

    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[12:28, 12:28, 12:28] = 1
    m = mask_metrics(_mask_file(tmp_path, arr))

    assert m["leak_fraction"] is None


def test_leak_flag_raised_in_validation(tmp_path):
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import validate_mask

    img = _skull_on_phantom(tmp_path, with_fat_blob=True)
    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[12:28, 12:28, 12:28] = 1
    arr[8:11, 8:11, 8:11] = 1
    check = validate_mask(_mask_file(tmp_path, arr), image_path=img,
                          max_leak_fraction=0.001)

    assert "MASK_LEAKAGE" in check["review_flags"]


# ---------------------------------------------------------------------------
# Asymmetry (geometry only — catches a one-sided cut)
# ---------------------------------------------------------------------------

def test_symmetric_mask_has_low_asymmetry(tmp_path):
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import mask_metrics

    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[10:30, 10:30, 10:30] = 1
    m = mask_metrics(_mask_file(tmp_path, arr))

    assert m["asymmetry"] < 0.05


def test_one_sided_cut_shows_up_as_asymmetry(tmp_path):
    """The failure mode volume cannot see: half the brain removed.

    A one-sided cut can leave the volume inside its window while losing a
    hemisphere, so only left/right balance catches it.
    """
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import mask_metrics

    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[20:30, 10:30, 10:30] = 1           # right side only
    m = mask_metrics(_mask_file(tmp_path, arr))

    assert m["asymmetry"] > 0.5


def test_asymmetry_flag_raised_in_validation(tmp_path):
    import numpy as np
    from preprocessing_steps.skull_stripping.validation import validate_mask

    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[20:30, 10:30, 10:30] = 1
    check = validate_mask(_mask_file(tmp_path, arr))

    assert "MASK_ASYMMETRIC" in check["review_flags"]


# ---------------------------------------------------------------------------
# Review volume band — calibrated 2026-09-09 on dropbox_33 (12 subjects x 4 tools)
# ---------------------------------------------------------------------------

def _mask_of_ml(tmp, millilitres, zooms=(5.0, 5.0, 5.0)):
    """A solid cube of a given volume. 5 mm isotropic keeps the array small
    while still reaching brain-sized volumes."""
    voxel_ml = float(np.prod(zooms)) / 1000.0
    side = int(round((millilitres / voxel_ml) ** (1 / 3)))
    arr = np.zeros((side + 4, side + 4, side + 4))
    arr[2:2 + side, 2:2 + side, 2:2 + side] = 1
    return _write_mask(tmp, arr, zooms)


def test_large_but_plausible_brain_is_not_flagged(tmp_path):
    """A 1900 ml mask must pass the default band.

    The old 1800 ml ceiling sat 58 ml below the worst known-bad mask in the
    calibration sample, so it separated nothing while flagging large heads.
    Since review flags advance the cascade, such a false alarm costs a correct
    mask — this test is the guard against quietly narrowing the band again.
    """
    res = validate_mask(_mask_of_ml(tmp_path, 1900))

    assert res["valid"] is True
    assert "MASK_VOLUME_TOO_LARGE" not in res["review_flags"]


def test_implausible_volumes_still_flagged(tmp_path):
    """The band is loose, not absent: it still catches frank nonsense."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    small = validate_mask(_mask_of_ml(tmp_path / "a", 700))
    large = validate_mask(_mask_of_ml(tmp_path / "b", 2300))

    assert "MASK_VOLUME_TOO_SMALL" in small["review_flags"]
    assert "MASK_VOLUME_TOO_LARGE" in large["review_flags"]
