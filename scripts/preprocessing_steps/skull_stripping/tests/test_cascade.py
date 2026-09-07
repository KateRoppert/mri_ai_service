from pathlib import Path
import numpy as np
import nibabel as nib

from preprocessing_steps.skull_stripping import dispatcher
from preprocessing_steps.skull_stripping import process_subject_skull_stripping
import preprocessing_steps.skull_stripping as ss
from preprocessing_steps.skull_stripping import gpu_pool


def test_build_cascade_orders_method_first_then_cascade():
    order = dispatcher.build_cascade_order(
        {"method": "hdbet", "cascade": ["synthstrip", "hdbet", "bet"]})
    assert order == ["hdbet", "synthstrip", "bet"]


def test_build_cascade_falls_back_to_single_fallback_method():
    order = dispatcher.build_cascade_order({"method": "hdbet", "fallback_method": "bet"})
    assert order == ["hdbet", "bet"]


def test_build_cascade_method_only():
    assert dispatcher.build_cascade_order({"method": "bet"}) == ["bet"]


def test_build_cascade_ignores_empty_and_duplicates():
    order = dispatcher.build_cascade_order({
        "method": "HDBET",
        "cascade": ["bet", "", "bet", "hdbet"],
    })
    assert order == ["hdbet", "bet"]


def _write_nifti(path, arr, zooms=(5.0, 5.0, 5.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.uint8), np.eye(4))
    img.header.set_zooms(zooms)
    nib.save(img, path)
    return path


def _passing_mask():
    # 20^3 voxels × 125 mm³ = 1000 ml — inside the hard 300–2500 window.
    arr = np.zeros((24, 24, 24), dtype=np.uint8)
    arr[2:22, 2:22, 2:22] = 1
    return arr


def _empty_mask():
    return np.zeros((24, 24, 24), dtype=np.uint8)


def _clean_mask():
    """Inside both the hard gates and the review windows (~1331 ml)."""
    arr = np.zeros((26, 26, 26), dtype=np.uint8)
    arr[2:24, 2:24, 2:24] = 1          # 22^3 x 125 mm3 = 1331 ml
    return arr


class _FakeStripper:
    def __init__(self, name, uses_gpu, mask_arr, available=True):
        self.name = name
        self.uses_gpu = uses_gpu
        self._mask_arr = mask_arr
        self._available = available
        self.calls = 0
        self.seen_params = None

    def is_available(self):
        return self._available

    def strip(self, input_path, output_path, mask_path=None, params=None):
        self.calls += 1
        self.seen_params = params or {}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if mask_path is not None:
            _write_nifti(mask_path, self._mask_arr)
            _write_nifti(output_path, self._mask_arr)
        return {
            "success": True,
            "output_path": str(output_path),
            "mask_path": str(mask_path),
            "processing_time": 0.0,
        }


def _subject_anat(tmp_path):
    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-001_ses-001_t1.nii.gz").write_bytes(b"x")
    return anat


def _register(monkeypatch, mapping):
    """mapping: method name -> factory that returns a stripper instance."""
    monkeypatch.setattr(ss, "STRIPPERS", dict(mapping))
    monkeypatch.setattr(dispatcher, "STRIPPERS", dict(mapping))


def test_cascade_retries_when_first_mask_fails_validation(monkeypatch, tmp_path):
    first = _FakeStripper("hdbet", True, _empty_mask())
    second = _FakeStripper("bet", False, _passing_mask())
    _register(monkeypatch, {
        "hdbet": lambda: first,
        "bet": lambda: second,
    })
    anat = _subject_anat(tmp_path)
    params = {
        "method": "hdbet",
        "fallback_method": "bet",
        "reference_modality": "t1",
        "apply_to_all": False,
        "cleanup": False,
    }
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params=params,
    )
    assert first.calls == 1
    assert second.calls == 1
    assert results["t1"]["success"] is True
    assert results["t1"]["method"] == "bet"
    assert results["t1"]["mask_validation"]["valid"] is True


def _flagged_mask():
    """Passes hard gates, trips review flags (small but not catastrophic)."""
    arr = np.zeros((24, 24, 24), dtype=np.uint8)
    arr[2:22, 2:22, 2:22] = 1          # 1000 ml
    return arr


def test_review_flags_move_on_to_the_next_stripper(monkeypatch, tmp_path):
    """Flags are the whole point of the cascade.

    A mask that trips review flags is not good enough to stop on while
    untried tools remain — otherwise validation detects a defect and then
    ships it anyway, which is what happened in production (MNI masks with
    eyes and holes were accepted with flags raised).
    """
    first = _FakeStripper("hdbet", True, _flagged_mask())     # flagged
    second = _FakeStripper("bet", False, _clean_mask())       # clean
    _register(monkeypatch, {"hdbet": lambda: first, "bet": lambda: second})
    anat = _subject_anat(tmp_path)
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "fallback_method": "bet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
            "validation": {"review_min_ml": 1200, "review_max_ml": 1800},
        },
    )
    assert first.calls == 1
    assert second.calls == 1
    assert results["t1"]["method"] == "bet"
    assert not (results["t1"]["mask_validation"].get("review_flags") or [])


def test_flagged_mask_is_kept_when_no_tool_is_left(monkeypatch, tmp_path):
    """Better a flagged mask than none: the patient still needs one."""
    only = _FakeStripper("hdbet", True, _flagged_mask())
    _register(monkeypatch, {"hdbet": lambda: only})
    anat = _subject_anat(tmp_path)
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
            "validation": {"review_min_ml": 1200, "review_max_ml": 1800},
        },
    )
    assert results["t1"]["success"] is True
    assert results["t1"]["method"] == "hdbet"
    assert "MASK_VOLUME_TOO_SMALL" in results["t1"]["mask_validation"]["review_flags"]


def test_first_flagged_candidate_wins_when_all_are_flagged(monkeypatch, tmp_path):
    """All flagged → keep the configured preference order.

    `method` is first in the cascade because the operator ranked it highest;
    with nothing clean to prefer, that ranking still decides.
    """
    first = _FakeStripper("hdbet", True, _flagged_mask())
    second = _FakeStripper("bet", False, _flagged_mask())
    _register(monkeypatch, {"hdbet": lambda: first, "bet": lambda: second})
    anat = _subject_anat(tmp_path)
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "fallback_method": "bet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
            "validation": {"review_min_ml": 1200, "review_max_ml": 1800},
        },
    )
    assert first.calls == 1
    assert second.calls == 1, "the cascade should have tried the alternative"
    assert results["t1"]["method"] == "hdbet", "fall back to the preferred tool"


def test_retry_on_review_can_be_switched_off(monkeypatch, tmp_path):
    """Escape hatch: keep the old accept-on-flags behaviour via config."""
    first = _FakeStripper("hdbet", True, _flagged_mask())
    second = _FakeStripper("bet", False, _clean_mask())
    _register(monkeypatch, {"hdbet": lambda: first, "bet": lambda: second})
    anat = _subject_anat(tmp_path)
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "fallback_method": "bet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
            "validation": {
                "review_min_ml": 1200,
                "review_max_ml": 1800,
                "retry_on_review": False,
            },
        },
    )
    assert second.calls == 0
    assert results["t1"]["method"] == "hdbet"


def test_cascade_logs_metrics_and_accept_without_retry(monkeypatch, tmp_path, caplog):
    import logging
    first = _FakeStripper("bet", False, _passing_mask())
    second = _FakeStripper("hdbet", True, _passing_mask())
    _register(monkeypatch, {
        "bet": lambda: first,
        "hdbet": lambda: second,
    })
    anat = _subject_anat(tmp_path)
    with caplog.at_level(logging.INFO, logger="preprocessing_steps.skull_stripping"):
        process_subject_skull_stripping(
            subject_dir=anat,
            output_dir=tmp_path / "out",
            transform_dir=tmp_path / "xfm",
            modalities=["t1"],
            params={
                "method": "bet",
                "fallback_method": "hdbet",
                "reference_modality": "t1",
                "apply_to_all": False,
                "cleanup": False,
            },
        )
    text = caplog.text
    assert "volume=" in text
    assert "ACCEPT 'bet'" in text
    assert "not trying remaining ['hdbet']" in text
    assert second.calls == 0


def test_cpu_fallback_does_not_hold_gpu_slot(monkeypatch, tmp_path):
    first = _FakeStripper("hdbet", True, _empty_mask())
    second = _FakeStripper("bet", False, _passing_mask())
    _register(monkeypatch, {
        "hdbet": lambda: first,
        "bet": lambda: second,
    })
    anat = _subject_anat(tmp_path)
    pool = gpu_pool.build_pool(["cuda:1"])
    params = {
        "method": "hdbet",
        "fallback_method": "bet",
        "reference_modality": "t1",
        "apply_to_all": False,
        "cleanup": False,
    }
    process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params=params,
        gpu_pool=pool,
    )
    assert first.calls == 1
    assert second.calls == 1
    assert first.seen_params["device"] == "cuda:1"
    assert "device" not in (second.seen_params or {})
    assert pool.get() == "cuda:1"


def test_unavailable_and_unknown_names_are_skipped(monkeypatch, tmp_path):
    missing = _FakeStripper("synthstrip", False, _passing_mask(), available=False)
    bet = _FakeStripper("bet", False, _passing_mask())
    _register(monkeypatch, {
        "synthstrip": lambda: missing,
        "bet": lambda: bet,
    })
    anat = _subject_anat(tmp_path)
    params = {
        "method": "synthstrip",
        "cascade": ["not_a_tool", "bet"],
        "reference_modality": "t1",
        "apply_to_all": False,
        "cleanup": False,
    }
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params=params,
    )
    assert missing.calls == 0
    assert bet.calls == 1
    assert results["t1"]["method"] == "bet"


def test_all_candidates_fail_returns_error(monkeypatch, tmp_path):
    first = _FakeStripper("hdbet", True, _empty_mask())
    second = _FakeStripper("bet", False, _empty_mask())
    _register(monkeypatch, {
        "hdbet": lambda: first,
        "bet": lambda: second,
    })
    anat = _subject_anat(tmp_path)
    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "fallback_method": "bet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
        },
    )
    assert results["t1"]["success"] is False
    assert "error" in results["t1"]


class _InputRecordingStripper(_FakeStripper):
    """Records what the input volume contained when strip() was called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_input_bytes = None

    def strip(self, input_path, output_path, mask_path=None, params=None):
        self.seen_input_bytes = Path(input_path).read_bytes()
        return super().strip(input_path, output_path, mask_path, params)


def test_retry_gets_the_original_volume_not_the_first_tools_output(
    monkeypatch, tmp_path
):
    """A rejected candidate must not corrupt the input for the next one.

    Stage 05 passes the SAME directory as subject_dir and output_dir
    (05_preprocessing.py: registered_anat == output_dir/sub/ses/anat), so a
    stripper writing its result to output_path overwrites the very volume the
    next cascade candidate has to read. The earlier retry test hid this by
    using two different directories.
    """
    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    original = b"ORIGINAL-REGISTERED-VOLUME-WITH-SKULL"
    (anat / "sub-001_ses-001_t1.nii.gz").write_bytes(original)

    first = _InputRecordingStripper("hdbet", True, _empty_mask())      # rejected
    second = _InputRecordingStripper("bet", False, _passing_mask())    # accepted
    _register(monkeypatch, {"hdbet": lambda: first, "bet": lambda: second})

    process_subject_skull_stripping(
        subject_dir=anat,
        # Production wiring: output lands in the same anat directory.
        output_dir=tmp_path,
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "fallback_method": "bet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
        },
    )

    assert first.seen_input_bytes == original
    assert second.seen_input_bytes == original, (
        "second candidate read the first one's output instead of the "
        "original volume"
    )


def test_rejected_candidate_does_not_leave_its_mask_behind(monkeypatch, tmp_path):
    """Only the accepted tool's mask may reach the final path.

    Otherwise a rejected mask sits at mask_path and gets applied to the other
    modalities in step 2.
    """
    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-001_ses-001_t1.nii.gz").write_bytes(b"original")

    first = _FakeStripper("hdbet", True, _empty_mask())     # rejected: empty
    second = _FakeStripper("bet", False, _passing_mask())   # accepted
    _register(monkeypatch, {"hdbet": lambda: first, "bet": lambda: second})

    results = process_subject_skull_stripping(
        subject_dir=anat,
        output_dir=tmp_path,
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params={
            "method": "hdbet",
            "fallback_method": "bet",
            "reference_modality": "t1",
            "apply_to_all": False,
            "cleanup": False,
        },
    )

    final_mask = tmp_path / "xfm" / "sub-001" / "ses-001" / "anat" / "sub-001_ses-001_brain_mask.nii.gz"
    assert final_mask.is_file()
    arr = np.asanyarray(nib.load(str(final_mask)).dataobj)
    assert arr.sum() > 0, "the accepted tool's mask must be the one that lands"
    assert results["t1"]["method"] == "bet"
