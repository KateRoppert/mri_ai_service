from preprocessing_steps.skull_stripping import (
    dispatcher,
    gpu_pool,
    process_subject_skull_stripping,
)
import preprocessing_steps.skull_stripping as ss  # package (__init__) namespace


class _FakeStripper:
    name = "hdbet"
    uses_gpu = True

    def __init__(self):
        self.seen_params = None

    def is_available(self):
        return True

    def strip(self, input_path, output_path, mask_path=None, params=None):
        self.seen_params = params or {}
        return {"success": True, "output_path": str(output_path),
                "mask_path": str(mask_path), "processing_time": 0.0}


def _setup(monkeypatch, tmp_path, fake):
    # Reference modality file must exist for the glob in the function.
    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-001_ses-001_t1.nii.gz").write_bytes(b"x")
    # Cascade instantiates via STRIPPERS[name](), not get_stripper().
    monkeypatch.setattr(ss, "STRIPPERS", {"hdbet": lambda: fake})
    monkeypatch.setattr(dispatcher, "STRIPPERS", {"hdbet": lambda: fake})
    return anat


def test_gpu_tool_receives_device_from_pool(monkeypatch, tmp_path):
    fake = _FakeStripper()
    anat = _setup(monkeypatch, tmp_path, fake)
    pool = gpu_pool.build_pool(["cuda:1"])
    params = {
        "method": "hdbet",
        "reference_modality": "t1",
        "apply_to_all": False,
        "cleanup": False,
        "disable_tta": True,
        # This test pins the GPU slot; mask integrity is covered elsewhere.
        "validation": {"enabled": False},
    }

    process_subject_skull_stripping(
        subject_dir=anat, output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm", modalities=["t1"],
        params=params, gpu_pool=pool,
    )

    assert fake.seen_params["device"] == "cuda:1"
    assert fake.seen_params["disable_tta"] is True
    # slot returned to the pool after the call
    assert pool.get() == "cuda:1"
