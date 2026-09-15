import asyncio
from pathlib import Path

import kappa_uploader


def _make_uploader():
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="r1",
        output_path="/tmp",
        token="tok",
        user_id=52,
        user_type_id=4,
        lesion_type="glioblastoma",
        preprocessing_config_path=cfg,
    )


def test_resolve_passes_user_id_to_get_dataset_id(monkeypatch):
    seen = {}

    def fake_get(user_id, lesion_type, prep):
        seen["args"] = (user_id, lesion_type, prep)
        return 777  # hit -> no network, returns immediately

    monkeypatch.setattr(kappa_uploader, "get_dataset_id", fake_get)

    up = _make_uploader()
    result = asyncio.run(up._resolve_dataset_id())

    assert result == 777
    assert seen["args"][0] == 52                 # current user_id
    assert seen["args"][1] == "glioblastoma"
