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


def test_new_dataset_tags_include_predefined_ml_tag(monkeypatch):
    captured = {}

    async def fake_list(token, user_id, user_type_id):
        return []

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return 321

    monkeypatch.setattr(kappa_uploader, "get_dataset_id", lambda u, l, p: None)  # miss -> create
    monkeypatch.setattr(kappa_uploader, "set_dataset_id", lambda *a, **k: None)
    monkeypatch.setattr(kappa_uploader, "list_user_datasets", fake_list)
    monkeypatch.setattr(kappa_uploader, "create_dataset", fake_create)

    up = _make_uploader()
    result = asyncio.run(up._resolve_dataset_id())

    assert result == 321
    # Kappa rejects dataset creation unless datasetTags has a predefined ML tag.
    # "Image Segmentation" is the predefined tag (Computer Vision is the task type).
    assert "Image Segmentation" in captured["dataset_tags"]
