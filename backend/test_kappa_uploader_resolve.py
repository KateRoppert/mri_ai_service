import asyncio
from pathlib import Path

import kappa_dataset_resolver
import kappa_uploader


def _make_uploader(**kwargs):
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="r1",
        output_path="/tmp",
        token="tok",
        user_id=52,
        user_type_id=4,
        lesion_type="glioblastoma",
        preprocessing_config_path=cfg,
        **kwargs,
    )


def test_fixed_dataset_id_is_used_without_resolving(monkeypatch):
    """The dataset chosen at run start (backend/numbering.py) must not be
    re-resolved at upload — resolving again could pick a different dataset if
    `current` was repointed while the run was in flight."""
    async def fail_resolve(**kwargs):
        raise AssertionError("must not resolve when dataset_id was fixed at start")
    monkeypatch.setattr(kappa_dataset_resolver, "resolve_or_create", fail_resolve)

    up = _make_uploader(dataset_id=337)
    result = asyncio.run(up._resolve_dataset_id())

    assert result == 337


def test_resolve_delegates_to_the_shared_resolver_with_the_right_user(monkeypatch):
    seen = {}

    async def fake_resolve(**kwargs):
        seen.update(kwargs)
        return 777

    monkeypatch.setattr(kappa_dataset_resolver, "resolve_or_create", fake_resolve)
    # A hit here means no dataset was created, so the empty-dataset warning
    # (which calls the network) is skipped.
    monkeypatch.setattr(kappa_uploader, "get_dataset_id", lambda *a: 777)

    up = _make_uploader()
    result = asyncio.run(up._resolve_dataset_id())

    assert result == 777
    assert seen["user_id"] == 52
    assert seen["lesion_type"] == "glioblastoma"


def test_warns_before_creating_when_mapping_has_no_dataset(monkeypatch):
    """A miss must trigger the empty-dataset check (KI-015) before resolving —
    that is the ordering _warn_if_empty_dataset_exists relies on."""
    calls = []

    async def fake_resolve(**kwargs):
        return 900

    async def fake_warn(self):
        calls.append("warned")

    monkeypatch.setattr(kappa_dataset_resolver, "resolve_or_create", fake_resolve)
    monkeypatch.setattr(kappa_uploader, "get_dataset_id", lambda *a: None)
    monkeypatch.setattr(kappa_uploader.KappaUploader, "_warn_if_empty_dataset_exists",
                        fake_warn)

    up = _make_uploader()
    asyncio.run(up._resolve_dataset_id())

    assert calls == ["warned"]


def test_no_warning_when_mapping_already_has_a_dataset(monkeypatch):
    calls = []

    async def fake_resolve(**kwargs):
        return 777

    async def fake_warn(self):
        calls.append("warned")

    monkeypatch.setattr(kappa_dataset_resolver, "resolve_or_create", fake_resolve)
    monkeypatch.setattr(kappa_uploader, "get_dataset_id", lambda *a: 777)
    monkeypatch.setattr(kappa_uploader.KappaUploader, "_warn_if_empty_dataset_exists",
                        fake_warn)

    up = _make_uploader()
    asyncio.run(up._resolve_dataset_id())

    assert calls == []
