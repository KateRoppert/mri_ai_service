import queue as _queue

import pytest

from preprocessing_steps.skull_stripping import gpu_pool


def _fake_gpus(monkeypatch, gpus):
    monkeypatch.setattr(gpu_pool, "enumerate_gpus", lambda: list(gpus))


def test_auto_picks_all_usable_gpus_highest_free_first(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 500}, {"index": 1, "free_mb": 19000}])
    # index 0 is below DEFAULT_MIN_FREE_MB (2000) -> dropped
    assert gpu_pool.resolve_devices({"device": "auto"}) == ["cuda:1"]


def test_auto_with_two_free_gpus_orders_by_free_desc(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 8000}, {"index": 1, "free_mb": 19000}])
    assert gpu_pool.resolve_devices({"device": "auto"}) == ["cuda:1", "cuda:0"]


def test_exclude_gpus_removes_display_card(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 8000}, {"index": 1, "free_mb": 19000}])
    assert gpu_pool.resolve_devices({"device": "auto", "exclude_gpus": [0]}) == ["cuda:1"]


def test_gpu_pool_size_caps_devices(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 9000}, {"index": 1, "free_mb": 9000}])
    assert gpu_pool.resolve_devices({"device": "auto", "gpu_pool_size": 1}) == ["cuda:0"]


def test_explicit_cuda_index_is_passed_through(monkeypatch):
    _fake_gpus(monkeypatch, [])  # must not consult enumeration
    assert gpu_pool.resolve_devices({"device": "cuda:1"}) == ["cuda:1"]


def test_explicit_cpu(monkeypatch):
    assert gpu_pool.resolve_devices({"device": "cpu"}) == ["cpu"]


def test_auto_with_no_usable_gpu_falls_back_to_cpu(monkeypatch):
    _fake_gpus(monkeypatch, [])
    assert gpu_pool.resolve_devices({"device": "auto"}) == ["cpu"]


def test_default_device_is_auto(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 19000}])
    assert gpu_pool.resolve_devices({}) == ["cuda:0"]


def test_build_pool_seeds_all_devices():
    pool = gpu_pool.build_pool(["cuda:0", "cuda:1"])
    got = {pool.get(), pool.get()}
    assert got == {"cuda:0", "cuda:1"}
    assert pool.empty()


def test_acquire_device_returns_slot_to_pool():
    pool = gpu_pool.build_pool(["cuda:0"])
    with gpu_pool.acquire_device(pool) as dev:
        assert dev == "cuda:0"
        assert pool.empty()          # slot held while in the block
    assert pool.get() == "cuda:0"    # released back afterwards


def test_acquire_device_releases_on_exception():
    pool = gpu_pool.build_pool(["cuda:0"])
    with pytest.raises(RuntimeError):
        with gpu_pool.acquire_device(pool):
            raise RuntimeError("boom")
    assert pool.get() == "cuda:0"    # released despite the error


def test_acquire_device_none_pool_yields_cpu():
    with gpu_pool.acquire_device(None) as dev:
        assert dev == "cpu"


def test_build_pool_without_manager_is_in_process_queue():
    pool = gpu_pool.build_pool(["cpu"])
    assert isinstance(pool, _queue.Queue)
