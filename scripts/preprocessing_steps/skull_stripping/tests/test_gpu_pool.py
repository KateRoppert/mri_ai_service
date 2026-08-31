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
