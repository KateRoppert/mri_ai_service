import pytest
from pathlib import Path

from preprocessing_steps.skull_stripping import dispatcher
from preprocessing_steps.skull_stripping.base import SkullStripperBase


class _FakeAvailable(SkullStripperBase):
    name = "fake_ok"
    def is_available(self): return True
    def strip(self, *a, **k): return {"success": True, "vram_used_gb": 0.0}


class _FakeUnavailable(SkullStripperBase):
    name = "fake_bad"
    def is_available(self): return False
    def strip(self, *a, **k): return {"success": True, "vram_used_gb": 0.0}


def test_get_stripper_returns_registered_instance():
    s = dispatcher.get_stripper("bet")
    assert s.name == "bet"


def test_get_stripper_unknown_method_raises():
    with pytest.raises(ValueError):
        dispatcher.get_stripper("nope")


def test_resolve_uses_primary_when_available(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_ok", _FakeAvailable)
    s = dispatcher.resolve_stripper({"method": "fake_ok"})
    assert s.name == "fake_ok"


def test_resolve_falls_back_when_primary_unavailable(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_bad", _FakeUnavailable)
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_ok", _FakeAvailable)
    s = dispatcher.resolve_stripper({"method": "fake_bad", "fallback_method": "fake_ok"})
    assert s.name == "fake_ok"


def test_resolve_raises_when_primary_and_fallback_unavailable(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_bad", _FakeUnavailable)
    with pytest.raises(RuntimeError):
        dispatcher.resolve_stripper({"method": "fake_bad", "fallback_method": "fake_bad"})


def test_resolve_raises_runtime_error_for_unknown_fallback(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_bad", _FakeUnavailable)
    with pytest.raises(RuntimeError):
        dispatcher.resolve_stripper({"method": "fake_bad", "fallback_method": "no_such_tool"})
