"""
Tests for skull stripper selection (scripts/preprocessing_steps/skull_stripping).

Tool selection has to be exercised without the tools themselves installed —
FSL and HD-BET live in the web container, not on a dev machine or CI — so
availability is stubbed here. What matters is the decision logic: which tool
gets picked, when the fallback engages, and that a bad config fails loudly
instead of silently stripping with the wrong tool.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from preprocessing_steps.skull_stripping import (  # noqa: E402
    BetStripper,
    HdBetStripper,
    SkullStripperUnavailable,
    get_stripper,
    get_tool_params,
)


@pytest.fixture
def all_available(monkeypatch):
    monkeypatch.setattr(BetStripper, "is_available", lambda self: True)
    monkeypatch.setattr(HdBetStripper, "is_available", lambda self: True)


@pytest.fixture
def only_bet_available(monkeypatch):
    monkeypatch.setattr(BetStripper, "is_available", lambda self: True)
    monkeypatch.setattr(HdBetStripper, "is_available", lambda self: False)


@pytest.fixture
def none_available(monkeypatch):
    monkeypatch.setattr(BetStripper, "is_available", lambda self: False)
    monkeypatch.setattr(HdBetStripper, "is_available", lambda self: False)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_selects_configured_method(all_available):
    assert get_stripper({"method": "hdbet"}).name == "hdbet"
    assert get_stripper({"method": "bet"}).name == "bet"


def test_defaults_to_bet_when_method_absent(all_available):
    # An old config with no `method:` must behave as it always did.
    assert get_stripper({}).name == "bet"


def test_method_is_case_insensitive(all_available):
    assert get_stripper({"method": "HD-BET".replace("-", "")}).name == "hdbet"
    assert get_stripper({"method": "BET"}).name == "bet"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def test_falls_back_when_primary_unavailable(only_bet_available, caplog):
    stripper = get_stripper({"method": "hdbet", "fallback_method": "bet"})

    assert stripper.name == "bet"
    # The substitution must be visible — a run silently stripped with a
    # different tool than configured is a reproducibility trap.
    assert any("falling back" in r.message.lower() or "falling back" in r.getMessage().lower()
               for r in caplog.records)


def test_raises_when_primary_unavailable_and_no_fallback(only_bet_available):
    with pytest.raises(SkullStripperUnavailable, match="no fallback_method"):
        get_stripper({"method": "hdbet"})


def test_raises_when_neither_primary_nor_fallback_available(none_available):
    with pytest.raises(SkullStripperUnavailable, match="neither"):
        get_stripper({"method": "hdbet", "fallback_method": "bet"})


def test_no_fallback_needed_when_primary_available(all_available, caplog):
    stripper = get_stripper({"method": "hdbet", "fallback_method": "bet"})

    assert stripper.name == "hdbet"
    assert not any("falling back" in r.getMessage().lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Bad configuration
# ---------------------------------------------------------------------------

def test_unknown_method_raises(all_available):
    with pytest.raises(SkullStripperUnavailable, match="unknown skull stripping method"):
        get_stripper({"method": "synthstrip"})  # spec'd, not implemented yet


def test_unknown_fallback_raises(only_bet_available):
    with pytest.raises(SkullStripperUnavailable, match="fallback_method"):
        get_stripper({"method": "hdbet", "fallback_method": "nonesuch"})


# ---------------------------------------------------------------------------
# Tool parameters
# ---------------------------------------------------------------------------

def test_tool_params_read_from_nested_block():
    params = {"method": "hdbet", "tool_params": {"device": "cpu", "disable_tta": True}}

    assert get_tool_params(params) == {"device": "cpu", "disable_tta": True}


def test_legacy_flat_bet_params_still_honoured():
    # Existing production config puts these directly under params.
    params = {"method": "bet", "fractional_intensity": 0.35, "vertical_gradient": -0.1}

    assert get_tool_params(params) == {
        "fractional_intensity": 0.35,
        "vertical_gradient": -0.1,
    }


def test_nested_block_wins_over_legacy_key():
    params = {
        "method": "bet",
        "fractional_intensity": 0.35,
        "tool_params": {"fractional_intensity": 0.5},
    }

    assert get_tool_params(params)["fractional_intensity"] == 0.5


def test_tool_params_empty_when_nothing_configured():
    assert get_tool_params({"method": "hdbet"}) == {}


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def test_manifests_exist_and_declare_compute_needs():
    # The MAS coordinator will route on these; a tool without a manifest is
    # invisible to it.
    for stripper in (BetStripper(), HdBetStripper()):
        manifest = stripper.manifest
        assert manifest, f"{stripper.name} has no manifest"
        assert manifest["name"] == stripper.name
        assert "compute" in manifest, f"{stripper.name} manifest lacks compute block"


def test_hdbet_manifest_declares_gpu_requirement():
    manifest = HdBetStripper().manifest

    assert manifest["compute"]["requires_gpu"] is True
    # CPU fallback is what makes it deployable on a machine without a GPU.
    assert manifest["compute"]["cpu_fallback"] is True
