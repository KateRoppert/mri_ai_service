from preprocessing_steps.skull_stripping import load_manifest, BetStripper


def test_bet_manifest_loads_with_required_keys():
    m = load_manifest("bet")
    assert m["name"] == "bet"
    assert m["tool_type"] == "skull_stripping"
    assert "compute" in m and "requires_gpu" in m["compute"]
    assert "mas_metadata" in m and "agent_type" in m["mas_metadata"]


def test_stripper_exposes_its_manifest():
    assert BetStripper().manifest["name"] == "bet"


def test_missing_manifest_returns_empty_dict():
    assert load_manifest("does-not-exist") == {}
