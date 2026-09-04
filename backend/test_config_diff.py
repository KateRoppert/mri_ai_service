"""Differences between a stopped run's settings and the current ones."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_diff import diff_configs


def _config(method="bet", reference="t1"):
    return {
        "steps": [
            {"name": "skull_stripping", "enabled": True,
             "params": {"method": method, "reference_modality": reference}},
        ]
    }


def test_identical_configs_have_no_differences():
    assert diff_configs(_config(), _config()) == []


def test_changed_skull_stripper_is_reported():
    result = diff_configs(_config(method="bet"), _config(method="hdbet"))

    assert len(result) == 1
    assert result[0]["was"] == "bet"
    assert result[0]["now"] == "hdbet"
    assert "череп" in result[0]["setting"].lower()


def test_changed_reference_modality_is_reported():
    result = diff_configs(_config(reference="t1"), _config(reference="t1c"))

    assert len(result) == 1
    assert result[0]["was"] == "t1"


def test_multiple_differences_all_reported():
    result = diff_configs(_config("bet", "t1"), _config("hdbet", "t1c"))

    assert len(result) == 2


def test_missing_section_does_not_crash():
    assert diff_configs({}, _config()) != []
