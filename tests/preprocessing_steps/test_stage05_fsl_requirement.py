"""Stage 05 must not demand FSL for work that does not use it.

FSL is used by exactly one step — skull stripping (BET). Reorient runs on
nibabel, bias correction and registration on ANTs. Requiring FSL when skull
stripping is switched off blocks any run that legitimately skips it, which is
how the benchmark builds its inputs (registration-space volumes, skull on).
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

stage05 = importlib.import_module("05_preprocessing")


def _config(skull_stripping_enabled: bool) -> dict:
    return {
        "fsl": {"fsl_dir": "/usr/share/fsl/6.0"},
        "steps": [
            {"name": "reorient", "enabled": True, "params": {}},
            {"name": "registration", "enabled": True, "params": {}},
            {"name": "skull_stripping", "enabled": skull_stripping_enabled,
             "params": {"method": "hdbet", "fallback_method": "bet"}},
        ],
    }


def test_fsl_required_when_skull_stripping_enabled():
    # BET is the universal fallback of the cascade, so an enabled step still
    # needs FSL even when the configured method is HD-BET.
    assert stage05.requires_fsl(_config(True)) is True


def test_fsl_not_required_when_skull_stripping_disabled():
    assert stage05.requires_fsl(_config(False)) is False


def test_fsl_required_when_step_absent_from_config():
    # Absent means "default on" elsewhere in the stage; stay conservative.
    assert stage05.requires_fsl({"steps": []}) is True
