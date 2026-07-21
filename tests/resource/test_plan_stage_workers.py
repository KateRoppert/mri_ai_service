import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from utils.resource_planner import plan_stage_workers
from utils.config_loader import load_resource_config

GB = 1_000_000_000

_CFG = {
    "safety_factor": 0.85,
    "min_workers": 1,
    "stages": {
        "stage_05_preprocessing": {"k_bytes_per_voxel": 17.7, "reserve_bytes": 2 * GB},
    },
}


def _nii(tmp_path, name, shape):
    p = tmp_path / name
    nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)), str(p))
    return p


def test_real_resource_config_loads_and_has_stages():
    cfg = load_resource_config()
    assert "stage_05_preprocessing" in cfg["stages"]
    assert cfg["stages"]["stage_05_preprocessing"]["k_bytes_per_voxel"] > 0


def test_large_input_reduces_workers(tmp_path):
    # 231M-voxel input * 17.7 B ~= 4.1 GB/worker; budget 20*0.85-2 = 15 GB -> 3 workers
    big = _nii(tmp_path, "big.nii.gz", (310, 864, 864))
    r = plan_stage_workers("stage_05_preprocessing", [big], requested=6,
                           config=_CFG, budget_bytes=20 * GB)
    assert r.actual_workers == 3


def test_small_input_keeps_requested(tmp_path):
    small = _nii(tmp_path, "small.nii.gz", (64, 64, 40))
    r = plan_stage_workers("stage_05_preprocessing", [small], requested=6,
                           config=_CFG, budget_bytes=20 * GB)
    assert r.actual_workers == 6


def test_unknown_stage_falls_back_to_requested(tmp_path):
    big = _nii(tmp_path, "big.nii.gz", (310, 864, 864))
    r = plan_stage_workers("stage_99_unknown", [big], requested=6,
                           config=_CFG, budget_bytes=20 * GB)
    assert r.actual_workers == 6


def test_cpu_cap_forwarded(tmp_path):
    small = _nii(tmp_path, "small.nii.gz", (64, 64, 40))
    r = plan_stage_workers("stage_05_preprocessing", [small], requested=6,
                           cpu_cap=2, config=_CFG, budget_bytes=20 * GB)
    assert r.actual_workers == 2


def test_malformed_yaml_falls_back_to_requested(tmp_path, monkeypatch):
    """A YAML parse error in resource_config.yaml must not raise into the stage."""
    from utils.resource_planner import plan_stage_workers
    bad = tmp_path / "resource_config.yaml"
    bad.write_text("stages:\n  stage_05_preprocessing: [unbalanced\n")  # invalid YAML
    monkeypatch.setattr(
        "utils.config_loader.load_resource_config",
        lambda: (_ for _ in ()).throw(Exception("simulated parse error")),
    )
    # No explicit config= passed, so plan_stage_workers must call (the now-broken)
    # load_resource_config() itself and NOT raise.
    result = plan_stage_workers("stage_05_preprocessing", [], requested=6)
    assert result.actual_workers == 6


def test_stage_missing_cost_key_falls_back_to_requested():
    from utils.resource_planner import plan_stage_workers
    broken_cfg = {
        "safety_factor": 0.85, "min_workers": 1,
        "stages": {"stage_05_preprocessing": {"reserve_bytes": 2_000_000_000}},  # no k_bytes_per_voxel
    }
    result = plan_stage_workers("stage_05_preprocessing", [], requested=6, config=broken_cfg)
    assert result.actual_workers == 6
