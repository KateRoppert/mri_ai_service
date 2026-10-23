import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nibabel as nib

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _expected_workers(stage, voxels, requested, budget_bytes, cpu_cap=None):
    """Worker count the planner should give, from configs/resource_config.yaml.
    Computed, not hard-coded: the per-voxel cost is recalibrated from measured
    peaks (35.0 -> 25.0 for stage 05 on 2026-10-12) and a literal answer
    breaks on every recalibration (KI-063)."""
    import math
    import yaml
    cfg = yaml.safe_load((PROJ_ROOT / "configs" / "resource_config.yaml").read_text())
    st = cfg["stages"][stage]
    usable = budget_bytes * cfg["safety_factor"] - st["reserve_bytes"]
    by_mem = max(cfg.get("min_workers", 1), math.floor(usable / (voxels * st["k_bytes_per_voxel"])))
    return min(requested, by_mem, cpu_cap or requested)


def test_stage05_uses_plan_stage_workers(tmp_path):
    """Stage 05 must size its pool via the planner, not raw args.workers."""
    mod = _load("preprocessing_stage05", "05_preprocessing.py")
    assert hasattr(mod, "_plan_workers_for_inputs"), \
        "stage 05 should expose a thin _plan_workers_for_inputs helper"
    big = tmp_path / "big.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((310, 864, 864), np.int16), np.eye(4)), str(big))
    plan = mod._plan_workers_for_inputs([big], requested=6, budget_bytes=32_000_000_000)
    expected = _expected_workers("stage_05_preprocessing", 310 * 864 * 864, 6, 32_000_000_000)
    assert 1 < expected < 6, "fixture no longer exercises the memory cap"
    assert plan.actual_workers == expected
