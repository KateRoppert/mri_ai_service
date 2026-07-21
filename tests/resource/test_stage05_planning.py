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


def test_stage05_uses_plan_stage_workers(tmp_path):
    """Stage 05 must size its pool via the planner, not raw args.workers."""
    mod = _load("preprocessing_stage05", "05_preprocessing.py")
    assert hasattr(mod, "_plan_workers_for_inputs"), \
        "stage 05 should expose a thin _plan_workers_for_inputs helper"
    big = tmp_path / "big.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((310, 864, 864), np.int16), np.eye(4)), str(big))
    # 231M * 17.7 ~= 4.1 GB/worker; 20*0.85-2 = 15 GB -> 3 workers
    plan = mod._plan_workers_for_inputs([big], requested=6, budget_bytes=20_000_000_000)
    assert plan.actual_workers == 3
