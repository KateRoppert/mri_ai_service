import importlib.util
import sys
from pathlib import Path

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


def test_stage04_uses_plan_stage_workers(tmp_path):
    mod = _load("quality_stage04", "04_assess_quality.py")
    assert hasattr(mod, "_plan_workers_for_inputs")
    big = tmp_path / "big.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((310, 864, 864), np.int16), np.eye(4)), str(big))
    # 231M * 6.9 ~= 1.6 GB/worker; 20*0.85-1.5 = 15.5 GB -> 9 workers, capped by requested=6
    plan = mod._plan_workers_for_inputs([big], requested=6, budget_bytes=20_000_000_000)
    assert plan.actual_workers == 6
    # tighter budget forces fewer
    plan2 = mod._plan_workers_for_inputs([big], requested=6, budget_bytes=6_000_000_000)
    assert plan2.actual_workers < 6
