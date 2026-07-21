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


def test_stage07_memory_and_cpu_both_cap(tmp_path):
    mod = _load("inverse_stage07", "07_inverse_transform.py")
    assert hasattr(mod, "_plan_workers_for_inputs")
    big = tmp_path / "big.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((310, 864, 864), np.int16), np.eye(4)), str(big))
    # memory: 231M * 22.5 ~= 5.2 GB/worker; 20*0.85-2 = 15 GB -> 2 workers
    plan = mod._plan_workers_for_inputs([big], requested=6, cpu_cap=5,
                                        budget_bytes=20_000_000_000)
    assert plan.actual_workers == 2  # memory is tighter than the cpu cap of 5
    # cpu tighter than memory
    small = tmp_path / "small.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 40), np.int16), np.eye(4)), str(small))
    plan2 = mod._plan_workers_for_inputs([small], requested=6, cpu_cap=2,
                                         budget_bytes=20_000_000_000)
    assert plan2.actual_workers == 2  # cpu cap wins
