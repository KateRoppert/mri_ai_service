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


def test_stage07_measures_all_modalities_not_just_reference(tmp_path):
    """Regression test for the post-review fix: _input_refs must be built from
    ALL native modalities per mask, not just args.reference_modality, because
    inverse_transform_subject_masks (registration.py) warps into every
    modality's own native image. Here the reference modality's file is small
    but a non-reference modality's file is large — the planner must size
    workers off the large file, not the small reference one.
    """
    mod = _load("inverse_stage07_multimod", "07_inverse_transform.py")
    subj, sess = "P001", "S001"
    anat_dir = tmp_path / "nifti" / subj / sess / "anat"
    anat_dir.mkdir(parents=True)

    # Reference modality (t1) is SMALL.
    ref_path = anat_dir / f"{subj}_{sess}_t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 40), np.int16), np.eye(4)), str(ref_path))

    # Non-reference modality (t2fl) is LARGE — same shape as the "big" fixture
    # above (231M voxels, caps to 2 workers under a 20GB budget).
    large_path = anat_dir / f"{subj}_{sess}_t2fl.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((310, 864, 864), np.int16), np.eye(4)), str(large_path))

    # t1c and t2 are intentionally left absent for this subject/session —
    # max_voxels() must silently skip missing files (verified elsewhere).
    modalities = ["t1", "t1c", "t2", "t2fl"]
    masks = [(None, subj, sess)]

    # Mirrors the fixed _input_refs construction in 07_inverse_transform.py:
    # one candidate path per (mask, modality), not just the reference modality.
    input_refs = [
        anat_dir / f"{subj}_{sess}_{modality}.nii.gz"
        for (_mask_path, s, se) in masks
        for modality in modalities
    ]

    plan = mod._plan_workers_for_inputs(input_refs, requested=6, cpu_cap=5,
                                        budget_bytes=20_000_000_000)
    # Must be capped as if the large t2fl file drove the estimate, not as if
    # the small reference t1 file were the only input considered.
    assert plan.actual_workers == 2

    # Regression guard: if only the reference modality had been measured (the
    # pre-fix behavior), the tiny t1 file would not trip the memory cap at
    # all, so cpu_cap=5 alone would bind.
    plan_ref_only = mod._plan_workers_for_inputs([ref_path], requested=6, cpu_cap=5,
                                                 budget_bytes=20_000_000_000)
    assert plan_ref_only.actual_workers == 5
    assert plan.actual_workers < plan_ref_only.actual_workers
