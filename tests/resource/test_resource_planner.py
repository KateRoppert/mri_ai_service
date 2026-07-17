import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from utils.resource_planner import (
    cgroup_memory_limit_bytes,
    max_voxels,
    plan_workers,
    PlanResult,
)

GB = 1_000_000_000


# --- cgroup_memory_limit_bytes ---
def test_cgroup_v2_integer(tmp_path):
    p = tmp_path / "memory.max"
    p.write_text("21474836480\n")  # 20 GiB
    assert cgroup_memory_limit_bytes(v2_path=str(p), v1_path=str(tmp_path / "none")) == 21474836480


def test_cgroup_v2_max_means_unlimited(tmp_path):
    p = tmp_path / "memory.max"
    p.write_text("max\n")
    assert cgroup_memory_limit_bytes(v2_path=str(p), v1_path=str(tmp_path / "none")) is None


def test_cgroup_v1_unlimited_sentinel(tmp_path):
    p = tmp_path / "limit_in_bytes"
    p.write_text("9223372036854771712\n")  # v1 "unlimited"
    assert cgroup_memory_limit_bytes(v2_path=str(tmp_path / "none"), v1_path=str(p)) is None


def test_cgroup_absent_returns_none(tmp_path):
    assert cgroup_memory_limit_bytes(v2_path=str(tmp_path / "a"), v1_path=str(tmp_path / "b")) is None


# --- max_voxels ---
def test_max_voxels_picks_largest(tmp_path):
    small = tmp_path / "s.nii.gz"
    big = tmp_path / "b.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.int16), np.eye(4)), str(small))
    nib.save(nib.Nifti1Image(np.zeros((20, 30, 40), np.int16), np.eye(4)), str(big))
    assert max_voxels([small, big]) == 20 * 30 * 40


def test_max_voxels_empty_is_zero():
    assert max_voxels([]) == 0


def test_max_voxels_skips_unreadable(tmp_path):
    good = tmp_path / "g.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), np.int16), np.eye(4)), str(good))
    assert max_voxels([tmp_path / "missing.nii.gz", good]) == 125


# --- plan_workers ---
def test_memory_caps_below_requested():
    # budget 12 GB - 2 GB reserve = 10 GB; 4 GB/worker -> 2 workers
    r = plan_workers(requested=6, per_worker_bytes=4 * GB, budget_bytes=12 * GB,
                     reserve_bytes=2 * GB)
    assert r.actual_workers == 2


def test_requested_is_the_ceiling():
    # budget allows 100 but config asked for 4
    r = plan_workers(requested=4, per_worker_bytes=1 * GB, budget_bytes=200 * GB,
                     reserve_bytes=0)
    assert r.actual_workers == 4


def test_cpu_cap_applies():
    r = plan_workers(requested=8, per_worker_bytes=1 * GB, budget_bytes=200 * GB,
                     reserve_bytes=0, cpu_cap=3)
    assert r.actual_workers == 3


def test_no_budget_means_no_memory_cap():
    # budget_bytes None and cgroup unreadable -> fall back to requested (capped by cpu only)
    r = plan_workers(requested=5, per_worker_bytes=4 * GB, budget_bytes=None, cpu_cap=None)
    assert r.actual_workers == 5


def test_at_least_one_worker_when_nothing_fits():
    r = plan_workers(requested=6, per_worker_bytes=50 * GB, budget_bytes=20 * GB,
                     reserve_bytes=2 * GB)
    assert r.actual_workers == 1


def test_zero_per_worker_uses_requested():
    r = plan_workers(requested=4, per_worker_bytes=0, budget_bytes=20 * GB)
    assert r.actual_workers == 4


def test_reason_is_populated():
    r = plan_workers(requested=6, per_worker_bytes=4 * GB, budget_bytes=12 * GB,
                     reserve_bytes=2 * GB)
    assert isinstance(r, PlanResult)
    assert r.reason  # non-empty explanation for the stage log
