"""
Memory-aware worker sizing (see docs/superpowers/specs/2026-07-17-memory-aware-worker-sizing-design.md).

Pure helpers a stage calls at startup to cap its parallelism at what the
container memory budget allows, given an input-size cost estimate. Fail-safe:
when no budget can be determined the planner returns the requested value, so a
working run is never broken.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def cgroup_memory_limit_bytes(
    v2_path: str = "/sys/fs/cgroup/memory.max",
    v1_path: str = "/sys/fs/cgroup/memory/memory.limit_in_bytes",
) -> Optional[int]:
    """Container memory limit in bytes, or None if unavailable/unlimited.

    Reads cgroup v2 (memory.max) then v1 (memory.limit_in_bytes). "max" or a
    near-2**63 sentinel means unlimited -> None.
    """
    for path in (v2_path, v1_path):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            val = int(raw)
        except ValueError:
            continue
        if val >= 2 ** 62:  # v1 unlimited sentinel
            return None
        return val
    return None


def max_voxels(nifti_paths: Iterable[Path]) -> int:
    """Largest voxel count (product of header dims) among the given NIfTI files.

    Header-only via nibabel (no voxel load). Unreadable files are skipped; an
    empty input returns 0.
    """
    import nibabel as nib

    best = 0
    for p in nifti_paths:
        try:
            shape = nib.load(str(p)).shape
        except Exception:
            continue
        best = max(best, int(np.prod(shape)))
    return best


@dataclass
class PlanResult:
    actual_workers: int
    reason: str


def plan_workers(
    requested: int,
    per_worker_bytes: float,
    budget_bytes: Optional[int] = None,
    cpu_cap: Optional[int] = None,
    safety_factor: float = 0.85,
    reserve_bytes: int = 1_500_000_000,
    min_workers: int = 1,
) -> PlanResult:
    """Cap `requested` by CPU and by the memory budget; never below min_workers.

    budget_bytes=None -> read the cgroup limit * safety_factor. If that is also
    unavailable, memory does not cap (fail-safe to requested/CPU).
    """
    caps = [requested]
    parts = [f"requested={requested}"]
    if cpu_cap is not None:
        caps.append(cpu_cap)
        parts.append(f"cpu={cpu_cap}")

    if budget_bytes is None:
        limit = cgroup_memory_limit_bytes()
        budget_bytes = int(limit * safety_factor) if limit is not None else None

    if budget_bytes is not None and per_worker_bytes > 0:
        usable = budget_bytes - reserve_bytes
        mem_workers = int(usable // per_worker_bytes) if usable > 0 else 0
        caps.append(mem_workers)
        parts.append(
            f"mem=({budget_bytes / 1e9:.1f}-{reserve_bytes / 1e9:.1f})GB"
            f"/{per_worker_bytes / 1e9:.2f}GB={mem_workers}"
        )
    else:
        parts.append("mem=unbounded")

    actual = max(min_workers, min(caps))
    return PlanResult(actual, f"min({', '.join(parts)}) -> {actual}")
