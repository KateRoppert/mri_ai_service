"""
Memory-aware worker sizing (see docs/superpowers/specs/2026-07-17-memory-aware-worker-sizing-design.md).

Pure helpers a stage calls at startup to cap its parallelism at what the
container memory budget allows, given an input-size cost estimate. Fail-safe:
when no budget can be determined the planner returns the requested value, so a
working run is never broken.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


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


def host_available_bytes(meminfo_path: str = "/proc/meminfo") -> Optional[int]:
    """Memory the machine can give out right now (MemAvailable), or None.

    Inside a container /proc/meminfo is the host's, so this already excludes
    what other containers (ms-seg's loaded model), the desktop and this
    container's own baseline hold. KI-058: a cgroup cap of 20g on a 15 GiB
    laptop let stage 07 plan against memory that did not exist.
    """
    try:
        text = Path(meminfo_path).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            try:
                return int(line.split()[1]) * 1024  # kB
            except (IndexError, ValueError):
                return None
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
    # True when even one worker's estimate exceeds the usable budget: the
    # plan falls back to min_workers and the run may still OOM.
    over_budget: bool = False
    per_worker_bytes: float = 0.0
    usable_bytes: Optional[int] = None


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

    budget_bytes=None -> min(cgroup limit, host MemAvailable) * safety_factor;
    either source may be missing. With neither, memory does not cap
    (fail-safe to requested/CPU). An injected budget_bytes is used as given.
    """
    caps = [requested]
    parts = [f"requested={requested}"]
    if cpu_cap is not None:
        caps.append(cpu_cap)
        parts.append(f"cpu={cpu_cap}")

    if budget_bytes is None:
        sources = [
            (name, value) for name, value in (
                ("cgroup", cgroup_memory_limit_bytes()),
                ("host-avail", host_available_bytes()),
            ) if value is not None
        ]
        if sources:
            budget_bytes = int(min(v for _, v in sources) * safety_factor)
            parts.append(
                "budget=min(" + ", ".join(f"{n} {v / 1e9:.1f}" for n, v in sources)
                + f")GBx{safety_factor}"
            )

    over_budget = False
    usable = None
    if budget_bytes is not None and per_worker_bytes > 0:
        usable = budget_bytes - reserve_bytes
        mem_workers = int(usable // per_worker_bytes) if usable > 0 else 0
        caps.append(mem_workers)
        parts.append(
            f"mem=({budget_bytes / 1e9:.1f}-{reserve_bytes / 1e9:.1f})GB"
            f"/{per_worker_bytes / 1e9:.2f}GB={mem_workers}"
        )
        if per_worker_bytes > usable:
            over_budget = True
    else:
        parts.append("mem=unbounded")

    actual = max(min_workers, min(caps))
    reason = f"min({', '.join(parts)}) -> {actual}"
    if over_budget:
        reason += (f"; OVER BUDGET: one task ~{per_worker_bytes / 1e9:.1f} GB"
                   f" > usable {max(usable, 0) / 1e9:.1f} GB")
    return PlanResult(actual, reason, over_budget=over_budget,
                      per_worker_bytes=per_worker_bytes, usable_bytes=usable)


def plan_stage_workers(
    stage_name: str,
    input_files: "Iterable[Path]",
    requested: int,
    cpu_cap: Optional[int] = None,
    config: Optional[dict] = None,
    budget_bytes: Optional[int] = None,
) -> PlanResult:
    """Stage-facing wrapper: read the stage's cost constants, estimate per-worker
    bytes from the largest input, and cap the worker count.

    Fail-safe: an unknown stage, missing config, malformed config file, or a
    stage entry missing a required cost key all yield `requested` (capped by
    `cpu_cap` if given) unchanged. `budget_bytes` is normally None (read from
    cgroup); tests inject it. When budget_bytes is provided, it is the raw
    cgroup limit; safety_factor is applied here.
    """
    try:
        if config is None:
            from utils.config_loader import load_resource_config
            config = load_resource_config()

        stage_cfg = config.get("stages", {}).get(stage_name)
        if not stage_cfg:
            return PlanResult(
                max(config.get("min_workers", 1), min([requested, cpu_cap] if cpu_cap else [requested])),
                f"no resource config for {stage_name}; using requested={requested}",
            )

        voxels = max_voxels(input_files)
        per_worker = voxels * float(stage_cfg["k_bytes_per_voxel"])

        # If budget_bytes is provided (by tests), apply safety_factor to it.
        # budget_bytes here is the raw/unscaled limit (e.g. from a test or a
        # future caller); plan_workers's own budget_bytes=None branch already
        # applies safety_factor when reading from cgroup, so we replicate that
        # scaling here for an explicitly-provided budget too, keeping the two
        # call paths consistent.
        safety_factor = float(config.get("safety_factor", 0.85))
        if budget_bytes is not None:
            budget_bytes = int(budget_bytes * safety_factor)

        plan = plan_workers(
            requested=requested,
            per_worker_bytes=per_worker,
            budget_bytes=budget_bytes,
            cpu_cap=cpu_cap,
            safety_factor=safety_factor,
            reserve_bytes=int(stage_cfg.get("reserve_bytes", 1_500_000_000)),
            min_workers=int(config.get("min_workers", 1)),
        )
        if plan.over_budget:
            logger.warning(
                "%s: одна задача по оценке ~%.1f ГБ (крупнейший вход %.1f Мвокс "
                "× k=%s Б/воксель), а доступно ~%.1f ГБ — возможен OOM",
                stage_name, per_worker / 1e9, voxels / 1e6,
                stage_cfg["k_bytes_per_voxel"], max(plan.usable_bytes or 0, 0) / 1e9,
            )
        return plan
    except Exception as exc:
        # Fail-safe: plan_stage_workers must never raise into a calling stage.
        # Any problem reading/parsing the resource config, or a malformed
        # per-stage cost entry (e.g. missing k_bytes_per_voxel), degrades to
        # the requested worker count (capped by cpu_cap if given).
        min_workers_default = 1
        actual_workers = max(
            min_workers_default,
            min([requested] + ([cpu_cap] if cpu_cap else [])),
        )
        logger.warning(
            "plan_stage_workers: resource config error for stage=%s (%s); "
            "falling back to requested=%d",
            stage_name, exc, requested,
        )
        return PlanResult(
            actual_workers,
            f"resource config error: {exc}; using requested={requested}",
        )


# ---------------------------------------------------------------------------
# Per-task peak memory (KI-058): what a worker really used, so k_stage can be
# checked and the next OOM explained from the log alone.
# ---------------------------------------------------------------------------

@dataclass
class TaskPeak:
    peak_bytes: Optional[int] = None


def _read_vmhwm(proc_dir: str) -> Optional[int]:
    try:
        for line in Path(proc_dir, "status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024  # kB
    except (OSError, IndexError, ValueError):
        pass
    return None


@contextmanager
def task_peak_meter(proc_dir: str = "/proc/self"):
    """Measure this process' peak RSS over the `with` block.

    Resets the high-water mark first (writing 5 to clear_refs), so in a
    process-pool worker that runs many tasks each reading is that task's own
    peak, not the worker's lifetime maximum. If the reset is not possible the
    reading would be cumulative and misleading, so peak_bytes stays None.
    """
    meter = TaskPeak()
    try:
        Path(proc_dir, "clear_refs").write_text("5")
        reset = True
    except OSError:
        reset = False
    try:
        yield meter
    finally:
        if reset:
            meter.peak_bytes = _read_vmhwm(proc_dir)


def format_task_peak(peak_bytes: Optional[float], voxels: int) -> Optional[str]:
    """One log line per task; the k figure is what resource_config.yaml wants."""
    if peak_bytes is None:
        return None
    line = f"peak RSS {peak_bytes / 1e9:.2f} GB"
    if voxels:
        line += (f" · max input {voxels / 1e6:.1f} Mvox"
                 f" · k={peak_bytes / voxels:.1f} B/voxel")
    return line
