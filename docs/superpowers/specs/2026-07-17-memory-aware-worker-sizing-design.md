# Memory-Aware Worker Sizing — Design

**Date:** 2026-07-17
**Status:** Approved (brainstorming), ready for implementation plan
**Related:** KI-042 (resource scaling with data resolution), KI-043 (no memory-aware admission control)

## Problem

Each pipeline stage runs a fixed number of parallel workers, taken verbatim from
`pipeline_config.yaml` (`--workers`). But per-worker memory scales with the input
volume size, which varies enormously: within a single SibBMS patient the input
NIfTI ranges from `[256,256,21]` (1.4M voxels, thick-slice 2010 scan) to 231M
voxels (0.35 mm) — a 165× range. A worker count safe for small volumes causes
OOM on large ones (this is what forced the manual `stage_05` retunes and the
cgroup OOM incidents). A count safe for large volumes wastes resources on small
ones.

The fix: compute the safe worker count at runtime from the container memory budget
and an estimate of per-worker cost derived from the actual input headers, capping
(never raising) the configured value.

## Scope (MVP)

- **In:** RAM-based worker sizing for the heavy, NIfTI-consuming stages —
  `stage_04_quality`, `stage_05_preprocessing`, `stage_07_inverse_transform`.
- **Out (future / other KIs):** VRAM budgeting for `stage_06_segmentation`
  (GPU-bound); dynamic k-calibration feedback loop; central admission control /
  backpressure (KI-043); light stages `01`/`03`/`08` (low memory, DICOM inputs).

Fail-safe principle: whenever the planner cannot compute a budget (not in a
container, missing config, no inputs), it does **not** cap — the stage falls back
to the configured `--workers`, i.e. today's behaviour. The planner can only make
things safer, never break a working run.

## Architecture & Data Flow

A stage, at startup, sizes its own pool via a shared pure utility. This extends
the pattern `stage_07_inverse_transform` already uses to cap workers by CPU.

```
Stage (04 / 05 / 07) at startup:
  1. Scan input NIfTI headers (dims only, no voxel load) -> max_voxels
  2. plan_workers(
         requested        = config --workers,          # ceiling
         per_worker_bytes = max_voxels * k_stage,       # from resource_config
         budget_bytes     = None,                       # None -> cgroup * safety
         reserve_bytes    = stage reserve,
         cpu_cap          = effective_cpu // threads)   # as stage 07 already does
  3. actual_workers = max(min_workers,
                          min(requested, cpu_cap, (budget - reserve) // per_worker))
  4. Log the decision (actual_workers + reason)
  5. Create the pool with actual_workers
```

The stage owns knowledge of its inputs; the utility owns the budget arithmetic.

## Components

### `utils/resource_planner.py`

Three small, independently testable functions.

```python
def cgroup_memory_limit_bytes() -> Optional[int]:
    """Container memory limit from /sys/fs/cgroup/memory.max (v2) or
    memory.limit_in_bytes (v1). None if unavailable or unlimited ("max")."""

def max_voxels(nifti_paths: Iterable[Path]) -> int:
    """max(product of header dims) over the given NIfTI files — the worst-case
    volume, since any worker may receive it. Reads headers only. 0 if empty."""

@dataclass
class PlanResult:
    actual_workers: int
    reason: str            # human-readable, for the stage log

def plan_workers(
    requested: int,
    per_worker_bytes: float,
    budget_bytes: Optional[int] = None,     # None -> cgroup_memory_limit_bytes() * safety_factor
    cpu_cap: Optional[int] = None,
    safety_factor: float = 0.85,
    reserve_bytes: int = 1_500_000_000,
    min_workers: int = 1,
) -> PlanResult:
    """actual = max(min_workers,
                    min(requested, cpu_cap?, (budget - reserve) // per_worker_bytes))."""
```

### `configs/resource_config.yaml`

Per-stage cost model plus global knobs. Separate from `pipeline_config.yaml`
because it describes resource behaviour, not pipeline structure.

```yaml
safety_factor: 0.85          # fraction of the cgroup limit the planner may use
min_workers: 1
stages:
  stage_04_quality:          { k_bytes_per_voxel: 6.9,  reserve_bytes: 1500000000 }
  stage_05_preprocessing:    { k_bytes_per_voxel: 17.7, reserve_bytes: 2000000000 }
  stage_07_inverse_transform:{ k_bytes_per_voxel: 22.5, reserve_bytes: 2000000000 }
```

### Cost model & calibration

`per_worker_bytes = max_voxels * k_bytes_per_voxel`.

`k_stage` is calibrated on the **worst-case volume** (peak memory occurs when
workers process the largest volumes): `k = measured_peak_per_worker_bytes /
worst_case_voxels`. Initial values from the SibBMS 3-patient run (worst-case
231M voxels): stage 04 ≈ 6.9, stage 05 ≈ 17.7, stage 07 ≈ 22.5 bytes/voxel.

This is a deliberately rough linear model. Stage 05 in particular has a large
shared/fixed component (KI-042: reducing 6→4 workers saved only 11%, not 33%),
absorbed by `reserve_bytes`; `safety_factor` covers the rest. The goal is "don't
OOM", not "extract the last worker". Values live in config and are refined as
more datasets are measured.

## Integration

- **Stage 04, 05:** currently pass `args.workers` straight into the pool. Insert
  a `plan_workers` call before pool creation; use `plan.actual_workers`.
- **Stage 07:** already computes a CPU cap (`effective_cpu // 4`). That value
  becomes `cpu_cap`; `plan_workers` takes the min of ceiling / memory / CPU. No
  behavioural conflict — memory is simply added as a third cap.
- Each stage logs `f"Workers: {plan.actual_workers} — {plan.reason}"` so the
  decision is visible (as stage 07 already logs its reasoning).

## Error Handling (all fail-safe toward current behaviour)

| Condition | Behaviour |
|---|---|
| cgroup limit unavailable (not in container) | `budget_bytes=None` → no memory cap; use `requested` (CPU cap still applies for 07) |
| `resource_config.yaml` missing / stage absent | Skip memory planning for that stage; use `requested`; log a warning |
| No input files found | `per_worker_bytes` undefined → use `requested` |
| Even 1 worker exceeds budget | Return `min_workers` (1) + WARNING (cannot go lower) |

## Testing

- **`plan_workers`** (pure): budget-limited, CPU-limited, ceiling-limited cases;
  edges — `per_worker_bytes=0`, `budget_bytes=None`, one-worker-doesn't-fit,
  `min_workers` floor. Deterministic.
- **`max_voxels`**: temp NIfTI files of differing dims → asserts max dim-product;
  empty input → 0.
- **`cgroup_memory_limit_bytes`**: inject the file path → parse v2 integer, v1
  integer, and `"max"` (→ None).
- **Integration (one focused test):** with a mocked tight cgroup budget and a
  large `max_voxels`, a stage's planning call yields fewer workers than the
  config; with a generous budget it yields the config value.

## Out of Scope / Follow-ups

- VRAM-aware sizing for stage 06 (segmentation) — different resource dimension.
- Dynamic k-calibration: write measured peaks back to refine k automatically.
- Central resource coordinator / admission control / backpressure (KI-043), the
  path toward the MAS resource-allocation vision.
