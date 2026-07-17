# Memory-Aware Worker Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cap each heavy pipeline stage's worker count at runtime from the container memory budget and an input-size-based cost estimate, so a config value safe for small volumes no longer OOMs on large ones (and vice-versa).

**Architecture:** A shared pure utility `utils/resource_planner.py` computes safe worker counts. Each heavy stage (04/05/07) calls a stage-facing wrapper `plan_stage_workers()` at startup with its input files; the wrapper reads per-stage cost constants from `configs/resource_config.yaml`, estimates per-worker bytes as `max_input_voxels × k_stage`, and returns `min(config_workers, cpu_cap, memory_budget / per_worker)`. Fail-safe: when no budget can be computed (not in a container, missing config, no inputs) it returns the configured value unchanged.

**Tech Stack:** Python 3.12, nibabel (header-only reads), PyYAML, pytest.

## Global Constraints

- Fail-safe: the planner may only *lower* the worker count, never raise it, and must fall back to the configured `--workers` on any missing input (no cgroup, no config, no files). It must never raise an exception into a stage.
- Pure/testable: `resource_planner` functions have no side effects except reading `/sys/fs/cgroup/*` and NIfTI headers; both are injectable for tests.
- `nibabel.load(path).shape` reads headers only (no voxel load) — use it for sizing; never load `.dataobj` in the planner.
- Scope is stages `stage_04_quality`, `stage_05_preprocessing`, `stage_07_inverse_transform` only. Do not touch 01/03/06/08.
- Initial calibrated constants (bytes/voxel), from the SibBMS 3-patient run at worst-case 231M voxels: stage 04 = 6.9, stage 05 = 17.7, stage 07 = 22.5.

---

### Task 1: `resource_planner` core utility

**Files:**
- Create: `utils/resource_planner.py`
- Test: `tests/resource/test_resource_planner.py`

**Interfaces:**
- Produces:
  - `cgroup_memory_limit_bytes(v2_path=..., v1_path=...) -> Optional[int]`
  - `max_voxels(nifti_paths: Iterable[Path]) -> int`
  - `@dataclass PlanResult(actual_workers: int, reason: str)`
  - `plan_workers(requested: int, per_worker_bytes: float, budget_bytes: Optional[int]=None, cpu_cap: Optional[int]=None, safety_factor: float=0.85, reserve_bytes: int=1_500_000_000, min_workers: int=1) -> PlanResult`

- [ ] **Step 1: Write the failing tests**

Create `tests/resource/test_resource_planner.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_resource_planner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.resource_planner'`

- [ ] **Step 3: Implement `utils/resource_planner.py`**

```python
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

    if budget_bytes is not None and per_worker_bytes and per_worker_bytes > 0:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_resource_planner.py -q`
Expected: PASS (all green)

- [ ] **Step 5: Commit**

```bash
git add -f tests/resource/test_resource_planner.py
git add utils/resource_planner.py
git commit -m "feat(resource): add resource_planner core (cgroup budget, max_voxels, plan_workers)"
```

---

### Task 2: Resource config + stage-facing wrapper

**Files:**
- Create: `configs/resource_config.yaml`
- Modify: `utils/config_loader.py` (add `load_resource_config`)
- Modify: `utils/resource_planner.py` (add `plan_stage_workers`)
- Test: `tests/resource/test_plan_stage_workers.py`

**Interfaces:**
- Consumes: `plan_workers`, `max_voxels`, `PlanResult` from Task 1.
- Produces:
  - `load_resource_config() -> dict` in `utils/config_loader.py`
  - `plan_stage_workers(stage_name: str, input_files: Iterable[Path], requested: int, cpu_cap: Optional[int]=None, config: Optional[dict]=None) -> PlanResult` in `utils/resource_planner.py`

- [ ] **Step 1: Create `configs/resource_config.yaml`**

```yaml
# Per-stage memory cost model for memory-aware worker sizing.
# k_bytes_per_voxel = measured peak bytes-per-worker / worst-case input voxels.
# Calibrated on the SibBMS 3-patient run (worst case 231M voxels). Refine as more
# datasets are measured. See docs/superpowers/specs/2026-07-17-memory-aware-worker-sizing-design.md
safety_factor: 0.85
min_workers: 1
stages:
  stage_04_quality:
    k_bytes_per_voxel: 6.9
    reserve_bytes: 1500000000
  stage_05_preprocessing:
    k_bytes_per_voxel: 17.7
    reserve_bytes: 2000000000
  stage_07_inverse_transform:
    k_bytes_per_voxel: 22.5
    reserve_bytes: 2000000000
```

- [ ] **Step 2: Write the failing tests**

Create `tests/resource/test_plan_stage_workers.py`:

```python
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
```

Note: `plan_stage_workers` must accept a `budget_bytes` passthrough for tests (add it to the signature — default None so production reads cgroup).

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_plan_stage_workers.py -q`
Expected: FAIL — `ImportError: cannot import name 'plan_stage_workers'` / `load_resource_config`

- [ ] **Step 4: Add `load_resource_config` to `utils/config_loader.py`**

Append this function to `utils/config_loader.py` (it already has `load_series_scoring_config` with the same pattern):

```python
def load_resource_config() -> Dict[str, Any]:
    """Load per-stage memory cost model from configs/resource_config.yaml.

    Used by memory-aware worker sizing (utils/resource_planner). Returns a dict
    with keys: safety_factor, min_workers, stages. Returns an empty-but-safe
    default if the file is missing so the planner degrades to config workers.
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'resource_config.yaml'
    if not config_path.exists():
        return {"safety_factor": 0.85, "min_workers": 1, "stages": {}}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Failed to parse resource_config.yaml: {e}")
    if not isinstance(config, dict):
        return {"safety_factor": 0.85, "min_workers": 1, "stages": {}}
    config.setdefault("safety_factor", 0.85)
    config.setdefault("min_workers", 1)
    config.setdefault("stages", {})
    return config
```

- [ ] **Step 5: Add `plan_stage_workers` to `utils/resource_planner.py`**

```python
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

    Fail-safe: an unknown stage or missing config yields `requested` unchanged.
    `budget_bytes` is normally None (read from cgroup); tests inject it.
    """
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
    return plan_workers(
        requested=requested,
        per_worker_bytes=per_worker,
        budget_bytes=budget_bytes,
        cpu_cap=cpu_cap,
        safety_factor=float(config.get("safety_factor", 0.85)),
        reserve_bytes=int(stage_cfg.get("reserve_bytes", 1_500_000_000)),
        min_workers=int(config.get("min_workers", 1)),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/ -q`
Expected: PASS (all green across both files)

- [ ] **Step 7: Commit**

```bash
git add configs/resource_config.yaml utils/config_loader.py utils/resource_planner.py
git add -f tests/resource/test_plan_stage_workers.py
git commit -m "feat(resource): resource_config + plan_stage_workers wrapper"
```

---

### Task 3: Integrate into stage_05_preprocessing

**Files:**
- Modify: `scripts/05_preprocessing.py` (compute workers via the planner before creating the pool)
- Test: `tests/resource/test_stage05_planning.py`

**Interfaces:**
- Consumes: `plan_stage_workers` from Task 2.

- [ ] **Step 1: Locate the worker/pool site**

In `scripts/05_preprocessing.py`, find `main()` where `args.workers` is resolved and passed into the parallel pool (search for `args.workers` and the `Pool(`/`ProcessPoolExecutor(` call). Note the list of input NIfTI files the stage discovered (the images it will process).

- [ ] **Step 2: Write the failing test**

Create `tests/resource/test_stage05_planning.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_stage05_planning.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute '_plan_workers_for_inputs'`

- [ ] **Step 4: Add the helper and call it in `scripts/05_preprocessing.py`**

Add this module-level helper (near the top, after imports):

```python
def _plan_workers_for_inputs(input_files, requested, cpu_cap=None, budget_bytes=None):
    """Memory-aware worker count for this stage (see utils.resource_planner)."""
    from utils.resource_planner import plan_stage_workers
    plan = plan_stage_workers(
        "stage_05_preprocessing", input_files, requested,
        cpu_cap=cpu_cap, budget_bytes=budget_bytes,
    )
    return plan
```

Then in `main()`, immediately before the pool is created, replace the raw worker count:

```python
# BEFORE:
#   workers = args.workers
#   ... Pool(processes=workers) / ProcessPoolExecutor(max_workers=workers) ...

# AFTER:
_plan = _plan_workers_for_inputs(input_nifti_files, args.workers)
logger.info(f"Workers: {_plan.actual_workers} — {_plan.reason}")
workers = _plan.actual_workers
```

Where `input_nifti_files` is the list of NIfTI paths the stage already discovered to process (the images passed into preprocessing). `_plan_workers_for_inputs` returns a `PlanResult`; the stage reads `.actual_workers`.

- [ ] **Step 5: Run test + regression**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_stage05_planning.py -q && python -m py_compile scripts/05_preprocessing.py`
Expected: PASS + compile OK

- [ ] **Step 6: Commit**

```bash
git add scripts/05_preprocessing.py
git add -f tests/resource/test_stage05_planning.py
git commit -m "feat(stage05): size preprocessing pool from memory budget"
```

---

### Task 4: Integrate into stage_04_quality

**Files:**
- Modify: `scripts/04_assess_quality.py` (`_process_parallel` is called from `main()` with a worker count)
- Test: `tests/resource/test_stage04_planning.py`

**Interfaces:**
- Consumes: `plan_stage_workers` from Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/resource/test_stage04_planning.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_stage04_planning.py -q`
Expected: FAIL — `AttributeError: ... '_plan_workers_for_inputs'`

- [ ] **Step 3: Add helper and wire into `scripts/04_assess_quality.py`**

Add the module-level helper (mirror stage 05, changing only the stage name):

```python
def _plan_workers_for_inputs(input_files, requested, cpu_cap=None, budget_bytes=None):
    """Memory-aware worker count for this stage (see utils.resource_planner)."""
    from utils.resource_planner import plan_stage_workers
    return plan_stage_workers(
        "stage_04_quality", input_files, requested,
        cpu_cap=cpu_cap, budget_bytes=budget_bytes,
    )
```

In `main()`, before the `_process_parallel(images, output_dir, workers, ...)` call, compute:

```python
_input_paths = [nifti_path for (nifti_path, *_rest) in images]
_plan = _plan_workers_for_inputs(_input_paths, args.workers)
self.logger.info(f"Workers: {_plan.actual_workers} — {_plan.reason}")  # or module logger
workers = _plan.actual_workers
```

(`images` is the list of `(nifti_path, patient_id, session_id, modality)` tuples the stage builds before parallel processing.)

- [ ] **Step 4: Run test + regression**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_stage04_planning.py tests/stage04 -q && python -m py_compile scripts/04_assess_quality.py`
Expected: PASS + compile OK (stage04 suite still green)

- [ ] **Step 5: Commit**

```bash
git add scripts/04_assess_quality.py
git add -f tests/resource/test_stage04_planning.py
git commit -m "feat(stage04): size quality pool from memory budget"
```

---

### Task 5: Integrate into stage_07_inverse_transform (reconcile CPU tuner)

**Files:**
- Modify: `scripts/07_inverse_transform.py:255-283` (existing CPU auto-tuner block)
- Test: `tests/resource/test_stage07_planning.py`

**Interfaces:**
- Consumes: `plan_stage_workers` from Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/resource/test_stage07_planning.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_stage07_planning.py -q`
Expected: FAIL — `AttributeError: ... '_plan_workers_for_inputs'`

- [ ] **Step 3: Add helper and reconcile with the CPU tuner**

Add the module-level helper:

```python
def _plan_workers_for_inputs(input_files, requested, cpu_cap=None, budget_bytes=None):
    """Memory-aware worker count for this stage (see utils.resource_planner)."""
    from utils.resource_planner import plan_stage_workers
    return plan_stage_workers(
        "stage_07_inverse_transform", input_files, requested,
        cpu_cap=cpu_cap, budget_bytes=budget_bytes,
    )
```

In the parallel branch of the auto-tuner (`scripts/07_inverse_transform.py:265-281`), the existing code computes `workers_by_cpu = max(1, effective_cpu // 4)`. Feed that as `cpu_cap` and let the planner take the min of ceiling/CPU/memory. Replace the `actual_workers = ...` computation:

```python
# BEFORE:
#   workers_by_tasks = min(args.workers, len(masks))
#   workers_by_cpu = max(1, effective_cpu // 4)
#   actual_workers = max(1, min(workers_by_tasks, workers_by_cpu))

# AFTER (keep workers_by_cpu; add memory via the planner):
workers_by_cpu = max(1, effective_cpu // 4)
# The memory driver in stage 07 is the native reference image each mask is
# warped into (loaded by inverse_transform_mask), not the small atlas mask.
# Build those native reference paths from the work list. `masks` is the list of
# (mask_path, subj, sess) the stage already assembled; `nifti_dir` and
# `args.reference_modality` are in scope in main().
_input_refs = [
    nifti_dir / subj / sess / "anat" / f"{subj}_{sess}_{args.reference_modality}.nii.gz"
    for (mask_path, subj, sess) in masks
]
_plan = _plan_workers_for_inputs(
    _input_refs, requested=min(args.workers, len(masks)), cpu_cap=workers_by_cpu,
)
actual_workers = _plan.actual_workers
threads = max(1, effective_cpu // actual_workers)
logger.info(f"Workers: {actual_workers} — {_plan.reason} | threads/worker: {threads}")
```

`max_voxels` reads only the largest reference and silently skips any path that is
missing or unreadable, so this is safe even if a particular reference is absent.

- [ ] **Step 4: Run test + regression**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest tests/resource/test_stage07_planning.py -q && python -m py_compile scripts/07_inverse_transform.py`
Expected: PASS + compile OK

- [ ] **Step 5: Commit**

```bash
git add scripts/07_inverse_transform.py
git add -f tests/resource/test_stage07_planning.py
git commit -m "feat(stage07): add memory cap to the existing CPU worker tuner"
```

---

## Notes for the implementer

- `tests/` is gitignored in this repo — commit test files with `git add -f` (as the commands above already do).
- Run the full relevant suites before finishing: `python -m pytest tests/resource tests/stage04 -q` and `cd backend && python -m pytest -q`.
- Do not change stage 01/03/06/08. Stage 06 is GPU-bound and out of scope.
- The planner must never raise into a stage: if in doubt, it returns `requested`.
