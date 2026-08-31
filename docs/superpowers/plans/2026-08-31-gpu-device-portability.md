# GPU Device Portability for Stage 05 Skull Stripping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage 05 skull stripping select GPUs safely and portably across machines (1 GPU, 2 GPUs with a display card, or N GPUs), and never freeze the desktop by running two HD-BET jobs on the same card at once.

**Architecture:** A device resolver turns skull-stripping config into a list of torch device strings; those seed a get/put **slot pool** (a `multiprocessing.Manager().Queue` in parallel mode, a plain `queue.Queue` in sequential mode). Each subject's GPU skull-stripping call acquires a device from the pool, runs HD-BET pinned to it (`cuda:N`), and releases it. Pool size 1 → serialized (freeze-safe); pool size N → N concurrent, one per card. Mirrors the segmentation services' existing GPU-pool pattern (`services/common/service_base.py:85-89`).

**Tech Stack:** Python 3.12, `multiprocessing`, `pynvml`/`torch`/`nvidia-smi` (GPU enumeration, with fallbacks), pytest. HD-BET invoked as a subprocess (unchanged).

**Spec:** `docs/superpowers/specs/2026-08-31-gpu-device-portability-design.md`

## Global Constraints

- Existing `scripts/preprocessing_steps/skull_stripping/tests/` (14 tests) must stay green.
- `process_subject_skull_stripping` keeps a backward-compatible signature: the new `gpu_pool` parameter defaults to `None` (direct callers and existing tests keep working).
- CPU tools (BET) must never touch the pool or a `device` param — only GPU-capable tools do.
- Config keys live flat under the `skull_stripping` step params: `device` (`auto`|`cuda`|`cuda:N`|`cpu`, default `auto`), `exclude_gpus` (list of ints, default `[]`), `gpu_pool_size` (int or null, default null), `disable_tta` (bool, optional).
- Device enumeration must degrade to CPU (never raise) when no CUDA GPU or query tool is available.
- Follow existing file style: module docstring referencing the spec, `logging` module logger, no new heavyweight deps.

---

### Task 1: GPU enumeration + device resolution

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/gpu_pool.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py`

**Interfaces:**
- Produces:
  - `enumerate_gpus() -> list[dict]` — each `{"index": int, "free_mb": int}`, highest `free_mb` first; `[]` if no CUDA GPU / no query method.
  - `resolve_devices(params: dict | None = None) -> list[str]` — torch device strings (`"cuda:1"`, `"cpu"`, …) seeding the pool; never empty.
  - Module constant `DEFAULT_MIN_FREE_MB = 2000`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py
from preprocessing_steps.skull_stripping import gpu_pool


def _fake_gpus(monkeypatch, gpus):
    monkeypatch.setattr(gpu_pool, "enumerate_gpus", lambda: list(gpus))


def test_auto_picks_all_usable_gpus_highest_free_first(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 500}, {"index": 1, "free_mb": 19000}])
    # index 0 is below DEFAULT_MIN_FREE_MB (2000) -> dropped
    assert gpu_pool.resolve_devices({"device": "auto"}) == ["cuda:1"]


def test_auto_with_two_free_gpus_orders_by_free_desc(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 8000}, {"index": 1, "free_mb": 19000}])
    assert gpu_pool.resolve_devices({"device": "auto"}) == ["cuda:1", "cuda:0"]


def test_exclude_gpus_removes_display_card(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 8000}, {"index": 1, "free_mb": 19000}])
    assert gpu_pool.resolve_devices({"device": "auto", "exclude_gpus": [0]}) == ["cuda:1"]


def test_gpu_pool_size_caps_devices(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 9000}, {"index": 1, "free_mb": 9000}])
    assert gpu_pool.resolve_devices({"device": "auto", "gpu_pool_size": 1}) == ["cuda:0"]


def test_explicit_cuda_index_is_passed_through(monkeypatch):
    _fake_gpus(monkeypatch, [])  # must not consult enumeration
    assert gpu_pool.resolve_devices({"device": "cuda:1"}) == ["cuda:1"]


def test_explicit_cpu(monkeypatch):
    assert gpu_pool.resolve_devices({"device": "cpu"}) == ["cpu"]


def test_auto_with_no_usable_gpu_falls_back_to_cpu(monkeypatch):
    _fake_gpus(monkeypatch, [])
    assert gpu_pool.resolve_devices({"device": "auto"}) == ["cpu"]


def test_default_device_is_auto(monkeypatch):
    _fake_gpus(monkeypatch, [{"index": 0, "free_mb": 19000}])
    assert gpu_pool.resolve_devices({}) == ["cuda:0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/test_gpu_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: ... gpu_pool` (module not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/preprocessing_steps/skull_stripping/gpu_pool.py
"""
GPU device selection + a cross-process slot pool for the Stage 05
skull-stripping step.

See docs/superpowers/specs/2026-08-31-gpu-device-portability-design.md.
The pool mirrors the segmentation services' GPU-pool pattern
(services/common/service_base.py): a queue of device slots, acquired before a
GPU job and released after, so N slots == N concurrent jobs. One slot == the
serialized, freeze-safe case that motivated this change.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# A GPU with less free VRAM than this is treated as busy/unusable for HD-BET.
DEFAULT_MIN_FREE_MB = 2000


def enumerate_gpus() -> List[Dict[str, int]]:
    """
    Visible CUDA GPUs as [{"index": i, "free_mb": m}, ...], most-free first.

    Tries pynvml, then torch, then nvidia-smi; returns [] if none work or no
    CUDA GPU is visible. Never raises — callers treat [] as "run on CPU".
    """
    # 1) pynvml (already a dependency of services/common/gpu_monitor.py)
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            out = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                out.append({"index": i, "free_mb": int(mem.free // (1024 * 1024))})
        finally:
            pynvml.nvmlShutdown()
        if out:
            return sorted(out, key=lambda g: g["free_mb"], reverse=True)
    except Exception as e:
        logger.debug("pynvml GPU enumeration unavailable: %s", e)

    # 2) torch
    try:
        import torch
        if torch.cuda.is_available():
            out = []
            for i in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(i)
                out.append({"index": i, "free_mb": int(free // (1024 * 1024))})
            if out:
                return sorted(out, key=lambda g: g["free_mb"], reverse=True)
    except Exception as e:
        logger.debug("torch GPU enumeration unavailable: %s", e)

    # 3) nvidia-smi
    try:
        if shutil.which("nvidia-smi"):
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            out = []
            for line in res.stdout.strip().splitlines():
                idx, free = (p.strip() for p in line.split(","))
                out.append({"index": int(idx), "free_mb": int(free)})
            if out:
                return sorted(out, key=lambda g: g["free_mb"], reverse=True)
    except Exception as e:
        logger.debug("nvidia-smi GPU enumeration unavailable: %s", e)

    return []


def resolve_devices(params: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Turn skull-stripping params into the list of torch device strings that
    seed the slot pool. List length == max concurrent GPU jobs. Never empty.

    device: "auto" (default) | "cuda" | "cuda:N" | "cpu"
    exclude_gpus: GPU indices never to use (e.g. [0] to protect a display card)
    gpu_pool_size: optional hard cap on concurrent GPU jobs
    """
    params = params or {}
    device = str(params.get("device", "auto")).lower()
    exclude = {int(x) for x in (params.get("exclude_gpus") or [])}
    cap = params.get("gpu_pool_size")

    if device == "cpu":
        devices = ["cpu"]
    elif device.startswith("cuda:"):
        devices = [device]
    else:  # "auto" or bare "cuda"
        gpus = [
            g for g in enumerate_gpus()
            if g["index"] not in exclude and g["free_mb"] >= DEFAULT_MIN_FREE_MB
        ]
        if gpus:
            devices = [f"cuda:{g['index']}" for g in gpus]
        else:
            logger.warning(
                "No usable CUDA GPU (device=%s, exclude=%s) — skull stripping "
                "will run on CPU (slower).", device, sorted(exclude)
            )
            devices = ["cpu"]

    if cap is not None and int(cap) > 0:
        devices = devices[: int(cap)]
    if not devices:
        devices = ["cpu"]

    logger.info("Skull stripping device pool: %s", devices)
    return devices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/test_gpu_pool.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/gpu_pool.py \
        scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py
git commit -m "feat(ss): GPU enumeration + device resolution for skull stripping"
```

---

### Task 2: GPU-slot pool + acquire context manager

**Files:**
- Modify: `scripts/preprocessing_steps/skull_stripping/gpu_pool.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py`

**Interfaces:**
- Consumes: `resolve_devices` (Task 1).
- Produces:
  - `build_pool(devices: list[str], manager=None)` — returns a queue seeded with `devices`; uses `manager.Queue()` (cross-process) when a `multiprocessing.Manager` is given, else `queue.Queue()`.
  - `acquire_device(pool)` — context manager yielding a device string; blocks until a slot is free and always returns it. `pool=None` yields `"cpu"` without gating.

- [ ] **Step 1: Write the failing test**

```python
# append to scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py
import queue as _queue
import pytest


def test_build_pool_seeds_all_devices():
    pool = gpu_pool.build_pool(["cuda:0", "cuda:1"])
    got = {pool.get(), pool.get()}
    assert got == {"cuda:0", "cuda:1"}
    assert pool.empty()


def test_acquire_device_returns_slot_to_pool():
    pool = gpu_pool.build_pool(["cuda:0"])
    with gpu_pool.acquire_device(pool) as dev:
        assert dev == "cuda:0"
        assert pool.empty()          # slot held while in the block
    assert pool.get() == "cuda:0"    # released back afterwards


def test_acquire_device_releases_on_exception():
    pool = gpu_pool.build_pool(["cuda:0"])
    with pytest.raises(RuntimeError):
        with gpu_pool.acquire_device(pool):
            raise RuntimeError("boom")
    assert pool.get() == "cuda:0"    # released despite the error


def test_acquire_device_none_pool_yields_cpu():
    with gpu_pool.acquire_device(None) as dev:
        assert dev == "cpu"


def test_build_pool_without_manager_is_in_process_queue():
    pool = gpu_pool.build_pool(["cpu"])
    assert isinstance(pool, _queue.Queue)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/test_gpu_pool.py -k "pool or acquire" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_pool'`.

- [ ] **Step 3: Write minimal implementation**

Append to `gpu_pool.py`:

```python
def build_pool(devices: List[str], manager=None):
    """
    A get/put queue seeded with `devices`. With a multiprocessing Manager the
    queue is shareable across ProcessPoolExecutor workers (parallel mode);
    without one it is an in-process queue.Queue (sequential mode).
    """
    if manager is not None:
        pool = manager.Queue()
    else:
        import queue
        pool = queue.Queue()
    for device in devices:
        pool.put(device)
    return pool


@contextmanager
def acquire_device(pool):
    """
    Block until a device slot is free, yield its device string, and always
    return it to the pool — even if the body raises. `pool=None` disables
    gating and yields "cpu" (used by direct callers without a pool).
    """
    if pool is None:
        yield "cpu"
        return
    device = pool.get()
    try:
        yield device
    finally:
        pool.put(device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/test_gpu_pool.py -v`
Expected: PASS (13 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/gpu_pool.py \
        scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py
git commit -m "feat(ss): cross-process GPU-slot pool + acquire context manager"
```

---

### Task 3: Wire the pool into `process_subject_skull_stripping`

**Files:**
- Modify: `scripts/preprocessing_steps/skull_stripping/base.py` (add `uses_gpu` marker)
- Modify: `scripts/preprocessing_steps/skull_stripping/hdbet.py` (set `uses_gpu = True`)
- Modify: `scripts/preprocessing_steps/skull_stripping/__init__.py` (`process_subject_skull_stripping` acquires a device)
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_gpu_dispatch.py`

**Interfaces:**
- Consumes: `resolve_devices`, `acquire_device` (Tasks 1-2); `get_stripper`, `get_tool_params` (existing dispatcher).
- Produces: `process_subject_skull_stripping(subject_dir, output_dir, transform_dir, modalities, params, gpu_pool=None)` — GPU tools now run pinned to a device acquired from `gpu_pool`; `device` and `disable_tta` are injected into the tool params passed to `stripper.strip`.
- `SkullStripperBase.uses_gpu: bool = False`; `HdBetStripper.uses_gpu = True`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/preprocessing_steps/skull_stripping/tests/test_gpu_dispatch.py
from pathlib import Path

from preprocessing_steps.skull_stripping import (
    gpu_pool,
    process_subject_skull_stripping,
)
import preprocessing_steps.skull_stripping as ss  # package (__init__) namespace


class _FakeStripper:
    name = "hdbet"
    uses_gpu = True

    def __init__(self):
        self.seen_params = None

    def is_available(self):
        return True

    def strip(self, input_path, output_path, mask_path=None, params=None):
        self.seen_params = params or {}
        return {"success": True, "output_path": str(output_path),
                "mask_path": str(mask_path), "processing_time": 0.0}


def _setup(monkeypatch, tmp_path, fake):
    # Reference modality file must exist for the glob in the function.
    anat = tmp_path / "sub-001" / "ses-001" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-001_ses-001_t1.nii.gz").write_bytes(b"x")
    monkeypatch.setattr(ss, "get_stripper", lambda params: fake)
    return anat


def test_gpu_tool_receives_device_from_pool(monkeypatch, tmp_path):
    fake = _FakeStripper()
    anat = _setup(monkeypatch, tmp_path, fake)
    pool = gpu_pool.build_pool(["cuda:1"])
    params = {"reference_modality": "t1", "apply_to_all": False,
              "cleanup": False, "disable_tta": True}

    process_subject_skull_stripping(
        subject_dir=anat, output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm", modalities=["t1"],
        params=params, gpu_pool=pool,
    )

    assert fake.seen_params["device"] == "cuda:1"
    assert fake.seen_params["disable_tta"] is True
    # slot returned to the pool after the call
    assert pool.get() == "cuda:1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/test_gpu_dispatch.py -v`
Expected: FAIL — `process_subject_skull_stripping()` has no `gpu_pool` parameter (TypeError).

- [ ] **Step 3: Write minimal implementation**

In `base.py`, add the marker to the class body (after `name` on line ~40):

```python
    #: True for tools that run on the GPU and must go through the device pool.
    uses_gpu: bool = False
```

In `hdbet.py`, in `class HdBetStripper` body (after `name = "hdbet"` on line ~40):

```python
    uses_gpu = True
```

In `__init__.py`:

Add to the imports near the top (after the `.dispatcher` import block, ~line 39):

```python
from .gpu_pool import acquire_device, resolve_devices
```

Change the signature of `process_subject_skull_stripping` (line 64) to add `gpu_pool=None`:

```python
def process_subject_skull_stripping(
    subject_dir: Path,
    output_dir: Path,
    transform_dir: Path,
    modalities: list,
    params: dict,
    gpu_pool=None,
) -> dict:
```

Replace the single `strip_result = stripper.strip(...)` call (lines 131-136) with a device-aware version:

```python
    def _run_strip(tool_params_local):
        return stripper.strip(
            input_path=ref_file,
            output_path=ref_output,
            mask_path=mask_path,
            params=tool_params_local,
        )

    if getattr(stripper, "uses_gpu", False):
        # A GPU tool: pin it to a device. With a pool (Stage 05) acquire a
        # slot so only pool-size jobs share the GPUs at once; without one
        # (direct callers / sequential resolve) pick a device inline.
        extra = {}
        if "disable_tta" in params:
            extra["disable_tta"] = params["disable_tta"]
        if gpu_pool is not None:
            with acquire_device(gpu_pool) as device:
                logger.info("Skull stripping on device %s", device)
                strip_result = _run_strip({**tool_params, "device": device, **extra})
        else:
            device = resolve_devices(params)[0]
            logger.info("Skull stripping on device %s", device)
            strip_result = _run_strip({**tool_params, "device": device, **extra})
    else:
        strip_result = _run_strip(tool_params)
```

(The `resolve_devices` fallback keeps direct/sequential callers correct even
when no pool is threaded in.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/test_gpu_dispatch.py preprocessing_steps/skull_stripping/tests/ -v`
Expected: PASS — new test passes and the existing 14 tests stay green.

- [ ] **Step 5: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/base.py \
        scripts/preprocessing_steps/skull_stripping/hdbet.py \
        scripts/preprocessing_steps/skull_stripping/__init__.py \
        scripts/preprocessing_steps/skull_stripping/tests/test_gpu_dispatch.py
git commit -m "feat(ss): acquire a GPU device from the pool per subject"
```

---

### Task 4: Build the pool in Stage 05 and thread it to workers

**Files:**
- Modify: `scripts/05_preprocessing.py`
  - imports (~line 36-42)
  - `process_single_subject` signature (line 296) + skull-stripping call (line 512)
  - `process_subject_wrapper` unpack (line 577) + call (line 592)
  - `main`: pool build + `processing_args` tuple (lines 838-842) + parallel re-tuple (line 962)

**Interfaces:**
- Consumes: `resolve_devices`, `build_pool` (Tasks 1-2); `process_subject_skull_stripping(..., gpu_pool=...)` (Task 3).
- Produces: both sequential and parallel paths pass a seeded pool down to skull stripping. `args_tuple` element order becomes `(..., lesion_type, gpu_pool, threads_per_worker)` — `gpu_pool` second-to-last so the existing `t[:-1] + (threads_per_worker,)` re-tuple at line 962 keeps it.

- [ ] **Step 1: Add the import**

In the skull-stripping import block (lines 36-41), append `resolve_devices`/`build_pool` via the package. After the existing:

```python
from preprocessing_steps.skull_stripping import (
    setup_fsl_environment,
    check_fsl_installed,
    process_subject_skull_stripping
)
```

add:

```python
from preprocessing_steps.skull_stripping.gpu_pool import build_pool, resolve_devices
```

- [ ] **Step 2: Thread `gpu_pool` through `process_single_subject`**

Change the signature (line 296) to add `gpu_pool=None` after `lesion_type`:

```python
def process_single_subject(
    anat_dir: Path,
    subject_id: str,
    session_id: str,
    output_dir: Path,
    transform_dir: Path,
    temp_dir: Path,
    atlas_path: Path,
    config: dict,
    modalities: List[str],
    lesion_type: str = 'glioblastoma',
    gpu_pool=None,
) -> dict:
```

In the skull-stripping call (line 512), pass the pool:

```python
            skull_strip_results = process_subject_skull_stripping(
                subject_dir=registered_anat,
                output_dir=output_dir,
                transform_dir=transform_dir,
                modalities=modalities,
                params=step_params,
                gpu_pool=gpu_pool,
            )
```

- [ ] **Step 3: Unpack `gpu_pool` in the worker wrapper**

Change the unpack (line 577) to the new order:

```python
    (anat_dir, subject_id, session_id, output_dir, transform_dir,
     base_temp_dir, atlas_path, config, modalities, lesion_type,
     gpu_pool, threads_per_worker) = args_tuple
```

Pass it in the `process_single_subject` call (line 592):

```python
        result = process_single_subject(
            anat_dir=anat_dir,
            subject_id=subject_id,
            session_id=session_id,
            output_dir=output_dir,
            transform_dir=transform_dir,
            temp_dir=worker_temp_dir,
            atlas_path=atlas_path,
            config=config,
            modalities=modalities,
            lesion_type=lesion_type,
            gpu_pool=gpu_pool,
        )
```

- [ ] **Step 4: Build the pool and seed `processing_args` in `main`**

Replace the `processing_args` construction (lines 836-842) with a version that
builds the pool first and inserts it into each tuple:

```python
        threads_for_parallel = None if args.mode == 'parallel' else None

        # GPU-slot pool for the skull-stripping step. Size = number of usable
        # devices, so one card -> serialized (freeze-safe), N cards -> N
        # concurrent. Manager().Queue crosses ProcessPoolExecutor workers;
        # sequential mode uses an in-process queue.
        steps_by_name = {s['name']: s for s in config.get('steps', [])}
        skull_cfg = steps_by_name.get('skull_stripping', {})
        gpu_manager = None
        gpu_pool = None
        if skull_cfg.get('enabled', True):
            devices = resolve_devices(skull_cfg.get('params', {}))
            if args.mode == 'parallel':
                from multiprocessing import Manager
                gpu_manager = Manager()
                gpu_pool = build_pool(devices, manager=gpu_manager)
            else:
                gpu_pool = build_pool(devices)

        processing_args = [
            (anat_dir, subject_id, session_id, preprocessed_dir, transform_dir,
             temp_dir, atlas_path, config, modalities, args.lesion_type,
             gpu_pool, threads_for_parallel)
            for anat_dir, subject_id, session_id in subjects
        ]
```

- [ ] **Step 5: Keep the parallel re-tuple correct**

The parallel branch rebuilds tuples with the computed `threads_per_worker`
(line 962). It uses `t[:-1] + (threads_per_worker,)`, which drops only the last
element (`threads_for_parallel`) and preserves `gpu_pool` in second-to-last
position. Confirm that line is unchanged and still reads:

```python
                processing_args = [t[:-1] + (threads_per_worker,) for t in processing_args]
```

No edit needed if it already matches; otherwise restore it to the above.

- [ ] **Step 6: Verify existing tests + a config-driven smoke check**

Run: `cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/ -v`
Expected: PASS (all prior tests green).

Run a syntax/import check on the edited script:
`cd scripts && python -c "import ast; ast.parse(open('05_preprocessing.py').read()); print('ok')"`
Expected: `ok`.

Manual (this 2-GPU host, requires FSL/HD-BET + data — do if available): a
parallel Stage-05 run with `exclude_gpus: [0]` in the config must show
`Skull stripping device pool: ['cuda:1']` in the log and, in `nvidia-smi`,
HD-BET load only on GPU 1 while the desktop stays responsive.

- [ ] **Step 7: Commit**

```bash
git add scripts/05_preprocessing.py
git commit -m "feat(stage05): seed a GPU-slot pool and pin skull stripping per subject"
```

---

### Task 5: Config + compose — portable defaults

**Files:**
- Modify: `configs/preprocessing_config.yaml` (skull_stripping params)
- Modify: `configs/preprocessing_config_ms60.yaml` (skull_stripping params)
- Modify: `docker-compose.yml` (web service GPU reservation)

**Interfaces:**
- Consumes: config keys read by `resolve_devices` (Task 1) and `process_subject_skull_stripping` (Task 3).

- [ ] **Step 1: Add device keys to `preprocessing_config.yaml`**

In the `skull_stripping` step's `params:` block (currently `method`, `fallback_method`, `reference_modality`, `apply_to_all`, `cleanup` at lines 56-68), add:

```yaml
      # GPU device selection (used by GPU tools such as hdbet; ignored by bet).
      #   auto    — pick usable CUDA GPUs by free VRAM, else CPU (default)
      #   cuda:N  — force a specific card
      #   cpu     — force CPU
      device: "auto"
      # GPU indices never to use. On a workstation, list the display card here
      # (e.g. [0]) so HD-BET never freezes the desktop.
      exclude_gpus: []
      # Optional hard cap on concurrent GPU skull-stripping jobs.
      # null = one job per usable card.
      gpu_pool_size: null
      # Test-time augmentation: better masks, slower + more VRAM. Left to the
      # tool default (on for GPU, off for CPU) unless set here.
      # disable_tta: false
```

- [ ] **Step 2: Mirror the keys in `preprocessing_config_ms60.yaml`**

Add the same `device` / `exclude_gpus` / `gpu_pool_size` block under that file's
`skull_stripping` params.

- [ ] **Step 3: Expose all GPUs to the web container**

In `docker-compose.yml`, the web service GPU reservation (around line 46) reads:

```yaml
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Change `count: 1` to `count: all` so the container sees every card; GPU
**selection** is now the app's job (`device`/`exclude_gpus`), portable across
1- and N-GPU hosts.

- [ ] **Step 4: Verify configs parse**

Run:
```bash
python -c "import yaml; [yaml.safe_load(open(p)) for p in ['configs/preprocessing_config.yaml','configs/preprocessing_config_ms60.yaml','docker-compose.yml']]; print('ok')"
```
Expected: `ok`.

Run the full skull-stripping suite once more:
`cd scripts && python -m pytest preprocessing_steps/skull_stripping/tests/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/preprocessing_config.yaml configs/preprocessing_config_ms60.yaml docker-compose.yml
git commit -m "config(ss): device/exclude_gpus/gpu_pool_size keys + compose count:all"
```

---

## Post-implementation

- On this 2-GPU workstation, set `exclude_gpus: [0]` in the machine's
  `preprocessing_config.yaml` to hard-protect the display GPU; other machines
  keep the portable `auto` default.
- Re-run the case that hung (`data/MS_5` → Stage 05, parallel) and confirm it
  completes through skull stripping with the desktop responsive.
- Update `KNOWN_ISSUES.md` (KI-056 neighbourhood) noting the fix once verified.
```
