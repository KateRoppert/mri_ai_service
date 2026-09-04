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
        gpus = sorted(
            (g for g in enumerate_gpus()
             if g["index"] not in exclude and g["free_mb"] >= DEFAULT_MIN_FREE_MB),
            key=lambda g: g["free_mb"], reverse=True,
        )
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
