"""
Memory regression tests for Stage 04 quality assessment.

Context: SibBMS high-resolution volumes (0.35 mm, ~231M voxels) made each
quality worker peak at ~5.5 GB RSS, which OOM-killed the container (and, before
the 20 GB cap, the whole OS). Two root causes:

  1. ``nib.Nifti1Image.get_fdata()`` upcasts to float64 — ~1.85 GB for one
     SibBMS volume before any metric runs.
  2. ``gradient_sharpness`` allocated three full-size gradient arrays plus
     squared temporaries simultaneously.

These tests lock in the fixes: float32 loading and in-place gradient
accumulation. They are written to FAIL on the pre-fix code.
"""

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from quality_metrics.gradient_sharpness import GradientSharpnessMetric


def _load_module(filename, module_name):
    """Import a scripts/ module whose filename is not a valid identifier."""
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reference_gradient_sharpness(data, fg_mask):
    """Original naive formula — the correctness reference the fix must match."""
    grad_x = np.gradient(data, axis=0)
    grad_y = np.gradient(data, axis=1)
    grad_z = np.gradient(data, axis=2)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    fg_gradients = gradient_magnitude[fg_mask]
    return float(np.var(fg_gradients))


def test_gradient_sharpness_matches_reference():
    """The optimized gradient_sharpness must return the original value."""
    rng = np.random.default_rng(0)
    data = rng.random((40, 50, 60)).astype(np.float32)
    mask = data > 0.5
    got = GradientSharpnessMetric().calculate(data, mask)
    expected = _reference_gradient_sharpness(data, mask)
    assert got == pytest.approx(expected, rel=1e-5)


def _peak_rss_worker(shape, queue):
    """Run gradient_sharpness in a fresh process; report peak RSS delta (bytes)."""
    import resource
    import sys as _sys

    import numpy as _np

    _sys.path.insert(0, str(SCRIPTS_DIR))
    from quality_metrics.gradient_sharpness import GradientSharpnessMetric as GSM

    data = _np.random.default_rng(0).random(shape).astype(_np.float32)
    mask = data > 0.5
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    GSM().calculate(data, mask)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KiB on Linux; convert the delta to bytes.
    queue.put((data.nbytes, (peak - base) * 1024))


def test_gradient_sharpness_peak_memory_bounded():
    """Peak allocation during gradient_sharpness must stay near ~2 arrays.

    Measured in a spawned subprocess so the RSS high-water mark reflects only
    this computation. The naive three-array version peaks at ~3.1 arrays; the
    in-place version holds ~2. The 2.5x threshold discriminates with headroom.
    """
    shape = (200, 200, 200)  # 8M voxels → 32 MB per float32 array
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_peak_rss_worker, args=(shape, queue))
    proc.start()
    nbytes, peak_delta = queue.get()
    proc.join()
    assert peak_delta < 2.5 * nbytes, (
        f"gradient_sharpness peaked at {peak_delta / 1e6:.0f} MB "
        f"({peak_delta / nbytes:.1f} arrays); expected < {2.5 * nbytes / 1e6:.0f} MB"
    )


def test_load_image_data_returns_float32(tmp_path):
    """Volume loading must return float32, not nibabel's default float64."""
    import nibabel as nib

    mod = _load_module("04_assess_quality.py", "assess_quality_mem")
    arr = (np.random.default_rng(0).random((10, 10, 10)) * 1000).astype(np.int16)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    path = tmp_path / "t1.nii.gz"
    nib.save(img, str(path))

    _loaded_img, data = mod._load_image_data(path)
    assert data.dtype == np.float32


def _worker_that_dies(x):
    """Simulate a worker killed mid-task, as the cgroup OOM killer does."""
    import os

    if x == 2:
        os._exit(1)
    return x


def test_pool_fails_fast_on_worker_death():
    """A killed worker must raise BrokenProcessPool, never hang.

    Stage 04 previously used multiprocessing.Pool.map(), which waits forever for
    a result a dead worker will never deliver — the stage hung for over an hour
    with idle workers at ~0% CPU after the OOM killer took a worker. This locks
    in the ProcessPoolExecutor behaviour the fix relies on. The SIGALRM guard
    turns a regression (deadlock) into a failure instead of a hanging suite.
    """
    import signal
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    def _on_timeout(signum, frame):
        raise AssertionError("pool deadlocked instead of failing fast")

    previous = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(30)
    try:
        with pytest.raises(BrokenProcessPool):
            with ProcessPoolExecutor(max_workers=2) as executor:
                list(executor.map(_worker_that_dies, [1, 2, 3, 4]))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_process_parallel_produces_reports(tmp_path):
    """The ProcessPoolExecutor path must still assess images end to end.

    Guards the pool-engine swap: the worker callable is a staticmethod that has
    to stay picklable, and results must aggregate as before.
    """
    import logging

    import nibabel as nib
    import yaml

    mod = _load_module("04_assess_quality.py", "assess_quality_par")
    config = yaml.safe_load((PROJ_ROOT / "configs" / "quality_config.yaml").read_text())
    assessor = mod.QualityAssessor(config, logging.getLogger("test_quality"))

    images = []
    for index, modality in enumerate(["t1", "t2"]):
        arr = (np.random.default_rng(index).random((20, 20, 20)) * 1000).astype(np.int16)
        path = tmp_path / "nifti" / f"sub-001_ses-001_{modality}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))
        images.append((path, "001", "001", modality))

    assessor._process_parallel(images, tmp_path / "quality", workers=2, skip_existing=False)
    assert assessor.stats["successful"] == 2
