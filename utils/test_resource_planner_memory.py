"""
KI-058: plan against the memory the machine really has, say when one task
does not fit, and measure what each task actually used.

On the 15 GiB demob laptop the web container's cap was 20g, so the planner
saw an 18.3 GB budget while ms-seg's model and the desktop held ~5 GB: stage
07 started 2 workers and the whole host ran out of memory (2026-09-30).
Stage 05's cost model said 4.1 GB per worker for a 231 Mvox session; the
worker was killed above 9.2 GB, and no log said how much it really took.
"""
import logging

import numpy as np
import pytest

from utils import resource_planner as rp
from utils.resource_planner import (
    format_task_peak,
    host_available_bytes,
    plan_stage_workers,
    plan_workers,
    task_peak_meter,
)

GB = 1_000_000_000
GiB = 1024 ** 3


# --- host MemAvailable -------------------------------------------------------

def test_host_available_reads_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       15987192 kB\n"
        "MemFree:          512000 kB\n"
        "MemAvailable:   12042820 kB\n"
    )
    assert host_available_bytes(str(meminfo)) == 12042820 * 1024


def test_host_available_missing_file_or_field_is_none(tmp_path):
    assert host_available_bytes(str(tmp_path / "nope")) is None
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       15987192 kB\n")
    assert host_available_bytes(str(meminfo)) is None


# --- budget = min(cgroup limit, host available) ------------------------------

def _sources(monkeypatch, cgroup, host):
    monkeypatch.setattr(rp, "cgroup_memory_limit_bytes", lambda *a, **k: cgroup)
    monkeypatch.setattr(rp, "host_available_bytes", lambda *a, **k: host)


def test_host_caps_a_cgroup_limit_above_physical_ram(monkeypatch):
    """The 2026-09-30 case: 20g cap, ~11 GB really free, ~3.85 GB per worker."""
    _sources(monkeypatch, cgroup=20 * GiB, host=11 * GB)
    r = plan_workers(requested=4, per_worker_bytes=3.85 * GB, reserve_bytes=2 * GB)
    # (11 * 0.85 - 2) / 3.85 = 1.9 -> 1, where cgroup alone gave 4
    assert r.actual_workers == 1
    assert "host" in r.reason and "cgroup" in r.reason


def test_cgroup_still_caps_when_host_has_more(monkeypatch):
    _sources(monkeypatch, cgroup=9 * GiB, host=30 * GB)
    r = plan_workers(requested=4, per_worker_bytes=2.1 * GB, reserve_bytes=2 * GB)
    # (9.66 * 0.85 - 2) / 2.1 = 2.9 -> 2
    assert r.actual_workers == 2


def test_host_alone_is_enough_without_a_container_limit(monkeypatch):
    _sources(monkeypatch, cgroup=None, host=10 * GB)
    r = plan_workers(requested=6, per_worker_bytes=2 * GB, reserve_bytes=1 * GB)
    # (10 * 0.85 - 1) / 2 = 3.75 -> 3
    assert r.actual_workers == 3


def test_no_source_at_all_means_no_memory_cap(monkeypatch):
    _sources(monkeypatch, cgroup=None, host=None)
    r = plan_workers(requested=5, per_worker_bytes=4 * GB)
    assert r.actual_workers == 5
    assert "unbounded" in r.reason


def test_explicit_budget_is_used_as_given(monkeypatch):
    """Callers/tests that inject budget_bytes keep today's meaning."""
    _sources(monkeypatch, cgroup=1 * GB, host=1 * GB)
    r = plan_workers(requested=6, per_worker_bytes=4 * GB, budget_bytes=12 * GB,
                     reserve_bytes=0)
    assert r.actual_workers == 3


# --- one task over budget ----------------------------------------------------

def test_one_task_over_budget_is_flagged(monkeypatch):
    _sources(monkeypatch, cgroup=9 * GiB, host=11 * GB)
    r = plan_workers(requested=4, per_worker_bytes=9.5 * GB, reserve_bytes=2 * GB)
    assert r.actual_workers == 1
    assert r.over_budget is True
    assert "OVER BUDGET" in r.reason


def test_fitting_plan_is_not_flagged(monkeypatch):
    _sources(monkeypatch, cgroup=9 * GiB, host=11 * GB)
    r = plan_workers(requested=4, per_worker_bytes=2 * GB, reserve_bytes=2 * GB)
    assert r.over_budget is False
    assert "OVER BUDGET" not in r.reason


def test_stage_wrapper_warns_when_one_task_does_not_fit(monkeypatch, tmp_path, caplog):
    _sources(monkeypatch, cgroup=9 * GiB, host=11 * GB)
    monkeypatch.setattr(rp, "max_voxels", lambda files: 231_400_000)
    cfg = {"safety_factor": 0.85, "min_workers": 1,
           "stages": {"stage_05_preprocessing": {"k_bytes_per_voxel": 45.0,
                                                 "reserve_bytes": 2 * GB}}}

    with caplog.at_level(logging.WARNING, logger="utils.resource_planner"):
        r = plan_stage_workers("stage_05_preprocessing", [tmp_path / "x.nii.gz"],
                               requested=4, config=cfg)

    assert r.actual_workers == 1
    warning = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warning) == 1
    assert "231.4" in warning[0]          # the input that drives the estimate
    assert "OOM" in warning[0]


# --- per-task peak memory ----------------------------------------------------

def test_task_peak_meter_sees_what_the_task_allocated():
    with task_peak_meter() as meter:
        block = np.ones(300 * 1024 * 1024 // 8)   # 300 MiB, touched
        block.sum()
        del block
    if meter.peak_bytes is None:
        pytest.skip("/proc not available")
    assert meter.peak_bytes >= 300 * 1024 * 1024


def test_task_peak_meter_is_per_task_not_cumulative():
    with task_peak_meter() as big:
        block = np.ones(400 * 1024 * 1024 // 8)
        block.sum()
        del block
    with task_peak_meter() as small:
        tiny = np.ones(1024)
        tiny.sum()
    if big.peak_bytes is None or small.peak_bytes is None:
        pytest.skip("/proc not available")
    assert small.peak_bytes < big.peak_bytes - 300 * 1024 * 1024


def test_task_peak_meter_without_proc_yields_none(tmp_path):
    with task_peak_meter(proc_dir=str(tmp_path / "noproc")) as meter:
        pass
    assert meter.peak_bytes is None


def test_format_task_peak_gives_the_k_figure():
    line = format_task_peak(6.2 * GB, 231_400_000)
    assert line == "peak RSS 6.20 GB · max input 231.4 Mvox · k=26.8 B/voxel"
    assert format_task_peak(None, 231_400_000) is None
    assert format_task_peak(1 * GB, 0) == "peak RSS 1.00 GB"
