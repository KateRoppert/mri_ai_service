"""Per-subject cascade trace — research instrumentation, off in production.

Calibrating the gates needs the numbers behind every decision, not just the
winner: which tools were tried, what each mask measured, which flags fired and
what the cascade did about them. The pipeline log has that in prose; a machine
-readable file per subject is what statistics and paper tables come from.

Writing it is opt-in via `validation.trace_dir`, so a production run without
that key behaves exactly as before.
"""
import json
from pathlib import Path

import numpy as np

from preprocessing_steps.skull_stripping import process_subject_skull_stripping
from .test_cascade import (
    _FakeStripper,
    _clean_mask,
    _flagged_mask,
    _register,
    _subject_anat,
)


def _params(tmp_path, **extra):
    validation = {"review_min_ml": 1200, "review_max_ml": 1800}
    validation.update(extra.pop("validation", {}))
    return {
        "method": "hdbet",
        "fallback_method": "bet",
        "reference_modality": "t1",
        "apply_to_all": False,
        "cleanup": False,
        "validation": validation,
        **extra,
    }


def _run(tmp_path, params):
    return process_subject_skull_stripping(
        subject_dir=_subject_anat(tmp_path),
        output_dir=tmp_path / "out",
        transform_dir=tmp_path / "xfm",
        modalities=["t1"],
        params=params,
    )


def test_no_trace_written_without_trace_dir(tmp_path, monkeypatch):
    """Production must be untouched: no key, no files."""
    _register(monkeypatch, {"hdbet": lambda: _FakeStripper("hdbet", True, _clean_mask())})
    _run(tmp_path, _params(tmp_path))

    assert not list(tmp_path.rglob("*_cascade.json"))


def test_trace_records_every_attempt_and_the_decision(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    first = _FakeStripper("hdbet", True, _flagged_mask())   # flagged -> moves on
    second = _FakeStripper("bet", False, _clean_mask())     # accepted
    _register(monkeypatch, {"hdbet": lambda: first, "bet": lambda: second})

    _run(tmp_path, _params(tmp_path, validation={"trace_dir": str(trace_dir)}))

    traces = list(trace_dir.rglob("*.json"))
    assert len(traces) == 1
    data = json.loads(traces[0].read_text())

    assert data["subject"] == "sub-001"
    assert data["cascade"] == ["hdbet", "bet"]
    assert [a["tool"] for a in data["attempts"]] == ["hdbet", "bet"]
    assert data["attempts"][0]["decision"] == "rejected_review"
    assert data["attempts"][1]["decision"] == "accepted"
    assert data["selected"]["tool"] == "bet"


def test_trace_carries_the_metrics_used_for_the_decision(tmp_path, monkeypatch):
    """The numbers are the point — thresholds get calibrated from these."""
    trace_dir = tmp_path / "traces"
    _register(monkeypatch, {"hdbet": lambda: _FakeStripper("hdbet", True, _clean_mask())})

    _run(tmp_path, _params(tmp_path, validation={"trace_dir": str(trace_dir)}))

    data = json.loads(next(trace_dir.rglob("*.json")).read_text())
    metrics = data["attempts"][0]["metrics"]
    for key in ("mask_volume_ml", "dominant_fraction", "hole_volume_ml",
                "asymmetry", "edge_touch_ratio"):
        assert key in metrics, f"{key} missing from the trace"


def test_trace_records_the_gates_in_force(tmp_path, monkeypatch):
    """A trace read months later must say which thresholds produced it."""
    trace_dir = tmp_path / "traces"
    _register(monkeypatch, {"hdbet": lambda: _FakeStripper("hdbet", True, _clean_mask())})

    _run(tmp_path, _params(tmp_path, validation={"trace_dir": str(trace_dir)}))

    data = json.loads(next(trace_dir.rglob("*.json")).read_text())
    assert data["gates"]["review_max_ml"] == 1800


def test_trace_failure_does_not_break_the_run(tmp_path, monkeypatch):
    """Instrumentation must never cost a patient their preprocessing."""
    _register(monkeypatch, {"hdbet": lambda: _FakeStripper("hdbet", True, _clean_mask())})
    # A path that cannot be created: an existing file where a directory is needed.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")

    results = _run(tmp_path, _params(tmp_path, validation={"trace_dir": str(blocker / "x")}))

    assert results["t1"]["success"] is True
