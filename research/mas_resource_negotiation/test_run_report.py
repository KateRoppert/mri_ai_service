"""Отчёт по прогону: доля дошедших сессий и разброс времени ожидания."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_report import find_timeline, report
from utils.task_timeline import TIMELINE_NAME


def _run(tmp_path, records):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / TIMELINE_NAME).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    return tmp_path


def test_a_missing_timeline_is_reported_not_raised(tmp_path):
    assert "error" in report(tmp_path)


def test_the_journal_is_found_under_logs(tmp_path):
    run = _run(tmp_path, [{"event": "stage_started", "stage": "s", "at": 0.0}])

    assert find_timeline(run) == run / "logs" / TIMELINE_NAME


def test_completion_rate_counts_what_did_not_arrive(tmp_path):
    """Клетка матрицы с потерянными сессиями недостоверна по времени.

    Стратегия, уронившая воркер по памяти, выглядит быстрее — поэтому доля
    дошедших печатается рядом с распределением, а не вместо него.
    """
    run = _run(tmp_path, [
        {"event": "stage_started", "stage": "stage_05", "at": 0.0},
        {"event": "task_done", "stage": "stage_05", "subject": "sub-001",
         "session": "ses-001", "status": "ok", "at": 10.0},
        {"event": "task_done", "stage": "stage_08", "subject": "sub-001",
         "session": "ses-001", "status": "ok", "at": 20.0},
        {"event": "task_done", "stage": "stage_05", "subject": "sub-002",
         "session": "ses-001", "status": "failed", "at": 11.0},
    ])

    data = report(run)

    assert data["sessions_done"] == 1
    assert data["sessions_attempted"] == 2
    assert data["completion_rate"] == 0.5
    assert data["incomplete"] == ["sub-002/ses-001"]


def test_the_spread_distinguishes_streaming_from_batching(tmp_path):
    """Главное, чего не видно по makespan: максимум один, разброс разный."""
    streamed = _run(tmp_path / "streamed", [
        {"event": "stage_started", "stage": "stage_08", "at": 0.0},
    ] + [
        {"event": "task_done", "stage": "stage_08", "subject": f"sub-{i}",
         "session": "ses-001", "status": "ok", "at": float(i + 1) * 10}
        for i in range(5)
    ])
    batched = _run(tmp_path / "batched", [
        {"event": "stage_started", "stage": "stage_08", "at": 0.0},
    ] + [
        {"event": "task_done", "stage": "stage_08", "subject": f"sub-{i}",
         "session": "ses-001", "status": "ok", "at": 50.0}
        for i in range(5)
    ])

    s, b = report(streamed)["turnaround_s"], report(batched)["turnaround_s"]

    assert s["max"] == b["max"] == 50.0
    assert s["median"] == 30.0 and b["median"] == 50.0
    assert s["min"] == 10.0 and b["min"] == 50.0


def test_stage_windows_are_durations_in_seconds(tmp_path):
    run = _run(tmp_path, [
        {"event": "stage_started", "stage": "stage_05", "at": 5.0},
        {"event": "stage_finished", "stage": "stage_05", "at": 65.0},
    ])

    assert report(run)["stage_windows_s"] == {"stage_05": 60.0}
