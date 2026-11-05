"""Журнал задач и распределение времени ожидания на пациента.

Makespan — метрика партии; врач ждёт свою сессию. При последовательной
обработке этапов эти величины почти совпадают, при конвейерной расходятся в
разы, и единственное число этого не покажет. Здесь проверяется и запись
журнала, и расчёт распределения по нему.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.task_timeline import (
    TIMELINE_NAME, read_timeline, stage_finished, stage_started, task_done,
    timeline_path, turnaround,
)


# ── путь журнала ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("log_file, expected_parent", [
    ("/run/logs/stages/stage_05_preprocessing.log", "/run/logs"),
    ("/run/logs/reorganize.log", "/run/logs"),
])
def test_one_timeline_per_run_whatever_the_log_layout(log_file, expected_parent):
    """Этапы кладут логи и в `logs/stages/`, и в `logs/` — журнал один."""
    assert timeline_path(log_file) == Path(expected_parent) / TIMELINE_NAME


def test_no_log_file_means_no_timeline():
    """Запуск без логов (отладка руками) не должен падать и не пишет журнал."""
    assert timeline_path(None) is None


# ── запись ────────────────────────────────────────────────────────────────


def test_each_call_appends_one_json_line(tmp_path):
    timeline = tmp_path / TIMELINE_NAME
    stage_started(timeline, "stage_05")
    task_done(timeline, "stage_05", "sub-001", "ses-001", duration_s=12.345)
    stage_finished(timeline, "stage_05", successful=1, failed=0)

    lines = timeline.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["stage_started", "task_done", "stage_finished"]


def test_the_record_carries_what_the_metric_needs(tmp_path):
    timeline = tmp_path / TIMELINE_NAME
    task_done(timeline, "stage_08", "sub-002", "ses-003",
              status="failed", duration_s=1.23456)

    record = json.loads(timeline.read_text(encoding="utf-8").splitlines()[0])

    assert record["stage"] == "stage_08"
    assert record["subject"] == "sub-002"
    assert record["session"] == "ses-003"
    assert record["status"] == "failed"
    assert record["duration_s"] == 1.235
    assert isinstance(record["at"], float)


def test_a_timeline_of_none_is_a_no_op():
    """Пишущие функции вызываются из этапов безусловно — без журнала молчат."""
    stage_started(None, "stage_05")
    task_done(None, "stage_05", "sub-001")
    stage_finished(None, "stage_05")


def test_an_unwritable_path_never_breaks_the_stage(tmp_path):
    """Инструментирование не имеет права потерять прогон пациентов."""
    blocked = tmp_path / "nope"
    blocked.write_text("i am a file, not a directory", encoding="utf-8")

    task_done(blocked / "sub" / TIMELINE_NAME, "stage_05", "sub-001")


def test_appends_from_several_processes_do_not_interleave(tmp_path):
    """Этапы пишут из родителя, но прогонов в одном каталоге бывает два."""
    timeline = tmp_path / TIMELINE_NAME
    children = []
    for i in range(4):
        pid = os.fork()
        if pid == 0:  # child
            for j in range(25):
                task_done(timeline, "stage_05", f"sub-{i:03d}", f"ses-{j:03d}")
            os._exit(0)
        children.append(pid)
    for pid in children:
        os.waitpid(pid, 0)

    lines = timeline.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 100
    assert all(json.loads(line)["event"] == "task_done" for line in lines)


def test_broken_lines_are_skipped_not_fatal(tmp_path):
    timeline = tmp_path / TIMELINE_NAME
    timeline.write_text(
        '{"event": "stage_started", "stage": "stage_05", "at": 1.0}\n'
        'не json\n'
        '\n'
        '{"event": "task_done", "stage": "stage_05", "at": 2.0}\n',
        encoding="utf-8",
    )

    assert len(read_timeline(timeline)) == 2


def test_a_missing_timeline_reads_as_empty(tmp_path):
    assert read_timeline(tmp_path / "absent.jsonl") == []


# ── распределение времени ожидания ────────────────────────────────────────


def _records(rows):
    return [dict(zip(("event", "stage", "subject", "session", "status", "at"), row))
            for row in rows]


def test_turnaround_is_measured_from_the_start_of_the_run():
    """Врач ждёт с момента отправки партии, а не с начала своей сессии."""
    records = _records([
        ("stage_started", "stage_05", None, None, None, 100.0),
        ("task_done", "stage_05", "sub-001", "ses-001", "ok", 130.0),
        ("task_done", "stage_08", "sub-001", "ses-001", "ok", 160.0),
    ])

    stats = turnaround(records)

    assert stats["sessions"][("sub-001", "ses-001")] == 60.0


def test_the_last_stage_decides_readiness_not_the_first():
    """Готовность — это последний этап сессии, а не любой пройденный."""
    records = _records([
        ("stage_started", "stage_05", None, None, None, 0.0),
        ("task_done", "stage_05", "sub-001", "ses-001", "ok", 10.0),
        ("task_done", "stage_07", "sub-001", "ses-001", "ok", 20.0),
        ("task_done", "stage_08", "sub-001", "ses-001", "ok", 30.0),
    ])

    assert turnaround(records)["sessions"][("sub-001", "ses-001")] == 30.0


def test_a_session_that_never_reached_the_end_is_not_in_the_distribution():
    """Иначе стратегия «выигрывает», потеряв работу (KI-058: сессии терялись)."""
    records = _records([
        ("stage_started", "stage_05", None, None, None, 0.0),
        ("task_done", "stage_05", "sub-001", "ses-001", "ok", 10.0),
        ("task_done", "stage_08", "sub-001", "ses-001", "ok", 20.0),
        ("task_done", "stage_05", "sub-002", "ses-001", "ok", 11.0),
    ])

    stats = turnaround(records, final_stage="stage_08")

    assert stats["n"] == 1
    assert stats["incomplete"] == [("sub-002", "ses-001")]


def test_a_failed_task_does_not_count_as_readiness():
    records = _records([
        ("stage_started", "stage_08", None, None, None, 0.0),
        ("task_done", "stage_08", "sub-001", "ses-001", "failed", 10.0),
    ])

    stats = turnaround(records, final_stage="stage_08")

    assert stats["n"] == 0
    assert stats["incomplete"] == [("sub-001", "ses-001")]


def test_the_distribution_is_what_makespan_hides():
    """Главный смысл метрики: одинаковый максимум, разная медиана.

    Поток (сессии готовы одна за другой) против пакета (все готовы в конце).
    """
    streamed = _records([("stage_started", "stage_08", None, None, None, 0.0)] + [
        ("task_done", "stage_08", f"sub-{i:03d}", "ses-001", "ok", float(i + 1) * 10)
        for i in range(10)
    ])
    batched = _records([("stage_started", "stage_08", None, None, None, 0.0)] + [
        ("task_done", "stage_08", f"sub-{i:03d}", "ses-001", "ok", 100.0)
        for i in range(10)
    ])

    s, b = turnaround(streamed), turnaround(batched)

    assert s["max_s"] == b["max_s"] == 100.0        # makespan одинаков
    assert s["median_s"] == 55.0 and b["median_s"] == 100.0
    assert s["min_s"] == 10.0 and b["min_s"] == 100.0


def test_p95_interpolates_instead_of_rounding_to_a_neighbour():
    """На трёх повторах округление до соседа даёт не то, что ждёт читатель."""
    records = _records([("stage_started", "s", None, None, None, 0.0)] + [
        ("task_done", "s", f"sub-{i}", "ses-001", "ok", float(v))
        for i, v in enumerate([10, 20, 100])
    ])

    stats = turnaround(records)

    assert stats["median_s"] == 20.0
    assert stats["p95_s"] == pytest.approx(92.0)  # 20 + 0.9 * (100 - 20)


def test_stage_windows_are_reported_for_the_allocation_figure():
    records = _records([
        ("stage_started", "stage_05", None, None, None, 5.0),
        ("stage_finished", "stage_05", None, None, None, 65.0),
        ("stage_started", "stage_08", None, None, None, 60.0),
    ])

    windows = turnaround(records)["stage_windows"]

    assert windows["stage_05"] == {"started_at": 5.0, "finished_at": 65.0}
    assert windows["stage_08"]["finished_at"] is None


def test_an_empty_timeline_does_not_raise():
    stats = turnaround([])

    assert stats["n"] == 0 and stats["median_s"] is None
