"""Монитор должен мерить всю работу, а не родительский процесс.

`PerformanceMonitor` сэмплировал `psutil.Process()` — текущий процесс. Но
этапы 04/05/07/08 делают работу в воркерах `ProcessPoolExecutor`, и родитель в
это время в основном ждёт. То есть `cpu_avg` и `memory_peak_mb` описывали
процесс, который не считает.

Это главная метрика RQ3 («выросла ли утилизация CPU»), поэтому прибор чинится
до матрицы, а не после. Заодно добавлены дисковые счётчики и iowait: прошлая
статья объясняет насыщение этапов 1-3 пропускной способностью диска, но
измеряла только форму кривой.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from performance_monitor import BenchmarkLogger, ExperimentMetrics, PerformanceMonitor


def _monitor():
    m = PerformanceMonitor(enabled=True)
    m.start()
    return m


# ── дефект, ради которого всё затевалось ──────────────────────────────────


def test_the_tree_includes_children_not_only_the_parent():
    """Воркеры — дети процесса этапа; без них метрика описывает ожидание."""
    parent, child = MagicMock(), MagicMock()
    parent.cpu_percent.return_value = 5.0
    parent.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
    child.cpu_percent.return_value = 180.0
    child.memory_info.return_value = MagicMock(rss=900 * 1024 * 1024)
    parent.children.return_value = [child]

    m = PerformanceMonitor(enabled=True)
    with patch("performance_monitor.psutil.Process", return_value=parent):
        m.start()
        m._sample()

    assert m.cpu_samples[-1] == pytest.approx(185.0)
    assert m.memory_samples[-1] == pytest.approx(1000.0)


def test_a_dead_child_does_not_break_sampling():
    """Воркер может умереть между обнаружением и опросом — это норма."""
    import psutil

    parent, child = MagicMock(), MagicMock()
    parent.cpu_percent.return_value = 1.0
    parent.memory_info.return_value = MagicMock(rss=10 * 1024 * 1024)
    child.cpu_percent.side_effect = psutil.NoSuchProcess(123)
    parent.children.return_value = [child]

    m = PerformanceMonitor(enabled=True)
    with patch("performance_monitor.psutil.Process", return_value=parent):
        m.start()
        m._sample()

    assert m.cpu_samples[-1] == pytest.approx(1.0)


def test_system_wide_load_is_recorded_separately():
    """Дерево отвечает «наша ли нагрузка», система — «загружена ли машина».

    RQ3 спрашивает именно про второе, поэтому хранятся оба.
    """
    m = _monitor()
    m._sample()

    assert m.system_cpu_samples, "системная утилизация не собрана"
    assert 0.0 <= m.system_cpu_samples[-1] <= 100.0 * (1 + 0)


# ── ввод-вывод: гипотеза прошлой статьи, которую надо проверить ───────────


def test_iowait_is_recorded():
    m = _monitor()
    m._sample()

    assert m.iowait_samples, "iowait не собран"
    assert m.iowait_samples[-1] >= 0.0


def test_disk_bytes_are_counted_from_the_start_of_the_run():
    """Нужна разница за прогон, а не счётчик с момента загрузки машины."""
    m = _monitor()
    m._sample()
    metrics = m.get_metrics()

    assert metrics["disk_read_mb"] >= 0.0
    assert metrics["disk_write_mb"] >= 0.0


def test_disk_counters_survive_their_absence():
    """В контейнере без привилегий disk_io_counters может вернуть None."""
    m = PerformanceMonitor(enabled=True)
    with patch("performance_monitor.psutil.disk_io_counters", return_value=None):
        m.start()
        m._sample()
        metrics = m.get_metrics()

    assert metrics["disk_read_mb"] is None
    assert metrics["disk_write_mb"] is None


# ── прежнее поведение не сломано ──────────────────────────────────────────


def test_disabled_monitor_still_returns_the_empty_shape():
    m = PerformanceMonitor(enabled=False)
    m.start()
    metrics = m.get_metrics()

    assert metrics["cpu_avg"] is None
    assert metrics["memory_peak_mb"] is None


def test_metrics_keys_are_a_superset_of_the_old_ones():
    """Старые ключи читают существующие вызывающие — их нельзя терять."""
    m = _monitor()
    m._sample()
    keys = set(m.get_metrics())

    assert {"cpu_avg", "cpu_max", "memory_avg_mb", "memory_peak_mb",
            "duration"} <= keys
    assert {"system_cpu_avg", "system_cpu_max", "system_memory_peak_mb",
            "iowait_avg", "iowait_max", "disk_read_mb", "disk_write_mb"} <= keys


# ── запись результата ─────────────────────────────────────────────────────


def test_the_new_fields_reach_the_record():
    metrics = ExperimentMetrics(
        experiment_id="t", timestamp="now", total_series=1, successful=1,
        failed=0, skipped=0, total_time=1.0, time_per_series=1.0,
        throughput=1.0,
        system_cpu_avg=31.6, system_cpu_max=88.0, system_memory_peak_mb=12000.0,
        iowait_avg=4.2, iowait_max=19.0,
        disk_read_mb=1024.0, disk_write_mb=512.0,
    )
    as_dict = metrics.to_dict()

    assert as_dict["system_cpu_avg"] == 31.6
    assert as_dict["iowait_max"] == 19.0
    assert as_dict["disk_read_mb"] == 1024.0


def test_the_csv_carries_the_new_columns(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    logger.log_metrics(ExperimentMetrics(
        experiment_id="t", timestamp="now", total_series=1, successful=1,
        failed=0, skipped=0, total_time=1.0, time_per_series=1.0,
        throughput=1.0, system_cpu_avg=31.6, iowait_avg=4.2,
        disk_read_mb=1024.0,
    ))

    header = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]

    for column in ("system_cpu_avg", "system_cpu_max", "system_memory_peak_mb",
                   "iowait_avg", "iowait_max", "disk_read_mb", "disk_write_mb"):
        assert column in header, f"нет колонки {column}"


def test_real_worker_cpu_is_actually_captured():
    """Тест на моках это пропустил, и дефект пережил свой же фикс.

    `children(recursive=True)` отдаёт НОВЫЕ объекты Process на каждый вызов, а
    первый `cpu_percent()` на объекте всегда возвращает 0.0 — он только ставит
    точку отсчёта. С пересозданием списка каждый тик дерево показывало 2% там,
    где машина показывала 13%. Мок возвращал один и тот же объект, поэтому
    суммирование проверялось, а кеширование — нет.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor

    m = PerformanceMonitor(enabled=True)
    m.start()
    with ProcessPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(_burn, 1.5) for _ in range(2)]
        for _ in range(4):
            time.sleep(0.4)
            m._sample()
        [f.result() for f in futures]
    m.stop()

    # Два воркера на 100% каждый дали бы ~200%; порог низкий, чтобы тест не
    # дрожал на загруженной машине, но 0% при двух горящих ядрах невозможен.
    assert max(m.cpu_samples) > 50.0, (
        f"работа воркеров не видна: {[round(s) for s in m.cpu_samples]}"
    )


def _burn(seconds):
    """Модуль верхнего уровня нужен, чтобы функция пиклилась для пула."""
    import time
    end = time.time() + seconds
    x = 0
    while time.time() < end:
        x += sum(i * i for i in range(5000))
    return x
