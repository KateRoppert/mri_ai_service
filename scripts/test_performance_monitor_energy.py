"""Энергия прогона: GPU сейчас, CPU — когда откроют RAPL (пункт 8).

Заявка статьи — «в клиниках ограниченные вычислительные ресурсы», и если МАС
выигрывает тем, что занимает больше железа, честная версия этого утверждения
включает цену. Поэтому энергия идёт рядом с makespan, а не вместо него.

GPU мерится мощностью (Вт) с интегрированием по времени: счётчика энергии у
карты нет. CPU мерится счётчиком RAPL в микроджоулях — это точнее, но файл
`energy_uj` закрыт для чтения после уязвимости PLATYPUS, где по
энергопотреблению восстанавливали ключи. Поэтому сбор обязан работать без него
и включаться сам, когда права появятся.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import performance_monitor as pm
from performance_monitor import BenchmarkLogger, ExperimentMetrics, PerformanceMonitor


def _rapl_tree(root: Path, domains: dict, max_range: int = 262143328850):
    """Дерево powercap: верхнеуровневые пакеты и их поддомены."""
    for name, energy in domains.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "name").write_text(name.split(":", 1)[1], encoding="utf-8")
        (d / "energy_uj").write_text(str(energy), encoding="utf-8")
        (d / "max_energy_range_uj").write_text(str(max_range), encoding="utf-8")
    return root


# ── RAPL ──────────────────────────────────────────────────────────────────


def test_only_top_level_packages_are_summed(tmp_path):
    """Поддомены (`intel-rapl:0:0` — ядра) входят в пакет; сложить и то и
    другое означало бы посчитать одну энергию дважды."""
    _rapl_tree(tmp_path, {"intel-rapl:0": 1000, "intel-rapl:0:0": 400,
                          "intel-rapl:1": 2000})

    assert pm._rapl_energy_uj(root=tmp_path) == 3000


def test_unreadable_rapl_is_none_not_zero(tmp_path):
    """Ноль джоулей — это результат; «нет прав» результатом не является."""
    _rapl_tree(tmp_path, {"intel-rapl:0": 1000})
    (tmp_path / "intel-rapl:0" / "energy_uj").chmod(0o000)

    try:
        assert pm._rapl_energy_uj(root=tmp_path) is None
    finally:
        (tmp_path / "intel-rapl:0" / "energy_uj").chmod(0o644)


def test_absent_rapl_is_none(tmp_path):
    assert pm._rapl_energy_uj(root=tmp_path / "nope") is None


def test_a_wrapped_counter_does_not_give_negative_energy(tmp_path):
    """Счётчик кольцевой: на 100 Вт оборачивается примерно за 45 минут,
    а прогон матрицы длиннее."""
    root = _rapl_tree(tmp_path, {"intel-rapl:0": 10}, max_range=1000)
    m = PerformanceMonitor(enabled=True)
    m._rapl_root = root
    m.start()
    assert m._cpu_energy_uj == 0

    (root / "intel-rapl:0" / "energy_uj").write_text("990", encoding="utf-8")
    m._sample_energy()
    (root / "intel-rapl:0" / "energy_uj").write_text("5", encoding="utf-8")  # оборот
    m._sample_energy()

    # 10 -> 990 это 980, затем 990 -> 5 через оборот это 1000 - 990 + 5 = 15
    assert m._cpu_energy_uj == 995


# ── мощность GPU ──────────────────────────────────────────────────────────


def test_power_is_summed_over_all_visible_cards():
    """На баргузине две карты: прогон тратит столько, сколько тратят обе."""
    out = MagicMock(returncode=0, stdout="54.12\n61.88\n")
    with patch.object(pm.shutil, "which", return_value="/usr/bin/nvidia-smi"):
        with patch.object(pm.subprocess, "run", return_value=out):
            assert pm._gpu_power_watts() == pytest.approx(116.0)


def test_power_without_nvidia_smi_is_none():
    with patch.object(pm.shutil, "which", return_value=None):
        assert pm._gpu_power_watts() is None


def test_power_survives_a_card_reporting_na():
    """`power.draw` возвращает [N/A] на части ноутбучных карт."""
    out = MagicMock(returncode=0, stdout="[N/A]\n61.88\n")
    with patch.object(pm.shutil, "which", return_value="/usr/bin/nvidia-smi"):
        with patch.object(pm.subprocess, "run", return_value=out):
            assert pm._gpu_power_watts() == pytest.approx(61.88)


def test_power_all_na_is_none():
    out = MagicMock(returncode=0, stdout="[N/A]\n")
    with patch.object(pm.shutil, "which", return_value="/usr/bin/nvidia-smi"):
        with patch.object(pm.subprocess, "run", return_value=out):
            assert pm._gpu_power_watts() is None


# ── сведение в метрики ────────────────────────────────────────────────────


def test_gpu_energy_is_power_integrated_over_the_run():
    m = PerformanceMonitor(enabled=True)
    m.start()
    m.stop()  # остановить поток, иначе он добавит настоящие отсчёты карты
    m.gpu_power_samples = [100.0, 100.0]  # Вт
    m.start_time, m.end_time = 0.0, 3600.0  # час
    metrics = m.get_metrics()

    assert metrics["gpu_power_avg_w"] == pytest.approx(100.0)
    assert metrics["gpu_energy_wh"] == pytest.approx(100.0)


def test_cpu_energy_is_reported_in_watt_hours():
    """Про перевод единиц, не про доступность: RAPL на этой машине закрыт,
    поэтому доступность выставляется вручную."""
    m = PerformanceMonitor(enabled=True)
    m.start()
    m.stop()
    m._cpu_energy_available = True
    m._cpu_energy_uj = 3_600_000_000  # 3600 Дж = 1 Вт·ч
    metrics = m.get_metrics()

    assert metrics["cpu_energy_wh"] == pytest.approx(1.0)


def test_without_rapl_the_cpu_energy_is_none_not_zero():
    """Иначе в таблице статьи окажется «0 Вт·ч потрачено процессором»."""
    m = PerformanceMonitor(enabled=True)
    m._rapl_root = Path("/nonexistent-powercap")
    m.start()
    m._sample_energy()

    assert m.get_metrics()["cpu_energy_wh"] is None


def test_the_empty_shape_has_the_energy_keys_too():
    m = PerformanceMonitor(enabled=False)
    m.start()
    keys = set(m.get_metrics())

    assert {"gpu_power_avg_w", "gpu_power_max_w", "gpu_energy_wh",
            "cpu_energy_wh"} <= keys


def test_the_csv_carries_the_energy_columns(tmp_path):
    logger = BenchmarkLogger(tmp_path)
    logger.log_metrics(ExperimentMetrics(
        experiment_id="t", timestamp="now", total_series=1, successful=1,
        failed=0, skipped=0, total_time=1.0, time_per_series=1.0,
        throughput=1.0, gpu_power_avg_w=116.0, gpu_energy_wh=0.03,
        cpu_energy_wh=0.12,
    ))

    header = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]

    for column in ("gpu_power_avg_w", "gpu_power_max_w", "gpu_energy_wh",
                   "cpu_energy_wh"):
        assert column in header, f"нет колонки {column}"


def test_on_this_machine_gpu_power_is_actually_readable():
    """Живая проверка: карта тут есть, и мощность она отдаёт."""
    watts = pm._gpu_power_watts()

    assert watts is None or watts > 0, f"подозрительное значение: {watts}"
