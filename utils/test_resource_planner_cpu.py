"""Сколько процессоров этапу реально можно занять (KI-078).

Потоки ANTs/ITK задавались от `os.cpu_count()`, то есть от процессоров ХОСТА:
ни `cpuset-cpus`, ни квота `cpus` контейнера не учитывались. При
`cpuset-cpus: "0-7"` на 24-ядерной машине этап выдавал 11 потоков на воркер и
переподписывал 8 доступных ядер. Та же слепота, что KI-058 исправил для
памяти, только по процессору.

Это нужно и продакшену (урезанный контейнер), и исследованию: варьировать
ядра лимитами контейнера нельзя, пока замер меряет не ту конфигурацию,
которую задали.
"""
import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import resource_planner
from utils.resource_planner import available_cpu_count, cgroup_cpu_limit

REPO = Path(__file__).resolve().parent.parent


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ── cgroup v2 ──────────────────────────────────────────────────────────────


def test_v2_quota_is_read_as_a_cpu_count(tmp_path):
    """`cpu.max` — это "квота период" в микросекундах: 200000/100000 = 2 ядра."""
    path = _write(tmp_path, "cpu.max", "200000 100000\n")

    assert cgroup_cpu_limit(v2_path=path) == 2.0


def test_v2_unlimited_means_no_limit(tmp_path):
    path = _write(tmp_path, "cpu.max", "max 100000\n")

    assert cgroup_cpu_limit(v2_path=path) is None


def test_v2_fractional_quota_is_kept_fractional(tmp_path):
    """Округляет не чтец, а потребитель — ему решать, в какую сторону."""
    path = _write(tmp_path, "cpu.max", "150000 100000\n")

    assert cgroup_cpu_limit(v2_path=path) == 1.5


# ── cgroup v1 (баргузин может быть на нём) ─────────────────────────────────


def test_v1_quota_and_period_are_read(tmp_path):
    quota = _write(tmp_path, "cpu.cfs_quota_us", "400000\n")
    period = _write(tmp_path, "cpu.cfs_period_us", "100000\n")

    assert cgroup_cpu_limit(
        v2_path=str(tmp_path / "absent"),
        v1_quota_path=quota,
        v1_period_path=period,
    ) == 4.0


def test_v1_minus_one_means_unlimited(tmp_path):
    quota = _write(tmp_path, "cpu.cfs_quota_us", "-1\n")
    period = _write(tmp_path, "cpu.cfs_period_us", "100000\n")

    assert cgroup_cpu_limit(
        v2_path=str(tmp_path / "absent"),
        v1_quota_path=quota,
        v1_period_path=period,
    ) is None


# ── отказоустойчивость чтения ──────────────────────────────────────────────


def test_missing_files_mean_no_limit(tmp_path):
    assert cgroup_cpu_limit(
        v2_path=str(tmp_path / "nope"),
        v1_quota_path=str(tmp_path / "nope"),
        v1_period_path=str(tmp_path / "nope"),
    ) is None


def test_garbage_does_not_raise(tmp_path):
    """Планировщик не имеет права падать в вызывающий этап (как и по памяти)."""
    path = _write(tmp_path, "cpu.max", "что-то не то\n")

    assert cgroup_cpu_limit(v2_path=path) is None


def test_zero_period_does_not_divide_by_zero(tmp_path):
    path = _write(tmp_path, "cpu.max", "100000 0\n")

    assert cgroup_cpu_limit(v2_path=path) is None


# ── итоговый бюджет процессора ─────────────────────────────────────────────


def test_affinity_caps_the_count(monkeypatch, tmp_path):
    """`cpuset-cpus: "0-7"` — ровно тот случай, из-за которого KI-078 завели."""
    monkeypatch.setattr(resource_planner.os, "sched_getaffinity",
                        lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 24)

    assert available_cpu_count(v2_path=str(tmp_path / "absent"),
                               v1_quota_path=str(tmp_path / "absent"),
                               v1_period_path=str(tmp_path / "absent")) == 8


def test_quota_caps_the_count(monkeypatch, tmp_path):
    """`--cpus=6`: привязка показывает все 24 ядра, ограничивает квота."""
    monkeypatch.setattr(resource_planner.os, "sched_getaffinity",
                        lambda pid: set(range(24)), raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 24)
    path = _write(tmp_path, "cpu.max", "600000 100000\n")

    assert available_cpu_count(v2_path=path) == 6


def test_the_tighter_of_the_two_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(resource_planner.os, "sched_getaffinity",
                        lambda pid: set(range(4)), raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 24)
    path = _write(tmp_path, "cpu.max", "600000 100000\n")

    assert available_cpu_count(v2_path=path) == 4


def test_a_fractional_quota_is_floored(monkeypatch, tmp_path):
    """Переподписка — это и есть исправляемый отказ, поэтому округляем вниз."""
    monkeypatch.setattr(resource_planner.os, "sched_getaffinity",
                        lambda pid: set(range(24)), raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 24)
    path = _write(tmp_path, "cpu.max", "150000 100000\n")

    assert available_cpu_count(v2_path=path) == 1


def test_a_tiny_quota_still_leaves_one_cpu(monkeypatch, tmp_path):
    """0.3 ядра -> 0 воркеров сделало бы этап невыполнимым."""
    monkeypatch.setattr(resource_planner.os, "sched_getaffinity",
                        lambda pid: set(range(24)), raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 24)
    path = _write(tmp_path, "cpu.max", "30000 100000\n")

    assert available_cpu_count(v2_path=path) == 1


def test_falls_back_to_cpu_count_without_affinity(monkeypatch, tmp_path):
    """`sched_getaffinity` есть не везде — на не-Linux его просто нет."""
    monkeypatch.delattr(resource_planner.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: 12)

    assert available_cpu_count(v2_path=str(tmp_path / "absent"),
                               v1_quota_path=str(tmp_path / "absent"),
                               v1_period_path=str(tmp_path / "absent")) == 12


def test_falls_back_when_everything_is_unknown(monkeypatch, tmp_path):
    """Фолбэк прежнего кода был `or 4` — сохраняем его, а не ноль."""
    monkeypatch.delattr(resource_planner.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(resource_planner.os, "cpu_count", lambda: None)

    assert available_cpu_count(v2_path=str(tmp_path / "absent"),
                               v1_quota_path=str(tmp_path / "absent"),
                               v1_period_path=str(tmp_path / "absent")) == 4


def test_on_this_machine_it_is_sane():
    """Без лимитов значение должно совпасть с числом процессоров машины."""
    value = available_cpu_count()

    assert 1 <= value <= (os.cpu_count() or 4)


# ── защита от повторного расползания ──────────────────────────────────────


def _cpu_count_calls(path: Path):
    """Строки, где читается число процессоров напрямую, в обход бюджета."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "cpu_count"
                and isinstance(func.value, ast.Name) and func.value.id in ("os", "multiprocessing")):
            hits.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id == "cpu_count":
            hits.append(node.lineno)
    return hits


STAGE_SCRIPTS = sorted(
    p for p in (REPO / "scripts").glob("*.py")
    if not p.name.startswith("test_")
)


@pytest.mark.parametrize("script", STAGE_SCRIPTS, ids=lambda p: p.name)
def test_no_stage_reads_the_processor_count_directly(script):
    """Шесть мест в пяти файлах читали хост напрямую — дальше только бюджет.

    Пока это можно сделать на месте, следующий этап снова скопирует образец у
    соседа вместе со слепотой к cpuset.
    """
    lines = _cpu_count_calls(script)

    assert lines == [], (
        f"{script.name} читает число процессоров напрямую (строки {lines}) — "
        f"используйте utils.resource_planner.available_cpu_count()"
    )


def test_the_guard_actually_detects_a_violation(tmp_path):
    """Проверка, которая утверждает отсутствие нарушений, должна уметь их видеть."""
    bad = tmp_path / "offender.py"
    bad.write_text(
        "import os\n"
        "from multiprocessing import cpu_count\n"
        "a = os.cpu_count()\n"
        "b = cpu_count()\n",
        encoding="utf-8",
    )

    assert _cpu_count_calls(bad) == [3, 4]


def test_the_guard_ignores_prose(tmp_path):
    prose = tmp_path / "prose.py"
    prose.write_text(
        '"""Раньше тут стоял os.cpu_count() — больше нет."""\n'
        "# и в комментарии cpu_count() не нарушение\n"
        "X = 'os.cpu_count()'\n",
        encoding="utf-8",
    )

    assert _cpu_count_calls(prose) == []
