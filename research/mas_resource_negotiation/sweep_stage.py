#!/usr/bin/env python3
"""Развёртка одного этапа по параллелизму: профиль насыщения (P0, P1b).

Что меряет. Пропускную способность этапа как функцию числа воркеров, то есть
ровно те кривые, которые в прошлой статье стали убеждениями агентов (Table I).
Точка насыщения определяется как в ней: наименьшее число процессов, после
которого прирост пропускной способности меньше 5%.

Как запускает. Через `orchestrator.py` с сгенерированным конфигом, а не вызовом
скрипта этапа напрямую. Имена аргументов этапов оркестратор подставляет буква в
букву (KI-071), и дублировать это знание здесь означало бы мерить не то, что
работает в бою.

Что обеспечивает сравнимость клеток:

* **одинаковый вход.** Подготовленное дерево копируется на каждую клетку, так
  что все начинают с побайтно равных данных. Копия дешева рядом с этапом:
  `nifti/` на 100 пациентов — сотни мегабайт, этап 05 — десятки минут;
* **выключенный `skip_existing`.** Иначе вторая клетка «пропустит» готовую
  работу и окажется бесконечно быстрой;
* **один режим кода.** Везде `mode: parallel`, в том числе при одном воркере:
  последовательная ветка этапа — другой путь исполнения, и подмешивать его в
  кривую нельзя;
* **сброс дискового кэша** перед клеткой, как в протоколе прошлой статьи. Без
  него I/O-этапы показывали ускорение завышенным в 2–4 раза. Требует прав;
  удалось или нет — пишется в артефакт, чтобы протокол ехал вместе с данными.

Возобновление. Результат каждой клетки дописывается строкой в `results.jsonl`;
при повторном запуске уже посчитанные клетки пропускаются. Развёртка на
десятки прогонов неизбежно будет прерываться.

    python research/mas_resource_negotiation/sweep_stage.py \\
        --stage stage_08_anatomical_analysis --workers 1 2 4 \\
        --work-dir <подготовленное дерево> --out <каталог результатов>
    ... --summary          # только таблица по уже посчитанному
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from performance_monitor import PerformanceMonitor  # noqa: E402
from utils.task_timeline import read_timeline, turnaround  # noqa: E402

RESULTS_NAME = "results.jsonl"


# ── планирование ──────────────────────────────────────────────────────────


def plan_cells(stage: str, workers: Iterable[int], repeats: int = 1) -> list:
    """Клетки развёртки. Повторы — отдельные клетки, а не усреднение на месте:
    разброс между повторами сам является результатом (прошлая статья
    показывала его полосой на графике)."""
    return [{"stage": stage, "workers": int(w), "repeat": r}
            for w in workers for r in range(1, repeats + 1)]


def cell_key(cell: dict) -> str:
    return f"{cell['stage']}|w{cell['workers']}|r{cell['repeat']}"


def done_cells(results_path: Path) -> set:
    """Ключи клеток, уже посчитанных ранее — для возобновления."""
    done = set()
    try:
        text = Path(results_path).read_text(encoding="utf-8")
    except OSError:
        return done
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        # Валидный JSON — ещё не наша запись: строка или число разберутся
        # успешно и уронили бы разбор на `.get`.
        if not isinstance(row, dict) or not row.get("ok"):
            continue
        try:
            done.add(cell_key(row))
        except KeyError:
            continue
    return done


# ── протокол ──────────────────────────────────────────────────────────────


def drop_caches() -> bool:
    """Сбросить дисковый кэш. True, если получилось.

    Требует прав: без них возвращаем False, а не падаем — прогон всё равно
    полезен, просто протокол в артефакте будет честно помечен.
    """
    try:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        time.sleep(2)  # стабилизация, как в протоколе прошлой статьи
        return True
    except OSError:
        return False


def _first_line(command: list) -> Optional[str]:
    """Вывод внешней программы одной строкой, или None если её нет."""
    try:
        if not shutil.which(command[0]):
            return None
        out = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        for line in (out.stdout or out.stderr).splitlines():
            if line.strip():
                return line.strip()
    except Exception:
        return None
    return None


def _cpu_model() -> Optional[str]:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def environment() -> dict:
    """Что надо записать про машину, чтобы числа потом что-то значили.

    Блок 0 плана требует фиксировать железо и версии до любого прогона. Имя
    хоста для этого недостаточно — на этой машине `platform.node()` возвращает
    бесполезную "1", — поэтому пишем ещё и модель процессора.

    Версии ANTs, FSL и dcm2niix берутся, только если бинарник есть: развёртка
    может идти и на хосте, где из них стоит не всё, и тогда честнее записать
    None, чем промолчать.
    """
    from utils.resource_planner import available_cpu_count, cgroup_memory_limit_bytes
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
    except Exception:
        total_ram = None
    gpus = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    try:
        import ants
        ants_version = getattr(ants, "__version__", None)
    except Exception:
        ants_version = None
    return {
        "host": platform.node() or None,
        "cpu_model": _cpu_model(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_budget": available_cpu_count(),
        "os_cpu_count": os.cpu_count(),
        "total_ram_bytes": total_ram,
        "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes(),
        "gpus": gpus,
        "nvidia_driver": _first_line(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "antspyx": ants_version,
        "fsl_bet": _first_line(["bet", "-h"]),
        "dcm2niix": _first_line(["dcm2niix", "-v"]),
        "git_commit": _first_line(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_dirty": _first_line(["git", "-C", str(REPO), "status", "--porcelain"])
                     is not None,
    }


# ── один прогон ───────────────────────────────────────────────────────────


def build_cell_config(base: dict, stage: str, workers: int,
                      work_dir: Path, max_subjects: Optional[int]) -> dict:
    """Конфиг, в котором включён ровно один этап с заданным параллелизмом."""
    cfg = copy.deepcopy(base)
    if stage not in cfg.get("stages", {}):
        raise KeyError(f"нет такого этапа в конфиге: {stage}")
    for name, spec in cfg["stages"].items():
        spec["enabled"] = (name == stage)
        # `script` разрешается относительно каталога КОНФИГА
        # (`utils/config_loader.validate_config`), а конфиг клетки лежит в
        # каталоге результатов, не в репозитории. Абсолютный путь снимает
        # зависимость от того, где файл оказался. Пути в `args` трогать не
        # надо: их оркестратор разрешает от своего собственного каталога.
        if spec.get("script") and not Path(spec["script"]).is_absolute():
            spec["script"] = str(REPO / spec["script"])
    args = cfg["stages"][stage].setdefault("args", {})
    args["workers"] = workers
    args["mode"] = "parallel"
    # Каждая клетка обязана делать работу заново. Решает это НЕ общий
    # `general.skip_existing` — оркестратор его только пишет в лог, — а
    # собственный ключ этапа в `args`: значение True он превращает в флаг
    # `--skip-existing`, а False просто не передаёт. Проверены оба написания,
    # потому что имена аргументов идут буква в букву (KI-071), и у этапов
    # они разные: `skip-existing` у 04–08, `force` у 01–04.
    for key in ("skip-existing", "skip_existing"):
        if key in args:
            args[key] = False
    if "force" in args:
        args["force"] = True
    cfg["general"]["root_input_dir"] = str(work_dir)
    cfg["general"]["root_output_dir"] = str(work_dir)
    # Обязательно: иначе клетка пропустит уже сделанную работу.
    cfg["general"]["skip_existing"] = False
    if max_subjects is None:
        cfg["general"].pop("max_subjects", None)
    else:
        cfg["general"]["max_subjects"] = max_subjects
    return cfg


def run_cell(cell: dict, base_config: dict, prepared: Path, out_dir: Path,
             max_subjects: Optional[int], want_drop_caches: bool,
             copy_input: bool = True) -> dict:
    """Одна клетка: свежий вход, сброс кэша, прогон, метрики."""
    work_dir = out_dir / f"work_{cell['workers']}w_r{cell['repeat']}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    if copy_input:
        shutil.copytree(prepared, work_dir)
    else:
        work_dir.mkdir(parents=True)

    config_path = out_dir / f"config_{cell['workers']}w_r{cell['repeat']}.yaml"
    cfg = build_cell_config(base_config, cell["stage"], cell["workers"],
                            work_dir, max_subjects)
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True,
                                          sort_keys=False), encoding="utf-8")

    caches_dropped = drop_caches() if want_drop_caches else None

    monitor = PerformanceMonitor(enabled=True)
    monitor.start()
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(REPO / "orchestrator.py"), "--config", str(config_path)],
        capture_output=True, text=True, cwd=str(REPO))
    wall = time.time() - started
    monitor.stop()
    metrics = monitor.get_metrics()

    # Сколько работы реально сделано — из журнала задач, а не из кода возврата.
    timeline = work_dir / "logs" / "task_timeline.jsonl"
    stats = turnaround(read_timeline(timeline)) if timeline.is_file() else None

    row = {
        **cell,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "wall_seconds": round(wall, 3),
        "caches_dropped": caches_dropped,
        "sessions_done": stats["n"] if stats else None,
        "sessions_incomplete": len(stats["incomplete"]) if stats else None,
        "turnaround_median_s": stats["median_s"] if stats else None,
        "turnaround_p95_s": stats["p95_s"] if stats else None,
        "cpu_tree_avg": metrics.get("cpu_avg"),
        "cpu_system_avg": metrics.get("system_cpu_avg"),
        "memory_peak_mb": metrics.get("memory_peak_mb"),
        "iowait_avg": metrics.get("iowait_avg"),
        "disk_read_mb": metrics.get("disk_read_mb"),
        "disk_write_mb": metrics.get("disk_write_mb"),
        "gpu_power_avg_w": metrics.get("gpu_power_avg_w"),
        "gpu_energy_wh": metrics.get("gpu_energy_wh"),
        "cpu_energy_wh": metrics.get("cpu_energy_wh"),
        "work_dir": str(work_dir),
    }
    if proc.returncode != 0:
        # Оркестратор пишет диагностику в stdout, а не в stderr — без этого
        # причина падения клетки не доезжает до артефакта.
        row["stdout_tail"] = proc.stdout[-3000:]
        row["stderr_tail"] = proc.stderr[-2000:]
    return row


# ── сведение ──────────────────────────────────────────────────────────────


def throughput_rows(rows: list) -> list:
    """Пропускная способность по числу воркеров: сессий в секунду.

    Единица — сессия, а не пациент (KI-042): «3 пациента» SibBMS это 11 сессий.
    """
    by_workers: dict = {}
    for row in rows:
        if not row.get("ok") or not row.get("sessions_done"):
            continue
        by_workers.setdefault(row["workers"], []).append(row)
    out = []
    for workers in sorted(by_workers):
        group = by_workers[workers]
        times = sorted(r["wall_seconds"] for r in group)
        # Медиана, а не среднее: повторов мало, один выброс сдвинул бы среднее.
        median = times[len(times) // 2] if len(times) % 2 else (
            (times[len(times) // 2 - 1] + times[len(times) // 2]) / 2)
        sessions = group[0]["sessions_done"]
        out.append({"workers": workers, "runs": len(group),
                    "seconds_median": median,
                    "seconds_min": times[0], "seconds_max": times[-1],
                    "sessions": sessions,
                    "throughput": sessions / median if median else None})
    return out


def saturation_point(rows: list, threshold: float = 0.05) -> Optional[int]:
    """Наименьшее число воркеров, после которого прирост меньше порога.

    Определение взято из прошлой статьи дословно, чтобы кривые были сравнимы
    с её Table I.
    """
    curve = [r for r in rows if r.get("throughput")]
    if len(curve) < 2:
        return None
    for i in range(len(curve) - 1):
        gain = curve[i + 1]["throughput"] / curve[i]["throughput"] - 1.0
        if gain < threshold:
            return curve[i]["workers"]
    return curve[-1]["workers"]


def print_summary(rows: list) -> None:
    curve = throughput_rows(rows)
    if not curve:
        print("нет успешных клеток")
        return
    base = curve[0]
    print(f"\n{'воркеров':>9} {'прогонов':>9} {'время, с':>12} {'разброс, с':>14} "
          f"{'сессий/с':>10} {'ускорение':>10} {'эффект.':>8}")
    for r in curve:
        speedup = r["throughput"] / base["throughput"] if base["throughput"] else 0
        ratio = base["workers"] / r["workers"] if r["workers"] else 1
        print(f"{r['workers']:>9} {r['runs']:>9} {r['seconds_median']:>12.2f} "
              f"{r['seconds_max'] - r['seconds_min']:>14.2f} "
              f"{r['throughput']:>10.4f} {speedup:>9.2f}× "
              f"{speedup * ratio:>7.2f}")
    sat = saturation_point(curve)
    print(f"\n  точка насыщения: {sat} (прирост < 5%, как в прошлой статье)")
    dropped = {r.get("caches_dropped") for r in rows if r.get("ok")}
    if False in dropped:
        print("  ВНИМАНИЕ: кэш сбрасывался не на всех клетках — "
              "I/O-этапы могут быть завышены")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", help="имя этапа как в pipeline_config.yaml")
    parser.add_argument("--workers", type=int, nargs="+",
                        help="уровни параллелизма")
    parser.add_argument("--work-dir", type=Path,
                        help="подготовленное дерево (вход этапа)")
    parser.add_argument("--out", type=Path, required=True,
                        help="каталог результатов развёртки")
    parser.add_argument("--base-config", type=Path,
                        default=REPO / "pipeline_config.yaml")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--no-drop-caches", action="store_true",
                        help="не пытаться сбрасывать кэш (нужны права)")
    parser.add_argument("--no-copy-input", action="store_true",
                        help="не копировать дерево на клетку (быстрее, но "
                             "клетки перестают быть независимыми)")
    parser.add_argument("--summary", action="store_true",
                        help="только таблица по уже посчитанному")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / RESULTS_NAME

    if args.summary:
        rows = [json.loads(l) for l in results_path.read_text(
            encoding="utf-8").splitlines() if l.strip()]
        print_summary(rows)
        return 0

    for required in ("stage", "workers", "work_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} обязателен без --summary")

    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    (args.out / "environment.json").write_text(
        json.dumps(environment(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    cells = plan_cells(args.stage, args.workers, args.repeats)
    already = done_cells(results_path)
    todo = [c for c in cells if cell_key(c) not in already]
    print(f"клеток всего {len(cells)}, посчитано {len(already)}, "
          f"осталось {len(todo)}")

    for i, cell in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {cell_key(cell)} … ", end="", flush=True)
        row = run_cell(cell, base_config, args.work_dir, args.out,
                       args.max_subjects, not args.no_drop_caches,
                       copy_input=not args.no_copy_input)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["ok"]:
            print(f"{row['wall_seconds']:.1f} с, сессий {row['sessions_done']}")
        else:
            print(f"ОШИБКА (код {row['returncode']})")

    rows = [json.loads(l) for l in results_path.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
