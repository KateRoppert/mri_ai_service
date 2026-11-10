#!/usr/bin/env python3
"""Отчёт по одному прогону: время ожидания на пациента и доля дошедших.

Одна клетка матрицы Блока 2 — это один прогон, и в таблицу от него идут не
только makespan и утилизация. Этот скрипт печатает то, что makespan скрывает:
распределение времени ожидания по сессиям и сколько сессий вообще дошло до
конца. Второе — условие достоверности первого: стратегия, потерявшая работу,
выглядит быстрее (KI-058: пациент терял две сессии на каждом прогоне).

Читает только `{run}/logs/task_timeline.jsonl`, который пишут сами этапы, —
ни GPU, ни данных, ни пайплайна для этого не нужно.

    python research/mas_resource_negotiation/run_report.py <каталог прогона> [--json]
    python research/mas_resource_negotiation/run_report.py <прогон1> <прогон2> ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.task_timeline import TIMELINE_NAME, read_timeline, turnaround


def find_timeline(run_dir: Path) -> Path | None:
    """Журнал лежит в `{run}/logs/`; на всякий случай ищем и глубже.

    Обход дерева глотает ошибки доступа намеренно: скрипту дают список
    прогонов, и один каталог без прав не должен ронять отчёт по остальным.
    """
    try:
        direct = run_dir / "logs" / TIMELINE_NAME
        if direct.is_file():
            return direct
        if run_dir.name == TIMELINE_NAME and run_dir.is_file():
            return run_dir
        found = sorted(run_dir.rglob(TIMELINE_NAME))
        return found[0] if found else None
    except OSError:
        return None


def report(run_dir: Path) -> dict:
    timeline = find_timeline(run_dir)
    if timeline is None:
        return {"run": str(run_dir), "error": f"нет {TIMELINE_NAME}"}

    # Нечитаемый или пустой журнал НЕ должен превращаться в честно
    # выглядящий отчёт «0/0 сессий»: это число пошло бы в таблицу статьи как
    # результат, хотя в действительности замера нет. Поэтому отличаем
    # «пусто» от «успешно и ничего не сделано».
    records = read_timeline(timeline)
    if not records:
        return {"run": str(run_dir), "timeline": str(timeline),
                "error": "журнал пуст или не читается"}

    stats = turnaround(records)
    if stats["run_started_at"] is None:
        return {"run": str(run_dir), "timeline": str(timeline),
                "error": "в журнале нет ни одного stage_started — "
                         "отсчитывать время ожидания не от чего"}

    attempted = stats["n"] + len(stats["incomplete"])
    return {
        "run": str(run_dir),
        "timeline": str(timeline),
        "final_stage": stats["final_stage"],
        "sessions_done": stats["n"],
        "sessions_attempted": attempted,
        "completion_rate": (round(stats["n"] / attempted, 4)
                            if attempted else None),
        "incomplete": [f"{s or '?'}/{ses or '?'}" for s, ses in stats["incomplete"]],
        "turnaround_s": {
            "min": stats["min_s"], "median": stats["median_s"],
            "p95": stats["p95_s"], "max": stats["max_s"],
        },
        "stage_windows_s": {
            stage: round(w["finished_at"] - w["started_at"], 1)
            for stage, w in sorted(stats["stage_windows"].items())
            if w["started_at"] is not None and w["finished_at"] is not None
        },
    }


def _print(data: dict) -> None:
    if "error" in data:
        print(f"{data['run']}: {data['error']}")
        return
    t = data["turnaround_s"]
    rate = data["completion_rate"]
    print(f"\n{data['run']}")
    print(f"  готовность по этапу : {data['final_stage']}")
    print(f"  дошло сессий        : {data['sessions_done']}/{data['sessions_attempted']}"
          + (f"  ({rate:.1%})" if rate is not None else ""))
    if data["incomplete"]:
        shown = ", ".join(data["incomplete"][:8])
        more = "" if len(data["incomplete"]) <= 8 else f" и ещё {len(data['incomplete']) - 8}"
        print(f"  не дошли            : {shown}{more}")
    if t["median"] is not None:
        print("  время ожидания, с   : "
              f"мин {t['min']:.1f} | медиана {t['median']:.1f} | "
              f"p95 {t['p95']:.1f} | макс {t['max']:.1f}")
        # Разброс и есть то, чего не видно по makespan: при пакетной обработке
        # он близок к нулю, при конвейерной — велик.
        spread = t["max"] - t["min"]
        print(f"  разброс (макс-мин)  : {spread:.1f} с"
              f"{' — сессии выходили потоком' if spread > 0.1 * t['max'] else ''}")
    for stage, seconds in data["stage_windows_s"].items():
        print(f"    {stage:34s} {seconds:8.1f} с")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="каталоги прогонов (root_output_dir)")
    parser.add_argument("--json", action="store_true",
                        help="машинно-читаемый вывод для сводной таблицы")
    args = parser.parse_args()

    reports = [report(d) for d in args.run_dirs]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for data in reports:
            _print(data)
    return 0 if all("error" not in r for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
