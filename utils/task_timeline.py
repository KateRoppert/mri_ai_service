"""Машинно-читаемый журнал задач прогона: кто когда закончил.

Зачем. Makespan — метрика партии, а клиническая мотивация работы — время
ожидания одного пациента. Эти две величины расходятся качественно: при
последовательной обработке этапов НИ ОДИН пациент не готов почти до конца
прогона, а при конвейерной они выходят потоком. Единственное число этого не
покажет, поэтому нужен журнал, из которого считается распределение.

Почему не разбор логов. Строку завершения задачи с субъектом, сессией и
временем пишет только этап 05; у остальных шести её нет. Добавлять
инструментирование всё равно пришлось бы, и машинно-читаемая форма надёжнее
текста, который меняется при правке формулировок.

Кто пишет. Родитель, по образцу KI-058: воркер измеряет, записывает родитель.
У родителя под рукой и `(subject, session)`, и момент завершения.

Формат — JSON Lines, одна запись на строку: дописывание из нескольких
процессов не портит файл, и его можно читать по мере роста. Записи:

    {"event": "stage_started",  "stage": ..., "at": ...}
    {"event": "stage_finished", "stage": ..., "at": ..., <счётчики>}
    {"event": "task_done", "stage": ..., "subject": ..., "session": ...,
     "status": "ok"|"failed"|"skipped", "at": ..., "duration_s": ...}

Единица у этапов разная: 01 работает по пациентам, 03 по сериям, 04 по
снимкам, 05–08 по сессиям. `task_done` пишут те, чья единица — сессия или
пациент; остальные ограничиваются границами этапа. Для turnaround этого
достаточно: он считается от старта прогона до последнего завершения сессии.

Ничего из этого не имеет права сломать прогон: все функции записи глотают
ошибки, как и планировщик ресурсов.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

TIMELINE_NAME = "task_timeline.jsonl"


def timeline_path(log_file: Optional[Any]) -> Optional[Path]:
    """Путь журнала рядом с логами прогона, или None если логов нет.

    Этапы получают `--log_file` вида `{run}/logs/stages/stage_05_*.log`, а
    иногда `{run}/logs/*.log`. Журнал в обоих случаях ложится в `{run}/logs/`,
    чтобы на прогон он был один.
    """
    if not log_file:
        return None
    try:
        parent = Path(log_file).resolve().parent
        if parent.name == "stages":
            parent = parent.parent
        return parent / TIMELINE_NAME
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("task timeline path unavailable: %s", e)
        return None


def _write(timeline: Optional[Path], payload: Dict[str, Any]) -> None:
    if timeline is None:
        return
    try:
        payload.setdefault("at", time.time())
        line = json.dumps(payload, ensure_ascii=False, default=str)
        Path(timeline).parent.mkdir(parents=True, exist_ok=True)
        # Одна запись = один write() строки с переводом: при O_APPEND короткие
        # строки не перемешиваются между процессами.
        with open(timeline, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.debug("task timeline write skipped: %s", e)


def stage_started(timeline: Optional[Path], stage: str) -> None:
    """Старт этапа. Самый ранний такой момент и есть точка отсчёта прогона."""
    _write(timeline, {"event": "stage_started", "stage": stage, "pid": os.getpid()})


def stage_finished(timeline: Optional[Path], stage: str, **counters: Any) -> None:
    _write(timeline, {"event": "stage_finished", "stage": stage, **counters})


def task_done(
    timeline: Optional[Path],
    stage: str,
    subject: Optional[str],
    session: Optional[str] = None,
    status: str = "ok",
    duration_s: Optional[float] = None,
) -> None:
    """Одна завершённая единица работы этапа."""
    _write(timeline, {
        "event": "task_done",
        "stage": stage,
        "subject": subject,
        "session": session,
        "status": status,
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
    })


def read_timeline(path: Any) -> List[Dict[str, Any]]:
    """Записи журнала. Битые строки пропускаются, а не роняют разбор."""
    records: List[Dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Линейная интерполяция, как у numpy, но без зависимости.

    Повторов в матрице мало, и округление до ближайшего элемента на выборке
    из трёх значений даёт не то, что ожидает читатель таблицы.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def turnaround(
    records: Iterable[Dict[str, Any]],
    final_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Время ожидания на сессию и его распределение.

    Turnaround одной сессии — от старта прогона (самый ранний
    `stage_started`) до её последнего успешного `task_done`. Так считает
    клиницист: он ждёт с момента, когда партия отправлена.

    `final_stage` задаёт, что считается готовностью. Без него берётся этап,
    который в журнале отметился последним. Сессии, не дошедшие до него,
    попадают в `incomplete`, а не в распределение: смешивать завершённые и
    потерянные нельзя — иначе стратегия «выигрывает», потеряв работу.
    """
    records = list(records)
    starts = [r["at"] for r in records
              if r.get("event") == "stage_started" and r.get("at") is not None]
    if not starts:
        return {"run_started_at": None, "sessions": {}, "incomplete": [],
                "n": 0, "median_s": None, "p95_s": None,
                "max_s": None, "min_s": None, "stage_windows": {}}
    run_started = min(starts)

    tasks = [r for r in records if r.get("event") == "task_done"]
    if final_stage is None and tasks:
        final_stage = max(tasks, key=lambda r: r.get("at") or 0).get("stage")

    last_ok: Dict[Tuple[Optional[str], Optional[str]], float] = {}
    reached_final = set()
    seen = set()
    for r in tasks:
        key = (r.get("subject"), r.get("session"))
        seen.add(key)
        if r.get("status") != "ok" or r.get("at") is None:
            continue
        last_ok[key] = max(last_ok.get(key, 0.0), r["at"])
        if r.get("stage") == final_stage:
            reached_final.add(key)

    sessions = {key: round(at - run_started, 3)
                for key, at in last_ok.items() if key in reached_final}
    incomplete = sorted(key for key in seen if key not in reached_final)
    values = list(sessions.values())

    windows: Dict[str, Dict[str, Optional[float]]] = {}
    for r in records:
        stage = r.get("stage")
        if not stage or r.get("at") is None:
            continue
        window = windows.setdefault(
            stage, {"started_at": None, "finished_at": None, "last_seen_at": None})
        if r.get("event") == "stage_started":
            window["started_at"] = (r["at"] if window["started_at"] is None
                                    else min(window["started_at"], r["at"]))
        elif r.get("event") == "stage_finished":
            window["finished_at"] = (r["at"] if window["finished_at"] is None
                                     else max(window["finished_at"], r["at"]))
        else:
            # Fallback for a stage that writes no stage_finished — either it
            # only marks its boundary (units finer than a session) or it died.
            # The last thing it recorded is the best end we have.
            last = window.get("last_seen_at")
            window["last_seen_at"] = r["at"] if last is None else max(last, r["at"])

    for window in windows.values():
        if window["finished_at"] is None:
            window["finished_at"] = window.pop("last_seen_at")
        else:
            window.pop("last_seen_at", None)

    return {
        "run_started_at": run_started,
        "final_stage": final_stage,
        "sessions": sessions,
        "incomplete": incomplete,
        "n": len(values),
        "median_s": _percentile(values, 50),
        "p95_s": _percentile(values, 95),
        "max_s": max(values) if values else None,
        "min_s": min(values) if values else None,
        "stage_windows": windows,
    }
