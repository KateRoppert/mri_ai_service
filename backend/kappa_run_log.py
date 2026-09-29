"""Per-run Kappa delivery log: {output_path}/logs/kappa.log

Why a separate file. The service log interleaves every run and every
subsystem, so answering "what happened to THIS run's upload" means grepping
a uuid through thousands of lines — and the lines that matter (which session
failed, and why) were never there to begin with.

What goes in: one line per event that changes what a human would conclude.
Attempts, per-session outcomes, verdicts. What stays out: HTTP plumbing,
tokens, JSON dumps, anything that repeats every tick. A log nobody can read
through is the same as no log.

Written in Russian, unlike the code around it: this file is an operator
artifact that sits in the run's own output folder, next to the reports.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Per-session error codes -> what the operator needs to understand.
_SESSION_ERRORS = {
    "name_clash": "номер уже занят другими данными — нужна ручная проверка",
    "no files": "нет файлов для выгрузки",
    "upload failed": "Kappa не приняла файлы",
    "duplicate": "уже в датасете, локальная запись не восстановлена",
}

_VERDICTS = {
    "done": "доставлено",
    "pending": "ждёт",
    "needs_attention": "требует внимания",
}

_REASONS = {
    "network": "Kappa недоступна",
    "no_session": "нет входа в Kappa",
    "name_clash": "номер занят другими данными",
    "missing_files": "нет файлов",
    "stuck": "не удаётся выгрузить больше суток",
}


def _log_path(output_path) -> Optional[Path]:
    if not output_path:
        return None
    return Path(output_path) / "logs" / "kappa.log"


def append(output_path, *lines: str) -> None:
    """Add lines to the run's Kappa log. Never raises.

    Delivery must not fail because a log file could not be written — the
    run's output directory may be on a disconnected mount, or gone entirely
    (which is itself one of the states we report).
    """
    path = _log_path(output_path)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(f"{stamp}  {line}\n")
    except OSError as e:
        logger.debug("Не удалось записать %s: %s", path, e)


def log_start(output_path, dataset_id: Optional[int], warning: Optional[str]) -> None:
    """The run has started: where its patients are headed."""
    if dataset_id:
        append(output_path, f"Запуск начат. Датасет Kappa: {dataset_id}.")
    else:
        append(output_path, "Запуск начат. Датасет Kappa ещё не определён.")
    if warning:
        append(output_path, f"  {warning}")


def log_attempt(
    output_path,
    source: str,
    result: Optional[Dict[str, Any]],
    exc: Optional[BaseException],
    verdict: Dict[str, Any],
) -> None:
    """One delivery attempt, start to verdict.

    `source` is who tried: 'после прогона', 'фоновая досылка', 'вручную'.
    """
    detail = verdict.get("detail") or {}
    attempt_no = detail.get("attempts") or 0

    if exc is not None:
        append(output_path, f"Попытка ({source}): сорвалась — {type(exc).__name__}")
    elif result is None:
        append(output_path, f"Попытка ({source}): загрузчик не отработал")
    elif result.get("error") == "no_session":
        append(output_path, f"Попытка ({source}): пропущена — нет входа в Kappa")
    elif result.get("error"):
        append(output_path,
               f"Попытка ({source}): не удалось определить датасет "
               f"({result['error']})")
    else:
        sessions = result.get("sessions") or []
        append(output_path,
               f"Попытка ({source}): сессий найдено {len(sessions)}")
        for item in sessions:
            name = item.get("session") or "?"
            if item.get("success"):
                entity = item.get("entity_id")
                suffix = f" (entity {entity})" if entity else ""
                # Имя в Kappa может отличаться от имени на диске: сессия
                # попала в датасет под другим номером. Писать локальное имя
                # значит утверждать, что в датасете лежит не то, что лежит.
                actual = item.get("kappa_name")
                if item.get("skipped_upload"):
                    mark = (f"уже в датасете под именем {actual}"
                            if actual and actual != name
                            else "уже была в датасете")
                else:
                    mark = "загружено"
                append(output_path, f"    {name} — {mark}{suffix}")
            else:
                code = item.get("error") or "неизвестная ошибка"
                append(output_path,
                       f"    {name} — не отправлено: "
                       f"{_SESSION_ERRORS.get(code, code)}")

    status = _VERDICTS.get(verdict.get("status"), verdict.get("status"))
    reason = detail.get("reason")
    tail = f" ({_REASONS[reason]})" if reason in _REASONS else ""
    counts = f"{detail.get('delivered', 0)} из {detail.get('total', 0)}"

    line = f"  Итог: {status} — {counts}{tail}"
    nxt = verdict.get("next_attempt")
    if nxt is not None:
        line += f", следующая попытка {nxt.strftime('%H:%M:%S')}"
    if attempt_no:
        line += f" [попыток: {attempt_no}]"
    append(output_path, line)
