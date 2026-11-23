"""Свод и события МРТ-исследований для платформы СИГМА.

Считает исследования, а не запуски: один запуск держит много сессий BIDS,
а у запуска один lesion_type. Класс сессии читается из отчёта, который
пайплайн уже записал. Запрос за период ничего не пишет. Успешный прогон
кладёт в logs этого запуска по одному событию на исследование.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

from database import PipelineRun

logger = logging.getLogger(__name__)

CLASS_KEYS = ("normal", "glioblastoma", "multiple_sclerosis")

# Пока две тестовые учётки. Имя в Kappa — не название центра, поэтому
# соответствие задано здесь, а в событие уходит английское название.
ORGANIZATION_BY_LOGIN = {
    "e.roppert": "Federal Center of Neurosurgery",
    "test_med1": "International Tomography Center",
}

_DESCRIPTION = {
    "normal": "normal",
    "glioblastoma": "glioblastoma",
    "multiple_sclerosis": "multiple sclerosis",
}

_REPORT_GLOB = {
    "glioblastoma": "*_volume_report.json",
    "multiple_sclerosis": "*_lesion_stats_report.json",
}


class PeriodBoundsError(ValueError):
    """Границы периода нельзя применить. Сообщение годится оператору."""


def parse_period(start: Optional[str], end: Optional[str]):
    """Разобрать ISO 8601. Без зоны — UTC. end обязан быть позже start."""
    if start is None or end is None or not str(start).strip() or not str(end).strip():
        raise PeriodBoundsError("Нужны границы периода start и end в ISO 8601")
    try:
        start_dt = _parse_iso(str(start))
        end_dt = _parse_iso(str(end))
    except ValueError as exc:
        raise PeriodBoundsError("start и end должны быть датой ISO 8601") from exc
    if end_dt <= start_dt:
        raise PeriodBoundsError("end должен быть позже start")
    return start_dt, end_dt


def write_run_events(
    output_path,
    lesion_type: str,
    *,
    finished_at: datetime,
    kappa_user_name: Optional[str] = None,
) -> Optional[Path]:
    """Положить события исследований этого запуска в его logs/.

    Одно событие на сессию, у которой есть класс. Id — пациент sub-XXX,
    не сессия и не больничный номер. Организация берётся по логину того,
    кто стартовал прогон; неизвестный логин не получает чужой центр.
    Ошибка записи не поднимается: прогон уже состоялся.
    """
    root = Path(output_path)
    path = root / "logs" / "sigma_events.json"
    source = ORGANIZATION_BY_LOGIN.get((kappa_user_name or "").strip(), "")
    if not source:
        logger.warning(
            "Событие СИГМА: логин %r не закреплён за организацией, source пустой",
            kappa_user_name,
        )

    run = SimpleNamespace(
        run_id=root.name,
        output_path=str(root),
        lesion_type=lesion_type,
        kappa_dataset_id=None,
    )
    stamp = _iso_z(finished_at)
    classes = _classes_for_run(run)
    events = []
    for session_key in sorted(classes):
        patient_id = session_key.split("_", 1)[0]
        events.append({
            "Id": patient_id,
            "modality": "Mri",
            "timestamp": stamp,
            "source": source,
            "description": _DESCRIPTION[classes[session_key]],
        })
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(events, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("События СИГМА не записаны в %s: %s", path, exc)
        return None
    logger.info("События СИГМА записаны: %s (%s)", path, len(events))
    return path


def build_period_stats(db, start: datetime, end: datetime) -> dict:
    """Документ свода. start и end — моменты; наивные считаются UTC.

    Период полуоткрытый: начало входит, конец нет. Одна и та же сессия
    внутри периода считается один раз, побеждает более поздний запуск.
    """
    start_utc = _as_utc(start)
    end_utc = _as_utc(end)

    runs = (
        db.query(PipelineRun)
        .filter(PipelineRun.status == "completed")
        .filter(PipelineRun.completed_at.isnot(None))
        .all()
    )
    placed = []
    for run in runs:
        stamp = _as_utc(run.completed_at)
        if start_utc <= stamp < end_utc:
            placed.append(run)
    placed.sort(key=lambda run: (_as_utc(run.completed_at), run.run_id))

    # Номер sub-XXX начинается заново в каждом датасете Kappa и в локальной
    # нумерации по типу поражения. Повтор внутри того же диапазона — одна
    # сессия, и побеждает более поздний запуск.
    winners: Dict[str, str] = {}
    for run in placed:
        for session_key, outcome in _classes_for_run(run).items():
            winners[_study_id(run, session_key)] = outcome

    by_class = {key: 0 for key in CLASS_KEYS}
    for outcome in winners.values():
        by_class[outcome] += 1
    studies_total = sum(by_class.values())
    return {
        "period": {"start": _iso_z(start_utc), "end": _iso_z(end_utc)},
        "modality": "Mri",
        "studies_total": studies_total,
        "detected_total": studies_total - by_class["normal"],
        "by_class": by_class,
    }


def _classes_for_run(run) -> Dict[str, str]:
    pattern = _REPORT_GLOB.get(run.lesion_type)
    if pattern is None:
        logger.warning(
            "Свод СИГМА: запуск %s пропущен, тип поражения %r не из свода",
            run.run_id, run.lesion_type,
        )
        return {}

    root = Path(run.output_path) / "segmentation"
    if not root.is_dir():
        logger.warning(
            "Свод СИГМА: запуск %s пропущен, нет каталога segmentation",
            run.run_id,
        )
        return {}

    grouped: Dict[str, set] = {}
    for path in sorted(root.rglob(pattern)):
        session_key = _session_key(path)
        if session_key is None:
            logger.warning(
                "Свод СИГМА: запуск %s, файл %s без ключа сессии",
                run.run_id, path.name,
            )
            continue
        outcome = _class_from_report(run.lesion_type, path)
        if outcome is None:
            logger.warning(
                "Свод СИГМА: запуск %s, сессия %s, отчёт %s не читается",
                run.run_id, session_key, path.name,
            )
            continue
        grouped.setdefault(session_key, set()).add(outcome)

    decided = {}
    for session_key, outcomes in grouped.items():
        if len(outcomes) != 1:
            logger.warning(
                "Свод СИГМА: запуск %s, сессия %s, отчёты расходятся — сессия не в своде",
                run.run_id, session_key,
            )
            continue
        decided[session_key] = next(iter(outcomes))
    return decided


def _class_from_report(lesion_type: str, path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    if lesion_type == "glioblastoma":
        tumor = data.get("total_tumor")
        if not isinstance(tumor, dict) or "voxel_count" not in tumor:
            return None
        voxels = _as_int(tumor.get("voxel_count"))
        if voxels is None:
            return None
        return "normal" if voxels == 0 else "glioblastoma"

    if lesion_type == "multiple_sclerosis":
        if "lesion_count" not in data:
            return None
        count = _as_int(data.get("lesion_count"))
        if count is None:
            return None
        return "normal" if count == 0 else "multiple_sclerosis"

    return None


def _as_int(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _study_id(run, session_key: str) -> str:
    if run.kappa_dataset_id is not None:
        return f"ds:{run.kappa_dataset_id}:{session_key}"
    lesion_type = run.lesion_type or "unknown"
    return f"local:{lesion_type}:{session_key}"


def _session_key(path: Path) -> Optional[str]:
    sub = ses = None
    for part in path.parts[:-1]:
        if sub is None and part.startswith("sub-"):
            sub = part
        elif ses is None and part.startswith("ses-"):
            ses = part
    if sub and ses:
        return f"{sub}_{ses}"

    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".nii"):
        name = name[:-4]
    if name.endswith(".json"):
        name = name[:-5]
    parts = name.split("_")
    for index, part in enumerate(parts[:-1]):
        if part.startswith("sub-") and parts[index + 1].startswith("ses-"):
            return f"{part}_{parts[index + 1]}"
    return None


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return _as_utc(parsed)


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _iso_z(moment: datetime) -> str:
    return _as_utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")
