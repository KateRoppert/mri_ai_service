"""
Версионирование конфигов препроцессинга.
Вычисляет стабильный hash от параметров, влияющих на результат обработки.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Legacy history file, READ-ONLY. New versions go to the preprocessing_versions
# table in the DB instead: this file is tracked by git and configs/ is mounted
# into the container read-write, so writing it from the application dirtied the
# working copy on every config change (KI-085). It still holds the one entry
# registered before that move, and that entry must stay resolvable.
VERSIONS_FILE = Path(__file__).parent.parent / "configs" / "preprocessing_versions.json"


def _extract_functional_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлечь из конфига только параметры, влияющие на результат обработки.
    Исключает: пути (fsl_dir, cache_dir, url), логирование, комментарии.
    """
    functional = {}

    # Atlas — только имя и filename (не пути и URL)
    if "atlas" in config:
        functional["atlas"] = {
            "name": config["atlas"].get("name"),
            "filename": config["atlas"].get("filename"),
        }

    # Steps — все шаги с параметрами
    if "steps" in config:
        functional["steps"] = []
        for step in config["steps"]:
            functional["steps"].append({
                "name": step.get("name"),
                "enabled": step.get("enabled"),
                "params": step.get("params", {}),
            })

    # Modalities
    if "modalities" in config:
        functional["modalities"] = sorted(config["modalities"])

    return functional


def compute_preprocessing_id(config_path: str) -> str:
    """
    Вычислить ID (hash) конфига препроцессинга.
    Возвращает первые 8 символов SHA256 от функциональных параметров.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    functional = _extract_functional_config(config)

    # Сериализуем в каноническую JSON-строку (sorted keys, no spaces)
    canonical = json.dumps(functional, sort_keys=True, ensure_ascii=False)
    full_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    short_id = full_hash[:8]

    logger.debug("Preprocessing ID: %s (from %s)", short_id, config_path)
    return short_id


def get_full_config(config_path: str) -> Dict[str, Any]:
    """Загрузить полный конфиг из файла."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def register_version(config_path: str) -> Optional[str]:
    """Record what this preprocessing_id means, so it can be resolved later.

    Called at run start, where the config on disk is still the one about to
    be used. Never raises: a run must start even when this bookkeeping fails,
    and an unreadable config returns None rather than inventing a label.
    """
    try:
        preprocessing_id = compute_preprocessing_id(config_path)
    except Exception as exc:  # noqa: BLE001 — старт важнее метки
        logger.warning(
            "Could not compute preprocessing id from %s: %s", config_path, exc,
        )
        return None

    try:
        from database import SessionLocal
        from registry_models import PreprocessingVersion

        with open(config_path, "r", encoding="utf-8") as fh:
            functional = _extract_functional_config(yaml.safe_load(fh))

        db = SessionLocal()
        try:
            if db.get(PreprocessingVersion, preprocessing_id) is None:
                db.add(PreprocessingVersion(
                    preprocessing_id=preprocessing_id,
                    config_json=json.dumps(
                        functional, ensure_ascii=False, sort_keys=True),
                ))
                db.commit()
                logger.info(
                    "Registered preprocessing version %s", preprocessing_id,
                )
            else:
                logger.debug(
                    "Preprocessing version already registered: %s",
                    preprocessing_id,
                )
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not store preprocessing version %s: %s",
            preprocessing_id, exc,
        )

    return preprocessing_id


def get_version_config(preprocessing_id: str) -> Optional[Dict[str, Any]]:
    """Settings behind an id: the DB first, then the legacy JSON file.

    The JSON is read-only now. It holds exactly one entry (1e2b93ad, created
    02.04.2026) from before this moved into the DB, and that entry has to
    stay resolvable.
    """
    if not preprocessing_id:
        return None

    try:
        from database import SessionLocal
        from registry_models import PreprocessingVersion

        db = SessionLocal()
        try:
            row = db.get(PreprocessingVersion, preprocessing_id)
            if row is not None:
                return json.loads(row.config_json)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not read preprocessing version %s: %s",
            preprocessing_id, exc,
        )

    entry = _load_versions().get(preprocessing_id)
    return entry.get("config") if entry else None


def list_versions() -> Dict[str, Any]:
    """Все известные версии: сначала унаследованный файл, поверх — из БД.

    Both sources, because the name promises all of them: the DB holds
    everything registered since the move, the file the one entry from before.
    """
    versions: Dict[str, Any] = dict(_load_versions())
    try:
        from database import SessionLocal
        from registry_models import PreprocessingVersion

        db = SessionLocal()
        try:
            for row in db.query(PreprocessingVersion).all():
                versions[row.preprocessing_id] = {
                    "created_at": (
                        row.created_at.isoformat() if row.created_at else None
                    ),
                    "config": json.loads(row.config_json),
                }
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list preprocessing versions: %s", exc)
    return versions


def _load_versions() -> Dict[str, Any]:
    """Загрузить файл версий."""
    if VERSIONS_FILE.exists():
        with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
