"""
Версионирование конфигов препроцессинга.
Вычисляет стабильный hash от параметров, влияющих на результат обработки.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Файл для хранения истории версий
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


def register_version(config_path: str) -> str:
    """
    Зарегистрировать текущий конфиг в истории версий.
    Если версия уже существует — ничего не делает.
    Возвращает preprocessing_id.
    """
    preprocessing_id = compute_preprocessing_id(config_path)

    # Загружаем историю
    versions = _load_versions()

    if preprocessing_id not in versions:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        functional = _extract_functional_config(config)

        versions[preprocessing_id] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(config_path),
            "config": functional,
        }

        _save_versions(versions)
        logger.info(
            "Registered new preprocessing version: %s", preprocessing_id
        )
    else:
        logger.debug(
            "Preprocessing version already registered: %s", preprocessing_id
        )

    return preprocessing_id


def get_version_config(preprocessing_id: str) -> Optional[Dict[str, Any]]:
    """Получить конфиг по preprocessing_id."""
    versions = _load_versions()
    entry = versions.get(preprocessing_id)
    if entry:
        return entry.get("config")
    return None


def list_versions() -> Dict[str, Any]:
    """Получить все зарегистрированные версии."""
    return _load_versions()


def _load_versions() -> Dict[str, Any]:
    """Загрузить файл версий."""
    if VERSIONS_FILE.exists():
        with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_versions(versions: Dict[str, Any]) -> None:
    """Сохранить файл версий."""
    VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)