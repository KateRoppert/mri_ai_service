"""
Маппинг (lesion_type + preprocessing_id) → dataset_id в Каппе.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

MAPPING_FILE = Path(__file__).parent.parent / "configs" / "kappa_datasets.yaml"


def _load_mapping() -> Dict:
    """Загрузить маппинг из файла."""
    if not MAPPING_FILE.exists():
        logger.warning("Mapping file not found: %s", MAPPING_FILE)
        return {"lesion_types": [], "datasets": {}}

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"lesion_types": [], "datasets": {}}


def _save_mapping(data: Dict) -> None:
    """Сохранить маппинг в файл."""
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def get_lesion_types(user_id: Optional[int]) -> List[Dict[str, Any]]:
    """Список типов поражений с dataset_id, привязанным к текущему пользователю."""
    data = _load_mapping()
    types = data.get("lesion_types", [])

    # Добавляем dataset_id (по 'current' маппингу пользователя) к каждому типу
    enriched = []
    for lt in types:
        item = dict(lt)
        item["dataset_id"] = get_dataset_id(user_id, lt["id"], "current")
        enriched.append(item)
    return enriched


def get_dataset_id(
    user_id: Optional[int], lesion_type: str, preprocessing_id: str
) -> Optional[int]:
    """
    dataset_id для (user_id, lesion_type, preprocessing_id).
    Сначала точное совпадение, затем ключ 'current' — всё в рамках user_id.
    Нет user_id (нет сессии) → None, чтобы вызывающий создал новый датасет.
    """
    if user_id is None:
        return None
    data = _load_mapping()
    datasets = data.get("datasets", {})

    # Точное совпадение
    exact_key = f"{user_id}:{lesion_type}:{preprocessing_id}"
    if exact_key in datasets:
        return datasets[exact_key]

    # Fallback на 'current'
    current_key = f"{user_id}:{lesion_type}:current"
    if current_key in datasets:
        return datasets[current_key]

    return None


def set_dataset_id(
    user_id: Optional[int], lesion_type: str, preprocessing_id: str, dataset_id: int
) -> None:
    """Установить dataset_id для (user_id, lesion_type, preprocessing_id) + 'current'."""
    if user_id is None:
        logger.warning(
            "set_dataset_id called without user_id (lesion=%s) — not stored",
            lesion_type,
        )
        return
    data = _load_mapping()
    if "datasets" not in data:
        data["datasets"] = {}

    key = f"{user_id}:{lesion_type}:{preprocessing_id}"
    data["datasets"][key] = dataset_id

    # Также обновляем 'current'
    current_key = f"{user_id}:{lesion_type}:current"
    data["datasets"][current_key] = dataset_id

    _save_mapping(data)
    logger.info(
        "Dataset mapping updated: %s → %d (also set as current)",
        key, dataset_id,
    )