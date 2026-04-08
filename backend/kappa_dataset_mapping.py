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


def get_lesion_types() -> List[Dict[str, Any]]:
    """Получить список доступных типов поражений с привязанными dataset_id."""
    data = _load_mapping()
    types = data.get("lesion_types", [])
    
    # Добавляем dataset_id (по 'current' маппингу) к каждому типу
    enriched = []
    for lt in types:
        item = dict(lt)
        item["dataset_id"] = get_dataset_id(lt["id"], "current")
        enriched.append(item)
    return enriched


def get_dataset_id(lesion_type: str, preprocessing_id: str) -> Optional[int]:
    """
    Получить dataset_id для комбинации (lesion_type, preprocessing_id).
    Сначала ищет точное совпадение, затем с ключом 'current'.
    """
    data = _load_mapping()
    datasets = data.get("datasets", {})

    # Точное совпадение
    exact_key = f"{lesion_type}:{preprocessing_id}"
    if exact_key in datasets:
        return datasets[exact_key]

    # Fallback на 'current'
    current_key = f"{lesion_type}:current"
    if current_key in datasets:
        return datasets[current_key]

    return None


def set_dataset_id(
    lesion_type: str, preprocessing_id: str, dataset_id: int
) -> None:
    """Установить dataset_id для комбинации."""
    data = _load_mapping()
    if "datasets" not in data:
        data["datasets"] = {}

    key = f"{lesion_type}:{preprocessing_id}"
    data["datasets"][key] = dataset_id

    # Также обновляем 'current'
    current_key = f"{lesion_type}:current"
    data["datasets"][current_key] = dataset_id

    _save_mapping(data)
    logger.info(
        "Dataset mapping updated: %s → %d (also set as current)",
        key, dataset_id,
    )