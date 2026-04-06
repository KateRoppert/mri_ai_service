"""
Локальный маппинг пациентов.
Хранит связку (bids_id ↔ original_patient_id ↔ kappa_entity_id).
Никогда не загружается в Каппу — только на сервере.

Формат: JSON-файл в configs/patient_registry.json
В будущем будет перенесён в БД (Фаза 5).
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILE = Path(__file__).parent.parent / "configs" / "patient_registry.json"


def _load_registry() -> List[Dict[str, Any]]:
    """Загрузить реестр."""
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error("Failed to load patient registry: %s", e)
            return []
    return []


def _save_registry(records: List[Dict[str, Any]]) -> None:
    """Сохранить реестр."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def register_patient(
    bids_id: str,
    study_hash: str,
    original_patient_id: str,
    patient_name: str = "",
    scan_date: str = "",
    study_instance_uid: str = "",
    kappa_entity_id: Optional[str] = None,
    kappa_dataset_id: Optional[int] = None,
    pipeline_run_id: str = "",
    lesion_type: str = "",
    preprocessing_id: str = "",
) -> Dict[str, Any]:
    """
    Зарегистрировать пациента/сессию в локальном реестре.
    Если запись с таким study_hash уже есть — обновляет kappa_entity_id.
    
    Returns:
        Созданная или обновлённая запись.
    """
    records = _load_registry()

    # Ищем существующую запись
    existing = None
    for record in records:
        if record.get("study_hash") == study_hash:
            existing = record
            break

    if existing:
        # Обновляем kappa-поля если они появились
        if kappa_entity_id:
            existing["kappa_entity_id"] = kappa_entity_id
        if kappa_dataset_id:
            existing["kappa_dataset_id"] = kappa_dataset_id
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Patient record updated: study_hash=%s, patient=%s",
            study_hash, original_patient_id,
        )
        _save_registry(records)
        return existing

    # Новая запись
    record = {
        "study_hash": study_hash,
        "bids_id": bids_id,
        "original_patient_id": original_patient_id,
        "patient_name": patient_name,
        "scan_date": scan_date,
        "study_instance_uid": study_instance_uid,
        "kappa_entity_id": kappa_entity_id,
        "kappa_dataset_id": kappa_dataset_id,
        "pipeline_run_id": pipeline_run_id,
        "lesion_type": lesion_type,
        "preprocessing_id": preprocessing_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    records.append(record)
    _save_registry(records)

    logger.info(
        "Patient registered: study_hash=%s, patient=%s, bids=%s",
        study_hash, original_patient_id, bids_id,
    )
    return record


def find_by_study_hash(study_hash: str) -> Optional[Dict[str, Any]]:
    """Найти запись по study_hash."""
    records = _load_registry()
    for record in records:
        if record.get("study_hash") == study_hash:
            return record
    return None


def find_by_patient_id(original_patient_id: str) -> List[Dict[str, Any]]:
    """Найти все записи по оригинальному ID пациента."""
    records = _load_registry()
    return [r for r in records if r.get("original_patient_id") == original_patient_id]


def find_by_kappa_entity(kappa_entity_id: str) -> Optional[Dict[str, Any]]:
    """Найти запись по ID сущности в Каппе."""
    records = _load_registry()
    for record in records:
        if record.get("kappa_entity_id") == kappa_entity_id:
            return record
    return None


def get_all_records() -> List[Dict[str, Any]]:
    """Получить все записи."""
    return _load_registry()