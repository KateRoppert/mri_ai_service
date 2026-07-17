"""
Локальный реестр пациентов (SQLite).
Хранит связку (bids_id ↔ original_patient_id ↔ kappa_entity_id).
Никогда не загружается в Каппу — только на сервере.

При первом запуске автоматически мигрирует существующий
patient_registry.json в таблицу patient_registry.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database import SessionLocal
from registry_models import PatientRegistry, init_registry_tables

logger = logging.getLogger(__name__)

# Путь к legacy JSON (для миграции)
LEGACY_JSON_FILE = Path(__file__).parent.parent / "configs" / "patient_registry.json"


def _migrate_from_json(db: Session) -> int:
    """
    Мигрировать legacy patient_registry.json в SQLite (если файл существует
    и таблица пустая).
    Возвращает количество мигрированных записей.
    """
    # Проверяем, есть ли уже записи в таблице
    existing_count = db.query(PatientRegistry).count()
    if existing_count > 0:
        return 0

    if not LEGACY_JSON_FILE.exists():
        return 0

    logger.info("Migrating patient_registry.json → SQLite")

    try:
        with open(LEGACY_JSON_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logger.error("Failed to read legacy JSON: %s", e)
        return 0

    if not isinstance(records, list):
        logger.warning("Legacy JSON is not a list, skipping migration")
        return 0

    count = 0
    for record in records:
        try:
            entry = PatientRegistry(
                study_hash=record.get("study_hash"),
                bids_id=record.get("bids_id", ""),
                original_patient_id=record.get("original_patient_id", ""),
                patient_name=record.get("patient_name") or None,
                scan_date=record.get("scan_date") or None,
                study_instance_uid=record.get("study_instance_uid") or None,
                kappa_entity_id=record.get("kappa_entity_id"),
                kappa_dataset_id=record.get("kappa_dataset_id"),
                pipeline_run_id=record.get("pipeline_run_id"),
                lesion_type=record.get("lesion_type"),
                preprocessing_id=record.get("preprocessing_id"),
            )
            db.add(entry)
            count += 1
        except Exception as e:
            logger.warning("Failed to migrate record: %s (%s)", record.get("study_hash"), e)

    db.commit()

    # Переименовываем JSON в .backup
    backup_path = LEGACY_JSON_FILE.with_suffix(".json.backup")
    try:
        LEGACY_JSON_FILE.rename(backup_path)
        logger.info("Legacy JSON renamed to: %s", backup_path)
    except Exception as e:
        logger.warning("Could not rename legacy JSON: %s", e)

    logger.info("Migrated %d records from JSON to SQLite", count)
    return count


def ensure_tables():
    """Создать таблицы и выполнить миграцию, если нужно."""
    init_registry_tables()

    db = SessionLocal()
    try:
        _migrate_from_json(db)
    finally:
        db.close()


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
    Если запись с таким study_hash уже есть — обновляет kappa-поля.
    """
    db = SessionLocal()
    try:
        existing = db.query(PatientRegistry).filter(
            PatientRegistry.study_hash == study_hash
        ).first()

        if existing:
            # Keep bids_id in sync with the allocator. The BIDS subject can change
            # between runs (a patient re-numbered by the persistent allocator), and
            # if we don't update it here the registry keeps the first-ever id while
            # run outputs use the new one — which desyncs the longitudinal view (a
            # patient's timeline endpoint can't find its sessions under the new id).
            if bids_id and existing.bids_id != bids_id:
                existing.bids_id = bids_id
            if kappa_entity_id:
                existing.kappa_entity_id = kappa_entity_id
            if kappa_dataset_id:
                existing.kappa_dataset_id = kappa_dataset_id
            if pipeline_run_id:
                existing.pipeline_run_id = pipeline_run_id
            existing.updated_at = datetime.now(timezone.utc)
            
            db.commit()
            db.refresh(existing)
            logger.info(
                "Patient record updated: study_hash=%s, patient=%s",
                study_hash, original_patient_id,
            )
            return _to_dict(existing)

        # Новая запись
        record = PatientRegistry(
            study_hash=study_hash,
            bids_id=bids_id,
            original_patient_id=original_patient_id,
            patient_name=patient_name or None,
            scan_date=scan_date or None,
            study_instance_uid=study_instance_uid or None,
            kappa_entity_id=kappa_entity_id,
            kappa_dataset_id=kappa_dataset_id,
            pipeline_run_id=pipeline_run_id or None,
            lesion_type=lesion_type or None,
            preprocessing_id=preprocessing_id or None,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "Patient registered: study_hash=%s, patient=%s, bids=%s",
            study_hash, original_patient_id, bids_id,
        )
        return _to_dict(record)
    finally:
        db.close()


def find_by_study_hash(study_hash: str) -> Optional[Dict[str, Any]]:
    """Найти запись по study_hash."""
    db = SessionLocal()
    try:
        record = db.query(PatientRegistry).filter(
            PatientRegistry.study_hash == study_hash
        ).first()
        return _to_dict(record) if record else None
    finally:
        db.close()


def find_by_patient_id(original_patient_id: str) -> List[Dict[str, Any]]:
    """Найти все записи по оригинальному ID пациента."""
    db = SessionLocal()
    try:
        records = db.query(PatientRegistry).filter(
            PatientRegistry.original_patient_id == original_patient_id
        ).all()
        return [_to_dict(r) for r in records]
    finally:
        db.close()


def find_by_bids_id(bids_id: str) -> List[Dict[str, Any]]:
    """Найти все записи по BIDS-идентификатору пациента (sub-XXX)."""
    db = SessionLocal()
    try:
        records = db.query(PatientRegistry).filter(
            PatientRegistry.bids_id == bids_id
        ).all()
        return [_to_dict(r) for r in records]
    finally:
        db.close()


def find_by_bids_subject(subject: str) -> List[Dict[str, Any]]:
    """
    Найти все записи BIDS-субъекта (все его сессии).

    bids_id хранится как "sub-001_ses-002" (субъект + сессия). Лонгитюд
    оперирует субъектом "sub-001", поэтому матчим по префиксу "{subject}_".
    Реестр небольшой — фильтруем на стороне Python, без LIKE-экранирования.
    """
    db = SessionLocal()
    try:
        records = db.query(PatientRegistry).all()
        prefix = f"{subject}_"
        return [_to_dict(r) for r in records if (r.bids_id or "").startswith(prefix)]
    finally:
        db.close()


def find_by_kappa_entity(kappa_entity_id: str) -> Optional[Dict[str, Any]]:
    """Найти запись по ID сущности в Каппе."""
    db = SessionLocal()
    try:
        record = db.query(PatientRegistry).filter(
            PatientRegistry.kappa_entity_id == kappa_entity_id
        ).first()
        return _to_dict(record) if record else None
    finally:
        db.close()


def find_by_run_id(pipeline_run_id: str) -> List[Dict[str, Any]]:
    """Найти все записи, связанные с конкретным запуском пайплайна."""
    db = SessionLocal()
    try:
        records = db.query(PatientRegistry).filter(
            PatientRegistry.pipeline_run_id == pipeline_run_id
        ).all()
        return [_to_dict(r) for r in records]
    finally:
        db.close()


def get_all_records() -> List[Dict[str, Any]]:
    """Получить все записи."""
    db = SessionLocal()
    try:
        records = db.query(PatientRegistry).order_by(
            PatientRegistry.created_at.desc()
        ).all()
        return [_to_dict(r) for r in records]
    finally:
        db.close()


def _resync_bids_id(current_bids_id: str, correct_subject: str) -> str:
    """Return the bids_id with its subject replaced by ``correct_subject``.

    bids_id is "sub-XXX_ses-YYY"; only the subject (sub-XXX) is governed by the
    allocator — the session part is per-run and kept as-is.
    """
    if "_" in (current_bids_id or ""):
        session_part = current_bids_id.split("_", 1)[1]
        return f"{correct_subject}_{session_part}"
    return correct_subject


def reconcile_registry_bids_ids(
    allocations_by_lesion_type: Optional[Dict[str, Dict[str, str]]] = None,
) -> int:
    """Sync each registry row's BIDS subject to the allocator (source of truth).

    register_patient historically did not update bids_id, so patients re-numbered
    by the persistent allocator kept their first-ever subject id. That desynced
    the registry from run outputs and broke the longitudinal view. This brings
    existing rows back in line without re-running the pipeline. Idempotent: rows
    already matching the allocator are left untouched, and patients with no
    allocator entry (e.g. processed before the allocator existed) are skipped.

    Args:
        allocations_by_lesion_type: injectable {lesion_type: {original_id: sub-XXX}}
            for testing; when omitted it is read from utils.bids_allocator.

    Returns:
        Number of registry rows updated.
    """
    db = SessionLocal()
    try:
        rows = db.query(PatientRegistry).all()
        alloc_cache: Dict[str, Dict[str, str]] = allocations_by_lesion_type or {}
        updated = 0
        for row in rows:
            lesion_type = row.lesion_type
            if not lesion_type:
                continue
            if lesion_type not in alloc_cache:
                # Imported lazily so the reconcile logic can be tested with
                # injected allocations without importing the allocator module.
                from utils.bids_allocator import get_allocations
                alloc_cache[lesion_type] = get_allocations(lesion_type)
            correct_subject = alloc_cache[lesion_type].get(row.original_patient_id)
            if not correct_subject:
                continue  # patient predates the allocator — leave as-is
            new_bids_id = _resync_bids_id(row.bids_id, correct_subject)
            if new_bids_id != row.bids_id:
                logger.info(
                    "Reconciling registry bids_id: %s -> %s (%s)",
                    row.bids_id, new_bids_id, row.original_patient_id,
                )
                row.bids_id = new_bids_id
                updated += 1
        if updated:
            db.commit()
        return updated
    finally:
        db.close()


def _to_dict(record: PatientRegistry) -> Dict[str, Any]:
    """Преобразовать SQLAlchemy объект в dict."""
    return {
        "id": record.id,
        "study_hash": record.study_hash,
        "bids_id": record.bids_id,
        "original_patient_id": record.original_patient_id,
        "patient_name": record.patient_name,
        "scan_date": record.scan_date,
        "study_instance_uid": record.study_instance_uid,
        "kappa_entity_id": record.kappa_entity_id,
        "kappa_dataset_id": record.kappa_dataset_id,
        "pipeline_run_id": record.pipeline_run_id,
        "lesion_type": record.lesion_type,
        "preprocessing_id": record.preprocessing_id,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }