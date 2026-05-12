"""
Сервис версионирования масок сегментации.
Хранит историю всех версий масок (от ИИ и от экспертов).
В Каппе всегда лежит актуальная (последняя) маска.
"""
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from database import SessionLocal
from registry_models import MaskVersion, init_registry_tables

logger = logging.getLogger(__name__)


def register_ai_mask(
    entity_id: str,
    dataset_id: int,
    file_path: str,
    kappa_file_id: str = None,
) -> Dict[str, Any]:
    """
    Зарегистрировать оригинальную маску от ИИ (версия 1).
    Вызывается автоматически при загрузке сущности в Каппу.
    """
    db = SessionLocal()
    try:
        # Проверяем, есть ли уже версия 1
        existing = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
            MaskVersion.version == 1,
        ).first()

        if existing:
            logger.debug("AI mask already registered for entity %s", entity_id)
            return _to_dict(existing)

        record = MaskVersion(
            entity_id=entity_id,
            dataset_id=dataset_id,
            version=1,
            source="ai",
            uploaded_by_user_id=None,
            uploaded_by_name="AI Pipeline",
            file_path=file_path,
            file_name=Path(file_path).name,
            kappa_file_id=kappa_file_id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info("AI mask registered: entity=%s, path=%s", entity_id, file_path)
        return _to_dict(record)
    finally:
        db.close()


def get_next_version(entity_id: str) -> int:
    """Получить следующий номер версии маски (max + 1)."""
    db = SessionLocal()
    try:
        last = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
        ).order_by(MaskVersion.version.desc()).first()

        return (last.version + 1) if last else 1
    finally:
        db.close()


def get_ai_mask_dir(entity_id: str) -> Optional[str]:
    """
    Получить директорию ИИ-маски (version 1) для данной сущности.
    Все версии масок пациента должны храниться в этой директории.
    """
    db = SessionLocal()
    try:
        record = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
            MaskVersion.version == 1,
        ).first()

        if record and record.file_path:
            return str(Path(record.file_path).parent)
        return None
    finally:
        db.close()


def register_expert_mask(
    entity_id: str,
    dataset_id: int,
    file_path: str,
    user_id: int,
    user_name: str = "",
    kappa_file_id: str = None,
) -> Dict[str, Any]:
    """
    Зарегистрировать маску, загруженную экспертом.
    Автоматически назначает следующий номер версии.
    """
    db = SessionLocal()
    try:
        # Определяем следующий номер версии
        last = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
        ).order_by(MaskVersion.version.desc()).first()

        next_version = (last.version + 1) if last else 1

        record = MaskVersion(
            entity_id=entity_id,
            dataset_id=dataset_id,
            version=next_version,
            source="expert",
            uploaded_by_user_id=user_id,
            uploaded_by_name=user_name,
            file_path=file_path,
            file_name=Path(file_path).name,
            kappa_file_id=kappa_file_id,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "Expert mask registered: entity=%s, version=%d, user=%s",
            entity_id, next_version, user_name,
        )
        return _to_dict(record)
    finally:
        db.close()


def get_current_mask(entity_id: str) -> Optional[Dict[str, Any]]:
    """Получить актуальную (последнюю) версию маски."""
    db = SessionLocal()
    try:
        record = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
        ).order_by(MaskVersion.version.desc()).first()

        return _to_dict(record) if record else None
    finally:
        db.close()


def get_mask_history(entity_id: str) -> List[Dict[str, Any]]:
    """Получить все версии масок для сущности (в хронологическом порядке)."""
    db = SessionLocal()
    try:
        records = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
        ).order_by(MaskVersion.version.asc()).all()

        return [_to_dict(r) for r in records]
    finally:
        db.close()


def get_mask_by_version(entity_id: str, version: int) -> Optional[Dict[str, Any]]:
    """Получить конкретную версию маски."""
    db = SessionLocal()
    try:
        record = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
            MaskVersion.version == version,
        ).first()

        return _to_dict(record) if record else None
    finally:
        db.close()


def delete_mask_version(entity_id: str, version: int) -> bool:
    """Удалить конкретную версию маски из БД."""
    db = SessionLocal()
    try:
        record = db.query(MaskVersion).filter(
            MaskVersion.entity_id == entity_id,
            MaskVersion.version == version,
        ).first()

        if not record:
            return False

        db.delete(record)
        db.commit()
        logger.info("Deleted mask version: entity=%s, version=%d", entity_id, version)
        return True
    finally:
        db.close()


def _to_dict(record: MaskVersion) -> Dict[str, Any]:
    """Преобразовать SQLAlchemy объект в dict."""
    return {
        "id": record.id,
        "entity_id": record.entity_id,
        "dataset_id": record.dataset_id,
        "version": record.version,
        "source": record.source,
        "uploaded_by_user_id": record.uploaded_by_user_id,
        "uploaded_by_name": record.uploaded_by_name,
        "file_path": record.file_path,
        "file_name": record.file_name,
        "kappa_file_id": record.kappa_file_id,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }