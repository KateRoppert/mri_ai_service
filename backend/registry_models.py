"""
SQLAlchemy модели для patient_registry и validations.
Добавляются к существующим моделям в database.py.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Integer, Text, Index
from sqlalchemy.orm import Session

from database import Base, engine


class PatientRegistry(Base):
    """
    Локальный реестр пациентов.
    Связка bids_id ↔ original_patient_id ↔ kappa_entity_id.
    Никогда не загружается в Каппу — только на сервере.
    """
    __tablename__ = "patient_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Ключевые идентификаторы
    study_hash = Column(String, nullable=False, index=True, unique=True)
    bids_id = Column(String, nullable=False, index=True)

    # Оригинальные данные пациента (DICOM)
    original_patient_id = Column(String, nullable=False, index=True)
    patient_name = Column(String, nullable=True)
    scan_date = Column(String, nullable=True)
    study_instance_uid = Column(String, nullable=True)

    # Связь с Каппой
    kappa_entity_id = Column(String, nullable=True, index=True)
    kappa_dataset_id = Column(Integer, nullable=True, index=True)

    # Связь с запуском пайплайна
    pipeline_run_id = Column(String, nullable=True, index=True)
    lesion_type = Column(String, nullable=True)
    preprocessing_id = Column(String, nullable=True)

    # Временные метки
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Validation(Base):
    """
    История валидаций сущностей датасета.
    Каждая запись — одно действие пользователя (confirm/reject/revoke).
    Полная история сохраняется, для вычисления текущего состояния
    берётся последняя запись каждого пользователя по сущности.
    """
    __tablename__ = "validations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Идентификаторы сущности в Каппе
    entity_id = Column(String, nullable=False, index=True)
    dataset_id = Column(Integer, nullable=False, index=True)

    # Пользователь, совершивший действие
    user_id = Column(Integer, nullable=False, index=True)
    user_name = Column(String, nullable=True)

    # Действие: 'confirm', 'reject', 'revoke'
    action = Column(String, nullable=False)

    # Опциональный комментарий
    comment = Column(Text, nullable=True)

    # Временная метка
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


# Составные индексы для быстрого поиска
Index("ix_validations_entity_user", Validation.entity_id, Validation.user_id)
Index("ix_patient_registry_patient_date", PatientRegistry.original_patient_id, PatientRegistry.scan_date)

class MaskVersion(Base):
    """
    История версий масок сегментации.
    Каждая запись — одна версия маски для сущности.
    В Каппе хранится только актуальная (последняя), здесь — все.
    """
    __tablename__ = "mask_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Связь с Каппой
    entity_id = Column(String, nullable=False, index=True)
    dataset_id = Column(Integer, nullable=False)

    # Версия (1 = оригинал от ИИ, 2+ = правки экспертов)
    version = Column(Integer, nullable=False)

    # Кто загрузил
    source = Column(String, nullable=False)  # 'ai' или 'expert'
    uploaded_by_user_id = Column(Integer, nullable=True)
    uploaded_by_name = Column(String, nullable=True)

    # Путь к файлу маски
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)

    # Временная метка
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


Index("ix_mask_versions_entity_version", MaskVersion.entity_id, MaskVersion.version)

def init_registry_tables():
    """Создать таблицы реестра и валидаций (если их ещё нет)."""
    Base.metadata.create_all(bind=engine)