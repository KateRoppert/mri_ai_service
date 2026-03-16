"""
Модуль для работы с базой данных
"""

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timezone
from typing import List, Optional
import uuid

from config import settings

# Создаём движок БД
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}  # Для SQLite
)

# Создаём фабрику сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для моделей
Base = declarative_base()


# ============================================
# МОДЕЛИ БД
# ============================================

class PipelineRun(Base):
    """Модель запуска pipeline"""
    __tablename__ = "pipeline_runs"
    
    run_id = Column(String, primary_key=True, index=True)
    input_path = Column(String, nullable=False)
    output_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    
    # Прогресс
    current_stage = Column(Integer, nullable=True)
    overall_progress = Column(Float, default=0.0)
    
    # Качество
    quality_score = Column(Float, nullable=True)
    quality_category = Column(String, nullable=True)
    
    # Временные метки
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Ошибки
    error_message = Column(Text, nullable=True)
    
    # Путь к конфигу
    config_path = Column(String, nullable=True)


class StageExecution(Base):
    """Модель выполнения отдельного этапа"""
    __tablename__ = "stage_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    stage_number = Column(Integer, nullable=False)
    stage_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    progress = Column(Float, default=0.0)
    
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    error_message = Column(Text, nullable=True)


# ============================================
# CRUD ОПЕРАЦИИ
# ============================================

def get_db() -> Session:
    """Получить сессию БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_pipeline_run(
    db: Session,
    input_path: str,
    output_path: str
) -> PipelineRun:
    """Создать новый запуск pipeline"""
    run_id = str(uuid.uuid4())
    
    run = PipelineRun(
        run_id=run_id,
        input_path=input_path,
        output_path=output_path,
        status="pending",
        created_at=datetime.now(timezone.utc)
    )
    
    db.add(run)
    db.commit()
    db.refresh(run)
    
    # Создаём записи для всех этапов
    stage_names = settings.stage_names_ru
    for stage_num in range(1, 7):
        stage = StageExecution(
            run_id=run_id,
            stage_number=stage_num,
            stage_name=stage_names[stage_num],
            status="pending"
        )
        db.add(stage)
    
    db.commit()
    
    return run


def get_pipeline_run(db: Session, run_id: str) -> Optional[PipelineRun]:
    """Получить информацию о запуске по ID"""
    return db.query(PipelineRun).filter(PipelineRun.run_id == run_id).first()


def update_pipeline_run(
    db: Session,
    run_id: str,
    **kwargs
) -> Optional[PipelineRun]:
    """Обновить информацию о запуске"""
    run = get_pipeline_run(db, run_id)
    if not run:
        return None
    
    for key, value in kwargs.items():
        if hasattr(run, key):
            setattr(run, key, value)
    
    db.commit()
    db.refresh(run)
    return run


def get_stage_executions(db: Session, run_id: str) -> List[StageExecution]:
    """Получить все этапы для запуска"""
    return db.query(StageExecution).filter(
        StageExecution.run_id == run_id
    ).order_by(StageExecution.stage_number).all()


def update_stage_execution(
    db: Session,
    run_id: str,
    stage_number: int,
    **kwargs
) -> Optional[StageExecution]:
    """Обновить информацию об этапе"""
    stage = db.query(StageExecution).filter(
        StageExecution.run_id == run_id,
        StageExecution.stage_number == stage_number
    ).first()
    
    if not stage:
        return None
    
    for key, value in kwargs.items():
        if hasattr(stage, key):
            setattr(stage, key, value)
    
    db.commit()
    db.refresh(stage)
    return stage


def get_pipeline_history(
    db: Session,
    limit: int = 50,
    offset: int = 0
) -> tuple[List[PipelineRun], int]:
    """Получить историю запусков с пагинацией"""
    query = db.query(PipelineRun).order_by(PipelineRun.created_at.desc())
    
    total = query.count()
    runs = query.limit(limit).offset(offset).all()
    
    return runs, total


# ============================================
# ИНИЦИАЛИЗАЦИЯ БД
# ============================================

def init_db():
    """Инициализировать базу данных"""
    Base.metadata.create_all(bind=engine)


def reset_db():
    """Удалить и пересоздать все таблицы (для разработки)"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)