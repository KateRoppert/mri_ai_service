"""
Pydantic модели для API
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class PipelineStatus(str, Enum):
    """Статусы выполнения pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityCategory(str, Enum):
    """Категории качества изображения"""
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    FAIR = "FAIR"
    POOR = "POOR"
    BAD = "BAD"

class NIfTIFile(BaseModel):
    """Информация о NIfTI файле"""
    filename: str = Field(..., description="Имя preprocessed файла")
    mask_filename: str = Field(..., description="Имя файла маски сегментации")
    patient_id: str = Field(..., description="ID пациента")
    session_id: str = Field(..., description="ID сессии")
    modality: str = Field(..., description="Модальность (T1, T2, FLAIR)")
    image_url: str = Field(..., description="URL для получения preprocessed файла")
    mask_url: str = Field(..., description="URL для получения маски")
    native_image_url: Optional[str] = Field(None, description="URL нативного изображения из nifti/")
    native_mask_url: Optional[str] = Field(None, description="URL нативной маски сегментации")

class NIfTIFilesResponse(BaseModel):
    """Список доступных NIfTI файлов"""
    total: int = Field(..., description="Количество файлов")
    files: List[NIfTIFile] = Field(..., description="Список файлов")

# ============================================
# МОДЕЛИ ДЛЯ ОТЧЁТА ОБ ОБЪЁМАХ ОПУХОЛИ
# ============================================

class VolumeClass(BaseModel):
    """Объём одного класса сегментации"""
    name: str = Field(..., description="Название класса")
    voxel_count: int = Field(..., description="Количество вокселей")
    volume_mm3: float = Field(..., description="Объём в мм³")
    volume_cm3: float = Field(..., description="Объём в см³")

class VolumeReportResponse(BaseModel):
    """Отчёт об объёмах опухоли для одной маски"""
    mask_file: str = Field(..., description="Имя файла маски")
    patient_id: str = Field(..., description="ID пациента")
    session_id: str = Field(..., description="ID сессии")
    report_text: str = Field(..., description="Полный текст отчёта")

class VolumeReportListResponse(BaseModel):
    """Список отчётов об объёмах"""
    total: int = Field(..., description="Количество отчётов")
    reports: List[VolumeReportResponse] = Field(..., description="Список отчётов")

class LobarClassVolume(BaseModel):
    """Объём одного класса поражения в доле"""
    name_en: str = Field(..., description="Название класса (EN)")
    name_ru: str = Field(..., description="Название класса (RU)")
    voxel_count: int = Field(..., description="Количество вокселей")
    volume_mm3: float = Field(..., description="Объём в мм³")
    volume_cm3: float = Field(..., description="Объём в см³")

class LobarResult(BaseModel):
    """Результат локализации для одной доли"""
    name_en: str = Field(..., description="Название доли (EN)")
    name_ru: str = Field(..., description="Название доли (RU)")
    color: str = Field(..., description="Цвет для визуализации")
    total_voxels: int = Field(..., description="Всего вокселей поражения в доле")
    total_volume_mm3: float = Field(..., description="Объём в мм³")
    total_volume_cm3: float = Field(..., description="Объём в см³")
    percent_of_lesion: float = Field(..., description="Процент от общего объёма поражения")
    classes: Dict[str, LobarClassVolume] = Field(default_factory=dict, description="Объёмы по классам")

class LobarReportResponse(BaseModel):
    """Отчёт о лобарной локализации для одной маски"""
    mask_file: str = Field(..., description="Имя файла маски")
    patient_id: str = Field(..., description="ID пациента")
    session_id: str = Field(..., description="ID сессии")
    atlas_name: str = Field(..., description="Название атласа")
    total_lesion_voxels: int = Field(..., description="Всего вокселей поражения")
    total_lesion_volume_mm3: float = Field(..., description="Общий объём поражения мм³")
    total_lesion_volume_cm3: float = Field(..., description="Общий объём поражения см³")
    lobes: Dict[str, LobarResult] = Field(default_factory=dict, description="Результаты по долям")

class LobarReportListResponse(BaseModel):
    """Список лобарных отчётов"""
    total: int = Field(..., description="Количество отчётов")
    reports: List[LobarReportResponse] = Field(..., description="Список отчётов")

# ============================================
# МОДЕЛИ ДЛЯ ЗАПУСКА PIPELINE
# ============================================

class PipelineStartRequest(BaseModel):
    """Запрос на запуск pipeline"""
    input_path: str = Field(
        ..., 
        description="Путь к директории с DICOM данными",
        min_length=1
    )
    output_path: Optional[str] = Field(
        None, 
        description="Путь для сохранения результатов (опционально)"
    )
    use_default_output: bool = Field(
        False, 
        description="Использовать путь по умолчанию для результатов"
    )
    kappa_session_id: Optional[str] = None
    lesion_type: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_path": "/data/dicom/patient_001",
                "output_path": "/data/results/patient_001",
                "use_default_output": False
            }
        }
    )


class PipelineStartResponse(BaseModel):
    """Ответ на запуск pipeline"""
    run_id: str = Field(..., description="Уникальный ID запуска")
    status: PipelineStatus = Field(..., description="Текущий статус")
    message: str = Field(..., description="Информационное сообщение")
    created_at: datetime = Field(..., description="Время создания задачи")


# ============================================
# МОДЕЛИ ДЛЯ СТАТУСА ВЫПОЛНЕНИЯ
# ============================================

class StageProgress(BaseModel):
    """Прогресс выполнения одного этапа"""
    stage_number: int = Field(..., ge=1, le=7, description="Номер этапа (1-7)")
    stage_name: str = Field(..., description="Название этапа на русском")
    status: str = Field(..., description="Статус: pending/running/completed/failed")
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Прогресс в процентах")
    started_at: Optional[datetime] = Field(None, description="Время начала этапа")
    completed_at: Optional[datetime] = Field(None, description="Время завершения этапа")
    error: Optional[str] = Field(None, description="Сообщение об ошибке (если есть)")


class PipelineStatusResponse(BaseModel):
    """Ответ с текущим статусом pipeline"""
    run_id: str = Field(..., description="ID запуска")
    status: PipelineStatus = Field(..., description="Общий статус pipeline")
    current_stage: Optional[int] = Field(None, description="Номер текущего этапа (1-6)")
    overall_progress: float = Field(0.0, ge=0.0, le=100.0, description="Общий прогресс")
    stages: List[StageProgress] = Field(default_factory=list, description="Детали по каждому этапу")
    created_at: datetime = Field(..., description="Время запуска")
    completed_at: Optional[datetime] = Field(None, description="Время завершения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    lesion_type: Optional[str] = Field(None, description="Тип поражения (glioblastoma / multiple_sclerosis)")


# ============================================
# МОДЕЛИ ДЛЯ ОТЧЁТА О КАЧЕСТВЕ
# ============================================

class QualityMetrics(BaseModel):
    """Метрики качества изображения"""
    snr: float = Field(..., description="Signal-to-Noise Ratio")
    cnr: float = Field(..., description="Contrast-to-Noise Ratio")
    efc: float = Field(..., description="Entropy Focus Criterion")
    fber: float = Field(..., description="Foreground-Background Energy Ratio")
    gradient_sharpness: float = Field(..., description="Gradient Sharpness")
    voxel_anisotropy: float = Field(..., description="Voxel Anisotropy")
    intensity_variance: float = Field(..., description="Intensity Variance")
    coefficient_of_variation: float = Field(..., description="Coefficient of Variation")


class QualityReportResponse(BaseModel):
    """Отчёт о качестве изображения"""
    file: str = Field(..., description="Имя файла")
    patient_id: str = Field(..., description="ID пациента")
    modality: str = Field(..., description="Модальность (t1, t2, flair и т.д.)")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Общая оценка качества")
    quality_category: QualityCategory = Field(..., description="Категория качества")
    quality_category_ru: str = Field(..., description="Категория качества на русском")
    metrics: QualityMetrics = Field(..., description="Подробные метрики")

class QualityReportListResponse(BaseModel):
    total: int
    reports: List[QualityReportResponse]
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "total": 3,
                "reports": [
                    {
                        "file": "sub-001_ses-001_t1_quality.json",
                        "patient_id": "sub-001",
                        "modality": "t1",
                        "quality_score": 85.64,
                        "quality_category": "GOOD",
                        "quality_category_ru": "Хорошее",
                        "metrics": {
                            "snr": 12.456,
                            "cnr": 8.234,
                            # ...
                        }
                    }
                ]
            }
        }
    )


# ============================================
# МОДЕЛИ ДЛЯ ИСТОРИИ ЗАПУСКОВ
# ============================================

class PipelineRunHistoryItem(BaseModel):
    """История одного запуска"""
    run_id: str = Field(..., description="ID запуска")
    input_path: str = Field(..., description="Путь к входным данным")
    output_path: str = Field(..., description="Путь к выходным данным")
    status: PipelineStatus = Field(..., description="Статус выполнения")
    current_stage: Optional[int] = Field(0, description="Текущий этап (1-6)") 
    quality_score: Optional[float] = Field(None, description="Оценка качества")
    quality_category: Optional[str] = Field(None, description="Категория качества")
    created_at: datetime = Field(..., description="Время создания")
    started_at: Optional[datetime] = Field(None, description="Время начала выполнения")
    completed_at: Optional[datetime] = Field(None, description="Время завершения")
    duration_seconds: Optional[int] = Field(None, description="Длительность в секундах")


class PipelineHistoryResponse(BaseModel):
    """Список истории запусков"""
    total: int = Field(..., description="Общее количество запусков")
    runs: List[PipelineRunHistoryItem] = Field(..., description="Список запусков")


# ============================================
# СЛУЖЕБНЫЕ МОДЕЛИ
# ============================================

class HealthCheckResponse(BaseModel):
    """Ответ health check"""
    status: str = Field("ok", description="Статус сервиса")
    timestamp: datetime = Field(..., description="Текущее время сервера")
    version: str = Field(..., description="Версия API")


# ============================================
# МОДЕЛИ ДЛЯ МС-ОТЧЁТА
# ============================================

class LesionStatsReport(BaseModel):
    """Статистика очагов МС для одной сессии"""
    patient_id: str
    session_id: str
    lesion_count: int
    total_volume_cm3: float
    mean_lesion_volume_cm3: float
    lesion_volumes_cm3: List[float]

class LesionStatsListResponse(BaseModel):
    total: int
    reports: List[LesionStatsReport]

class LongitudinalPoint(BaseModel):
    """Одна точка лонгитюдного ряда"""
    session_id: str
    scan_date: Optional[str] = None
    total_volume_cm3: float
    lesion_count: Optional[int] = None

class LongitudinalResponse(BaseModel):
    patient_id: str
    lesion_type: str
    points: List[LongitudinalPoint]