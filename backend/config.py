"""
Конфигурация backend приложения
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Общие настройки
    app_name: str = "Brain Lesion Segmentation API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Настройки сервера
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # База данных
    database_url: str = "sqlite:///./brain_lesion.db"
    
    # Pipeline
    pipeline_root: Path = Path(__file__).parent.parent.resolve()
    pipeline_config_template: str = "pipeline_config.yaml"
    default_output_dir: Path = Path.home() / "brain_lesion_results"
    
    # Лимиты
    max_upload_size_mb: int = 5000  # 5GB для DICOM данных
    
    # Таймауты
    pipeline_timeout_seconds: int = 7200  # 2 часа
    
    # Логирование
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Отладка
    keep_runtime_configs: bool = False  # Если True, runtime конфиги не удаляются
    
    # Названия этапов на русском
    stage_names_ru: dict[int, str] = {
        1: "Стандартизация файловой структуры",
        2: "Извлечение метаданных",
        3: "Конвертация в NIfTI",
        4: "Оценка качества изображения",
        5: "Предобработка",
        6: "Сегментация"
    }
    
    # Маппинг категорий качества
    quality_category_mapping: dict[str, str] = {
        "GOOD": "Хорошее",
        "ACCEPTABLE": "Приемлемое",
        "FAIR": "Приемлемое",
        "POOR": "Плохое",
        "BAD": "Плохое"
    }
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    def get_stage_name_ru(self, stage_number: int) -> str:
        """Получить русское название этапа"""
        return self.stage_names_ru.get(stage_number, f"Этап {stage_number}")
    
    def get_quality_category_ru(self, category: str) -> str:
        """Получить русское название категории качества"""
        return self.quality_category_mapping.get(category.upper(), "Неизвестно")


# Создаём глобальный экземпляр настроек
settings = Settings()


# Создаём директорию для результатов по умолчанию
settings.default_output_dir.mkdir(parents=True, exist_ok=True)