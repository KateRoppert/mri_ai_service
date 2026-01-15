"""
Модуль для управления запуском и мониторингом pipeline
"""

import subprocess
import yaml
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from config import settings

logger = logging.getLogger(__name__)


class PipelineManager:
    """Менеджер для запуска и управления pipeline"""
    
    def __init__(self):
        self.pipeline_root = settings.pipeline_root
        self.config_template = self.pipeline_root / settings.pipeline_config_template
        self.orchestrator_script = self.pipeline_root / "orchestrator.py"
        
    def create_runtime_config(
        self,
        run_id: str,
        input_path: str,
        output_path: str
    ) -> Path:
        """
        Создаёт runtime конфигурацию для запуска pipeline
        
        Args:
            run_id: ID запуска
            input_path: Путь к входным данным
            output_path: Путь для сохранения результатов
            
        Returns:
            Путь к созданному конфиг-файлу
        """
        # Проверяем существование базового конфига
        if not self.config_template.exists():
            raise FileNotFoundError(
                f"Базовый конфиг не найден: {self.config_template}"
            )
        
        # Загружаем базовый конфиг
        with open(self.config_template, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Перезаписываем пути
        config['general']['root_input_dir'] = input_path
        config['general']['root_output_dir'] = output_path
        
        # ВАЖНО: Преобразуем относительные пути к скриптам в абсолютные
        # Иначе они будут искаться относительно runtime_configs директории
        for stage_name, stage_config in config.get('stages', {}).items():
            if 'script' in stage_config:
                script_path = stage_config['script']
                # Если путь относительный, делаем его абсолютным
                if not Path(script_path).is_absolute():
                    absolute_script_path = self.pipeline_root / script_path
                    stage_config['script'] = str(absolute_script_path)
                    logger.debug(f"Преобразован путь скрипта {stage_name}: {script_path} -> {absolute_script_path}")
        
        # Создаём директорию для runtime конфигов
        runtime_configs_dir = self.pipeline_root / "runtime_configs"
        runtime_configs_dir.mkdir(exist_ok=True)
        
        # Сохраняем runtime конфиг
        runtime_config_path = runtime_configs_dir / f"config_{run_id}.yaml"
        with open(runtime_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"Runtime конфиг создан: {runtime_config_path}")
        
        return runtime_config_path
    
    def validate_input_path(self, input_path: str) -> bool:
        """
        Проверяет существование и валидность входной директории
        
        Args:
            input_path: Путь к директории с DICOM данными
            
        Returns:
            True если путь валиден, иначе False
        """
        path = Path(input_path)
        
        if not path.exists():
            logger.error(f"Входная директория не существует: {input_path}")
            return False
        
        if not path.is_dir():
            logger.error(f"Путь не является директорией: {input_path}")
            return False
        
        # Проверяем наличие хотя бы одного файла
        if not any(path.iterdir()):
            logger.error(f"Входная директория пуста: {input_path}")
            return False
        
        return True
    
    def prepare_output_path(self, output_path: str) -> bool:
        """
        Подготавливает выходную директорию
        
        Args:
            output_path: Путь для сохранения результатов
            
        Returns:
            True если директория готова, иначе False
        """
        path = Path(output_path)
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Выходная директория готова: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка создания выходной директории {output_path}: {e}")
            return False
    
    def start_pipeline(
        self,
        run_id: str,
        input_path: str,
        output_path: str
    ) -> Optional[subprocess.Popen]:
        """
        Запускает pipeline как subprocess
        
        Args:
            run_id: ID запуска
            input_path: Путь к входным данным
            output_path: Путь для результатов
            
        Returns:
            subprocess.Popen объект или None при ошибке
        """
        # Валидация входного пути
        if not self.validate_input_path(input_path):
            logger.error(f"Валидация входного пути не прошла: {input_path}")
            return None
        
        # Подготовка выходного пути
        if not self.prepare_output_path(output_path):
            logger.error(f"Не удалось подготовить выходную директорию: {output_path}")
            return None
        
        try:
            # Создаём runtime конфиг
            config_path = self.create_runtime_config(run_id, input_path, output_path)
            
            # Формируем команду запуска
            cmd = [
                "python",
                str(self.orchestrator_script),
                "--config",
                str(config_path)
            ]
            
            logger.info(f"Запуск pipeline с командой: {' '.join(cmd)}")
            
            # Запускаем процесс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.pipeline_root)
            )
            
            logger.info(f"Pipeline запущен, PID: {process.pid}")
            
            return process
            
        except Exception as e:
            logger.error(f"Ошибка запуска pipeline: {e}")
            return None
    
    def get_log_file(self, output_path: str) -> Optional[Path]:
        """
        Получает путь к master лог-файлу pipeline
        
        Args:
            output_path: Выходная директория pipeline
            
        Returns:
            Path к лог-файлу или None
        """
        log_path = Path(output_path) / "logs" / "pipeline_master.log"
        if log_path.exists():
            return log_path
        return None
    
    def parse_log_for_progress(self, log_path: Path) -> Dict[str, Any]:
        """
        Парсит лог-файл для определения прогресса
        
        Args:
            log_path: Путь к лог-файлу
            
        Returns:
            Словарь с информацией о прогрессе
        """
        progress_info = {
            'current_stage': 0,
            'overall_progress': 0.0,
            'stages': {}
        }
        
        if not log_path.exists():
            return progress_info
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Парсим лог построчно
            for line in lines:
                # Ищем маркеры начала/завершения этапов
                if "STARTED" in line:
                    # Пример: [stage_01] STARTED
                    if "[stage_" in line:
                        stage_num = int(line.split("[stage_")[1].split("]")[0])
                        progress_info['current_stage'] = stage_num
                        progress_info['stages'][stage_num] = {
                            'status': 'running',
                            'progress': 0.0
                        }
                
                elif "SUCCESS" in line:
                    # Пример: [stage_01] SUCCESS (1m 23s)
                    if "[stage_" in line:
                        stage_num = int(line.split("[stage_")[1].split("]")[0])
                        progress_info['stages'][stage_num] = {
                            'status': 'completed',
                            'progress': 100.0
                        }
                
                elif "FAILED" in line:
                    # Пример: [stage_01] FAILED (1m 23s)
                    if "[stage_" in line:
                        stage_num = int(line.split("[stage_")[1].split("]")[0])
                        progress_info['stages'][stage_num] = {
                            'status': 'failed',
                            'progress': 0.0
                        }
            
            # Вычисляем общий прогресс
            completed_stages = sum(
                1 for stage in progress_info['stages'].values()
                if stage['status'] == 'completed'
            )
            progress_info['overall_progress'] = (completed_stages / 6) * 100
            
        except Exception as e:
            logger.error(f"Ошибка парсинга лог-файла: {e}")
        
        return progress_info
    
    def get_quality_report(self, output_path: str) -> Optional[Dict[str, Any]]:
        """
        Получает отчёт о качестве из JSON файла
        
        Args:
            output_path: Выходная директория pipeline
            
        Returns:
            Словарь с данными отчёта или None
        """
        # Ищем JSON файлы с отчётом о качестве в stage_04
        quality_dir = Path(output_path) / "stage_04_quality"
        
        if not quality_dir.exists():
            return None
        
        # Ищем первый JSON файл с качеством
        quality_files = list(quality_dir.glob("*_quality.json"))
        
        if not quality_files:
            return None
        
        try:
            import json
            with open(quality_files[0], 'r', encoding='utf-8') as f:
                quality_data = json.load(f)
            
            # Добавляем русское название категории
            category = quality_data.get('quality_category', '').upper()
            quality_data['quality_category_ru'] = settings.get_quality_category_ru(category)
            
            return quality_data
            
        except Exception as e:
            logger.error(f"Ошибка чтения отчёта о качестве: {e}")
            return None
    
    def cleanup_runtime_config(self, run_id: str, keep_for_debug: bool = False):
        """
        Удаляет runtime конфиг после завершения
        
        Args:
            run_id: ID запуска
            keep_for_debug: Если True, конфиг не удаляется (для отладки)
        """
        if keep_for_debug:
            config_path = self.pipeline_root / "runtime_configs" / f"config_{run_id}.yaml"
            logger.info(f"Runtime конфиг сохранён для отладки: {config_path}")
            return
            
        config_path = self.pipeline_root / "runtime_configs" / f"config_{run_id}.yaml"
        if config_path.exists():
            try:
                config_path.unlink()
                logger.info(f"Runtime конфиг удалён: {config_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить runtime конфиг: {e}")