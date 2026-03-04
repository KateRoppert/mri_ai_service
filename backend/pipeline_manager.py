"""
Модуль для управления запуском и мониторингом pipeline
"""

import subprocess
import yaml
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json

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
        Парсит лог-файл для определения прогресса (базовый вариант)
        
        Args:
            log_path: Путь к лог-файлу
            
        Returns:
            Словарь с информацией о прогрессе
        """
        # Инициализируем статусы всех этапов
        stages_status = {
            i: {"status": "pending", "progress": 0.0}
            for i in range(1, 7)
        }
        
        if not log_path.exists():
            return {
                'current_stage': 0,
                'overall_progress': 0.0,
                'stages': stages_status
            }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Парсим лог построчно
            for line in lines:
                # Ищем маркеры начала/завершения этапов
                
                # [stage_01_reorganize] STARTED
                if "[stage_" in line and "STARTED" in line:
                    try:
                        # Извлекаем номер этапа: [stage_01_reorganize] -> 1
                        stage_part = line.split("[stage_")[1].split("]")[0]
                        # Берём первые 2 символа после "stage_" как номер (01, 02, и т.д.)
                        stage_num_str = stage_part.split("_")[0]  # "01_reorganize" -> "01"
                        stage_num = int(stage_num_str)  # "01" -> 1
                        stages_status[stage_num] = {"status": "running", "progress": 50.0}
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Не удалось распарсить номер этапа из строки: {line.strip()}")
                
                # [stage_01_reorganize] SUCCESS
                elif "[stage_" in line and "SUCCESS" in line:
                    try:
                        stage_part = line.split("[stage_")[1].split("]")[0]
                        stage_num_str = stage_part.split("_")[0]  # "01_reorganize" -> "01"
                        stage_num = int(stage_num_str)
                        stages_status[stage_num] = {"status": "completed", "progress": 100.0}
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Не удалось распарсить номер этапа из строки: {line.strip()}")
                
                # [stage_01_reorganize] FAILED
                elif "[stage_" in line and "FAILED" in line:
                    try:
                        stage_part = line.split("[stage_")[1].split("]")[0]
                        stage_num_str = stage_part.split("_")[0]  # "01_reorganize" -> "01"
                        stage_num = int(stage_num_str)
                        stages_status[stage_num] = {"status": "failed", "progress": 0.0}
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Не удалось распарсить номер этапа из строки: {line.strip()}")
            
            # Вычисляем общий прогресс
            total_progress = sum(stage["progress"] for stage in stages_status.values())
            overall_progress = round(total_progress / 6, 2)
            
            # Определяем текущий этап
            current_stage = 0
            for stage_num in range(1, 7):
                if stages_status[stage_num]["status"] == "running":
                    current_stage = stage_num
                    break
            
            # Если нет running этапов, но есть completed - ищем последний completed
            if current_stage == 0:
                for stage_num in range(6, 0, -1):
                    if stages_status[stage_num]["status"] == "completed":
                        current_stage = stage_num
                        break
            
        except Exception as e:
            logger.error(f"Ошибка парсинга лог-файла: {e}")
        
        return {
            'current_stage': current_stage,
            'overall_progress': overall_progress,
            'stages': stages_status
        }
    
    def get_quality_report(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить все отчёты о качестве из quality_reports/
        
        Структура: quality_reports/sub-XXX/ses-XXX/anat/*_quality.json
        
        Returns:
            Список словарей с отчётами для каждого файла
        """
        quality_dir = Path(output_path) / "quality_reports"
        
        if not quality_dir.exists():
            logger.warning(f"Директория качества не найдена: {quality_dir}")
            return None
        
        # Рекурсивно ищем JSON файлы с отчётами
        quality_files = list(quality_dir.rglob("*_quality.json"))
        
        if not quality_files:
            logger.warning(f"Файлы отчётов не найдены в {quality_dir}")
            return None
        
        logger.info(f"Найдено {len(quality_files)} файлов отчётов")
        
        reports = []
        
        for quality_file in quality_files:
            try:
                with open(quality_file, 'r') as f:
                    report_data = json.load(f)
                
                # Добавляем русский перевод категории качества
                if 'quality_category' in report_data:
                    report_data['quality_category_ru'] = settings.get_quality_category_ru(
                        report_data['quality_category']
                    )
                
                reports.append(report_data)
                logger.info(f"Отчёт загружен: {quality_file.name}, quality_score={report_data.get('quality_score')}")
            
            except Exception as e:
                logger.error(f"Ошибка чтения отчёта {quality_file}: {e}")
                continue
        
        if not reports:
            logger.warning("Не удалось загрузить ни одного отчёта")
            return None
        
        logger.info(f"Успешно загружено {len(reports)} отчётов")
        return reports

    def get_volume_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить все отчёты об объёмах из segmentation/
        
        Структура: segmentation/sub-XXX/ses-XXX/anat/*_volume_report.txt
        
        Returns:
            Список словарей с отчётами
        """
        seg_dir = Path(output_path) / "segmentation"
        
        if not seg_dir.exists():
            logger.warning(f"Директория сегментации не найдена: {seg_dir}")
            return None
        
        report_files = list(seg_dir.rglob("*_volume_report.txt"))
        
        if not report_files:
            logger.warning(f"Отчёты об объёмах не найдены в {seg_dir}")
            return None
        
        logger.info(f"Найдено {len(report_files)} отчётов об объёмах")
        
        reports = []
        
        for report_file in report_files:
            try:
                report_text = report_file.read_text(encoding='utf-8')
                
                # Парсим patient_id и session_id из имени файла
                # sub-001_ses-001_T1w_volume_report.txt
                name = report_file.name.replace("_volume_report.txt", "")
                parts = name.split("_")
                patient_id = parts[0] if len(parts) > 0 else "unknown"
                session_id = parts[1] if len(parts) > 1 else "ses-001"
                
                # Имя маски — тот же базовый файл но с _segmask.nii.gz
                mask_file = name + "_segmask.nii.gz"
                
                reports.append({
                    "mask_file": mask_file,
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "report_text": report_text,
                })
                
                logger.info(f"Отчёт загружен: {report_file.name}")
            
            except Exception as e:
                logger.error(f"Ошибка чтения отчёта {report_file}: {e}")
                continue
        
        if not reports:
            logger.warning("Не удалось загрузить ни одного отчёта об объёмах")
            return None
        
        logger.info(f"Успешно загружено {len(reports)} отчётов об объёмах")
        return reports
    
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