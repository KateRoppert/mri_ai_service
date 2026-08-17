"""
Модуль для управления запуском и мониторингом pipeline
"""

import re
import subprocess
import yaml
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json

from config import settings

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# scripts/metadata_extractor.py does `from performance_monitor import ...` (bare,
# not `scripts.performance_monitor`) — it only resolves when 01_reorganize_folders.py
# runs as a direct script (Python auto-adds the running script's own directory to
# sys.path). Importing it as a module from here needs scripts/ on sys.path too.
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from utils.dicom_file_ops import find_dicom_files, copy_and_anonymize_series, MODALITY_BIDS_SUFFIX
from utils.config_loader import load_lesion_type_config
from scripts.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

# BIDS ID format enforced by IDMapper/bids_allocator (scripts/01_reorganize_folders.py) —
# relabel_series() uses patient_id/session_id (API path parameters, not sanitized by
# FastAPI beyond excluding '/') to build filesystem paths and a shutil.move target, so
# they're validated against this pattern before any path construction happens, rather
# than relying on the dict lookup that follows to implicitly reject anything else.
_BIDS_PATIENT_ID_PATTERN = re.compile(r'^sub-\d+$')
_BIDS_SESSION_ID_PATTERN = re.compile(r'^ses-\d+$')

# Per-stage "who got lost and why" reports — already written by stages
# 01/03/04/05/06 into their own output directories, but never read by
# anything (KI-054 in KNOWN_ISSUES.md). Stages 07/08 don't write this
# report yet — not covered here.
_LOSS_REPORT_FILES = [
    ("01_reorganize", "bids_organized/incomplete_data/01_reorganize_folders_incomplete_data.json"),
    ("03_convert", "nifti/incomplete_data/03_convert_to_nifti_incomplete_data.json"),
    ("04_quality", "quality_reports/incomplete_data/04_assess_quality_incomplete_data.json"),
    ("05_preprocessing", "preprocessed/incomplete_data/preprocessing_incomplete_data.json"),
    ("06_segmentation", "segmentation/incomplete_data/segmentation_incomplete_data.json"),
]


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
        output_path: str,
        lesion_type: str = "glioblastoma",
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
        
        # Пробрасываем lesion_type — для stages 06/07/08 (диспетчеризация сервиса +
        # подпапка выхода). Дефолт — glioblastoma для обратной совместимости.
        config['general']['lesion_type'] = lesion_type
        
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
    
    def estimate_pipeline_timeout(self, input_path: str) -> int:
        """
        Estimate a generous timeout for the whole multi-stage pipeline,
        scaled by how many patients are in the input directory — see
        KI-052 in KNOWN_ISSUES.md. Counts non-hidden top-level
        subdirectories, mirroring DatasetScanner.scan_dataset's own simple
        patient-counting convention in scripts/01_reorganize_folders.py
        (not imported here — that scanner also handles nested single-patient
        layouts, which the backend doesn't need just to size a timeout).
        """
        path = Path(input_path)
        if not path.is_dir():
            return settings.pipeline_timeout_base_seconds

        patient_count = sum(
            1 for entry in path.iterdir()
            if entry.is_dir() and not entry.name.startswith('.')
        )
        return settings.pipeline_timeout_base_seconds + settings.pipeline_timeout_per_patient_seconds * patient_count

    def start_pipeline(
        self,
        run_id: str,
        input_path: str,
        output_path: str,
        lesion_type: str = "glioblastoma",
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
            config_path = self.create_runtime_config(
                run_id, input_path, output_path, lesion_type=lesion_type
            )
            
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
                cwd=str(self.pipeline_root),
                start_new_session=True,  # own process group — see KI-052 / _kill_process_tree in app.py
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
        Парсит лог-файл для определения прогресса.
        
        Маппинг: оркестратор пишет stage_01/03/04/05/06 (02 отключён),
        фронтенд показывает этапы 1-5.
        """
        # Маппинг: номер в логе -> номер на фронте
        LOG_TO_DISPLAY = {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
        TOTAL_STAGES = 7
        
        stages_status = {
            i: {"status": "pending", "progress": 0.0}
            for i in range(1, TOTAL_STAGES + 1)
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
            
            for line in lines:
                if "[stage_" in line and "STARTED" in line:
                    try:
                        stage_part = line.split("[stage_")[1].split("]")[0]
                        stage_num_str = stage_part.split("_")[0]
                        log_num = int(stage_num_str)
                        display_num = LOG_TO_DISPLAY.get(log_num)
                        if display_num:
                            stages_status[display_num] = {"status": "running", "progress": 50.0}
                    except (ValueError, IndexError):
                        logger.warning(f"Не удалось распарсить номер этапа: {line.strip()}")
                
                elif "[stage_" in line and "SUCCESS" in line:
                    try:
                        stage_part = line.split("[stage_")[1].split("]")[0]
                        stage_num_str = stage_part.split("_")[0]
                        log_num = int(stage_num_str)
                        display_num = LOG_TO_DISPLAY.get(log_num)
                        if display_num:
                            stages_status[display_num] = {"status": "completed", "progress": 100.0}
                    except (ValueError, IndexError):
                        logger.warning(f"Не удалось распарсить номер этапа: {line.strip()}")
                
                elif "[stage_" in line and "FAILED" in line:
                    try:
                        stage_part = line.split("[stage_")[1].split("]")[0]
                        stage_num_str = stage_part.split("_")[0]
                        log_num = int(stage_num_str)
                        display_num = LOG_TO_DISPLAY.get(log_num)
                        if display_num:
                            stages_status[display_num] = {"status": "failed", "progress": 0.0}
                    except (ValueError, IndexError):
                        logger.warning(f"Не удалось распарсить номер этапа: {line.strip()}")
            
            total_progress = sum(stage["progress"] for stage in stages_status.values())
            overall_progress = round(total_progress / TOTAL_STAGES, 2)
            
            current_stage = 0
            for stage_num in range(1, TOTAL_STAGES + 1):
                if stages_status[stage_num]["status"] == "running":
                    current_stage = stage_num
                    break
            
            if current_stage == 0:
                for stage_num in range(TOTAL_STAGES, 0, -1):
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
        
        Ищет JSON-версии (*_volume_report.json), с fallback на txt.
        
        Returns:
            Список словарей с отчётами
        """
        seg_dir = Path(output_path) / "segmentation"
        
        if not seg_dir.exists():
            logger.warning(f"Директория сегментации не найдена: {seg_dir}")
            return None
        
        # Сначала ищем JSON
        report_files = list(seg_dir.rglob("*_volume_report.json"))
        use_json = True
        
        if not report_files:
            # Fallback на txt
            report_files = list(seg_dir.rglob("*_volume_report.txt"))
            use_json = False
        
        if not report_files:
            logger.warning(f"Отчёты об объёмах не найдены в {seg_dir}")
            return None
        
        logger.info(f"Найдено {len(report_files)} отчётов об объёмах (format={'json' if use_json else 'txt'})")
        
        reports = []
        
        for report_file in report_files:
            try:
                name_base = report_file.name.replace("_volume_report.json", "").replace("_volume_report.txt", "")
                parts = name_base.split("_")
                patient_id = parts[0] if len(parts) > 0 else "unknown"
                session_id = parts[1] if len(parts) > 1 else "ses-001"
                mask_file = name_base + "_segmask.nii.gz"
                
                if use_json:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    # Читаем txt тоже для фронтенда
                    txt_path = report_file.with_suffix(".txt")
                    report_text = txt_path.read_text(encoding='utf-8') if txt_path.exists() else ""
                    reports.append({
                        "mask_file": mask_file,
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "report_data": report_data,
                        "report_text": report_text,
                    })
                else:
                    report_text = report_file.read_text(encoding='utf-8')
                    reports.append({
                        "mask_file": mask_file,
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "report_data": None,
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

    def get_lobar_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить все отчёты о лобарной локализации из segmentation/
        
        Структура: segmentation/sub-XXX/ses-XXX/anat/*_lobar_report.json
        
        Returns:
            Список словарей с отчётами
        """
        seg_dir = Path(output_path) / "segmentation"
        
        if not seg_dir.exists():
            logger.warning(f"Директория сегментации не найдена: {seg_dir}")
            return None
        
        report_files = list(seg_dir.rglob("*_lobar_report.json"))
        
        if not report_files:
            logger.warning(f"Лобарные отчёты не найдены в {seg_dir}")
            return None
        
        logger.info(f"Найдено {len(report_files)} лобарных отчётов")
        
        reports = []
        
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                reports.append(report_data)
                
                affected = len(report_data.get("lobes", {}))
                logger.info(f"Отчёт загружен: {report_file.name}, поражённых долей: {affected}")
            
            except Exception as e:
                logger.error(f"Ошибка чтения отчёта {report_file}: {e}")
                continue
        
        if not reports:
            logger.warning("Не удалось загрузить ни одного лобарного отчёта")
            return None
        
        logger.info(f"Успешно загружено {len(reports)} лобарных отчётов")
        return reports

    def get_mcdonald_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Получить все отчёты о McDonald-классификации очагов МС из segmentation/

        Структура: segmentation/sub-XXX/ses-XXX/anat/multiple_sclerosis/*_mcdonald_report.json

        Returns:
            Список словарей с отчётами
        """
        seg_dir = Path(output_path) / "segmentation"

        if not seg_dir.exists():
            logger.warning(f"Директория сегментации не найдена: {seg_dir}")
            return None

        report_files = list(seg_dir.rglob("*_mcdonald_report.json"))

        if not report_files:
            logger.warning(f"McDonald-отчёты не найдены в {seg_dir}")
            return None

        logger.info(f"Найдено {len(report_files)} McDonald-отчётов")

        reports = []
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                reports.append(report_data)
                logger.info(f"McDonald-отчёт загружен: {report_file.name}, очагов: {report_data.get('total_lesion_count')}")
            except Exception as e:
                logger.error(f"Ошибка чтения McDonald-отчёта {report_file}: {e}")
                continue

        if not reports:
            logger.warning("Не удалось загрузить ни одного McDonald-отчёта")
            return None

        logger.info(f"Успешно загружено {len(reports)} McDonald-отчётов")
        return reports

    def _dataset_mapping_path(self, output_path: str) -> Path:
        return Path(output_path) / "bids_organized" / "dataset_mapping.json"

    def _build_metadata_extractor(self) -> Optional['MetadataExtractor']:
        """
        Same anonymization config stage 01 uses (configs/dicom_tags.yaml, see
        pipeline_config.yaml's stage_01_reorganize.args.config) — relabel_series()
        must anonymize exactly like the original reorganize pass, not skip it,
        since this is still the first time these bytes are written into the
        (supposedly anonymized) BIDS tree.
        """
        tags_config_path = _PROJECT_ROOT / "configs" / "dicom_tags.yaml"
        if not tags_config_path.exists():
            logger.error(f"Tags config not found: {tags_config_path} — cannot anonymize relabeled series")
            return None
        with open(tags_config_path, 'r', encoding='utf-8') as f:
            tags_config = yaml.safe_load(f)
        return MetadataExtractor(tags_config, logger)

    def get_incomplete_patients(
        self, output_path: str, lesion_type: str = 'glioblastoma',
        current_run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read dataset_mapping.json and return the doctor-review queue for a
        run: every incomplete session, every complete session that still has
        excluded_series (e.g. a dedup loser that lost to the winner but is
        still a plausible alternative), every discarded session (a
        permanent audit-trail record — discarding is a deliberate decision
        the doctor shouldn't have to remember making), and any complete,
        fully-resolved session that was just manually touched (relabeled) —
        but only ONCE. manually_reviewed is a one-shot confirmation flag: the
        moment it's the sole reason a resolved session gets included, it's
        cleared, so the doctor sees "стала полной" exactly once (right after
        acting) and it stops cluttering the queue on the next check. A
        session nobody has ever needed to look at (always complete, no
        alternatives, never touched) is excluded from the start.
        """
        mapping_file = self._dataset_mapping_path(output_path)
        if not mapping_file.exists():
            return []

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        try:
            required = set(load_lesion_type_config(lesion_type)['required_modalities'])
        except KeyError:
            required = {'t1', 't1c', 't2', 't2fl'}

        results: List[Dict[str, Any]] = []
        mapping_changed = False
        for patient_id, patient_data in mapping_data.get('patients', {}).items():
            for session_id, session_data in patient_data.get('sessions', {}).items():
                status = session_data.get('status')
                has_alternatives = bool(session_data.get('excluded_series'))
                manually_reviewed = session_data.get('manually_reviewed', False)
                # A merged donor is visible only from the SAME run_id it was
                # merged under (or when current_run_id isn't given at all,
                # preserving old callers' behavior) — see KI-055 in
                # KNOWN_ISSUES.md. Once a real requeue produces a NEW
                # run_id and stage 1 has re-run, the merge has already
                # served its "confirm this happened" purpose and shouldn't
                # keep cluttering the queue.
                is_visible_merge = status == 'merged' and (
                    current_run_id is None
                    or session_data.get('merged_at_run_id') == current_run_id
                )
                needs_review = (
                    status == 'incomplete'
                    or (status == 'complete' and has_alternatives)
                    or status == 'discarded'
                    or is_visible_merge
                    or manually_reviewed
                )
                if not needs_review:
                    continue

                # manually_reviewed is the ONLY reason this session is
                # included when it's fully resolved (complete, nothing left
                # to reconsider) — a one-shot confirmation, not a permanent
                # audit-trail entry like "discarded". Clear it now so the
                # NEXT fetch (doctor reopens the queue later) excludes it.
                if manually_reviewed and status == 'complete' and not has_alternatives:
                    session_data['manually_reviewed'] = False
                    mapping_changed = True

                available = sorted(session_data.get('series', {}).keys())
                results.append({
                    "patient_id": patient_id,
                    "original_id": patient_data.get('original_id', ''),
                    "session_id": session_id,
                    "date": session_data.get('original_date', ''),
                    "status": status,
                    "available": available,
                    "missing": sorted(required - set(available)),
                    "excluded_series": session_data.get('excluded_series', []),
                    "merged_into_session_id": session_data.get('merged_into_session_id'),
                })

        if mapping_changed:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        return results

    def get_pipeline_losses(self, output_path: str) -> List[Dict[str, Any]]:
        """
        Aggregate the per-stage {stage}_incomplete_data.json reports that
        stages 01/03/04/05/06 already write into their own output
        directories, into one flat "who got lost, at which stage, and why"
        list for the doctor (KI-054 in KNOWN_ISSUES.md). Stages 07/08
        don't write this report yet — a known gap, not covered here.

        Each stage's report uses a different shape (patient_id sometimes
        BIDS-style "sub-XXX", sometimes the bare original id; the "why" is
        sometimes an explicit reason string, sometimes only missing/
        available modality lists) — this reports each entry exactly as
        that stage describes it, without cross-referencing identities
        across stages.
        """
        losses: List[Dict[str, Any]] = []
        base = Path(output_path)

        for stage_label, relative_path in _LOSS_REPORT_FILES:
            report_file = base / relative_path
            if not report_file.exists():
                continue
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            for patient_entry in report.get('incomplete_data', []):
                patient_id = patient_entry.get('patient_id', '')
                for session_entry in patient_entry.get('incomplete_sessions', []):
                    losses.append({
                        "stage": stage_label,
                        "patient_id": patient_id,
                        "session_id": session_entry.get('session_id', ''),
                        "reason": self._describe_loss_reason(session_entry),
                    })

        return losses

    @staticmethod
    def _describe_loss_reason(session_entry: Dict[str, Any]) -> str:
        """Build a human-readable reason string from whichever fields a
        given stage's incomplete_data.json entry happens to carry."""
        reason = session_entry.get('reason')
        missing = session_entry.get('missing_in_output') or session_entry.get('missing')
        if reason and missing:
            return f"{reason} (нет: {', '.join(missing)})"
        if reason:
            return reason
        if missing:
            return f"не хватает модальностей: {', '.join(missing)}"
        return "причина не указана в отчёте этапа"

    def discard_session(self, output_path: str, patient_id: str, session_id: str) -> None:
        """Mark a session as intentionally skipped by the doctor. Still
        returned by get_incomplete_patients() (with status "discarded") as
        an audit-trail record that it was reviewed and rejected — no data
        is deleted."""
        # Same validation as relabel_series — patient_id/session_id are API path
        # parameters (not sanitized by FastAPI beyond excluding '/'). This method
        # only does a dict lookup (no path construction), so the risk is lower,
        # but validating consistently keeps the invariant obvious at every call site.
        if not _BIDS_PATIENT_ID_PATTERN.match(patient_id):
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")

        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        session_data = mapping_data['patients'][patient_id]['sessions'][session_id]
        session_data['status'] = 'discarded'
        session_data['manually_reviewed'] = True

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

    def merge_sessions(
        self, output_path: str, patient_id: str,
        primary_session_id: str, donor_session_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pull a donor session's series (both currently-assigned and its own
        leftover excluded_series) into the primary session's excluded_series
        pool as candidates, tagged reason="from_other_session". The doctor
        then assigns them through the existing relabel_series flow — this
        method does no file copying or anonymization itself, since every
        series already carries its original_path back to the raw DICOM
        source, which relabel_series already knows how to re-copy+anonymize.

        The donor session becomes a permanent 'merged' audit-trail entry
        (like 'discarded', but semantically distinct — merged means
        consolidated into another session, not skipped) pointing at
        primary_session_id via merged_into_session_id. No physical files
        are touched or deleted.

        Returns:
            Dict with status, primary_session_id, donor_session_id, and
            pulled_series (how many new candidates were added — 0 if
            merging again after everything was already pulled).

        Raises:
            ValueError if patient_id/primary_session_id/donor_session_id
            are malformed, or primary and donor are the same session.
        """
        if not _BIDS_PATIENT_ID_PATTERN.match(patient_id):
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(primary_session_id):
            raise ValueError(f"Invalid primary_session_id: {primary_session_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(donor_session_id):
            raise ValueError(f"Invalid donor_session_id: {donor_session_id!r}")
        if primary_session_id == donor_session_id:
            raise ValueError("primary_session_id and donor_session_id must be different")

        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        sessions = mapping_data['patients'][patient_id]['sessions']
        primary_session_data = sessions[primary_session_id]
        donor_session_data = sessions[donor_session_id]

        primary_session_data.setdefault('excluded_series', [])
        existing_paths = {u['original_path'] for u in primary_session_data['excluded_series']}

        pulled = 0
        for modality, series_info in donor_session_data.get('series', {}).items():
            if series_info['original_path'] in existing_paths:
                continue
            primary_session_data['excluded_series'].append({
                'original_path': series_info['original_path'],
                'series_description': series_info['series_description'],
                'slice_count': series_info['slice_count'],
                'detected_modality': modality,
                'reason': 'from_other_session',
            })
            existing_paths.add(series_info['original_path'])
            pulled += 1

        for entry in donor_session_data.get('excluded_series', []):
            if entry['original_path'] in existing_paths:
                continue
            primary_session_data['excluded_series'].append({
                'original_path': entry['original_path'],
                'series_description': entry['series_description'],
                'slice_count': entry['slice_count'],
                'detected_modality': entry.get('detected_modality'),
                'reason': 'from_other_session',
            })
            existing_paths.add(entry['original_path'])
            pulled += 1

        donor_session_data['status'] = 'merged'
        donor_session_data['merged_into_session_id'] = primary_session_id
        donor_session_data['merged_at_run_id'] = run_id

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        return {
            'status': 'merged',
            'primary_session_id': primary_session_id,
            'donor_session_id': donor_session_id,
            'pulled_series': pulled,
        }

    def relabel_series(
        self,
        output_path: str,
        patient_id: str,
        session_id: str,
        original_path: str,
        modality: str,
        lesion_type: str = 'glioblastoma',
    ) -> Dict[str, Any]:
        """
        Manually assign a modality to a series the automatic detector
        couldn't classify (piece B's excluded_series). Copies it into
        the correct BIDS location, updates dataset_mapping.json, and moves
        the whole session out of _incomplete/ into the main tree if this
        was the last missing modality. If the target modality slot is
        already occupied, this replaces it — the previous occupant is
        preserved in excluded_series rather than discarded.

        Returns:
            Dict with the updated session's "status" and "available" modalities.

        Raises:
            ValueError if patient_id/session_id/modality are malformed, or if the
            patient/session/series isn't found in dataset_mapping.json.
        """
        # Validate BEFORE any path construction — patient_id/session_id come from
        # the API path and are not sanitized beyond FastAPI excluding '/'; a bare
        # ".." segment would otherwise reach the shutil.move()/Path() calls below.
        if not _BIDS_PATIENT_ID_PATTERN.match(patient_id):
            raise ValueError(f"Invalid patient_id: {patient_id!r}")
        if not _BIDS_SESSION_ID_PATTERN.match(session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")
        if modality not in MODALITY_BIDS_SUFFIX:
            raise ValueError(
                f"Invalid modality: {modality!r} (must be one of {sorted(MODALITY_BIDS_SUFFIX)})"
            )

        mapping_file = self._dataset_mapping_path(output_path)
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)

        session_data = mapping_data['patients'][patient_id]['sessions'][session_id]

        excluded = session_data.get('excluded_series', [])
        matches = [u for u in excluded if u['original_path'] == original_path]
        if not matches:
            raise ValueError(
                f"No excluded series with original_path={original_path!r} "
                f"in {patient_id}/{session_id}"
            )
        series_entry = matches[0]

        bids_dir = Path(output_path) / "bids_organized"
        was_incomplete = session_data.get('status') == 'incomplete'
        current_root = (bids_dir / "_incomplete") if was_incomplete else bids_dir
        target_dir = current_root / patient_id / session_id / "anat" / modality
        target_dir.mkdir(parents=True, exist_ok=True)

        # Replacing an already-filled slot: target_dir holds the previous
        # occupant's files under the same deterministic naming scheme
        # (..._0001.dcm, _0002.dcm, ...) that copy_and_anonymize_series is
        # about to reuse. Without clearing first, a replacement with FEWER
        # files than the previous occupant only overwrites the first N —
        # the old occupant's tail files survive, so the modality directory
        # ends up holding a silent mix of two different series while
        # dataset_mapping.json reports it as a single, complete one. Clear
        # unconditionally (a no-op for the fill case, where target_dir is
        # freshly created and already empty).
        for stale_file in target_dir.iterdir():
            if stale_file.is_file():
                stale_file.unlink()

        metadata_extractor = self._build_metadata_extractor()
        if metadata_extractor is None:
            raise ValueError(
                "Anonymization config (configs/dicom_tags.yaml) not found — "
                "refusing to copy patient DICOM data without anonymizing it"
            )

        source_files = find_dicom_files(Path(original_path))
        copied = copy_and_anonymize_series(
            source_files, target_dir, patient_id, session_id, modality,
            metadata_extractor=metadata_extractor, logger=logger,
        )
        if copied != len(source_files):
            # Partial/failed copy — do NOT touch session_data, recompute status,
            # move the session directory, or write the mapping file back. Silently
            # accepting this would let a session with fewer files than expected
            # flow downstream to segmentation as "complete" (the exact failure
            # mode this incomplete-patient feature exists to prevent).
            raise ValueError(
                f"Copy failed: only {copied}/{len(source_files)} files copied for "
                f"{patient_id}/{session_id}/{modality} (source: {original_path})"
            )

        # Remove the newly-chosen series from the excluded pool.
        remaining_excluded = [u for u in excluded if u['original_path'] != original_path]

        # If modality was already occupied, its previous occupant is not
        # discarded — bumped back into excluded_series so the doctor can
        # switch back later. This is what makes relabel a genuine replace,
        # not just a fill-the-empty-slot operation.
        previous = session_data['series'].get(modality)
        if previous is not None:
            remaining_excluded.append({
                'original_path': previous['original_path'],
                'series_description': previous['series_description'],
                'slice_count': previous['slice_count'],
                'detected_modality': modality,
                'reason': 'replaced_by_manual_relabel',
            })

        session_data['excluded_series'] = remaining_excluded
        session_data['series'][modality] = {
            'original_path': original_path,
            'slice_count': len(source_files),
            'series_description': series_entry['series_description'],
        }

        # Recompute completeness — same lesion-type-aware required set as
        # CompletenessChecker.check_session() in 01_reorganize_folders.py
        # (piece B), not a hardcoded glioblastoma-only set — MS requires
        # t1/t2/t2fl, no t1c.
        try:
            required = set(load_lesion_type_config(lesion_type)['required_modalities'])
        except KeyError:
            required = {'t1', 't1c', 't2', 't2fl'}
        is_complete = required.issubset(session_data['series'].keys())
        session_data['status'] = 'complete' if is_complete else 'incomplete'
        session_data['manually_reviewed'] = True

        # If now complete and it was previously under _incomplete/, move the whole session
        if is_complete and was_incomplete:
            source_session_dir = bids_dir / "_incomplete" / patient_id / session_id
            target_session_dir = bids_dir / patient_id / session_id
            target_session_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_session_dir), str(target_session_dir))
            # Clean up now-empty _incomplete/<patient_id> if this was its only session
            parent_dir = bids_dir / "_incomplete" / patient_id
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()

        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)

        return {
            'status': session_data['status'],
            'available': sorted(session_data['series'].keys()),
        }

    def get_segmask_label_path(self, output_path: str, subject_id: str, session_id: str) -> Optional[Path]:
        """
        Locate the per-lesion labeled mask (*_segmask_labels.nii.gz) Stage 08
        writes for a specific MS session — used for cross-session lesion diffing.

        Args:
            output_path: pipeline run's output directory.
            subject_id: BIDS subject, e.g. "sub-001".
            session_id: BIDS session, e.g. "ses-001".
        """
        seg_dir = Path(output_path) / "segmentation" / subject_id / session_id / "anat" / "multiple_sclerosis"
        if not seg_dir.exists():
            return None
        matches = list(seg_dir.glob("*_segmask_labels.nii.gz"))
        return matches[0] if matches else None

    def get_preprocessed_reference_path(
        self, output_path: str, subject_id: str, session_id: str, modality: str = "t1"
    ) -> Optional[Path]:
        """Locate a session's preprocessed (atlas-space, skull-stripped) reference
        image — used to assess inter-session co-registration for the diff guardrail.

        Path: preprocessed/sub-XXX/ses-XXX/anat/sub-XXX_ses-XXX_<modality>.nii.gz
        """
        prep_path = (
            Path(output_path) / "preprocessed" / subject_id / session_id / "anat"
            / f"{subject_id}_{session_id}_{modality}.nii.gz"
        )
        return prep_path if prep_path.exists() else None

    def get_lesion_stats_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Read lesion_stats_report.json files produced by Stage 08 for MS cases.
        Pattern: segmentation/**/*_lesion_stats_report.json
        """
        seg_dir = Path(output_path) / "segmentation"
        if not seg_dir.exists():
            return None

        report_files = list(seg_dir.rglob("*_lesion_stats_report.json"))
        if not report_files:
            return None

        reports = []
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                reports.append(data)
                logger.info(f"Lesion stats loaded: {report_file.name}")
            except Exception as e:
                logger.error(f"Failed to read {report_file}: {e}")

        return reports or None

    def get_patient_map(self, output_path: str) -> Dict[str, str]:
        """
        Read Stage 01's mapping of BIDS subject → original patient id.

        Source: bids_organized/dataset_mapping.json, always written by Stage 01
        regardless of Kappa (so this works in the pure clinical flow). Used by the
        clinical UI to show the real patient behind "sub-001"; deliberately NOT
        sent to Kappa, where the expert flow stays anonymous.

        Returns:
            {"sub-001": "P000915", ...}; empty dict if the mapping is absent.
        """
        mapping_file = Path(output_path) / "bids_organized" / "dataset_mapping.json"
        if not mapping_file.exists():
            return {}
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read patient map {mapping_file}: {e}")
            return {}

        patients = data.get("patients", {})
        # Stage 01 stores {"sub-001": {"original_id": "P000915", ...}}.
        return {
            bids_id: info.get("original_id", "")
            for bids_id, info in patients.items()
            if isinstance(info, dict)
        }

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