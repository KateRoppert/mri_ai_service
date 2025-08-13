# scripts/dicom_to_bids/organizing/organizer.py

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from .services.file_operations import FileOperationService, FileOperation
from .services.bids_mapping import BidsMappingService
from .processors.modality_processor import ModalityProcessor, SeriesInfo


class BidsOrganizer:
    """Упрощённый BIDS органайзер с разделением ответственностей"""
    
    REQUIRED_MODALITIES = ['t1', 't1c', 't2', 't2fl']
    
    def __init__(
        self,
        output_dir: str,
        detector,  # ModalityDetector будет внедряться
        action_type: str = 'copy',
        max_workers: Optional[int] = None
    ):
        self.output_dir = output_dir
        self.detector = detector
        self.action = FileOperation(action_type)
        self.max_workers = max_workers or mp.cpu_count()
        
        # Инициализируем сервисы
        self.file_service = FileOperationService()
        self.mapping_service = BidsMappingService()
        self.modality_processor = ModalityProcessor(
            file_service=self.file_service,
            mapping_service=self.mapping_service,
            output_dir=output_dir,
            operation=self.action
        )
        
        self.logger = logging.getLogger(__name__)
    
    def organize_to_bids(self, collected_data: Dict):
        """
        Главный метод организации в BIDS
        
        Args:
            collected_data: Словарь с данными пациентов
        """
        self.logger.info("=" * 70)
        self.logger.info("Starting BIDS Organization")
        self.logger.info("=" * 70)
        
        # 1. Создаём маппинги
        patient_ids = list(collected_data.keys())
        patient_mapping = self.mapping_service.create_patient_mapping(patient_ids)
        
        self._log_patient_mapping(patient_mapping, collected_data)
        
        # 2. Обрабатываем пациентов
        if self.max_workers == 1:
            self._process_sequentially(collected_data, patient_mapping)
        else:
            self._process_parallel(collected_data, patient_mapping)
        
        # 3. Сохраняем результаты
        self._save_results()
        
        self.logger.info("=" * 70)
        self.logger.info("BIDS Organization Completed")
        self.logger.info("=" * 70)
    
    def _process_sequentially(
        self, 
        collected_data: Dict, 
        patient_mapping: Dict[str, str]
    ):
        """Последовательная обработка пациентов"""
        for patient_id, patient_data in collected_data.items():
            bids_patient_id = patient_mapping[patient_id]
            self._process_patient(patient_id, patient_data, bids_patient_id)
    
    def _process_parallel(
        self, 
        collected_data: Dict,
        patient_mapping: Dict[str, str]
    ):
        """Параллельная обработка пациентов"""
        # Подготавливаем задачи
        tasks = []
        for patient_id, patient_data in collected_data.items():
            bids_patient_id = patient_mapping[patient_id]
            tasks.append((patient_id, patient_data, bids_patient_id))
        
        self.logger.info(
            f"Processing {len(tasks)} patients using {self.max_workers} workers"
        )
        
        # Используем процессы для параллельной обработки
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Запускаем задачи
            future_to_patient = {
                executor.submit(
                    process_patient_worker,
                    task,
                    self.output_dir,
                    self.action.value
                ): task[0]
                for task in tasks
            }
            
            # Собираем результаты
            completed = 0
            for future in as_completed(future_to_patient):
                patient_id = future_to_patient[future]
                completed += 1
                
                try:
                    result = future.result(timeout=300)
                    self._handle_patient_result(result)
                    
                    self.logger.info(
                        f"[{completed}/{len(tasks)}] Completed {patient_id}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {patient_id}: {e}")
                    self.mapping_service.add_failed_case(
                        patient_id=patient_id,
                        reason=str(e)
                    )
    
    def _process_patient(
        self, 
        patient_id: str, 
        patient_data, 
        bids_patient_id: str
    ):
        """Обработка одного пациента"""
        self.logger.info(f"Processing patient: {patient_id} -> {bids_patient_id}")
        
        # Создаём маппинг сессий
        study_dates = {
            study_uid: study_info.study_datetime
            for study_uid, study_info in patient_data.studies.items()
        }
        
        session_mapping = self.mapping_service.create_session_mapping(
            patient_id=patient_id,
            study_uids=list(patient_data.studies.keys()),
            study_dates=study_dates
        )
        
        # Обрабатываем каждую сессию
        for study_uid, study_info in patient_data.studies.items():
            bids_session_id = session_mapping[study_uid]
            self._process_session(
                study_info=study_info,
                patient_id=patient_id,
                bids_patient_id=bids_patient_id,
                bids_session_id=bids_session_id
            )
    
    def _process_session(
        self,
        study_info,
        patient_id: str,
        bids_patient_id: str,
        bids_session_id: str
    ):
        """Обработка одной сессии"""
        self.logger.info(
            f"  Processing session {bids_session_id} for {bids_patient_id}"
        )
        
        # Группируем серии по модальностям
        modality_groups = self._group_series_by_modality(study_info.series)
        
        # Проверяем наличие всех модальностей
        found_modalities = list(modality_groups.keys())
        missing_modalities = [
            m for m in self.REQUIRED_MODALITIES 
            if m not in found_modalities
        ]
        
        if missing_modalities:
            self.logger.warning(
                f"    Missing modalities: {', '.join(missing_modalities)}"
            )
            self.mapping_service.add_failed_case(
                patient_id=patient_id,
                session_id=bids_session_id,
                missing_modalities=missing_modalities,
                found_modalities=found_modalities,
                reason="Missing required modalities"
            )
        
        # Обрабатываем найденные модальности
        for modality, series_list in modality_groups.items():
            result = self.modality_processor.process_modality_group(
                series_list=series_list,
                modality=modality,
                patient_id=bids_patient_id,
                session_id=bids_session_id
            )
            
            if not result.success:
                self.logger.error(
                    f"    Failed to process {modality}: {result.errors}"
                )
    
    def _group_series_by_modality(
        self, 
        series_dict: Dict
    ) -> Dict[str, List[SeriesInfo]]:
        """Группировка серий по модальностям"""
        modality_groups = {}
        
        for series_info in series_dict.values():
            # Определяем модальность
            modality = self.detector.determine_modality(
                series_info.first_dataset,
                series_info.files[0]
            )
            
            if modality == 'unknown':
                self.logger.debug(f"Skipping unknown modality: {series_info.uid}")
                continue
            
            # Создаём упрощённую структуру SeriesInfo
            simple_series = SeriesInfo(
                uid=series_info.uid,
                files=series_info.files,
                series_number=series_info.series_number,
                protocol_name=series_info.protocol_name,
                series_desc=series_info.series_desc
            )
            
            if modality not in modality_groups:
                modality_groups[modality] = []
            
            modality_groups[modality].append(simple_series)
        
        return modality_groups
    
    def _handle_patient_result(self, result: Dict):
        """Обработка результата обработки пациента"""
        # Обновляем статистику failed cases
        if 'failed_cases' in result:
            for case in result['failed_cases']:
                self.mapping_service.add_failed_case(**case)
    
    def _save_results(self):
        """Сохранение результатов обработки"""
        self.mapping_service.save_mappings(self.output_dir)
        
        # Логируем итоговую статистику
        stats = self.mapping_service.get_summary_statistics()
        self.logger.info("Processing Summary:")
        self.logger.info(f"  Total patients: {stats['total_patients']}")
        self.logger.info(f"  Total sessions: {stats['total_sessions']}")
        
        if stats['patients_with_issues'] > 0:
            self.logger.warning(
                f"  Patients with issues: {stats['patients_with_issues']}"
            )
        
        if stats['completely_failed_sessions'] > 0:
            self.logger.warning(
                f"  Completely failed sessions: {stats['completely_failed_sessions']}"
            )
    
    def _log_patient_mapping(self, patient_mapping: Dict, collected_data: Dict):
        """Логирование маппинга пациентов"""
        self.logger.info("=" * 50)
        self.logger.info("PATIENT MAPPING:")
        self.logger.info("=" * 50)
        
        for orig_id, bids_id in sorted(patient_mapping.items(), key=lambda x: x[1]):
            num_sessions = len(collected_data[orig_id].studies)
            self.logger.info(f"  {orig_id} -> {bids_id} ({num_sessions} sessions)")
        
        self.logger.info("=" * 50)


# Вспомогательная функция для параллельной обработки
def process_patient_worker(task: Tuple, output_dir: str, action: str) -> Dict:
    """Worker function для обработки пациента в отдельном процессе"""
    patient_id, patient_data, bids_patient_id = task
    
    # Создаём временные экземпляры сервисов для процесса
    file_service = FileOperationService()
    mapping_service = BidsMappingService()
    modality_processor = ModalityProcessor(
        file_service=file_service,
        mapping_service=mapping_service,
        output_dir=output_dir,
        operation=FileOperation(action)
    )
    
    # Здесь должна быть логика обработки пациента
    # Возвращаем результат
    return {
        'patient_id': patient_id,
        'bids_patient_id': bids_patient_id,
        'status': 'completed',
        'failed_cases': []
    }