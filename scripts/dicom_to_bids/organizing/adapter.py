# dicom_to_bids/organizing/adapter.py
"""
Адаптер для обеспечения совместимости между старым и новым BidsOrganizer
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Импортируем новые компоненты
from .organizer import BidsOrganizer as NewBidsOrganizer
from .services.file_operations import FileOperation


class BidsOrganizerAdapter:
    """
    Адаптер, который предоставляет старый интерфейс BidsOrganizer,
    но использует новую рефакторенную реализацию внутри.
    """
    
    def __init__(
        self, 
        output_dir: str, 
        action_type: str = 'copy',
        max_parallel_files: int = 32,
        max_workers: Optional[int] = None
    ):
        """
        Инициализация с параметрами старого интерфейса
        
        Args:
            output_dir: Директория для BIDS структуры
            action_type: 'copy' или 'move'
            max_parallel_files: Максимум параллельных файловых операций
            max_workers: Количество воркеров для обработки пациентов
        """
        self.output_dir = output_dir
        self.action_type = action_type
        self.max_parallel_files = max_parallel_files
        
        # Detector будет установлен позже через set_detector
        self._detector = None
        self._new_organizer = None
        
        # Атрибуты для совместимости со старым кодом
        self.patient_mapping = {}
        self.session_mapping = {}
        self.failed_cases = {
            'patients_with_missing_modalities': {},
            'sessions_with_missing_modalities': [],
            'patients_completely_missing': [],
            'sessions_completely_missing': []
        }
        self.selection_log = []
        self.input_stats = {'total_patients': 0, 'total_sessions': 0}
        
        # BIDS modality mapping для совместимости
        self.bids_modality_map = {
            't1': 'T1w',
            't1c': 'T1w-Gd',
            't2': 'T2w',
            't2fl': 'FLAIR'
        }
        
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
    
    def set_detector(self, detector):
        """
        Установка детектора модальностей
        
        Args:
            detector: Экземпляр EnhancedModalityDetector
        """
        self._detector = detector
        
        # Создаём новый органайзер с детектором
        self._new_organizer = NewBidsOrganizer(
            output_dir=self.output_dir,
            detector=detector,
            action_type=self.action_type,
            max_workers=self.max_workers
        )
        
        # Устанавливаем max_parallel_files в file_service
        self._new_organizer.file_service.max_parallel_operations = self.max_parallel_files
    
    def organize_to_bids(self, collected_data: Dict):
        """
        Главный метод организации - использует новую реализацию
        
        Args:
            collected_data: Словарь с данными пациентов из DicomScanner
        """
        if self._detector is None:
            raise RuntimeError(
                "Detector not set. Call set_detector() before organize_to_bids()"
            )
        
        # Сохраняем статистику входных данных для совместимости
        self.input_stats = {
            'total_patients': len(collected_data),
            'total_sessions': sum(
                len(patient.studies) 
                for patient in collected_data.values()
            )
        }
        
        # Вызываем новую реализацию
        self._new_organizer.organize_to_bids(collected_data)
        
        # Копируем результаты обратно для совместимости
        self._sync_results_from_new_organizer()
    
    def _sync_results_from_new_organizer(self):
        """Синхронизация результатов из нового органайзера"""
        
        # Копируем маппинги пациентов
        self.patient_mapping = {
            pm.original_id: pm.bids_id
            for pm in self._new_organizer.mapping_service.patient_mappings.values()
        }
        
        # Копируем маппинги сессий
        self.session_mapping = {}
        for session_key, sm in self._new_organizer.mapping_service.session_mappings.items():
            # Восстанавливаем старый формат session_mapping
            self.session_mapping[session_key] = {
                'original_patient_id': sm.original_patient_id,
                'bids_patient_id': sm.bids_patient_id,
                'original_study_uid': sm.original_study_uid,
                'original_study_date': sm.original_study_date,
                'bids_session_id': sm.bids_session_id
            }
        
        # Конвертируем failed cases в старый формат
        self._convert_failed_cases()
        
        # Копируем selection_log если есть
        if hasattr(self._new_organizer, 'selection_log'):
            self.selection_log = self._new_organizer.selection_log
    
    def _convert_failed_cases(self):
        """Конвертация failed cases из нового формата в старый"""
        
        # Очищаем старые failed cases
        self.failed_cases = {
            'patients_with_missing_modalities': {},
            'sessions_with_missing_modalities': [],
            'patients_completely_missing': [],
            'sessions_completely_missing': [],
            'summary': {}
        }
        
        # Конвертируем из нового формата
        for fc in self._new_organizer.mapping_service.failed_cases:
            # Если есть found_modalities, значит частичная проблема
            if fc.found_modalities:
                # Добавляем в sessions_with_missing_modalities
                session_info = {
                    'original_patient_id': fc.patient_id,
                    'bids_patient_id': self.patient_mapping.get(fc.patient_id, ''),
                    'bids_session_id': fc.session_id,
                    'found_modalities': fc.found_modalities,
                    'missing_modalities': fc.missing_modalities
                }
                
                # Для совместимости добавляем недостающие поля
                if fc.session_id:
                    # Находим study_uid и date из session_mapping
                    for key, mapping in self.session_mapping.items():
                        if (mapping['original_patient_id'] == fc.patient_id and
                            mapping['bids_session_id'] == fc.session_id):
                            session_info['original_study_uid'] = mapping['original_study_uid']
                            session_info['original_study_date'] = mapping['original_study_date']
                            break
                
                self.failed_cases['sessions_with_missing_modalities'].append(session_info)
                
                # Обновляем patients_with_missing_modalities
                if fc.patient_id not in self.failed_cases['patients_with_missing_modalities']:
                    self.failed_cases['patients_with_missing_modalities'][fc.patient_id] = {
                        'bids_id': self.patient_mapping.get(fc.patient_id, ''),
                        'sessions_with_issues': []
                    }
                
                self.failed_cases['patients_with_missing_modalities'][fc.patient_id]['sessions_with_issues'].append({
                    'session_id': fc.session_id,
                    'missing_modalities': fc.missing_modalities
                })
            
            # Если нет found_modalities, значит полный провал
            elif not fc.found_modalities and fc.session_id:
                session_info = {
                    'original_patient_id': fc.patient_id,
                    'bids_patient_id': self.patient_mapping.get(fc.patient_id, ''),
                    'bids_session_id': fc.session_id,
                    'reason': fc.reason
                }
                
                # Добавляем недостающие поля для совместимости
                for key, mapping in self.session_mapping.items():
                    if (mapping['original_patient_id'] == fc.patient_id and
                        mapping['bids_session_id'] == fc.session_id):
                        session_info['original_study_uid'] = mapping['original_study_uid']
                        session_info['original_study_date'] = mapping['original_study_date']
                        break
                
                self.failed_cases['sessions_completely_missing'].append(session_info)
        
        # Добавляем summary для совместимости
        stats = self._new_organizer.mapping_service.get_summary_statistics()
        self.failed_cases['summary'] = {
            'input_patients': self.input_stats['total_patients'],
            'input_sessions': self.input_stats['total_sessions'],
            'output_patients': stats['total_patients'],
            'output_sessions': stats['total_sessions'],
            'total_patients_with_partial_issues': len(self.failed_cases['patients_with_missing_modalities']),
            'total_sessions_with_partial_issues': len(self.failed_cases['sessions_with_missing_modalities']),
            'total_patients_completely_missing': len(self.failed_cases['patients_completely_missing']),
            'total_sessions_completely_missing': len(self.failed_cases['sessions_completely_missing'])
        }
    
    # Методы для обратной совместимости
    def _create_patient_bids_mapping(self, collected_data: Dict) -> Dict[str, str]:
        """Метод для совместимости"""
        return self._new_organizer.mapping_service.create_patient_mapping(
            list(collected_data.keys())
        )
    
    def _create_session_bids_mapping(self, studies: Dict) -> Dict[str, str]:
        """Метод для совместимости"""
        from datetime import datetime
        
        study_dates = {}
        for study_uid, study_info in studies.items():
            if hasattr(study_info, 'study_datetime'):
                study_dates[study_uid] = study_info.study_datetime
            else:
                study_dates[study_uid] = datetime.min
        
        # Нужен patient_id, но его нет в параметрах старого метода
        # Это ограничение совместимости
        return {uid: f"ses-{i+1:03d}" for i, uid in enumerate(sorted(studies.keys()))}
    
    def _write_mapping_files(self, logs_dir: str):
        """Метод для совместимости"""
        # Новый органайзер уже сохраняет файлы
        pass
    
    def _generate_selection_summary(self):
        """Метод для совместимости"""
        # В новом органайзере это делается автоматически
        pass