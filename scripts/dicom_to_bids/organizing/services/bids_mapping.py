# dicom_to_bids/organizing/services/bids_mapping.py

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PatientMapping:
    """Маппинг пациента"""
    original_id: str
    bids_id: str
    session_count: int = 0


@dataclass 
class SessionMapping:
    """Маппинг сессии"""
    original_patient_id: str
    bids_patient_id: str
    original_study_uid: str
    original_study_date: str
    bids_session_id: str


@dataclass
class FailedCase:
    """Информация о неудачной обработке"""
    patient_id: str
    session_id: Optional[str]
    missing_modalities: List[str]
    found_modalities: List[str]
    reason: str


class BidsMappingService:
    """Сервис для управления BIDS маппингами"""
    
    def __init__(self):
        self.patient_mappings: Dict[str, PatientMapping] = {}
        self.session_mappings: Dict[str, SessionMapping] = {}
        self.failed_cases: List[FailedCase] = []
        self.logger = logging.getLogger(__name__)
    
    def create_patient_mapping(
        self, 
        patient_ids: List[str]
    ) -> Dict[str, str]:
        """
        Создание маппинга пациентов
        
        Args:
            patient_ids: Список оригинальных ID пациентов
            
        Returns:
            Словарь {original_id: bids_id}
        """
        sorted_ids = sorted(patient_ids)
        mapping = {}
        
        for i, patient_id in enumerate(sorted_ids):
            bids_id = f"sub-{i+1:03d}"
            self.patient_mappings[patient_id] = PatientMapping(
                original_id=patient_id,
                bids_id=bids_id
            )
            mapping[patient_id] = bids_id
        
        self.logger.info(f"Created mapping for {len(mapping)} patients")
        return mapping
    
    def create_session_mapping(
        self, 
        patient_id: str,
        study_uids: List[str],
        study_dates: Dict[str, datetime]
    ) -> Dict[str, str]:
        """
        Создание маппинга сессий для пациента
        
        Args:
            patient_id: ID пациента
            study_uids: Список UID исследований
            study_dates: Словарь дат исследований
            
        Returns:
            Словарь {study_uid: bids_session_id}
        """
        # Сортируем по дате
        sorted_uids = sorted(
            study_uids, 
            key=lambda uid: study_dates.get(uid, datetime.min)
        )
        
        bids_patient_id = self.patient_mappings[patient_id].bids_id
        mapping = {}
        
        for i, study_uid in enumerate(sorted_uids):
            bids_session_id = f"ses-{i+1:03d}"
            session_key = f"{patient_id}_{study_uid}"
            
            self.session_mappings[session_key] = SessionMapping(
                original_patient_id=patient_id,
                bids_patient_id=bids_patient_id,
                original_study_uid=study_uid,
                original_study_date=study_dates[study_uid].strftime("%Y-%m-%d"),
                bids_session_id=bids_session_id
            )
            
            mapping[study_uid] = bids_session_id
        
        # Обновляем количество сессий у пациента
        self.patient_mappings[patient_id].session_count = len(mapping)
        
        return mapping
    
    def add_failed_case(
        self, 
        patient_id: str,
        session_id: Optional[str] = None,
        missing_modalities: Optional[List[str]] = None,
        found_modalities: Optional[List[str]] = None,
        reason: str = "Unknown"
    ):
        """Добавление информации о неудачной обработке"""
        self.failed_cases.append(
            FailedCase(
                patient_id=patient_id,
                session_id=session_id,
                missing_modalities=missing_modalities or [],
                found_modalities=found_modalities or [],
                reason=reason
            )
        )
    
    def get_summary_statistics(self) -> Dict:
        """Получение статистики обработки"""
        total_patients = len(self.patient_mappings)
        total_sessions = len(self.session_mappings)
        
        # Пациенты с проблемами
        patients_with_issues = set(
            fc.patient_id for fc in self.failed_cases
        )
        
        # Сессии полностью пропущенные
        completely_failed_sessions = [
            fc for fc in self.failed_cases 
            if not fc.found_modalities
        ]
        
        return {
            'total_patients': total_patients,
            'total_sessions': total_sessions,
            'patients_with_issues': len(patients_with_issues),
            'completely_failed_sessions': len(completely_failed_sessions),
            'total_failed_cases': len(self.failed_cases)
        }
    
    def save_mappings(self, output_dir: str):
        """Сохранение маппингов в файлы"""
        import os
        
        # Создаём директорию для логов
        logs_dir = os.path.join(output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Сохраняем маппинги
        mapping_file = os.path.join(logs_dir, 'bids_mapping.json')
        mapping_data = {
            'patients': {
                pm.original_id: pm.bids_id 
                for pm in self.patient_mappings.values()
            },
            'sessions': {
                sm.original_study_uid: asdict(sm)
                for sm in self.session_mappings.values()
            }
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        self.logger.info(f"Saved mappings to {mapping_file}")
        
        # Сохраняем failed cases
        if self.failed_cases:
            failed_file = os.path.join(logs_dir, 'failed_cases.json')
            failed_data = {
                'summary': self.get_summary_statistics(),
                'cases': [asdict(fc) for fc in self.failed_cases]
            }
            
            with open(failed_file, 'w') as f:
                json.dump(failed_data, f, indent=2)
            
            self.logger.info(f"Saved failed cases to {failed_file}")
    
    def get_bids_filename(
        self,
        patient_id: str,
        session_id: str,
        modality: str,
        instance: int
    ) -> str:
        """
        Генерация BIDS-совместимого имени файла
        
        Args:
            patient_id: BIDS ID пациента
            session_id: BIDS ID сессии  
            modality: Модальность (уже в BIDS формате)
            instance: Номер инстанса
            
        Returns:
            BIDS имя файла
        """
        return f"{patient_id}_{session_id}_{modality}_instance-{instance:03d}.dcm"