# dicom_to_bids/organizing/processors/modality_processor.py

import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pydicom

from ..services.file_operations import FileOperationService, FileOperation
from ..services.bids_mapping import BidsMappingService


@dataclass
class SeriesInfo:
    """Информация о серии DICOM"""
    uid: str
    files: List[str]
    series_number: int
    protocol_name: str
    series_desc: str
    

@dataclass
class ProcessingResult:
    """Результат обработки модальности"""
    success: bool
    modality: str
    files_processed: int
    errors: List[str] = None


class ModalityProcessor:
    """Процессор для обработки отдельных модальностей"""
    
    # BIDS модальности
    BIDS_MODALITY_MAP = {
        't1': 'T1w',
        't1c': 'T1w-Gd',
        't2': 'T2w',
        't2fl': 'FLAIR'
    }
    
    def __init__(
        self, 
        file_service: FileOperationService,
        mapping_service: BidsMappingService,
        output_dir: str,
        operation: FileOperation = FileOperation.COPY
    ):
        self.file_service = file_service
        self.mapping_service = mapping_service
        self.output_dir = output_dir
        self.operation = operation
        self.logger = logging.getLogger(__name__)
    
    def process_modality_group(
        self,
        series_list: List[SeriesInfo],
        modality: str,
        patient_id: str,
        session_id: str
    ) -> ProcessingResult:
        """
        Обработка группы серий для одной модальности
        
        Args:
            series_list: Список серий для модальности
            modality: Тип модальности (t1, t2, etc.)
            patient_id: BIDS ID пациента
            session_id: BIDS ID сессии
            
        Returns:
            Результат обработки
        """
        if not series_list:
            return ProcessingResult(
                success=False,
                modality=modality,
                files_processed=0,
                errors=["No series to process"]
            )
        
        # Выбираем лучшую серию
        selected_series = self._select_best_series(series_list, modality)
        
        if not selected_series:
            return ProcessingResult(
                success=False,
                modality=modality, 
                files_processed=0,
                errors=["Could not select series"]
            )
        
        # Обрабатываем выбранную серию
        return self._process_series(
            selected_series, 
            modality, 
            patient_id, 
            session_id
        )
    
    def _select_best_series(
        self, 
        series_list: List[SeriesInfo], 
        modality: str
    ) -> Optional[SeriesInfo]:
        """
        Выбор лучшей серии из списка
        
        Упрощённая логика выбора:
        1. Если одна серия - берём её
        2. Если несколько - берём с наибольшим количеством файлов
        3. При равенстве - берём с меньшим номером серии
        """
        if not series_list:
            return None
        
        if len(series_list) == 1:
            self.logger.info(
                f"Selected only series for {modality}: {series_list[0].uid}"
            )
            return series_list[0]
        
        # Сортируем по количеству файлов (убывание) и номеру серии (возрастание)
        sorted_series = sorted(
            series_list,
            key=lambda s: (-len(s.files), s.series_number)
        )
        
        selected = sorted_series[0]
        self.logger.info(
            f"Selected series for {modality}: {selected.uid} "
            f"({len(selected.files)} files) from {len(series_list)} candidates"
        )
        
        return selected
    
    def _process_series(
        self,
        series: SeriesInfo,
        modality: str,
        patient_id: str,
        session_id: str
    ) -> ProcessingResult:
        """Обработка одной серии"""
        errors = []
        
        # Получаем BIDS имя модальности
        bids_modality = self.BIDS_MODALITY_MAP.get(modality, modality)
        
        # Создаём путь для модальности
        modality_dir = os.path.join(
            self.output_dir,
            patient_id,
            session_id,
            'anat',
            modality
        )
        
        # Сортируем файлы по InstanceNumber
        sorted_files = self._sort_files_by_instance(series.files)
        
        # Подготавливаем операции копирования
        file_operations = []
        for idx, source_file in enumerate(sorted_files, 1):
            bids_filename = self.mapping_service.get_bids_filename(
                patient_id=patient_id,
                session_id=session_id,
                modality=bids_modality,
                instance=idx
            )
            
            destination = os.path.join(modality_dir, bids_filename)
            file_operations.append((source_file, destination))
        
        # Выполняем операции
        self.logger.info(
            f"Processing {len(file_operations)} files for "
            f"{modality} -> {bids_modality}"
        )
        
        results = self.file_service.process_files(
            file_operations,
            operation=self.operation,
            use_parallel=len(file_operations) > 10
        )
        
        # Собираем ошибки
        failed = [r for r in results if not r.success]
        if failed:
            errors = [r.error for r in failed if r.error]
            self.logger.warning(
                f"Failed to process {len(failed)} files for {modality}"
            )
        
        return ProcessingResult(
            success=len(failed) == 0,
            modality=modality,
            files_processed=len(results) - len(failed),
            errors=errors
        )
    
    def _sort_files_by_instance(self, file_paths: List[str]) -> List[str]:
        """Сортировка файлов по InstanceNumber"""
        files_with_instance = []
        
        for file_path in file_paths:
            try:
                # Читаем только нужный тег
                ds = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=True,
                    specific_tags=[(0x0020, 0x0013)]  # InstanceNumber
                )
                
                instance_number = getattr(ds, 'InstanceNumber', None)
                if instance_number:
                    instance_number = int(instance_number)
                else:
                    instance_number = float('inf')
                    
            except Exception as e:
                self.logger.debug(f"Could not read instance number from {file_path}: {e}")
                instance_number = float('inf')
            
            files_with_instance.append((instance_number, file_path))
        
        # Сортируем и возвращаем только пути
        files_with_instance.sort(key=lambda x: x[0])
        return [path for _, path in files_with_instance]