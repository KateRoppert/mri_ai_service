# dicom_to_bids/organizing/services/file_operations.py

import os
import shutil
import logging
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum


class FileOperation(Enum):
    COPY = "copy"
    MOVE = "move"


@dataclass
class FileOperationResult:
    """Результат файловой операции"""
    success: bool
    source: str
    destination: str
    error: Optional[str] = None


class FileOperationService:
    """Сервис для работы с файловыми операциями"""
    
    def __init__(self, max_parallel_operations: int = 32):
        self.max_parallel_operations = max_parallel_operations
        self.logger = logging.getLogger(__name__)
    
    def process_files(
        self, 
        file_pairs: List[Tuple[str, str]], 
        operation: FileOperation = FileOperation.COPY,
        use_parallel: bool = True
    ) -> List[FileOperationResult]:
        """
        Обработка списка файлов
        
        Args:
            file_pairs: Список кортежей (source_path, destination_path)
            operation: Тип операции (copy или move)
            use_parallel: Использовать параллельную обработку
            
        Returns:
            Список результатов операций
        """
        if not file_pairs:
            return []
        
        # Для малого количества файлов используем последовательную обработку
        if len(file_pairs) < 10 or not use_parallel:
            return self._process_sequentially(file_pairs, operation)
        
        return self._process_parallel(file_pairs, operation)
    
    def _process_sequentially(
        self, 
        file_pairs: List[Tuple[str, str]], 
        operation: FileOperation
    ) -> List[FileOperationResult]:
        """Последовательная обработка файлов"""
        results = []
        
        for source, destination in file_pairs:
            result = self._process_single_file(source, destination, operation)
            results.append(result)
        
        return results
    
    def _process_parallel(
        self, 
        file_pairs: List[Tuple[str, str]], 
        operation: FileOperation
    ) -> List[FileOperationResult]:
        """Параллельная обработка файлов"""
        results = []
        max_workers = min(self.max_parallel_operations, os.cpu_count() * 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи
            future_to_pair = {
                executor.submit(
                    self._process_single_file, 
                    source, 
                    dest, 
                    operation
                ): (source, dest)
                for source, dest in file_pairs
            }
            
            # Собираем результаты
            for future in as_completed(future_to_pair):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    source, dest = future_to_pair[future]
                    results.append(
                        FileOperationResult(
                            success=False,
                            source=source,
                            destination=dest,
                            error=str(e)
                        )
                    )
        
        # Логируем итоги
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        self.logger.info(
            f"File operation completed: {successful} successful, {failed} failed"
        )
        
        return results
    
    def _process_single_file(
        self, 
        source: str, 
        destination: str, 
        operation: FileOperation
    ) -> FileOperationResult:
        """Обработка одного файла"""
        try:
            # Создаём директорию если нужно
            dest_dir = os.path.dirname(destination)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Выполняем операцию
            if operation == FileOperation.MOVE:
                shutil.move(source, destination)
            else:
                shutil.copyfile(source, destination)
            
            return FileOperationResult(
                success=True,
                source=source,
                destination=destination
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to {operation.value} {source} -> {destination}: {e}"
            )
            return FileOperationResult(
                success=False,
                source=source,
                destination=destination,
                error=str(e)
            )
    
    def ensure_directory(self, path: str) -> bool:
        """Создание директории если она не существует"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except OSError as e:
            self.logger.error(f"Cannot create directory {path}: {e}")
            return False