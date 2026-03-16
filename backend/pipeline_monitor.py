"""
Фоновый мониторинг выполнения pipeline
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

from database import SessionLocal, get_pipeline_run, update_pipeline_run, update_stage_execution
from pipeline_manager import PipelineManager
from websocket_manager import ws_manager
from config import settings

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Мониторит выполнение pipeline и отправляет обновления через WebSocket"""
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.monitoring_tasks = {}  # run_id -> asyncio.Task
    
    async def start_monitoring(self, run_id: str, output_path: str):
        """
        Запускает мониторинг для конкретного run_id
        
        Args:
            run_id: ID запуска
            output_path: Путь к выходной директории
        """
        if run_id in self.monitoring_tasks:
            logger.warning(f"Мониторинг для run_id {run_id} уже запущен")
            return
        
        logger.info(f"Запуск мониторинга для run_id: {run_id}")
        
        # Создаём задачу мониторинга
        task = asyncio.create_task(self._monitor_loop(run_id, output_path))
        self.monitoring_tasks[run_id] = task
    
    async def stop_monitoring(self, run_id: str):
        """
        Останавливает мониторинг для run_id
        
        Args:
            run_id: ID запуска
        """
        if run_id not in self.monitoring_tasks:
            return
        
        logger.info(f"Остановка мониторинга для run_id: {run_id}")
        
        task = self.monitoring_tasks[run_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.monitoring_tasks[run_id]
    
    async def _monitor_loop(self, run_id: str, output_path: str):
        """
        Основной цикл мониторинга
        
        Args:
            run_id: ID запуска
            output_path: Путь к выходной директории
        """
        db = SessionLocal()
        
        try:
            # Даём время на создание записей в БД и первый REST-запрос от фронта
            await asyncio.sleep(2)
            
            while True:
                # Получаем текущий статус из БД
                run = get_pipeline_run(db, run_id)
                
                if not run:
                    logger.error(f"Run {run_id} не найден в БД")
                    break
                
                # Если pipeline завершён или упал - останавливаем мониторинг
                if run.status in ["completed", "failed"]:
                    logger.info(f"Pipeline {run_id} завершён со статусом: {run.status}")
                    
                    # Отправляем финальное обновление
                    await self._send_update(run_id, output_path, db)
                    break
                
                # Парсим логи и отправляем обновление
                await self._send_update(run_id, output_path, db)
                
                # Ждём перед следующей проверкой
                await asyncio.sleep(1)  # Проверяем каждую секунду
        
        except asyncio.CancelledError:
            logger.info(f"Мониторинг для run_id {run_id} отменён")
        
        except Exception as e:
            logger.error(f"Ошибка в цикле мониторинга для run_id {run_id}: {e}")
        
        finally:
            db.close()
    
    async def _send_update(self, run_id: str, output_path: str, db):
        """
        Парсит логи и отправляет обновление через WebSocket
        
        Args:
            run_id: ID запуска
            output_path: Путь к выходной директории
            db: Сессия БД
        """
        # Получаем лог-файл
        log_path = self.pipeline_manager.get_log_file(output_path)
        
        if not log_path:
            # Логи ещё не созданы
            return
        
        # Парсим прогресс
        progress_info = self.pipeline_manager.parse_log_for_progress(log_path)
        
        # Обновляем БД
        if progress_info['current_stage'] > 0:
            update_pipeline_run(
                db,
                run_id,
                current_stage=progress_info['current_stage'],
                overall_progress=progress_info['overall_progress']
            )
            
            # Обновляем статусы этапов
            for stage_num, stage_data in progress_info['stages'].items():
                update_stage_execution(
                    db,
                    run_id,
                    stage_number=stage_num,
                    status=stage_data['status'],
                    progress=stage_data['progress'],
                    started_at=datetime.utcnow() if stage_data['status'] == 'running' else None,
                    completed_at=datetime.utcnow() if stage_data['status'] == 'completed' else None
                )
        
        # Формируем сообщение для WebSocket
        run = get_pipeline_run(db, run_id)
        
        message = {
            "type": "progress_update",
            "run_id": run_id,
            "status": run.status,
            "current_stage": progress_info['current_stage'],
            "overall_progress": progress_info['overall_progress'],
            "stages": {
                stage_num: {
                    "stage_number": stage_num,
                    "stage_name": settings.get_stage_name_ru(stage_num),
                    "status": stage_data['status'],
                    "progress": stage_data['progress']
                }
                for stage_num, stage_data in progress_info['stages'].items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Отправляем через WebSocket
        await ws_manager.broadcast(run_id, message)


# Создаём глобальный экземпляр
pipeline_monitor = PipelineMonitor()