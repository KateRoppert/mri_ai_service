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

# Путь к конфигу препроцессинга (относительно корня проекта)
PREPROCESSING_CONFIG = Path(__file__).parent.parent / "configs" / "preprocessing_config.yaml"


class PipelineMonitor:
    """Мониторит выполнение pipeline и отправляет обновления через WebSocket"""
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.monitoring_tasks = {}  # run_id -> asyncio.Task
    
    async def start_monitoring(
        self,
        run_id: str,
        output_path: str,
        kappa_session_id: Optional[str] = None,
        lesion_type: Optional[str] = None,
    ):
        """Запускает мониторинг для конкретного run_id"""
        if run_id in self.monitoring_tasks:
            logger.warning(f"Мониторинг для run_id {run_id} уже запущен")
            return
        
        logger.info(f"Запуск мониторинга для run_id: {run_id}")
        
        task = asyncio.create_task(
            self._monitor_loop(run_id, output_path, kappa_session_id, lesion_type)
        )
        self.monitoring_tasks[run_id] = task
    
    async def stop_monitoring(self, run_id: str):
        """Останавливает мониторинг для run_id"""
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
    
    async def _monitor_loop(
        self,
        run_id: str,
        output_path: str,
        kappa_session_id: Optional[str] = None,
        lesion_type: Optional[str] = None,
    ):
        """Основной цикл мониторинга"""
        db = SessionLocal()
        
        # Инициализация Kappa uploader
        kappa_uploader = None
        if kappa_session_id and lesion_type:
            kappa_uploader = self._create_kappa_uploader(
                run_id, output_path, kappa_session_id, lesion_type
            )
        
        try:
            while True:
                run = get_pipeline_run(db, run_id)
                
                if not run:
                    logger.error(f"Run {run_id} не найден в БД")
                    break
                
                if run.status in ["completed", "failed"]:
                    logger.info(f"Pipeline {run_id} завершён со статусом: {run.status}")
                    await self._send_update(run_id, output_path, db)
                    
                    # Загрузка в Kappa после завершения пайплайна
                    if kappa_uploader and run.status == "completed":
                        logger.info("Starting Kappa upload for completed run %s", run_id)
                        asyncio.create_task(
                            self._kappa_upload_safe(kappa_uploader)
                        )
                    break
                
                await self._send_update(run_id, output_path, db)
                await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            logger.info(f"Мониторинг для run_id {run_id} отменён")
        
        except Exception as e:
            logger.error(f"Ошибка в цикле мониторинга для run_id {run_id}: {e}")
        
        finally:
            db.close()
    
    def _create_kappa_uploader(
        self,
        run_id: str,
        output_path: str,
        kappa_session_id: str,
        lesion_type: str,
    ):
        """Создать KappaUploader из session_id"""
        try:
            from kappa_auth import get_session
            from kappa_uploader import KappaUploader

            session = get_session(kappa_session_id)
            if not session:
                logger.warning("Kappa session not found: %s", kappa_session_id)
                return None

            config_path = str(PREPROCESSING_CONFIG)
            if not PREPROCESSING_CONFIG.exists():
                logger.error("Preprocessing config not found: %s", config_path)
                return None

            uploader = KappaUploader(
                run_id=run_id,
                output_path=output_path,
                token=session["kappa_token"],
                user_id=session["user_id"],
                user_type_id=session["user_type_id"],
                lesion_type=lesion_type,
                preprocessing_config_path=config_path,
            )
            logger.info("KappaUploader created for run %s", run_id)
            return uploader

        except Exception as e:
            logger.error("Failed to create KappaUploader: %s", e)
            return None
    
    async def _kappa_upload_safe(self, uploader):
        """Обёртка для безопасного вызова upload_results"""
        try:
            results = await uploader.upload_results()
            logger.info("Kappa upload results: %s", results)
        except Exception as e:
            logger.error("Kappa upload error: %s", e)
    
    async def _send_update(self, run_id: str, output_path: str, db):
        """Парсит логи и отправляет обновление через WebSocket."""
        log_path = self.pipeline_manager.get_log_file(output_path)
        
        if not log_path:
            return None
        
        progress_info = self.pipeline_manager.parse_log_for_progress(log_path)
        
        if progress_info['current_stage'] > 0:
            update_pipeline_run(
                db,
                run_id,
                current_stage=progress_info['current_stage'],
                overall_progress=progress_info['overall_progress']
            )
            
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
        
        await ws_manager.broadcast(run_id, message)
        
        return progress_info


# Создаём глобальный экземпляр
pipeline_monitor = PipelineMonitor()