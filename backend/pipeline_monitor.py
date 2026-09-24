"""
Фоновый мониторинг выполнения pipeline
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

from database import SessionLocal, get_pipeline_run, update_pipeline_run, update_stage_execution
from kappa_delivery import NO_SESSION
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

        try:
            while True:
                run = get_pipeline_run(db, run_id)

                if not run:
                    logger.error(f"Run {run_id} не найден в БД")
                    break

                if run.status in ["completed", "failed"]:
                    logger.info(f"Pipeline {run_id} завершён со статусом: {run.status}")
                    await self._send_update(run_id, output_path, db)

                    # Kappa uploader is built here, at completion, not at the
                    # top of the loop: a long run would otherwise finish
                    # holding a token snapshotted hours earlier.
                    if kappa_session_id and lesion_type and run.status == "completed":
                        kappa_uploader = self._create_kappa_uploader(
                            run_id, output_path, kappa_session_id, lesion_type
                        )
                        if kappa_uploader:
                            logger.info("Starting Kappa upload for completed run %s", run_id)
                            asyncio.create_task(
                                self._kappa_upload_safe(kappa_uploader, run_id)
                            )
                        else:
                            # Session was valid at start but is gone now (or the
                            # preprocessing config is missing) — the background
                            # worker will retry once a session exists again.
                            self._record_delivery(run_id, NO_SESSION, None)
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

            # The dataset was fixed at run start (backend/numbering.py) and
            # recorded on the run — reuse it rather than resolving again, so
            # upload cannot land in a different dataset than the one Stage 01
            # numbered subjects for. None for runs started before this
            # existed; KappaUploader falls back to resolving in that case.
            from database import SessionLocal as _DBSessionLocal
            from database import get_pipeline_run as _get_pipeline_run
            db = _DBSessionLocal()
            try:
                run = _get_pipeline_run(db, run_id)
                dataset_id = run.kappa_dataset_id if run else None
            finally:
                db.close()

            uploader = KappaUploader(
                run_id=run_id,
                output_path=output_path,
                token=session["kappa_token"],
                user_id=session["user_id"],
                user_type_id=session["user_type_id"],
                lesion_type=lesion_type,
                preprocessing_config_path=config_path,
                dataset_id=dataset_id,
            )
            logger.info("KappaUploader created for run %s (dataset_id=%s)",
                        run_id, dataset_id)
            return uploader

        except Exception as e:
            logger.error("Failed to create KappaUploader: %s", e)
            return None
    
    def _record_delivery(self, run_id, result, exc):
        """Classify one upload attempt and persist the verdict.

        Kept separate from the upload itself so the decision is testable and
        so all three callers (post-run, background worker, manual retry) reach
        the same policy. Returns None (without writing anything) for a run
        whose kappa_upload_status is NULL — that means it never intended to
        upload (CLI, no Kappa session at start).
        """
        from datetime import datetime, timezone

        from database import get_kappa_delivery, set_kappa_delivery
        from kappa_delivery import classify

        db = SessionLocal()
        try:
            run = get_pipeline_run(db, run_id)
            if run is None or run.kappa_upload_status is None:
                return None
            state = get_kappa_delivery(run)
            verdict = classify(result, exc, state, datetime.now(timezone.utc))
            set_kappa_delivery(
                db, run_id, verdict["status"],
                verdict["next_attempt"], verdict["detail"],
            )
            logger.info(
                "Kappa delivery for %s: %s (%d/%d, reason=%s)",
                run_id, verdict["status"],
                verdict["detail"].get("delivered", 0),
                verdict["detail"].get("total", 0),
                verdict["detail"].get("reason"),
            )
            return verdict
        finally:
            db.close()

    def _seed_delivery_progress(self, run_id):
        """Write disk/registry counters before upload_results() returns.

        The history column otherwise stays at 0/0 for the whole attempt,
        and a first-try Kappa outage has no '3 of 4 already there' to show.
        """
        from datetime import datetime, timezone

        from database import get_kappa_delivery, set_kappa_delivery
        from kappa_delivery import (
            count_local_progress,
            mark_in_progress,
            merge_local_counters,
        )

        db = SessionLocal()
        try:
            run = get_pipeline_run(db, run_id)
            if run is None or run.kappa_upload_status != "pending":
                return None
            local = count_local_progress(
                run_id, run.output_path, run.kappa_dataset_id
            )
            state = merge_local_counters(
                get_kappa_delivery(run), local["total"], local["delivered"],
            )
            verdict = mark_in_progress(
                state, datetime.now(timezone.utc),
                state["total"], state["delivered"],
            )
            set_kappa_delivery(
                db, run_id, verdict["status"],
                verdict["next_attempt"], verdict["detail"],
            )
            return verdict
        finally:
            db.close()

    async def _kappa_upload_safe(self, uploader, run_id: str = None):
        """Обёртка для безопасного вызова upload_results"""
        if run_id:
            self._seed_delivery_progress(run_id)

        results = None
        failure = None
        try:
            results = await uploader.upload_results()
            logger.info("Kappa upload results: %s", results)
        except Exception as e:
            failure = e
            logger.error("Kappa upload error: %s", e)

        verdict = self._record_delivery(run_id, results, failure) if run_id else None

        # Уведомляем фронт о завершении загрузки в Каппу. results.get("error")
        # is upload_results() reporting a failure BY RETURN VALUE (e.g. Kappa
        # unreachable) — that is not a set of uploaded sessions to announce.
        if run_id and results and not results.get("error"):
            entities = []
            for s in results.get("sessions", []):
                if s.get("entity_id"):
                    entities.append({
                        "entity_id": s["entity_id"],
                        "dataset_id": results.get("dataset_id"),
                        "session": s.get("session"),
                    })

            # Также находим entity из реестра для дубликатов
            if not entities:
                from patient_registry import find_by_run_id
                records = find_by_run_id(run_id)
                for r in records:
                    if r.get("kappa_entity_id") and r.get("kappa_dataset_id"):
                        entities.append({
                            "entity_id": r["kappa_entity_id"],
                            "dataset_id": r["kappa_dataset_id"],
                            "session": r.get("bids_id"),
                        })

            message = {
                "type": "kappa_upload_complete",
                "run_id": run_id,
                "entities": entities,
            }

            # Уведомления о дубликатах
            duplicates = [
                s for s in results.get("sessions", [])
                if s.get("error") == "duplicate"
            ]
            if duplicates:
                message["warnings"] = [s["message"] for s in duplicates]

            await ws_manager.broadcast(run_id, message)

        if run_id and verdict and verdict["status"] != "done":
            await ws_manager.broadcast(run_id, {
                "type": "kappa_upload_deferred",
                "run_id": run_id,
                "status": verdict["status"],
                "detail": verdict["detail"],
            })
    
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