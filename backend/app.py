"""
Основное FastAPI приложение
"""

import asyncio
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import asynccontextmanager
import json
import logging
import subprocess
import uvicorn
import os
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from lesion_diff import compare_labeled_masks
from patient_registry import find_by_patient_id, find_by_bids_id, find_by_bids_subject

from config import settings
from models import (
    PipelineStartRequest,
    PipelineStartResponse,
    PipelineStatusResponse,
    PipelineStatus,
    StageProgress,
    PipelineHistoryResponse,
    PipelineRunHistoryItem,
    HealthCheckResponse,
    QualityReportResponse,
    QualityReportListResponse,
    QualityMetrics,
    NIfTIFile,
    NIfTIFilesResponse,
    VolumeReportListResponse,
    LobarReportListResponse,
    McDonaldReportListResponse,
    LesionStatsReport,
    LesionStatsListResponse,
    LongitudinalPoint,
    LongitudinalResponse,
    LongitudinalDiffResponse,
    LongitudinalDiffPair,
    LesionDiffEntry,
)
from database import (
    get_db,
    init_db,
    create_pipeline_run,
    get_pipeline_run,
    update_pipeline_run,
    get_stage_executions,
    update_stage_execution,
    get_pipeline_history
)
from websocket_manager import ws_manager
from pipeline_monitor import pipeline_monitor
from pipeline_manager import PipelineManager
from fastapi.middleware.cors import CORSMiddleware

# Настройка логирования
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Lifespan context manager для инициализации/завершения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info(f"Запуск {settings.app_name} v{settings.app_version}")
    init_db()
    from patient_registry import ensure_tables
    ensure_tables()
    logger.info("Таблицы реестра и валидаций инициализированы")
    logger.info("База данных инициализирована")
    
    yield
    
    # Shutdown (если нужно что-то делать при завершении)
    logger.info("Завершение работы приложения")


# Создаём приложение с lifespan
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API для веб-интерфейса AI-сервиса распознавания поражений головного мозга",
    lifespan=lifespan
)

# CORS middleware для production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для Docker - фронт и бэк на одном порту
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаём экземпляр менеджера pipeline
pipeline_manager = PipelineManager()


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def run_pipeline_background(
    run_id: str,
    input_path: str,
    output_path: str,
    db: Session,
    lesion_type: str = "glioblastoma",
):
    """
    Фоновая задача для запуска pipeline
    
    Args:
        run_id: ID запуска
        input_path: Путь к входным данным
        output_path: Путь к результатам
        db: Сессия БД
    """
    logger.info(f"Фоновый запуск pipeline для run_id: {run_id}")
    
    # Обновляем статус на "running"
    update_pipeline_run(db, run_id, status="running", started_at=datetime.now(timezone.utc))
    
    # Запускаем pipeline
    process = pipeline_manager.start_pipeline(
        run_id, input_path, output_path,
        lesion_type=lesion_type,
    )
    
    if not process:
        # Ошибка запуска
        update_pipeline_run(
            db,
            run_id,
            status="failed",
            error_message="Не удалось запустить pipeline",
            completed_at=datetime.now(timezone.utc)
        )
        logger.error(f"Не удалось запустить pipeline для run_id: {run_id}")
        return
    
    # Ждём завершения процесса
    try:
        stdout, stderr = process.communicate(timeout=settings.pipeline_timeout_seconds)
        return_code = process.returncode
        
        if return_code == 0:
            # Успешное завершение
            logger.info(f"Pipeline успешно завершён для run_id: {run_id}")
            
            # Получаем отчёты о качестве (теперь это список)
            quality_reports = pipeline_manager.get_quality_report(output_path)
            quality_score = None
            quality_category = None

            if quality_reports and len(quality_reports) > 0:
                # Берём первый отчёт для общей оценки
                first_report = quality_reports[0]
                quality_score = first_report.get('quality_score')
                quality_category = first_report.get('quality_category')
                
                logger.info(f"Качество: {quality_score} ({quality_category}), всего отчётов: {len(quality_reports)}")
            
            # Обновляем статус
            update_pipeline_run(
                db,
                run_id,
                status="completed",
                overall_progress=100.0,
                quality_score=quality_score,
                quality_category=quality_category,
                completed_at=datetime.utcnow()
            )
        else:
            # Ошибка выполнения
            logger.error(f"Pipeline завершился с ошибкой для run_id: {run_id}, код: {return_code}")
            
            # Логируем полный stderr для отладки
            if stderr:
                logger.error(f"STDERR от pipeline:\n{stderr}")
            if stdout:
                logger.info(f"STDOUT от pipeline:\n{stdout}")
            
            error_msg = stderr[-500:] if stderr else "Неизвестная ошибка"
            
            update_pipeline_run(
                db,
                run_id,
                status="failed",
                error_message=error_msg,
                completed_at=datetime.utcnow()
            )
            
    except subprocess.TimeoutExpired:
        # Таймаут
        logger.error(f"Таймаут выполнения pipeline для run_id: {run_id}")
        process.kill()
        
        update_pipeline_run(
            db,
            run_id,
            status="failed",
            error_message="Превышено время ожидания выполнения",
            completed_at=datetime.utcnow()
        )
    
    except Exception as e:
        # Другие ошибки
        logger.error(f"Ошибка выполнения pipeline для run_id: {run_id}: {e}")
        
        update_pipeline_run(
            db,
            run_id,
            status="failed",
            error_message=str(e),
            completed_at=datetime.utcnow()
        )
    
    finally:
        # Мониторинг остановится автоматически когда pipeline завершится
        # (проверка статуса в _monitor_loop)
        
        # Очистка runtime конфига (с учётом настройки отладки)
        pipeline_manager.cleanup_runtime_config(run_id, keep_for_debug=settings.keep_runtime_configs)


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/api/health", response_model=HealthCheckResponse)
async def root():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc),
        version=settings.app_version
    )


@app.post("/api/pipeline/start", response_model=PipelineStartResponse)
async def start_pipeline(
    request: PipelineStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Запуск pipeline для обработки DICOM данных
    
    Args:
        request: Параметры запуска (пути входа/выхода)
        background_tasks: FastAPI background tasks
        db: Сессия БД
        
    Returns:
        Информация о созданном запуске
    """
    logger.info(f"Получен запрос на запуск pipeline: {request.input_path}")
    
    # Определяем output_path
    if request.use_default_output:
        output_path = str(settings.default_output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    elif request.output_path:
        output_path = request.output_path
    else:
        output_path = str(settings.default_output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Валидация входного пути
    if not pipeline_manager.validate_input_path(request.input_path):
        raise HTTPException(
            status_code=400,
            detail="Входная директория не существует или пуста"
        )
    
    # Создаём запись в БД
    run = create_pipeline_run(
        db,
        input_path=request.input_path,
        output_path=output_path,
        lesion_type=(request.lesion_type or "glioblastoma"),
    )
    
    # Запускаем pipeline в фоновой задаче
    background_tasks.add_task(
        run_pipeline_background,
        run.run_id,
        run.input_path,
        run.output_path,
        db,
        lesion_type=(request.lesion_type or "glioblastoma"),
    )

    # Запускаем мониторинг (из асинхронного контекста)
    asyncio.create_task(pipeline_monitor.start_monitoring(
        run.run_id, run.output_path, request.kappa_session_id, request.lesion_type
    ))

    logger.info(f"Pipeline запущен с run_id: {run.run_id}")
    
    return PipelineStartResponse(
        run_id=run.run_id,
        status=PipelineStatus.PENDING,
        message="Pipeline запущен и будет выполнен в фоновом режиме",
        created_at=run.created_at,
        lesion_type=run.lesion_type,
    )


@app.get("/api/pipeline/status/{run_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить текущий статус выполнения pipeline
    
    Args:
        run_id: ID запуска
        db: Сессия БД
        
    Returns:
        Текущий статус и прогресс выполнения
    """
    # Получаем информацию о запуске
    run = get_pipeline_run(db, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Запуск не найден")
    
    # Получаем информацию об этапах
    stage_executions = get_stage_executions(db, run_id)
    
    # Если pipeline запущен, обновляем прогресс из логов
    if run.status == "running":
        log_path = pipeline_manager.get_log_file(run.output_path)
        if log_path:
            progress_info = pipeline_manager.parse_log_for_progress(log_path)
            
            # Обновляем current_stage
            if progress_info['current_stage'] > 0:
                update_pipeline_run(
                    db,
                    run_id,
                    current_stage=progress_info['current_stage'],
                    overall_progress=progress_info['overall_progress']
                )
                run.current_stage = progress_info['current_stage']
                run.overall_progress = progress_info['overall_progress']
    
    # Формируем ответ
    stages = [
        StageProgress(
            stage_number=stage.stage_number,
            stage_name=stage.stage_name,
            status=stage.status,
            progress=stage.progress,
            started_at=stage.started_at,
            completed_at=stage.completed_at,
            error=stage.error_message
        )
        for stage in stage_executions
    ]
    
    return PipelineStatusResponse(
        run_id=run.run_id,
        status=PipelineStatus(run.status),
        current_stage=run.current_stage,
        overall_progress=run.overall_progress,
        stages=stages,
        created_at=run.created_at,
        completed_at=run.completed_at,
        error=run.error_message,
        lesion_type=getattr(run, "lesion_type", None),
    )


@app.get("/api/pipeline/history", response_model=PipelineHistoryResponse)
async def get_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    Получить историю запусков pipeline
    
    Args:
        limit: Количество записей
        offset: Смещение
        db: Сессия БД
        
    Returns:
        Список запусков с пагинацией
    """
    runs, total = get_pipeline_history(db, limit, offset)
    
    history_items = [
        PipelineRunHistoryItem(
            run_id=run.run_id,
            input_path=run.input_path,
            output_path=run.output_path,
            status=PipelineStatus(run.status),
            current_stage=run.current_stage if run.current_stage is not None else 0,
            quality_score=run.quality_score,
            quality_category=run.quality_category,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            duration_seconds=(
                int((run.completed_at - run.started_at).total_seconds())
                if run.completed_at and run.started_at else None
            ),
            lesion_type=getattr(run, 'lesion_type', None) or 'glioblastoma',
        )
        for run in runs
    ]
    
    return PipelineHistoryResponse(
        total=total,
        runs=history_items
    )


@app.get("/api/quality-report/{run_id}", response_model=QualityReportListResponse)
async def get_quality_report_endpoint(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Получить отчёты о качестве для всех обработанных файлов"""
    logger.info(f"Запрос отчётов о качестве для run_id: {run_id}")
    
    # Получаем run из БД
    run = get_pipeline_run(db, run_id)
    
    if not run:
        logger.error(f"Run не найден: {run_id}")
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    logger.info(f"Output path: {run.output_path}, Current stage: {run.current_stage}")
    
    # Проверяем что 3-й этап завершён
    if run.current_stage < 3:
        logger.warning(f"Этап 3 ещё не завершён для run_id: {run_id}")
        raise HTTPException(
            status_code=400, 
            detail="Quality assessment stage not yet completed"
        )
    
    # Получаем отчёты (список)
    reports = pipeline_manager.get_quality_report(run.output_path)
    
    if not reports:
        logger.error(f"Отчёты не найдены для run_id: {run_id}")
        raise HTTPException(status_code=404, detail="Quality reports not found")
    
    logger.info(f"Найдено {len(reports)} отчётов для run_id: {run_id}")
    
    return QualityReportListResponse(
        total=len(reports),
        reports=reports
    )

@app.websocket("/ws/pipeline/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint для real-time обновлений прогресса pipeline
    
    Args:
        websocket: WebSocket соединение
        run_id: ID запуска pipeline
    """
    await ws_manager.connect(run_id, websocket)
    
    try:
        # Отправляем приветственное сообщение
        await websocket.send_json({
            "type": "connected",
            "run_id": run_id,
            "message": "Подключено к мониторингу pipeline"
        })
        
        # Держим соединение открытым
        while True:
            # Ждём сообщения от клиента (ping для keep-alive)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket отключён для run_id: {run_id}")
    
    except Exception as e:
        logger.error(f"Ошибка WebSocket для run_id {run_id}: {e}")
    
    finally:
        ws_manager.disconnect(run_id, websocket)

@app.get("/api/nifti/{run_id}/{file_type}/{filename}")
async def get_nifti_file(
    run_id: str,
    file_type: str,  # "preprocessed" или "segmentation"
    filename: str,
    db: Session = Depends(get_db)
):
    """
    Получить NIfTI файл для визуализации
    
    Args:
        run_id: ID запуска pipeline
        file_type: Тип файла (preprocessed/segmentation)
        filename: Имя файла
    """
    logger.info(f"Запрос NIfTI файла: {file_type}/{filename} для run_id: {run_id}")
    
    # Получаем run из БД
    run = get_pipeline_run(db, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    # Проверяем что 5-й этап завершён
    if run.current_stage < 5:
        raise HTTPException(
            status_code=400,
            detail="Segmentation stage not yet completed"
        )
    
    if file_type not in ["preprocessed", "segmentation", "nifti"]:
        raise HTTPException(status_code=400, detail="Invalid file_type. Must be 'preprocessed', 'segmentation', or 'nifti'")
    
    # Формируем путь к файлу
    base_dir = Path(run.output_path) / file_type
    
    # Рекурсивный поиск файла
    file_path = None
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            file_path = Path(root) / filename
            break
    
    if not file_path or not file_path.exists():
        logger.error(f"Файл не найден: {filename} в {base_dir}")
        raise HTTPException(status_code=404, detail="NIfTI file not found")
    
    logger.info(f"Отдаём файл: {file_path}")
    
    return FileResponse(
        path=str(file_path),
        media_type="application/gzip",
        filename=filename
    )

@app.get("/api/nifti-files/{run_id}", response_model=NIfTIFilesResponse)
async def get_nifti_files_list(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить список доступных NIfTI файлов для визуализации
    
    Args:
        run_id: ID запуска pipeline
    """
    logger.info(f"Запрос списка NIfTI файлов для run_id: {run_id}")
    
    # Получаем run из БД
    run = get_pipeline_run(db, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    # Проверяем что 5-й этап завершён
    if run.current_stage < 5:
        raise HTTPException(
            status_code=400,
            detail="Segmentation stage not yet completed"
        )
    
    # Ищем файлы в preprocessed и segmentation директориях
    output_base = Path(run.output_path)
    preprocessed_dir = output_base / "preprocessed"
    segmentation_dir = output_base / "segmentation"
    nifti_dir = output_base / "nifti"
    
    if not preprocessed_dir.exists():
        raise HTTPException(status_code=404, detail="Preprocessed directory not found")
    
    if not segmentation_dir.exists():
        raise HTTPException(status_code=404, detail="Segmentation directory not found")
    
    nifti_files = []
    
    # Сначала находим все preprocessed файлы
    preprocessed_files = {}
    for nifti_path in preprocessed_dir.rglob("*.nii.gz"):
        filename = nifti_path.name
        
        # Создаём ключ без расширения для сопоставления с маской
        base_name = filename.replace(".nii.gz", "")
        preprocessed_files[base_name] = filename
    
    logger.info(f"Найдено {len(preprocessed_files)} preprocessed файлов")
    
    # Теперь находим сегментационные маски и сопоставляем с preprocessed
    for mask_path in segmentation_dir.rglob("*_segmask.nii.gz"):
        mask_filename = mask_path.name
        
        # Извлекаем base name без _segmask
        base_name = mask_filename.replace("_segmask.nii.gz", "")
        parts = base_name.split("_")
        
        try:
            patient_id = parts[0]  # sub-001
            session_id = parts[1] if len(parts) > 1 else "ses-001"
            
            # Формируем префикс пациент+сессия
            prefix = f"{patient_id}_{session_id}_"
            
            # Ищем ВСЕ preprocessed файлы с этим префиксом
            for prep_base, prep_filename in preprocessed_files.items():
                if prep_base.startswith(prefix):
                    prep_parts = prep_base.split("_")
                    modality = prep_parts[2] if len(prep_parts) > 2 else "unknown"
                    
                    # Ищем нативные файлы
                    native_image_url = None
                    native_mask_url = None
                    
                    # Нативное изображение в nifti/
                    native_image_name = f"{patient_id}_{session_id}_{modality}.nii.gz"
                    native_image_path = nifti_dir / patient_id / session_id / "anat" / native_image_name
                    if native_image_path.exists():
                        native_image_url = f"/api/nifti/{run_id}/nifti/{native_image_name}"
                    
                    # Нативная маска в segmentation/
                    native_mask_name = f"{base_name}_segmask_native_{modality}.nii.gz"
                    for native_mask_path in segmentation_dir.rglob(native_mask_name):
                        native_mask_url = f"/api/nifti/{run_id}/segmentation/{native_mask_name}"
                        break
                    
                    # Resolve labeled mask (MS: one label per lesion instance)
                    labels_name = mask_filename.replace("_segmask.nii.gz", "_segmask_labels.nii.gz")
                    mask_labels_url = None
                    for labels_path in segmentation_dir.rglob(labels_name):
                        mask_labels_url = f"/api/nifti/{run_id}/segmentation/{labels_name}"
                        break

                    # Resolve per-lesion volume map from the lesion stats JSON
                    volumes_by_label = None
                    stats_name = f"{base_name}_lesion_stats_report.json"
                    for stats_p in segmentation_dir.rglob(stats_name):
                        try:
                            with open(stats_p, "r", encoding="utf-8") as sf:
                                volumes_by_label = json.load(sf).get("lesion_volumes_by_label")
                        except Exception:
                            volumes_by_label = None
                        break

                    nifti_files.append(NIfTIFile(
                        filename=prep_filename,
                        mask_filename=mask_filename,
                        patient_id=patient_id,
                        session_id=session_id,
                        modality=modality.upper(),
                        image_url=f"/api/nifti/{run_id}/preprocessed/{prep_filename}",
                        mask_url=f"/api/nifti/{run_id}/segmentation/{mask_filename}",
                        native_image_url=native_image_url,
                        native_mask_url=native_mask_url,
                        mask_labels_url=mask_labels_url,
                        lesion_volumes_by_label=volumes_by_label,
                    ))
                    
                    logger.info(f"Добавлен файл: {prep_filename} с маской {mask_filename}"
                               f"{' (+native)' if native_mask_url else ''}")
        
        except Exception as e:
            logger.warning(f"Не удалось распарсить имя файла {mask_filename}: {e}")
            continue
    
    logger.info(f"Итого подготовлено {len(nifti_files)} пар файлов для визуализации")
    
    return NIfTIFilesResponse(
        total=len(nifti_files),
        files=nifti_files
    )

@app.get("/api/volume-reports/{run_id}", response_model=VolumeReportListResponse)
async def get_volume_reports(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить отчёты об объёмах опухоли
    """
    logger.info(f"Запрос отчётов об объёмах для run_id: {run_id}")
    
    run = get_pipeline_run(db, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    if run.current_stage < 5:
        raise HTTPException(
            status_code=400,
            detail="Segmentation stage not yet completed"
        )
    
    reports = pipeline_manager.get_volume_reports(run.output_path)
    
    if not reports:
        raise HTTPException(status_code=404, detail="Volume reports not found")
    
    logger.info(f"Найдено {len(reports)} отчётов об объёмах для run_id: {run_id}")
    
    return VolumeReportListResponse(
        total=len(reports),
        reports=reports
    )

@app.get("/api/lobar-reports/{run_id}", response_model=LobarReportListResponse)
async def get_lobar_reports(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить отчёты о лобарной локализации поражений
    """
    logger.info(f"Запрос лобарных отчётов для run_id: {run_id}")
    
    run = get_pipeline_run(db, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    if run.current_stage < 7:
        raise HTTPException(
            status_code=400,
            detail="Lobar localization stage not yet completed"
        )
    
    reports = pipeline_manager.get_lobar_reports(run.output_path)
    
    if not reports:
        raise HTTPException(status_code=404, detail="Lobar reports not found")
    
    logger.info(f"Найдено {len(reports)} лобарных отчётов для run_id: {run_id}")
    
    return LobarReportListResponse(
        total=len(reports),
        reports=reports
    )

@app.get("/api/mcdonald-reports/{run_id}", response_model=McDonaldReportListResponse)
async def get_mcdonald_reports(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить отчёты о McDonald-классификации очагов МС (зональная локализация)
    """
    logger.info(f"Запрос McDonald-отчётов для run_id: {run_id}")

    run = get_pipeline_run(db, run_id)

    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    if run.current_stage < 7:
        raise HTTPException(
            status_code=400,
            detail="Anatomical analysis stage not yet completed"
        )

    reports = pipeline_manager.get_mcdonald_reports(run.output_path)

    if not reports:
        raise HTTPException(status_code=404, detail="McDonald reports not found (MS only)")

    logger.info(f"Найдено {len(reports)} McDonald-отчётов для run_id: {run_id}")

    return McDonaldReportListResponse(
        total=len(reports),
        reports=reports
    )

@app.get("/api/lesion-stats/{run_id}", response_model=LesionStatsListResponse)
async def get_lesion_stats(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Статистика очагов МС (количество, объёмы) для run_id"""
    logger.info(f"Запрос lesion stats для run_id: {run_id}")
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    if run.current_stage < 7:
        raise HTTPException(status_code=400, detail="Lobar localization not yet completed")

    reports = pipeline_manager.get_lesion_stats_reports(run.output_path)
    if not reports:
        raise HTTPException(status_code=404, detail="Lesion stats not found (MS only)")

    return LesionStatsListResponse(
        total=len(reports),
        reports=[LesionStatsReport(**r) for r in reports]
    )


@app.get("/api/longitudinal/{patient_id}", response_model=LongitudinalResponse)
async def get_longitudinal(
    patient_id: str,
    lesion_type: str = "multiple_sclerosis",
    db: Session = Depends(get_db)
):
    """
    Лонгитюдный анализ: все сессии пациента по данному lesion_type.

    patient_id — original_patient_id (напр. "P000915").
    Матчинг: bids_id в registry == patient_id в stats файле (оба "sub-P000915").
    """
    from patient_registry import (
        find_by_patient_id, find_by_bids_id, find_by_bids_subject,
    )

    # The frontend passes the BIDS subject ("sub-001"). The registry stores
    # original_patient_id ("P000915") and a per-session bids_id ("sub-001_ses-002").
    # Resolve in order: original_patient_id → exact bids_id → bids subject prefix
    # (the last one is what actually matches "sub-001" → its ses-* records).
    all_records = find_by_patient_id(patient_id)
    if not all_records:
        all_records = find_by_bids_id(patient_id)
    if not all_records:
        all_records = find_by_bids_subject(patient_id)

    records = [r for r in all_records if r.get("lesion_type") == lesion_type]

    if not records:
        raise HTTPException(status_code=404, detail="No sessions found for this patient/lesion_type")

    # Each registry record is one session. A single run can hold several
    # sessions, so iterate every record (not unique run_ids) and cache each
    # run's stats to avoid re-reading. Match a stats entry to a record by the
    # full session key: stats files store patient_id="sub-001" + session_id
    # ="ses-002", whereas the registry bids_id is the combined "sub-001_ses-002".
    run_stats_cache: dict = {}
    all_stats = []
    for record in records:
        run_id = record.get("pipeline_run_id")  # registry key for the run
        if not run_id:
            continue
        if run_id not in run_stats_cache:
            run = get_pipeline_run(db, run_id)
            run_stats_cache[run_id] = (
                pipeline_manager.get_lesion_stats_reports(run.output_path) or []
                if run and run.output_path else []
            )
        bids_id = record.get("bids_id", "")
        for s in run_stats_cache[run_id]:
            session_key = f"{s.get('patient_id')}_{s.get('session_id')}"
            if session_key == bids_id:
                all_stats.append({**s, "_scan_date": record.get("scan_date")})

    # Deduplicate by session_id, sort by scan_date
    seen_sessions: set = set()
    points = []
    for s in sorted(all_stats, key=lambda x: x.get("_scan_date") or ""):
        sid = s.get("session_id", "")
        if sid in seen_sessions:
            continue
        seen_sessions.add(sid)
        points.append(LongitudinalPoint(
            session_id=sid,
            scan_date=str(s["_scan_date"]) if s.get("_scan_date") else None,
            total_volume_cm3=s.get("total_volume_cm3", 0.0),
            lesion_count=s.get("lesion_count"),
        ))

    if len(points) < 2:
        raise HTTPException(
            status_code=404,
            detail="Not enough sessions for longitudinal analysis (need >= 2)"
        )

    return LongitudinalResponse(
        patient_id=patient_id,
        lesion_type=lesion_type,
        points=points,
    )


def _split_bids_id(bids_id: str) -> tuple:
    """'sub-001_ses-002' -> ('sub-001', 'ses-002')"""
    idx = bids_id.find("_ses-")
    if idx == -1:
        return bids_id, ""
    return bids_id[:idx], bids_id[idx + 1:]


@app.get("/api/longitudinal/{patient_id}/diff", response_model=LongitudinalDiffResponse)
async def get_longitudinal_diff(
    patient_id: str,
    lesion_type: str = "multiple_sclerosis",
    db: Session = Depends(get_db)
):
    """
    Детекция новых/растущих/разрешившихся очагов между каждой парой соседних
    по времени сессий пациента (МС). Не использует ту же фильтрацию по
    pipeline_run_id, что get_longitudinal — резолвит output_path на лету.
    """
    all_records = find_by_patient_id(patient_id)
    if not all_records:
        all_records = find_by_bids_id(patient_id)
    if not all_records:
        all_records = find_by_bids_subject(patient_id)

    records = [r for r in all_records if r.get("lesion_type") == lesion_type]
    if not records:
        raise HTTPException(status_code=404, detail="No sessions found for this patient/lesion_type")

    sortable = sorted(records, key=lambda r: r.get("scan_date") or "")
    seen_bids_ids = set()
    ordered = []
    for r in sortable:
        bid = r.get("bids_id", "")
        if bid and bid not in seen_bids_ids:
            seen_bids_ids.add(bid)
            ordered.append(r)

    if len(ordered) < 2:
        raise HTTPException(status_code=404, detail="Not enough sessions for diff analysis (need >= 2)")

    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "ms_longitudinal_config.yaml"
    with open(config_path, "r") as f:
        diff_config = yaml.safe_load(f)

    pairs = []
    for prev_record, curr_record in zip(ordered[:-1], ordered[1:]):
        prev_run = get_pipeline_run(db, prev_record.get("pipeline_run_id"))
        curr_run = get_pipeline_run(db, curr_record.get("pipeline_run_id"))
        if not prev_run or not curr_run:
            continue

        prev_subject, prev_session = _split_bids_id(prev_record.get("bids_id", ""))
        curr_subject, curr_session = _split_bids_id(curr_record.get("bids_id", ""))

        prev_label_path = pipeline_manager.get_segmask_label_path(prev_run.output_path, prev_subject, prev_session)
        curr_label_path = pipeline_manager.get_segmask_label_path(curr_run.output_path, curr_subject, curr_session)
        if not prev_label_path or not curr_label_path:
            logger.warning(
                f"Skipping diff {prev_session}->{curr_session} for {patient_id}: "
                f"missing labeled mask (prev={prev_label_path}, curr={curr_label_path})"
            )
            continue

        try:
            result = compare_labeled_masks(
                prev_label_path, curr_label_path,
                growth_threshold_relative=diff_config.get("growth_threshold_relative", 0.20),
                growth_threshold_absolute_cm3=diff_config.get("growth_threshold_absolute_cm3", 0.03),
                dilation_voxels=diff_config.get("dilation_voxels", 1),
            )
        except ValueError as e:
            logger.warning(f"Skipping diff {prev_session}->{curr_session} for {patient_id}: {e}")
            continue

        pairs.append(LongitudinalDiffPair(
            from_session_id=prev_session,
            to_session_id=curr_session,
            new_count=result["new_count"],
            growing_count=result["growing_count"],
            stable_count=result["stable_count"],
            resolved_count=result["resolved_count"],
            lesions=[LesionDiffEntry(**l) for l in result["lesions"]],
        ))

    return LongitudinalDiffResponse(patient_id=patient_id, pairs=pairs)


@app.get("/api/lobar-atlas/{run_id}")
async def get_lobar_atlas(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить лобарный атлас, ресэмплированный под preprocessed данные запуска
    """
    import nibabel as nib_local
    import numpy as np
    from scipy.ndimage import affine_transform
    import tempfile
    
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    try:
        # Find any preprocessed file as reference
        prep_dir = Path(run.output_path) / "preprocessed"
        ref_files = list(prep_dir.rglob("*.nii.gz"))
        if not ref_files:
            raise HTTPException(status_code=404, detail="No preprocessed files found")
        
        ref_nii = nib_local.load(str(ref_files[0]))
        
        # Determine which lobar atlas to use from preprocessing config
        import yaml as yaml_local
        preprocessing_config_path = settings.pipeline_root / "configs" / "preprocessing_config.yaml"
        lobar_config_path = settings.pipeline_root / "configs" / "lobar_atlas_config.yaml"
        
        with open(preprocessing_config_path, 'r') as f:
            prep_config = yaml_local.safe_load(f)
        with open(lobar_config_path, 'r') as f:
            lobar_config = yaml_local.safe_load(f)
        
        template_name = prep_config.get("atlas", {}).get("name", "SRI24")
        templates = lobar_config.get("templates", {})
        
        if template_name not in templates:
            raise HTTPException(status_code=404, detail=f"No lobar atlas for template {template_name}")
        
        atlas_rel = templates[template_name]["file"]
        atlas_path = settings.pipeline_root / atlas_rel
        
        if not atlas_path.exists():
            raise HTTPException(status_code=404, detail="Lobar atlas file not found")
        
        atlas_nii = nib_local.load(str(atlas_path))
        atlas_data = np.asarray(atlas_nii.dataobj).astype(np.float64)
        ref_data = np.squeeze(ref_nii.get_fdata())
        
        # Check if resampling needed
        if atlas_nii.shape[:3] == ref_nii.shape[:3] and np.allclose(atlas_nii.affine, ref_nii.affine):
            # Exact match — serve as-is
            return FileResponse(
                path=str(atlas_path),
                media_type="application/gzip",
                filename=atlas_path.name
            )
        
        # Resample atlas to match preprocessed file
        logger.info(f"Resampling lobar atlas {atlas_nii.shape} -> {ref_nii.shape[:3]}")
        
        transform = np.linalg.inv(atlas_nii.affine) @ ref_nii.affine
        matrix = transform[:3, :3]
        offset = transform[:3, 3]
        
        resampled = affine_transform(
            atlas_data, matrix, offset=offset,
            output_shape=ref_nii.shape[:3],
            order=0, mode='constant', cval=0
        ).astype(np.int16)
        
        # Save to temp file with preprocessed affine
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        out_nii = nib_local.Nifti1Image(resampled, ref_nii.affine)
        nib_local.save(out_nii, tmp.name)
        
        logger.info(f"Lobar atlas resampled: {resampled.shape}, non-zero: {(resampled > 0).sum()}")
        
        return FileResponse(
            path=tmp.name,
            media_type="application/gzip",
            filename=f"lobar_atlas_{template_name}.nii.gz"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения лобарного атласа: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# KAPPA CONFIG ENDPOINTS
# ============================================

from kappa_dataset_mapping import get_lesion_types


@app.get("/api/kappa/lesion-types")
async def get_lesion_types_endpoint():
    """Список доступных типов поражений"""
    return get_lesion_types()

# ============================================
# VALIDATION ENDPOINTS
# ============================================

from pydantic import BaseModel as PydanticBaseModel
from fastapi.responses import FileResponse
import tempfile
import zipfile


@app.get("/api/validation/download-package/{run_id}")
async def download_slicer_package(run_id: str, session_id: str):
    """
    Скачать zip-пакет для редактирования в 3D Slicer.
    Содержит: 4 preprocessed NIfTI + 4 native NIfTI + 1 маска + README.
    """
    from kappa_auth import get_session
    from patient_registry import find_by_run_id

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия не найдена")

    # Находим output_path через pipeline_runs
    from database import SessionLocal, get_pipeline_run
    db = SessionLocal()
    try:
        run = get_pipeline_run(db, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Запуск не найден")
        output_path = Path(run.output_path)
    finally:
        db.close()

    # Находим entity info из реестра
    records = find_by_run_id(run_id)
    if not records:
        raise HTTPException(status_code=404, detail="Нет связанных сущностей")

    record = records[0]
    bids_id = record.get("bids_id", "unknown")
    sub, ses = bids_id.split("_", 1) if "_" in bids_id else (bids_id, "ses-001")

    # Собираем файлы
    preprocessed_dir = output_path / "preprocessed" / sub / ses / "anat"
    nifti_dir = output_path / "nifti" / sub / ses / "anat"
    segmentation_dir = output_path / "segmentation" / sub / ses / "anat"

    files_to_pack = []

    # Preprocessed NIfTI
    if preprocessed_dir.exists():
        for f in sorted(preprocessed_dir.glob("*.nii.gz")):
            files_to_pack.append(("preprocessed", f))

    # Native NIfTI
    if nifti_dir.exists():
        for f in sorted(nifti_dir.glob("*.nii.gz")):
            files_to_pack.append(("native", f))

    # Маска сегментации (основная, без native)
    if segmentation_dir.exists():
        for f in sorted(segmentation_dir.glob("*_segmask.nii.gz")):
            if "_native_" not in f.name:
                files_to_pack.append(("segmentation", f))

    if not files_to_pack:
        raise HTTPException(status_code=404, detail="Файлы не найдены")

    # README
    readme_text = f"""# Пакет для редактирования сегментации в 3D Slicer
# Пациент: {bids_id}
# Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Содержимое
- preprocessed/ — предобработанные NIfTI (4 модальности в пространстве атласа)
- native/ — оригинальные NIfTI (4 модальности в нативном пространстве)
- segmentation/ — маска сегментации от ИИ

## Инструкция по редактированию в 3D Slicer

1. Откройте 3D Slicer
2. File → Add Data → выберите ВСЕ файлы из этого архива
3. В выпадающем меню Volume выберите preprocessed модальность (например T1)
4. Перейдите в модуль Segment Editor (в верхнем меню или через поиск модулей)
5. Нажмите кнопку с иконкой стрелки вниз (Import Labelmap) рядом с "Add"
6. Выберите файл маски (*_segmask.nii.gz)
7. Маска загрузится с 4 сегментами:
   - Сегмент 1 (красный): NCR — Некротическое ядро
   - Сегмент 2 (зелёный): ED — Отёк
   - Сегмент 3 (жёлтый): NET — Неусиливающаяся опухоль
   - Сегмент 4 (синий): ET — Усиливающаяся опухоль

## Инструменты редактирования
- Paint (Кисть): рисовать области выбранного класса
- Erase (Ластик): стирать области
- Scissors (Ножницы): вырезать/оставить область
- Threshold: выделение по интенсивности
- Grow from Seeds: полуавтоматическая сегментация

## Сохранение результата
1. В Segment Editor нажмите кнопку экспорта (Export Labelmap)
2. File → Save → выберите формат NIfTI (.nii.gz)
3. Сохраните файл маски
4. Загрузите отредактированную маску через интерфейс сервиса
   (кнопка "Загрузить маску" в окне визуализации)
"""

    # Создаём zip
    tmp_dir = tempfile.mkdtemp()
    zip_name = f"slicer_package_{bids_id}.zip"
    zip_path = Path(tmp_dir) / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme_text)
        for folder, fpath in files_to_pack:
            arcname = f"{folder}/{fpath.name}"
            zf.write(fpath, arcname)

    logger.info(
        "Slicer package created: %s (%d files, %.1f MB)",
        zip_name, len(files_to_pack), zip_path.stat().st_size / 1024 / 1024,
    )

    return FileResponse(
        path=str(zip_path),
        filename=zip_name,
        media_type="application/zip",
    )


# ============================================
# ЗАГРУЗКА ОТРЕДАКТИРОВАННОЙ МАСКИ
# ============================================

from fastapi import UploadFile, File, Form

@app.post("/api/validation/upload-mask")
async def upload_edited_mask(
    entity_id: str = Form(...),
    dataset_id: int = Form(...),
    session_id: str = Form(...),
    run_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Загрузить отредактированную маску от эксперта.
    
    Шаги:
    1. Валидация сессии и файла
    2. Сохранение файла локально (в директорию ИИ-маски пациента)
    3. Загрузка в Каппу (получение kappa_file_id)
    4. Регистрация новой версии в mask_versions
    5. Возврат информации о новой версии
    """
    import shutil
    from kappa_auth import get_session
    from kappa_client import replace_entity_file
    from mask_service import register_expert_mask, get_next_version, get_ai_mask_dir

    # 1. Валидация сессии
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    # Валидация файла
    if not file.filename or not file.filename.endswith((".nii.gz", ".nii")):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть в формате NIfTI (.nii.gz или .nii)",
        )

    # 2. Определяем путь для сохранения — директория ИИ-маски пациента
    ai_mask_dir = get_ai_mask_dir(entity_id)
    if not ai_mask_dir:
        # Fallback: берём из текущего run (для первого запуска)
        from database import SessionLocal as DBSessionLocal, get_pipeline_run
        from patient_registry import find_by_run_id
        db = DBSessionLocal()
        try:
            run = get_pipeline_run(db, run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Запуск не найден")
            output_path = Path(run.output_path)
        finally:
            db.close()

        records = find_by_run_id(run_id)
        if not records:
            raise HTTPException(status_code=404, detail="Нет связанных сущностей")
        record = records[0]
        bids_id = record.get("bids_id", "unknown")
        sub, ses = bids_id.split("_", 1) if "_" in bids_id else (bids_id, "ses-001")
        segmentation_dir = output_path / "segmentation" / sub / ses / "anat"
    else:
        segmentation_dir = Path(ai_mask_dir)

    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # Определяем номер версии — единый источник из БД
    next_version = get_next_version(entity_id)

    # Имя файла: берём базу из ИИ-маски или формируем
    original_masks = list(segmentation_dir.glob("*_segmask.nii.gz"))
    if original_masks:
        base_name = original_masks[0].name.replace(".nii.gz", "")
    else:
        # Определяем bids_id для имени
        from patient_registry import find_by_run_id
        records = find_by_run_id(run_id) if not ai_mask_dir else []
        if records:
            bids_id = records[0].get("bids_id", "unknown")
            sub, ses = bids_id.split("_", 1) if "_" in bids_id else (bids_id, "ses-001")
        base_name = f"{sub}_{ses}_segmask" if 'sub' in dir() else "segmask"

    versioned_name = f"{base_name}_v{next_version}.nii.gz"
    save_path = segmentation_dir / versioned_name

    # Сохраняем файл
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info("Expert mask saved: %s (%.1f MB)", save_path, len(content) / 1024 / 1024)
    except Exception as e:
        logger.error("Failed to save expert mask: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {e}")

    # 3. Заливаем в Каппу (получаем kappa_file_id)
    kappa_file_id = None
    try:
        kappa_file_id = await replace_entity_file(
            token=session["kappa_token"],
            user_id=session["user_id"],
            user_type_id=session["user_type_id"],
            dataset_id=dataset_id,
            entity_id=entity_id,
            file_path=save_path,
        )
    except Exception as kappa_err:
        logger.error("Kappa upload exception (non-fatal): %s", kappa_err)

    kappa_ok = kappa_file_id is not None and kappa_file_id != "uploaded_no_file_id"

    if not kappa_ok:
        logger.warning(
            "Mask saved locally but Kappa upload issue: entity=%s, file_id=%s",
            entity_id, kappa_file_id,
        )

    # 4. Регистрируем в mask_versions (с kappa_file_id)
    user_name = ""
    first_name = session.get("first_name") or ""
    last_name = session.get("last_name") or ""
    if first_name or last_name:
        user_name = f"{first_name} {last_name}".strip()

    mask_info = register_expert_mask(
        entity_id=entity_id,
        dataset_id=dataset_id,
        file_path=str(save_path),
        user_id=session["user_id"],
        user_name=user_name,
        kappa_file_id=kappa_file_id if kappa_ok else None,
    )

    # 5. Ответ
    return {
        "success": True,
        "mask_version": mask_info,
        "kappa_uploaded": kappa_ok,
        "file_name": versioned_name,
        "message": (
            f"Маска v{mask_info['version']} сохранена"
            + (" и загружена в Каппу" if kappa_ok else ", но загрузка в Каппу не удалась")
        ),
    }


# ============================================
# ПОЛУЧЕНИЕ ИСТОРИИ ВЕРСИЙ МАСОК
# ============================================

@app.get("/api/validation/mask-versions/{entity_id}")
async def get_mask_versions(entity_id: str, session_id: str):
    """Получить историю версий масок для сущности."""
    from kappa_auth import get_session
    from mask_service import get_mask_history, get_current_mask

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    history = get_mask_history(entity_id)
    current = get_current_mask(entity_id)

    # Помечаем доступность и размер каждой версии
    for v in history:
        has_kappa = v.get("kappa_file_id") and v["kappa_file_id"] != "uploaded_no_file_id"
        local_path = Path(v.get("file_path", "")) if v.get("file_path") else None
        has_local = local_path.exists() if local_path else False
        v["available"] = has_kappa or has_local
        v["file_size"] = local_path.stat().st_size if has_local else None

    return {
        "entity_id": entity_id,
        "versions": history,
        "current_version": current["version"] if current else None,
        "total": len(history),
    }


@app.post("/api/validation/sync-masks/{entity_id}")
async def sync_masks_with_kappa(entity_id: str, session_id: str):
    """
    Синхронизировать mask_versions БД с Каппой.
    Удаляет записи, для которых нет ни kappa_file_id, ни локального файла.
    Для записей с kappa_file_id проверяет наличие файла в Каппе.
    """
    from kappa_auth import get_session
    from kappa_client import get_entity_details
    from mask_service import get_mask_history, delete_mask_version

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    history = get_mask_history(entity_id)
    if not history:
        return {"entity_id": entity_id, "removed": 0, "remaining": 0}

    # Получаем список файлов в Каппе
    dataset_id = history[0].get("dataset_id")
    kappa_files = set()
    try:
        details = await get_entity_details(
            token=session["kappa_token"],
            user_id=session["user_id"],
            user_type_id=session["user_type_id"],
            dataset_id=dataset_id,
            entity_id=entity_id,
        )
        if details and "files" in details:
            kappa_files = {f["fileId"] for f in details["files"] if f.get("fileId")}
            # Также собираем имена файлов для сопоставления
            kappa_filenames = {f["fileName"] for f in details["files"] if f.get("fileName")}
    except Exception as e:
        logger.warning("Failed to fetch entity details from Kappa: %s", e)
        kappa_filenames = set()

    removed = 0
    for v in history:
        has_local = Path(v.get("file_path", "")).exists() if v.get("file_path") else False
        has_kappa_id = v.get("kappa_file_id") and v["kappa_file_id"] in kappa_files
        has_kappa_name = v.get("file_name") in kappa_filenames if kappa_filenames else False

        if not has_local and not has_kappa_id and not has_kappa_name:
            delete_mask_version(entity_id, v["version"])
            removed += 1
            logger.info("Removed orphan mask version: entity=%s, v=%d, file=%s",
                       entity_id, v["version"], v.get("file_name"))

    remaining = len(history) - removed
    return {
        "entity_id": entity_id,
        "removed": removed,
        "remaining": remaining,
    }


@app.get("/api/validation/mask-file/{entity_id}/{version}")
async def serve_mask_version(entity_id: str, version: int, session_id: str):
    """
    Отдать файл маски конкретной версии.
    Если есть kappa_file_id — проксирует из Каппы.
    Иначе — отдаёт локальный файл (fallback).
    """
    from kappa_auth import get_session
    from kappa_client import download_entity_file
    from mask_service import get_mask_by_version

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    mask = get_mask_by_version(entity_id, version)
    if not mask:
        raise HTTPException(status_code=404, detail="Версия маски не найдена")

    # Если есть kappa_file_id — загружаем из Каппы
    if mask.get("kappa_file_id") and mask["kappa_file_id"] != "uploaded_no_file_id":
        content = await download_entity_file(
            token=session["kappa_token"],
            user_id=session["user_id"],
            user_type_id=session["user_type_id"],
            dataset_id=mask["dataset_id"],
            file_id=mask["kappa_file_id"],
        )
        if content:
            return Response(
                content=content,
                media_type="application/gzip",
                headers={"Content-Disposition": f'inline; filename="{mask["file_name"]}"'},
            )
        logger.warning("Kappa download failed for mask v%d, trying local fallback", version)

    # Fallback: локальный файл
    file_path = Path(mask["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл маски не найден")

    return FileResponse(
        path=str(file_path),
        media_type="application/gzip",
        filename=file_path.name,
    )


@app.get("/api/validation/entity-run-info/{entity_id}")
async def get_entity_run_info(entity_id: str):
    """
    Получить run_id и другие данные пайплайна по entity_id из Каппы.
    Нужен для вкладки Валидации, где данные приходят из Каппы
    и run_id изначально неизвестен.
    """
    from patient_registry import find_by_kappa_entity

    record = find_by_kappa_entity(entity_id)
    if not record:
        raise HTTPException(status_code=404, detail="Сущность не найдена в реестре")

    return {
        "entity_id": entity_id,
        "run_id": record.get("pipeline_run_id"),
        "bids_id": record.get("bids_id"),
        "dataset_id": record.get("kappa_dataset_id"),
    }


# ============================================
# ИНТЕГРАЦИЯ С 3D SLICER (ЧЕРЕЗ АГЕНТА)
# ============================================

SLICER_AGENT_URL = "http://host.docker.internal:8001"


@app.get("/api/slicer/status")
async def slicer_agent_status():
    """Проверить доступность Slicer Agent."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{SLICER_AGENT_URL}/health")
        if response.is_success:
            return response.json()
        return {"status": "error", "slicer_found": False}
    except Exception:
        return {"status": "unavailable", "slicer_found": False}


@app.post("/api/slicer/open/{run_id}")
async def open_in_slicer(
    run_id: str,
    session_id: Optional[str] = None,
    selected_mask_version: Optional[int] = None,
    entity_id: Optional[str] = None,
):
    """
    Открыть данные пациента в 3D Slicer через агента.
    Собирает пути к файлам из результатов пайплайна и отправляет агенту.

    entity_id — обязательный для multi-patient/multi-session runs:
        идентифицирует конкретную сущность (пациент+сессия) в пределах run'а.
        Если не передан и в run'е больше одной записи — 400.
    selected_mask_version — если указана, эта версия маски будет default в Slicer.
    """
    from patient_registry import find_by_run_id
    from database import SessionLocal as DBSessionLocal
    import httpx

    db = DBSessionLocal()
    try:
        run = get_pipeline_run(db, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Запуск не найден")
        output_path = Path(run.output_path)
        # Capture lesion_type while the session is open (avoid DetachedInstance).
        slicer_lesion_type = getattr(run, "lesion_type", None) or "glioblastoma"
    finally:
        db.close()

    # Находим данные пациента
    records = find_by_run_id(run_id)
    if not records:
        raise HTTPException(status_code=404, detail="Нет данных пациента")

    # Выбираем нужную запись по entity_id.
    # Если запись одна — допускаем отсутствие entity_id (single-session run).
    # Иначе entity_id обязателен, чтобы корректно различать пациентов/сессии.
    if entity_id:
        record = next(
            (r for r in records if r.get("kappa_entity_id") == entity_id),
            None,
        )
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Запись с entity_id={entity_id} не найдена в run {run_id}",
            )
    elif len(records) == 1:
        record = records[0]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"В run {run_id} {len(records)} записей; "
                "необходимо передать entity_id для выбора конкретной"
            ),
        )

    bids_id = record.get("bids_id", "unknown")
    sub, ses = bids_id.split("_", 1) if "_" in bids_id else (bids_id, "ses-001")

    # Собираем пути к файлам
    preprocessed_dir = output_path / "preprocessed" / sub / ses / "anat"
    segmentation_dir = output_path / "segmentation" / sub / ses / "anat"

    # Preprocessed изображения (атласное пространство)
    image_paths = sorted([str(p) for p in preprocessed_dir.glob("*.nii.gz")]) if preprocessed_dir.exists() else []

    # Маски из БД (mask_versions) — единственный источник правды
    from mask_service import get_mask_history
    entity_id = record.get("kappa_entity_id", "")
    mask_history = get_mask_history(entity_id) if entity_id else []

    ai_masks = []
    expert_masks = []
    for mv in mask_history:
        fp = mv.get("file_path", "")
        if not fp or not Path(fp).exists():
            continue
        if mv["source"] == "ai":
            ai_masks.append(fp)
        else:
            expert_masks.append(fp)

    # Маска по умолчанию: выбранная в UI, или последняя экспертная, или ИИ
    default_mask = ""
    if selected_mask_version:
        # Ищем конкретную версию в mask_history
        for mv in mask_history:
            fp = mv.get("file_path", "")
            if mv["version"] == selected_mask_version and fp and Path(fp).exists():
                default_mask = fp
                break

    if not default_mask:
        if expert_masks:
            default_mask = expert_masks[-1]
        elif ai_masks:
            default_mask = ai_masks[-1]

    logger.info(
        "Slicer open: patient=%s, default_mask=%s, ai=%d, expert=%d",
        sub, Path(default_mask).name if default_mask else "none",
        len(ai_masks), len(expert_masks),
    )

    if not image_paths and not default_mask:
        raise HTTPException(status_code=404, detail="Файлы результатов не найдены")

    # Отправляем запрос агенту
    payload = {
        "image_paths": image_paths,
        "mask_path": default_mask,
        "ai_masks": ai_masks,
        "expert_masks": expert_masks,
        "patient_id": sub,
        "session_id": ses,
        # Контекст для обратной отправки маски
        "entity_id": entity_id,
        "dataset_id": record.get("kappa_dataset_id", 0),
        "run_id": run_id,
        "segmentation_dir": str(segmentation_dir),
        "kappa_session_id": session_id or "",
        "lesion_type": slicer_lesion_type,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{SLICER_AGENT_URL}/open", json=payload)

        if response.is_success:
            result = response.json()
            logger.info("Slicer opened for %s: %s", bids_id, result)
            return result
        else:
            detail = response.json().get("detail", "Ошибка агента")
            raise HTTPException(status_code=response.status_code, detail=detail)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Slicer Agent недоступен. Убедитесь, что slicer_agent.py запущен.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Slicer Agent не ответил вовремя",
        )


class ValidationActionRequest(PydanticBaseModel):
    entity_id: str
    dataset_id: int
    session_id: str
    action: str  # 'confirm' / 'reject' / 'revoke'
    comment: Optional[str] = None


@app.post("/api/validation/action")
async def validation_action(request: ValidationActionRequest):
    """Записать действие валидации (confirm/reject/revoke) и обновить статус в Каппе."""
    from kappa_auth import get_session
    from kappa_client import update_entity_status
    from validation_service import record_action, get_current_votes

    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    if request.action not in ("confirm", "reject", "revoke"):
        raise HTTPException(status_code=400, detail="Недопустимое действие")

    # Записываем действие в локальную БД
    user_name = None
    first_name = session.get("first_name") or ""
    last_name = session.get("last_name") or ""
    if first_name or last_name:
        user_name = f"{first_name} {last_name}".strip()

    record_action(
        entity_id=request.entity_id,
        dataset_id=request.dataset_id,
        user_id=session["user_id"],
        user_name=user_name,
        action=request.action,
        comment=request.comment,
    )

    # Обновляем статус в Каппе на основе новых голосов
    votes = get_current_votes(request.entity_id)
    new_status = _compute_entity_status(votes)

    await update_entity_status(
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        dataset_id=request.dataset_id,
        entity_id=request.entity_id,
        status=new_status,
    )

    return {
        "success": True,
        "votes": votes,
        "kappa_status": new_status,
    }


def _compute_entity_status(votes: Dict[str, Any]) -> int:
    """
    Вычислить статус Каппы на основе голосов.
    3 = Labeled (нет голосов)
    4 = Under Verification (1+ голос, но меньше 2 confirms)
    5 = Verified (2+ подтверждения)
    """
    confirms = votes.get("confirms_count", 0)
    total = votes.get("total_votes", 0)

    if confirms >= 2:
        return 5  # Verified
    if total >= 1:
        return 4  # Under Verification
    return 3  # Labeled


@app.get("/api/validation/entity/{entity_id}")
async def get_entity_validation(entity_id: str, session_id: str):
    """Получить текущее состояние валидации сущности."""
    from kappa_auth import get_session
    from validation_service import get_current_votes, get_user_current_vote, get_entity_history

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    votes = get_current_votes(entity_id)
    my_vote = get_user_current_vote(entity_id, session["user_id"])
    history = get_entity_history(entity_id)

    return {
        "votes": votes,
        "my_vote": my_vote,
        "history": history,
    }


@app.get("/api/validation/entities-for-run/{run_id}")
async def get_entities_for_run(run_id: str):
    """
    Получить список сущностей Каппы, связанных с запуском пайплайна.
    Нужен для отображения кнопок валидации в ProgressMonitor и PipelineHistory.
    """
    from patient_registry import find_by_run_id

    records = find_by_run_id(run_id)
    entities = [
        {
            "entity_id": r["kappa_entity_id"],
            "dataset_id": r["kappa_dataset_id"],
            "bids_id": r["bids_id"],
            "study_hash": r["study_hash"],
        }
        for r in records
        if r.get("kappa_entity_id") and r.get("kappa_dataset_id")
    ]

    return {"entities": entities, "total": len(entities)}
        
# ============================================
# KAPPA AUTH ENDPOINTS
# ============================================

from kappa_auth import kappa_login, get_session, delete_session
from pydantic import BaseModel as PydanticBaseModel


class KappaLoginRequest(PydanticBaseModel):
    login_id: str
    passwd: str


@app.post("/api/kappa/login")
async def kappa_login_endpoint(request: KappaLoginRequest):
    """Авторизация в Kappa"""
    result = await kappa_login(request.login_id, request.passwd)
    if not result:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль Kappa")
    return result


@app.get("/api/kappa/me")
async def kappa_me(session_id: str):
    """Проверка текущей сессии Kappa"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия не найдена или истекла")
    return {
        "user_name": session["user_name"],
        "first_name": session["first_name"],
        "last_name": session["last_name"],
        "token_expiry": session["token_expiry"],
    }


@app.post("/api/kappa/logout")
async def kappa_logout(session_id: str):
    """Выход из Kappa"""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"status": "ok"}

@app.get("/api/kappa/entities/{dataset_id}")
async def get_kappa_entities(dataset_id: int, session_id: str):
    """Список сущностей датасета Kappa"""
    from kappa_auth import get_session
    from kappa_client import get_dataset_entities

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    entities = await get_dataset_entities(
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        dataset_id=dataset_id,
    )

    if entities is None:
        return {"entities": [], "total": 0}

    return {"entities": entities, "total": len(entities)}


@app.get("/api/kappa/entity/{dataset_id}/{entity_id}")
async def get_kappa_entity_details(dataset_id: int, entity_id: str, session_id: str):
    """Детали одной сущности (с информацией о файлах)"""
    from kappa_auth import get_session
    from kappa_client import get_entity_details

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    details = await get_entity_details(
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        dataset_id=dataset_id,
        entity_id=entity_id,
    )

    if details is None:
        raise HTTPException(status_code=404, detail="Сущность не найдена")

    return details

from fastapi.responses import Response


@app.get("/api/kappa/file/{dataset_id}/{file_id}")
async def get_kappa_file(dataset_id: int, file_id: str, session_id: str):
    """Скачать файл сущности из Kappa"""
    from kappa_auth import get_session
    from kappa_client import download_entity_file

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Сессия Kappa не найдена")

    content = await download_entity_file(
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        dataset_id=dataset_id,
        file_id=file_id,
    )

    if content is None:
        raise HTTPException(status_code=404, detail="Файл не найден")

    return Response(content=content, media_type="application/gzip")

# ============================================
# СТАТИКА ФРОНТЕНДА (React production build)
# ============================================

from fastapi.staticfiles import StaticFiles

# Путь к собранному фронтенду
frontend_dist_path = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist_path.exists():
    # Монтируем React статику на корень
    # ВАЖНО: это должно быть ПОСЛЕДНИМ, чтобы не перекрывать API роуты
    app.mount("/", StaticFiles(directory=str(frontend_dist_path), html=True), name="frontend")
    logger.info(f"Фронтенд смонтирован из: {frontend_dist_path}")
else:
    logger.warning(f"Директория фронтенда не найдена: {frontend_dist_path}")

# ============================================
# ЗАПУСК СЕРВЕРА
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )