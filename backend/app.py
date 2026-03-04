"""
Основное FastAPI приложение
"""

import asyncio
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import asynccontextmanager
import logging
import subprocess
import uvicorn
import os
from pydantic import BaseModel
from typing import Optional, List

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
    VolumeReportListResponse 
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
    db: Session
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
    process = pipeline_manager.start_pipeline(run_id, input_path, output_path)
    
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
        output_path=output_path
    )
    
    # Запускаем pipeline в фоновой задаче
    background_tasks.add_task(
        run_pipeline_background,
        run.run_id,
        run.input_path,
        run.output_path,
        db
    )

    # Запускаем мониторинг (из асинхронного контекста)
    asyncio.create_task(pipeline_monitor.start_monitoring(run.run_id, run.output_path))

    logger.info(f"Pipeline запущен с run_id: {run.run_id}")
    
    return PipelineStartResponse(
        run_id=run.run_id,
        status=PipelineStatus.PENDING,
        message="Pipeline запущен и будет выполнен в фоновом режиме",
        created_at=run.created_at
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
        error=run.error_message
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
            )
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
    
    # Проверяем что 4-й этап завершён
    if run.current_stage < 4:
        logger.warning(f"Этап 4 ещё не завершён для run_id: {run_id}")
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
    
    # Проверяем что 6-й этап завершён
    if run.current_stage < 6:
        raise HTTPException(
            status_code=400,
            detail="Segmentation stage not yet completed"
        )
    
    # Проверяем допустимость file_type
    if file_type not in ["preprocessed", "segmentation"]:
        raise HTTPException(status_code=400, detail="Invalid file_type. Must be 'preprocessed' or 'segmentation'")
    
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
    
    # Проверяем что 6-й этап завершён
    if run.current_stage < 6:
        raise HTTPException(
            status_code=400,
            detail="Segmentation stage not yet completed"
        )
    
    # Ищем файлы в preprocessed и segmentation директориях
    output_base = Path(run.output_path)
    preprocessed_dir = output_base / "preprocessed"
    segmentation_dir = output_base / "segmentation"
    
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
                    
                    nifti_files.append(NIfTIFile(
                        filename=prep_filename,
                        mask_filename=mask_filename,
                        patient_id=patient_id,
                        session_id=session_id,
                        modality=modality.upper(),
                        image_url=f"/api/nifti/{run_id}/preprocessed/{prep_filename}",
                        mask_url=f"/api/nifti/{run_id}/segmentation/{mask_filename}",
                    ))
                    
                    logger.info(f"Добавлен файл: {prep_filename} с маской {mask_filename}")
        
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
    
    if run.current_stage < 6:
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