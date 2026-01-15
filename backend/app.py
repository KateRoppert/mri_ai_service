"""
Основное FastAPI приложение
"""

import asyncio
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
import logging
import subprocess
import uvicorn

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
    QualityMetrics
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

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
    update_pipeline_run(db, run_id, status="running", started_at=datetime.utcnow())

    # Запускаем мониторинг через WebSocket
    asyncio.create_task(pipeline_monitor.start_monitoring(run_id, output_path))
    
    # Запускаем pipeline
    process = pipeline_manager.start_pipeline(run_id, input_path, output_path)
    
    if not process:
        # Ошибка запуска
        update_pipeline_run(
            db,
            run_id,
            status="failed",
            error_message="Не удалось запустить pipeline",
            completed_at=datetime.utcnow()
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
            
            # Получаем отчёт о качестве
            quality_report = pipeline_manager.get_quality_report(output_path)
            quality_score = None
            quality_category = None
            
            if quality_report:
                quality_score = quality_report.get('quality_score')
                quality_category = quality_report.get('quality_category')
            
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

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="ok",
        timestamp=datetime.utcnow(),
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
            quality_score=run.quality_score,
            quality_category=run.quality_category,
            created_at=run.created_at,
            completed_at=run.completed_at,
            duration_seconds=(
                int((run.completed_at - run.created_at).total_seconds())
                if run.completed_at else None
            )
        )
        for run in runs
    ]
    
    return PipelineHistoryResponse(
        total=total,
        runs=history_items
    )


@app.get("/api/quality-report/{run_id}", response_model=QualityReportResponse)
async def get_quality_report(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Получить отчёт о качестве изображения
    
    Args:
        run_id: ID запуска
        db: Сессия БД
        
    Returns:
        Отчёт о качестве
    """
    # Проверяем существование запуска
    run = get_pipeline_run(db, run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail="Запуск не найден")
    
    # Получаем отчёт о качестве
    quality_data = pipeline_manager.get_quality_report(run.output_path)
    
    if not quality_data:
        raise HTTPException(
            status_code=404,
            detail="Отчёт о качестве не найден. Возможно, 4-й этап ещё не завершён."
        )
    
    # Формируем ответ
    return QualityReportResponse(
        file=quality_data['file'],
        patient_id=quality_data['patient_id'],
        modality=quality_data['modality'],
        quality_score=quality_data['quality_score'],
        quality_category=quality_data['quality_category'],
        quality_category_ru=quality_data['quality_category_ru'],
        metrics=QualityMetrics(**quality_data['metrics'])
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