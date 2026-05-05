#!/usr/bin/env python3
"""
Slicer Agent — лёгкий HTTP-сервис на хост-машине.
Принимает команды на запуск 3D Slicer с данными пациента.

Запуск:
    pip install fastapi uvicorn
    python slicer_agent.py

По умолчанию слушает localhost:8001.
"""
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("slicer_agent")

app = FastAPI(title="Slicer Agent", version="1.0.0")

# Разрешаем CORS — фронт будет обращаться с другого порта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

def find_slicer() -> Optional[str]:
    """Найти исполняемый файл 3D Slicer на хосте."""
    # Стандартные пути установки
    candidates = []

    system = platform.system()
    if system == "Linux":
        candidates = [
            "/home/ubuntu/Загрузки/Slicer-5.10.0-linux-amd64/Slicer",
            "/usr/local/bin/Slicer",
            "/opt/Slicer/Slicer",
            str(Path.home() / "Slicer" / "Slicer"),
            str(Path.home() / "opt" / "Slicer" / "Slicer"),
        ]
        # Поиск в home по паттерну Slicer-*
        home = Path.home()
        for d in sorted(home.glob("Slicer-*"), reverse=True):
            candidates.append(str(d / "Slicer"))
        for d in sorted(Path("/opt").glob("Slicer-*"), reverse=True):
            candidates.append(str(d / "Slicer"))
        # Поиск в ~/Загрузки (русская локаль)
        downloads_ru = home / "Загрузки"
        if downloads_ru.exists():
            for d in sorted(downloads_ru.glob("Slicer-*"), reverse=True):
                candidates.append(str(d / "Slicer"))
        # Поиск в ~/Downloads
        downloads_en = home / "Downloads"
        if downloads_en.exists():
            for d in sorted(downloads_en.glob("Slicer-*"), reverse=True):
                candidates.append(str(d / "Slicer"))
    elif system == "Darwin":
        candidates = [
            "/Applications/Slicer.app/Contents/MacOS/Slicer",
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Program Files\Slicer\Slicer.exe",
            r"C:\Program Files (x86)\Slicer\Slicer.exe",
        ]

    # Также ищем в PATH
    slicer_in_path = shutil.which("Slicer")
    if slicer_in_path:
        candidates.insert(0, slicer_in_path)

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.info("Found Slicer: %s", path)
            return path

    return None


SLICER_PATH = find_slicer()
LOADER_SCRIPT = Path(__file__).parent / "load_in_slicer.py"


# ============================================
# МОДЕЛИ
# ============================================

class SlicerOpenRequest(BaseModel):
    """Запрос на открытие данных в Slicer."""
    # Пути к файлам на локальной файловой системе
    image_paths: list[str]         # Preprocessed MRI (.nii.gz)
    mask_path: str                 # Маска сегментации (.nii.gz)
    native_image_paths: list[str] = []   # Нативные изображения (опционально)
    native_mask_path: Optional[str] = None  # Нативная маска (опционально)
    patient_id: str = ""
    session_id: str = ""
    # Или URL-ы для скачивания (если файлы на сервере)
    image_urls: list[str] = []
    mask_url: str = ""


class SlicerStatusResponse(BaseModel):
    slicer_found: bool
    slicer_path: Optional[str]
    agent_version: str = "1.0.0"


# ============================================
# ЭНДПОИНТЫ
# ============================================

@app.get("/health")
async def health():
    """Проверка доступности агента."""
    return {
        "status": "ok",
        "slicer_found": SLICER_PATH is not None,
        "slicer_path": SLICER_PATH,
    }


@app.get("/status")
async def status():
    """Подробный статус агента."""
    return SlicerStatusResponse(
        slicer_found=SLICER_PATH is not None,
        slicer_path=SLICER_PATH,
    )


@app.post("/open")
async def open_in_slicer(request: SlicerOpenRequest):
    """
    Запустить 3D Slicer и загрузить данные пациента.
    Slicer откроется с Segment Editor для редактирования маски.
    """
    global SLICER_PATH

    if not SLICER_PATH:
        # Повторная попытка найти Slicer
        SLICER_PATH = find_slicer()
        if not SLICER_PATH:
            raise HTTPException(
                status_code=503,
                detail="3D Slicer не найден. Убедитесь, что Slicer установлен.",
            )

    if not LOADER_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Скрипт загрузки не найден: {LOADER_SCRIPT}",
        )

    # Проверяем, что файлы существуют (для локальных путей)
    all_paths = request.image_paths + [request.mask_path]
    if request.native_mask_path:
        all_paths.append(request.native_mask_path)
    all_paths.extend(request.native_image_paths)

    missing = [p for p in all_paths if p and not Path(p).exists()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Файлы не найдены: {missing}",
        )

    # Формируем параметры для Slicer-скрипта
    params = {
        "image_paths": request.image_paths,
        "mask_path": request.mask_path,
        "native_image_paths": request.native_image_paths,
        "native_mask_path": request.native_mask_path,
        "patient_id": request.patient_id,
        "session_id": request.session_id,
    }

    # Сохраняем параметры во временный файл
    params_file = Path(tempfile.mktemp(suffix=".json", prefix="slicer_params_"))
    params_file.write_text(json.dumps(params, ensure_ascii=False))

    # Запускаем Slicer БЕЗ аргументов (--python-script крашит Slicer 5.10).
    # Вместо этого записываем скрипт загрузки в ~/.slicerrc.py,
    # который Slicer выполняет автоматически при старте.
    
    # Генерируем .slicerrc.py с одноразовой загрузкой
    slicerrc_path = Path.home() / ".slicerrc.py"
    
    # Сохраняем оригинальный .slicerrc.py (если есть)
    slicerrc_backup = Path.home() / ".slicerrc.py.backup"
    if slicerrc_path.exists() and not slicerrc_backup.exists():
        shutil.copy2(slicerrc_path, slicerrc_backup)
    
    # Записываем одноразовый .slicerrc.py
    slicerrc_content = f'''# Auto-generated by Slicer Agent — will self-clean after execution
import os, json

_params_file = r"{params_file}"

def _load_patient_data():
    try:
        if not os.path.exists(_params_file):
            print(f"Slicer Agent: params file not found: {{_params_file}}")
            return
        
        with open(_params_file, "r") as f:
            params = json.load(f)
        
        image_paths = params.get("image_paths", [])
        mask_path = params.get("mask_path", "")
        patient_id = params.get("patient_id", "")
        
        print(f"Loading patient data: {{patient_id}}")
        
        # Загружаем изображения
        for img_path in image_paths:
            if os.path.exists(img_path):
                node = slicer.util.loadVolume(img_path)
                if node:
                    basename = os.path.basename(img_path).replace(".nii.gz", "")
                    parts = basename.split("_")
                    modality = parts[-1] if parts else basename
                    node.SetName(f"{{patient_id}}_{{modality}}")
                    print(f"  Loaded: {{img_path}}")
        
        # Загружаем маску как сегментацию
        if mask_path and os.path.exists(mask_path):
            print(f"  Loading mask: {{mask_path}}")
            labelmap_node = slicer.util.loadLabelVolume(mask_path)
            if labelmap_node:
                seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                seg_node.SetName(f"{{patient_id}}_segmentation")
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmap_node, seg_node
                )
                slicer.mrmlScene.RemoveNode(labelmap_node)
                
                # Именуем сегменты
                segment_names = {{1: "NCR", 2: "ED", 3: "NET", 4: "ET"}}
                segmentation = seg_node.GetSegmentation()
                for i in range(segmentation.GetNumberOfSegments()):
                    segment = segmentation.GetNthSegment(i)
                    if (i + 1) in segment_names:
                        segment.SetName(segment_names[i + 1])
                
                # Открываем Segment Editor
                slicer.util.selectModule("SegmentEditor")
                print("  Segment Editor opened")
        
        print(f"Patient {{patient_id}} loaded successfully!")
        
        # Удаляем временный файл параметров
        try:
            os.remove(_params_file)
        except Exception:
            pass
    except Exception as e:
        print(f"Slicer Agent error: {{e}}")
        import traceback
        traceback.print_exc()

# Восстанавливаем оригинальный .slicerrc.py
_backup = r"{slicerrc_backup}"
_slicerrc = r"{slicerrc_path}"
try:
    if os.path.exists(_backup):
        import shutil
        shutil.copy2(_backup, _slicerrc)
        os.remove(_backup)
    else:
        os.remove(_slicerrc)
except Exception:
    pass

# Запускаем загрузку после инициализации Slicer
import qt
qt.QTimer.singleShot(2000, _load_patient_data)
'''
    
    slicerrc_path.write_text(slicerrc_content, encoding="utf-8")
    logger.info("Generated .slicerrc.py for patient %s", request.patient_id)

    cmd = [SLICER_PATH]

    logger.info("Launching Slicer: %s", " ".join(cmd))

    try:
        # Передаём DISPLAY для GUI + путь к параметрам через env
        env = os.environ.copy()
        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"
        env["SLICER_PARAMS_FILE"] = str(params_file)  # Стандартный X11 display

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        logger.info("Slicer launched, PID: %d", process.pid)

        return {
            "success": True,
            "message": f"3D Slicer запущен (PID: {process.pid})",
            "pid": process.pid,
            "patient_id": request.patient_id,
        }
    except Exception as e:
        logger.error("Failed to launch Slicer: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось запустить Slicer: {e}",
        )


# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    if not SLICER_PATH:
        logger.warning("3D Slicer not found! Agent will start but /open will fail.")
        logger.warning("Install Slicer and restart the agent.")
    else:
        logger.info("Slicer found: %s", SLICER_PATH)

    logger.info("Starting Slicer Agent on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")