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
    image_paths: list[str]
    mask_path: str
    ai_masks: list[str] = []
    expert_masks: list[str] = []
    patient_id: str = ""
    session_id: str = ""
    # Контекст для обратной отправки маски
    entity_id: str = ""
    dataset_id: int = 0
    run_id: str = ""
    segmentation_dir: str = ""
    kappa_session_id: str = ""


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
    all_paths.extend(request.ai_masks)
    all_paths.extend(request.expert_masks)

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
        "ai_masks": request.ai_masks,
        "expert_masks": request.expert_masks,
        "patient_id": request.patient_id,
        "session_id": request.session_id,
        # Контекст для кнопки «Сохранить и отправить»
        "entity_id": request.entity_id,
        "dataset_id": request.dataset_id,
        "run_id": request.run_id,
        "segmentation_dir": request.segmentation_dir,
    }

    # Сохраняем параметры во временный файл
    params_file = Path(tempfile.mktemp(suffix=".json", prefix="slicer_params_"))
    params_file.write_text(json.dumps(params, ensure_ascii=False))

    # Сохраняем session_id для /upload-mask
    if request.kappa_session_id:
        session_file = Path(tempfile.gettempdir()) / "slicer_kappa_session.txt"
        session_file.write_text(request.kappa_session_id)

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
        ai_masks = params.get("ai_masks", [])
        expert_masks = params.get("expert_masks", [])
        patient_id = params.get("patient_id", "")
        
        print(f"Loading patient data: {{patient_id}}")
        
        # === 1. Загружаем preprocessed изображения ===
        volume_nodes = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                node = slicer.util.loadVolume(img_path)
                if node:
                    basename = os.path.basename(img_path).replace(".nii.gz", "")
                    parts = basename.split("_")
                    modality = parts[-1] if parts else basename
                    node.SetName(f"{{patient_id}}_{{modality}}")
                    volume_nodes.append(node)
                    print(f"  [volume] {{modality}}")
        
        ref_volume = volume_nodes[0] if volume_nodes else None
        if ref_volume:
            import vtk as _vtk
            _m = _vtk.vtkMatrix4x4()
            ref_volume.GetIJKToRASMatrix(_m)
            _ox = round(_m.GetElement(0,3),1)
            _oy = round(_m.GetElement(1,3),1)
            _oz = round(_m.GetElement(2,3),1)
            _dims = ref_volume.GetImageData().GetDimensions()
            print("  [ref_volume] origin=(%s, %s, %s) dims=%s" % (_ox, _oy, _oz, str(_dims)))
        
        # === 2. Вспомогательная функция загрузки маски ===
        def _load_mask(mask_file, seg_name):
            if not mask_file or not os.path.exists(mask_file):
                return None
            labelmap_node = slicer.util.loadLabelVolume(mask_file)
            if not labelmap_node:
                return None
            # Диагностика: проверяем геометрию labelmap
            import vtk as _vtk
            _lm = _vtk.vtkMatrix4x4()
            labelmap_node.GetIJKToRASMatrix(_lm)
            _lox = round(_lm.GetElement(0,3),1)
            _loy = round(_lm.GetElement(1,3),1)
            _loz = round(_lm.GetElement(2,3),1)
            _ldims = labelmap_node.GetImageData().GetDimensions()
            print("  [labelmap] %s: origin=(%s, %s, %s) dims=%s" % (seg_name, _lox, _loy, _loz, str(_ldims)))
            seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            seg_node.SetName(seg_name)
            # Импортируем labelmap — передаём ref_volume для корректного ресэмплинга
            if ref_volume:
                seg_node.SetReferenceImageGeometryParameterFromVolumeNode(ref_volume)
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmap_node, seg_node, ""
                )
            else:
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
            return seg_node
        
        # === 3. Загружаем маску по умолчанию (видимая) ===
        main_seg_node = None
        if mask_path:
            mask_bn = os.path.basename(mask_path).replace(".nii.gz", "")
            if "_segmask_v" in mask_bn:
                ver = mask_bn.split("_segmask_v")[1]
                label = f"expert v{{ver}}"
            else:
                label = "ai"
            main_seg_node = _load_mask(mask_path, f"{{patient_id}}_segmentation ({{label}})")
            if main_seg_node:
                print(f"  [mask:default] {{label}}")
        
        # === 4. Загружаем остальные маски (скрытые) ===
        for m in ai_masks:
            if m != mask_path and os.path.exists(m):
                node = _load_mask(m, f"{{patient_id}}_segmentation (ai)")
                if node:
                    node.SetDisplayVisibility(False)
                    print(f"  [mask:ai] hidden")
        
        for m in expert_masks:
            if m != mask_path and os.path.exists(m):
                bn = os.path.basename(m)
                ver = "expert"
                if "_segmask_v" in bn:
                    ver = f"expert v{{bn.split('_segmask_v')[1].replace('.nii.gz', '')}}"
                node = _load_mask(m, f"{{patient_id}}_segmentation ({{ver}})")
                if node:
                    node.SetDisplayVisibility(False)
                    print(f"  [mask:{{ver}}] hidden")
        
        # === 5. Настраиваем визуализацию ===
        if ref_volume:
            layout_manager = slicer.app.layoutManager()
            for view_name in ["Red", "Yellow", "Green"]:
                slice_widget = layout_manager.sliceWidget(view_name)
                if slice_widget:
                    slice_logic = slice_widget.sliceLogic()
                    composite = slice_logic.GetSliceCompositeNode()
                    composite.SetBackgroundVolumeID(ref_volume.GetID())
                    slice_logic.FitSliceToAll()
        
        # Открываем Segment Editor
        if main_seg_node and ref_volume:
            slicer.util.selectModule("SegmentEditor")
            editor_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentEditorNode")
            if not editor_node:
                editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
            editor_node.SetAndObserveSegmentationNode(main_seg_node)
            editor_node.SetAndObserveMasterVolumeNode(ref_volume)
            print("  Segment Editor opened")
        
        print(f"Patient {{patient_id}} loaded: {{len(volume_nodes)}} volumes, default mask={{'yes' if main_seg_node else 'no'}}")
        
        # === КНОПКА «СОХРАНИТЬ И ОТПРАВИТЬ» ===
        _add_upload_button(params, main_seg_node)
        
        # Удаляем временный файл параметров
        try:
            os.remove(_params_file)
        except Exception:
            pass
    except Exception as e:
        print(f"Slicer Agent error: {{e}}")
        import traceback
        traceback.print_exc()


def _add_upload_button(params, seg_node):
    """Добавить кнопку 'Сохранить и отправить' в toolbar Slicer."""
    import qt
    
    entity_id = params.get("entity_id", "")
    dataset_id = params.get("dataset_id", 0)
    run_id = params.get("run_id", "")
    segmentation_dir = params.get("segmentation_dir", "")
    patient_id = params.get("patient_id", "")
    
    if not entity_id or not run_id:
        print("  Warning: no entity_id/run_id, upload button disabled")
        return
    
    def _do_save_and_upload():
        """Экспортировать сегментацию и отправить на сервер."""
        try:
            import urllib.request, urllib.parse, urllib.error, tempfile
            
            # Находим сегментацию
            seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
            if not seg_nodes:
                qt.QMessageBox.warning(None, "Ошибка", "Сегментация не найдена")
                return
            
            active_seg = seg_nodes[0]
            
            # Экспортируем сегментацию в labelmap
            labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            
            # Slicer 5.10: ExportAllSegmentsToLabelmapNode(segNode, labelmapNode)
            # без reference volume — использует геометрию сегментации
            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                active_seg, labelmap
            )
            
            # Сохраняем в файл
            save_dir = segmentation_dir if segmentation_dir else tempfile.gettempdir()
            
            # Проверяем права записи, если нет — используем /tmp
            try:
                os.makedirs(save_dir, exist_ok=True)
                test_file = os.path.join(save_dir, ".write_test")
                with open(test_file, "w") as tf:
                    tf.write("test")
                os.remove(test_file)
            except (PermissionError, OSError):
                print(f"  No write access to {{save_dir}}, using /tmp")
                save_dir = tempfile.gettempdir()
            
            save_path = os.path.join(save_dir, f"{{patient_id}}_edited_mask.nii.gz")
            
            slicer.util.saveNode(labelmap, save_path)
            slicer.mrmlScene.RemoveNode(labelmap)
            
            # Проверяем, что файл сохранился
            if not os.path.exists(save_path):
                qt.QMessageBox.critical(
                    None, "Ошибка",
                    f"Файл маски не сохранился:\\n{{save_path}}\\n\\n"
                    f"Папка: {{save_dir}}\\n"
                    f"Папка существует: {{os.path.exists(save_dir)}}"
                )
                return
            
            file_size = os.path.getsize(save_path)
            print(f"  Mask saved: {{save_path}} ({{file_size}} bytes)")
            
            # Отправляем на агент
            agent_url = "http://localhost:8001/upload-mask"
            query = urllib.parse.urlencode({{
                "file_path": save_path,
                "entity_id": "{request.entity_id}",
                "dataset_id": {request.dataset_id},
                "run_id": "{request.run_id}",
            }})
            url = f"{{agent_url}}?{{query}}"
            
            req = urllib.request.Request(url, method="POST", data=b"")
            try:
                resp = urllib.request.urlopen(req, timeout=60)
                result = resp.read().decode()
            except urllib.error.HTTPError as http_err:
                error_body = http_err.read().decode() if http_err.fp else ""
                raise Exception(f"HTTP Error {{http_err.code}}: {{error_body[:200]}}")
            
            print(f"  Upload result: {{result}}")
            qt.QMessageBox.information(
                None, "Готово",
                "Маска сохранена и отправлена на сервер!\\n"
                f"Файл: {{os.path.basename(save_path)}}"
            )
        except Exception as e:
            print(f"  Upload error: {{e}}")
            import traceback
            traceback.print_exc()
            qt.QMessageBox.critical(
                None, "Ошибка",
                f"Не удалось отправить маску:\\n{{e}}"
            )
    
    # Создаём toolbar с кнопкой
    toolbar = qt.QToolBar("MRI AI Service")
    toolbar.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
    
    upload_action = qt.QAction("Сохранить и отправить маску", toolbar)
    upload_action.setToolTip("Экспортировать сегментацию и отправить на сервер MRI AI Service")
    upload_action.connect("triggered()", _do_save_and_upload)
    toolbar.addAction(upload_action)
    
    # Добавляем toolbar в главное окно
    main_window = slicer.util.mainWindow()
    main_window.addToolBar(toolbar)
    
    print("  Upload button added to Slicer toolbar")


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
# ПРИЁМ МАСКИ ОТ SLICER → ПЕРЕСЫЛКА НА БЭКЕНД
# ============================================

BACKEND_URL = "http://localhost:8000"


@app.post("/upload-mask")
async def upload_mask_from_slicer(
    file_path: str,
    entity_id: str,
    dataset_id: int,
    run_id: str,
    session_id: str = "",
):
    """
    Принимает путь к файлу маски от Slicer, пересылает на бэкенд.
    Вызывается из Slicer Python через HTTP.
    """
    mask_file = Path(file_path)
    if not mask_file.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {file_path}")

    import httpx as httpx_client

    # Если session_id не передан — берём из localStorage аналога
    if not session_id:
        # Пробуем прочитать из файла, который сохраняет агент при /open
        session_file = Path(tempfile.gettempdir()) / "slicer_kappa_session.txt"
        if session_file.exists():
            session_id = session_file.read_text().strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id не указан. Откройте сервис в браузере и залогиньтесь.",
        )

    logger.info(
        "Uploading mask from Slicer: file=%s, entity=%s, run=%s",
        file_path, entity_id, run_id,
    )

    # Пересылаем файл на бэкенд
    try:
        with open(mask_file, "rb") as f:
            files = [("file", (mask_file.name, f, "application/gzip"))]
            data = {
                "entity_id": entity_id,
                "dataset_id": str(dataset_id),
                "session_id": session_id,
                "run_id": run_id,
            }

            async with httpx_client.AsyncClient(
                timeout=httpx_client.Timeout(30.0, read=120.0, write=120.0),
            ) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/validation/upload-mask",
                    data=data,
                    files=files,
                )

        if response.is_success:
            result = response.json()
            logger.info("Mask uploaded successfully: %s", result.get("message"))
            return result
        else:
            detail = response.text[:300]
            logger.error("Backend rejected mask: %s", detail)
            raise HTTPException(status_code=response.status_code, detail=detail)
    except httpx_client.ConnectError:
        raise HTTPException(status_code=503, detail="Бэкенд недоступен")
    except Exception as e:
        logger.error("Error uploading mask: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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