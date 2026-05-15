#!/usr/bin/env python3
import glob
import os
import io
import traceback
import yaml
#from requests_toolbelt import MultipartEncoder
from quart import make_response, jsonify, abort, Quart, flash, request, redirect, url_for, send_file, render_template, send_from_directory
from werkzeug.utils import secure_filename
import random
import secrets 
import string
#import predict_multi
from dataclasses import dataclass
from typing import List
import subprocess
import torch
import shutil
import aiofiles 
import asyncio 
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy 
import queue
import threading
from functools import partial
import pynvml 
from common.gpu_monitor import GPUMonitor

# Enable Ampere optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("DEBUG: Ampere optimizations enabled (TF32, cuDNN benchmark)")

# GPU Pool для распределения задач
available_gpus = queue.Queue()

def get_available_gpu():
    """Получить свободный GPU из пула (блокирующий вызов)."""
    gpu_id = available_gpus.get()
    print(f"DEBUG: Allocated GPU {gpu_id}")
    return gpu_id

def release_gpu(gpu_id):
    """Вернуть GPU в пул."""
    available_gpus.put(gpu_id)
    print(f"DEBUG: Released GPU {gpu_id}")

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print(f"DEBUG: Using PyTorch {torch.__version__}")
print(f"DEBUG: Using NumPy {numpy.__version__}")

if torch.cuda.is_available():
    print(f"DEBUG: CUDA {torch.version.cuda} available with {torch.cuda.device_count()} GPUs")
    # Проверим совместимость GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if props.major < 7:
            print(f"WARNING: GPU {i} ({props.name}) has compute capability {props.major}.{props.minor} which may not be fully supported")
else:
    print("DEBUG: Using CPU mode")

# Для старых моделей nnUNet - патчим torch.load для совместимости с PyTorch 2.6+
try:
    # Добавляем numpy.core.multiarray.scalar в whitelist
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
        print("DEBUG: Added numpy.core.multiarray.scalar to safe globals")
    
    # КРИТИЧЕСКОЕ: Monkey patch torch.load для обратной совместимости
    # Старые модели nnUNet v1 несовместимы с weights_only=True
    _original_torch_load = torch.load
    
    def patched_torch_load(*args, **kwargs):
        """Патч torch.load для загрузки старых моделей nnUNet"""
        # Принудительно устанавливаем weights_only=False для совместимости
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    torch.load = patched_torch_load
    print("DEBUG: Patched torch.load to use weights_only=False")
    
except Exception as e:
    print(f"WARNING: Could not patch torch.load: {e}")
    print("Model loading may fail with PyTorch 2.6+")

# --- Server Configuration Loading ---
def load_server_config():
    """
    Loads server configuration from server_config.yaml or falls back to environment variables.
    
    Priority:
    1. YAML file specified in SERVER_CONFIG env var
    2. server_config.yaml in the same directory as this script
    3. Environment variables (for backward compatibility)
    4. Hardcoded defaults
    """
    # Try to find config file
    config_path = os.getenv('SERVER_CONFIG')
    
    if not config_path:
        # Look for config in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'server_config.yaml')
    
    # Try to load YAML config
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            config = data.get('server', {})
            print(f"DEBUG: Loaded configuration from {config_path}")
            return config
        except Exception as e:
            print(f"WARNING: Failed to load config from {config_path}: {e}")
            print("DEBUG: Falling back to environment variables")
    
    # Fallback to environment variables or defaults
    print("DEBUG: Using environment variables or defaults")
    return {
        'nnunet_path': os.getenv('NNUNET_PATH', '/media/storage2/pavlovskiy/Slicer_plugin/server/nnUNet'),
        'nnunet_models': os.getenv('NNUNET_MODELS', '/media/storage2/pavlovskiy/Slicer_plugin/server/nnUNetv1_data'),
        'task_name': os.getenv('TASK_NAME', 'Task115_AllData5foldsMeta'),
        'gpu_ids': [int(x) for x in os.getenv('GPU_IDS', '0,1,2').split(',')],
        'host': os.getenv('SERVER_HOST', '0.0.0.0'),
        'port': int(os.getenv('SERVER_PORT', '5000')),
        'debug': os.getenv('SERVER_DEBUG', 'True').lower() == 'true',
        'max_parallel_tasks': int(os.getenv('MAX_PARALLEL_TASKS', '3')),
        'task_timeout': int(os.getenv('TASK_TIMEOUT', '600')),
    }

# Load configuration at startup
server_config = load_server_config()

# Initialize GPU pool from config
for gpu_id in server_config.get('gpu_ids', [0, 1, 2]):
    available_gpus.put(gpu_id)
    print(f"DEBUG: Added GPU {gpu_id} to pool")

# Directories - can be configured or use defaults
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if 'input_dir' in server_config and server_config['input_dir']:
    INPUT_DIR = os.path.join(BASE_DIR, server_config['input_dir'])
else:
    INPUT_DIR = os.path.join(BASE_DIR, "data", "input")

if 'output_dir' in server_config and server_config['output_dir']:
    OUTPUT_DIR = os.path.join(BASE_DIR, server_config['output_dir'])
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

if 'tmp_dir' in server_config and server_config['tmp_dir']:
    TMP_DIR = os.path.join(BASE_DIR, server_config['tmp_dir'])
else:
    TMP_DIR = os.path.join(BASE_DIR, "data", "tmp")

print(f"DEBUG: Using directories:")
print(f"  INPUT: {INPUT_DIR}")
print(f"  OUTPUT: {OUTPUT_DIR}")
print(f"  TMP: {TMP_DIR}")

segmentation_semaphore = asyncio.Semaphore(server_config.get('max_parallel_tasks', 3))
segmentation_queue = asyncio.Queue()  # Очередь задач
is_processing = False  # Флаг занятости

# Добавляем путь к nnUNet в Python path
import sys
nnunet_path = server_config.get('nnunet_path', '/media/storage2/pavlovskiy/Slicer_plugin/server/nnUNet')
if nnunet_path not in sys.path:
    sys.path.insert(0, nnunet_path)
    print(f"DEBUG: Added nnUNet path: {nnunet_path}")

# Базовая директория для nnUNet v1
nnunet_v1_base = server_config.get('nnunet_models', '/media/storage2/pavlovskiy/Slicer_plugin/server/nnUNetv1_data')

# Установим переменные окружения для nnUNet v1
os.environ['nnUNet_raw_data_base'] = os.path.join(nnunet_v1_base, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(nnunet_v1_base, 'nnUNet_preprocessed') 
os.environ['RESULTS_FOLDER'] = nnunet_v1_base

print(f"DEBUG: nnUNet v1 environment variables:")
print(f"  nnUNet_raw_data_base: {os.environ.get('nnUNet_raw_data_base')}")
print(f"  nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed')}")
print(f"  RESULTS_FOLDER: {os.environ.get('RESULTS_FOLDER')}")

# Создаем директории если их нет
for env_var in ['nnUNet_raw_data_base', 'nnUNet_preprocessed']:
    path = os.environ.get(env_var)
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"DEBUG: Created directory: {path}")

# Проверяем что RESULTS_FOLDER существует и содержит модели
results_path = os.environ.get('RESULTS_FOLDER')
task_name = server_config.get('task_name', 'Task115_AllData5foldsMeta')
task_path = os.path.join(results_path, 'nnUNet', '3d_fullres', task_name, 'nnUNetTrainerV2__nnUNetPlansv2.1')

if os.path.exists(task_path):
    print(f"DEBUG: Found Task115 models at: {task_path}")
    folds = [f for f in os.listdir(task_path) if f.startswith('fold_')]
    print(f"DEBUG: Available folds: {folds}")
else:
    print(f"ERROR: Task115 models not found at: {task_path}")

# Оптимальный пул потоков для асинхронных задач
executor = ThreadPoolExecutor(max_workers=3)

# Словарь для отслеживания статусов задач
task_status = {}

app = Quart(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1000 * 1000
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

@dataclass
class Model:
    type: str
    version: str
    description: str
    name: str
    path: str
    labels: List[str]

MODELS = {
    "Unet": Model("segmentation", "1", "NSU Unet mode ", "Unet", "n/a", ["Necr", "EnTu","NenTu", "Ed"]),
    "Unet+Folds": Model("segmentation", "1", "NSU Unet mode with folds", "Unet+Folds", "n/a", ["Necr", "EnTu","NenTu", "Ed"]),
    "Unet+Folds+TTA": Model("segmentation", "1", "NSU Unet mode with folds+tta", "Unet+Folds+TTA", "n/a", ["Necr", "EnTu","NenTu", "Ed"]),
    "Unet_ISBI": Model("segmentation", "1", "NSU Unet mode (Multiple sclerosis, ISBI)","Unet_ISBI" , "n/a", ["lesion"]),
    "Unet+TTA_ISBI": Model("segmentation", "1", "NSU Unet mode (Multiple sclerosis, ISBI)","Unet+TTA_ISBI" , "n/a", ["lesion"]),
}

@app.route('/v1/models', methods=['GET'])
async def get_models():
    return jsonify(list(MODELS.values()))

@app.route('/v1/info', methods=['GET'])
async def get_server_info():
    """Returns server configuration and status information."""
    return jsonify({
        "status": "ready",
        "version": "1.0",
        "models_location": os.environ.get('RESULTS_FOLDER'),
        "task_name": server_config.get('task_name', 'Unknown'),
        "gpu_ids": server_config.get('gpu_ids', []),
        "available_models": list(MODELS.keys()),
        "max_parallel_tasks": server_config.get('max_parallel_tasks', 3)
    })

@app.route('/uploads/<name>')
async def download_file(name):
    return await send_from_directory(OUTPUT_DIR, name)

@app.route('/', methods=['GET', 'POST'])
async def upload_file():
    if request.method == 'POST':
        t1 = False
        t1c = False
        t2fl = False
        t2 = False
        t1_fname = None
        t1c_fname = None
        t2fl_fname = None
        t2_fname = None
        
        form = await request.form
        req_files = await request.files
        net_name = form['net']
        model_desc = MODELS[net_name]
        net_type = model_desc.type
        net_path = model_desc.path
        
        try:
            f_id = form['id']
            prefix = f_id + "_"
            f_id = form['client_id']
            if f_id != "":
                prefix = f_id + "_"
        except Exception:
            if f_id is None or f_id == "":
                return await render_template('index.html', error="Please specify patient ID! E.g. P000018")
        
        out_file = os.path.join(OUTPUT_DIR, prefix + "result.nii.gz")

        async def save_file(file, base_prefix):
            if file.filename.endswith(".nii.gz"):
                fname = base_prefix + prefix + secure_filename(file.filename)
                full_path = os.path.join(INPUT_DIR, fname)
                print("filename: ", fname)
                print(f"full path: {full_path}")
                
                async with aiofiles.open(full_path, 'wb') as f:
                    content = file.read()
                    if asyncio.iscoroutine(content):
                        content = await content
                    await f.write(content)
                return fname  # Возвращаем имя файла
            return None

        # Сохраняем файлы и получаем реальные имена
        if 'file_t1' in req_files:
            t1_fname = await save_file(req_files['file_t1'], "t1_")
            t1 = t1_fname is not None
            
        if 'file_t1c' in req_files:
            t1c_fname = await save_file(req_files['file_t1c'], "t1c_")
            t1c = t1c_fname is not None
            
        if 'file_t2fl' in req_files:
            t2fl_fname = await save_file(req_files['file_t2fl'], "t2fl_")
            t2fl = t2fl_fname is not None
            
        if 'file_t2' in req_files:
            t2_fname = await save_file(req_files['file_t2'], "t2_")
            t2 = t2_fname is not None

        print(f"DEBUG: Saved files:")
        print(f"  t1: {t1_fname} (exists: {t1})")
        print(f"  t1c: {t1c_fname} (exists: {t1c})")
        print(f"  t2fl: {t2fl_fname} (exists: {t2fl})")
        print(f"  t2: {t2_fname} (exists: {t2})")

        if t1 and t1c and t2fl and net_path:
            if model_desc.name == "Resnet":
                rs = "Resnet disabled"
            elif model_desc.name.startswith("Unet") and t2:
                files = {
                    "t1": os.path.join(INPUT_DIR, t1_fname), 
                    "t1c": os.path.join(INPUT_DIR, t1c_fname), 
                    "t2fl": os.path.join(INPUT_DIR, t2fl_fname), 
                    "t2": os.path.join(INPUT_DIR, t2_fname)
                }
                
                print(f"DEBUG: Files for UNet:")
                for k, v in files.items():
                    print(f"  {k}: {v}")
                    print(f"    exists: {os.path.exists(v)}")
                
                in_path, out_path = await prepare_files_for_unet(files, prefix)
                
                try:
                    import inference as nnUNet_inference
                    print("DEBUG: nnUNet_inference imported successfully")
                except Exception as e:
                    print(f"ERROR: Failed to import nnUNet_inference: {e}")
                    return await render_template('index.html', error=f"nnUNet import error: {e}")
                
                task_name = "Task115_AllData5foldsMeta"
                
                if model_desc.name.startswith("Unet+Folds+TTA"):
                    rs = nnUNet_inference.predict_for_api(in_path, out_path, True, (0, 1, 2, 3, 4), task_name)
                elif model_desc.name.startswith("Unet+Folds"):
                    rs = nnUNet_inference.predict_for_api(in_path, out_path, False, (0, 1, 2, 3, 4), task_name)
                elif model_desc.name.startswith("Unet"):
                    rs = nnUNet_inference.predict_for_api(in_path, out_path, False, (0,), task_name)
                    
                if rs == "":
                    try:
                        result_files = glob.glob(f'{out_path}/*.nii.gz')
                        if result_files:
                            shutil.move(result_files[0], out_file)
                            print(f"DEBUG: Moved result from {result_files[0]} to {out_file}")
                        else:
                            rs = "Cannot find response file from UNET model."
                    except Exception as e:
                        rs = f"Error moving result file: {str(e)}"
                else:
                    print(f"DEBUG: nnUNet_inference returned: {rs}")

                if rs == "":
                    # Проверяем что файл действительно существует
                    if os.path.exists(out_file):
                        print(f"DEBUG: Result file exists: {out_file}")
                        return redirect(url_for('download_file', name=prefix + "result.nii.gz"))
                    else:
                        print(f"ERROR: Result file not found: {out_file}")
                        return await render_template('index.html', error=f"Result file not found: {out_file}")
                else:
                    print(f"ERROR: Segmentation failed: {rs}")
                    return await render_template('index.html', error=rs)
                
    return await render_template('index.html', error="No input data")
                
@app.route('/v1/inference', methods=['POST'])
async def autosegmentation():
    try:
        form = await request.form
        req_files = await request.files
        net_name = request.args.get('net')
        
        if net_name not in MODELS:
            return jsonify({"error": f"Unknown model: {net_name}"}), 400
            
        model_desc = MODELS[net_name]
        prefix = str(request.args.get('client_id')) + "_"
        
        # ИСПРАВЛЕНИЕ 1: Используем правильные директории
        dir = INPUT_DIR  # Вместо "/data/input/"
        files = dict()
        
        from datetime import datetime
        str_dt = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        out_fname = prefix + str_dt + "_result.nii.gz"
        out_file = os.path.join(OUTPUT_DIR, out_fname)  # Вместо "/data/output/" + out_fname

        # ИСПРАВЛЕНИЕ 2: Правильная асинхронная функция сохранения
        async def save_file(file, fname):
            full_path = os.path.join(dir, fname)
            async with aiofiles.open(full_path, 'wb') as f:
                content = file.read()
                # Проверяем, является ли content корутиной
                if asyncio.iscoroutine(content):
                    content = await content
                await f.write(content)

        for file in req_files:
            ftype = file.split("_")[1]  # "file_t1" -> "t1", "file_t2fl" -> "t2fl"
            
            # Get original filename to extract extension
            original_file = req_files[file]
            original_filename = original_file.filename
            
            # Extract extension (should be .nii.gz)
            if original_filename.endswith('.nii.gz'):
                ext = '.nii.gz'
            elif original_filename.endswith('.nii'):
                ext = '.nii'
            else:
                ext = ''
            
            # Form filename: prefix + file_key + extension
            fname = secure_filename(prefix + file) + ext
            
            await save_file(req_files[file], fname)
            files[ftype] = fname

        if model_desc.name == "Resnet":
            rs = predict_multi.prediction_volume_resnet(
                model_desc.path, 
                os.path.join(dir, files["t1"]), 
                os.path.join(dir, files["t1c"]), 
                os.path.join(dir, files["t2fl"]), 
                out_file
            )
        elif model_desc.name.startswith("Unet"):
            files_with_dir = {
                "t1": os.path.join(dir, files["t1"]), 
                "t1c": os.path.join(dir, files["t1c"]), 
                "t2fl": os.path.join(dir, files["t2fl"]), 
                "t2": os.path.join(dir, files["t2"])
            }
            in_path, out_path = await prepare_files_for_unet(files_with_dir, prefix)
            
            import inference as nnUNet_inference
            task_name = "Task115_AllData5foldsMeta"
            
            if model_desc.name.startswith("Unet+Folds+TTA"):
                rs = nnUNet_inference.predict_for_api(in_path, out_path, True, (0, 1, 2, 3, 4), task_name)
            elif model_desc.name.startswith("Unet+Folds"):
                rs = nnUNet_inference.predict_for_api(in_path, out_path, False, (0, 1, 2, 3, 4), task_name)
            elif model_desc.name.startswith("Unet"):
                rs = nnUNet_inference.predict_for_api(in_path, out_path, False, (0,), task_name)
            else:
                rs = f"Unknown UNET model: {model_desc}"
                
            if rs == "":
                files = glob.glob(out_path + "/*.nii.gz")
                if len(files) == 1:
                    # Перемещаем результат в выходную директорию
                    shutil.move(files[0], out_file)
                else:
                    rs = f"Cannot find response file from UNET model: in dir {out_path} we can see only: {files}"
        else:
            rs = f"Unknown model: {model_desc}"
            
        if rs == "":
            # Проверяем, существует ли файл
            if not os.path.exists(out_file):
                return jsonify({"error": "Segmentation result file not found"}), 500

            # Логируем путь к файлу
            print(f"Sending result file: {out_file}")

            # Передаем путь к файлу в send_file
            return await send_file(
                out_file,
                mimetype='application/octet-stream',
                as_attachment=False
            )
        else:
            return jsonify({"error": f"Segmentation error: {rs}"}), 500
            
    except Exception as err:
        # ИСПРАВЛЕНИЕ 3: Правильная обработка ошибок
        print(f"Error in autosegmentation: {str(err)}")
        traceback.print_tb(err.__traceback__)
        
        # Используем await для make_response
        response = await make_response(jsonify({"error": str(err)}), 500)
        response.headers["Content-Type"] = "application/json"
        return response
    
@app.route('/get_status/<task_id>', methods=['GET'])
async def get_status(task_id):
    """Получение статуса задачи по task_id"""
    if task_id in task_status:
        return jsonify(task_status[task_id])
    return jsonify({"error": "Task not found"}), 404

@app.route('/test_task', methods=['POST'])
async def create_test_task():
    """Тестовый endpoint для создания задачи (временный)"""
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "processing", 
        "progress": 0, 
        "message": "Task created"
    }
    
    # Симуляция работы задачи
    async def simulate_work():
        await asyncio.sleep(2)
        task_status[task_id] = {
            "status": "processing", 
            "progress": 50, 
            "message": "Halfway done"
        }
        await asyncio.sleep(3)
        task_status[task_id] = {
            "status": "completed", 
            "progress": 100, 
            "message": "Task completed successfully"
        }
    
    # Запускаем симуляцию асинхронно
    asyncio.create_task(simulate_work())
    
    return jsonify({"task_id": task_id})

@app.route('/v1/inference_async', methods=['POST'])
async def inference_async():
    """Асинхронный endpoint для сегментации с отслеживанием прогресса"""

    global is_processing
    
    print(f"=== INFERENCE_ASYNC: Received request ===")
    print(f"=== Current is_processing: {is_processing} ===")
    
    if is_processing:
        task_id = str(uuid.uuid4())
        task_status[task_id] = {
            "status": "queued", 
            "progress": 0, 
            "message": "Another segmentation is in progress. Your task is queued."
        }
        print(f"=== INFERENCE_ASYNC: Server busy, queued task {task_id} ===")
        return jsonify({"task_id": task_id, "status": "queued", "message": "Server busy, task queued"})

    try:
        # Создаем новую задачу
        task_id = str(uuid.uuid4())
        print(f"=== INFERENCE_ASYNC: Created task {task_id} ===")
        
        task_status[task_id] = {
            "status": "processing", 
            "progress": 5, 
            "message": "Initializing task"
        }
        
        # Получаем параметры
        net_name = request.args.get('net', 'Unet')
        client_id = request.args.get('client_id', 'unknown')
        
        print(f"=== INFERENCE_ASYNC: net_name={net_name}, client_id={client_id} ===")
        
        if net_name not in MODELS:
            task_status[task_id] = {
                "status": "failed", 
                "progress": 0, 
                "error": f"Unknown model: {net_name}"
            }
            return jsonify({"task_id": task_id, "error": f"Unknown model: {net_name}"}), 400
        
        model_desc = MODELS[net_name]
        prefix = client_id + "_"
        
        print(f"DEBUG: Created async task {task_id} for model {net_name}, client {client_id}")
        
        # Обновляем статус - начинаем загрузку файлов
        task_status[task_id] = {
            "status": "processing", 
            "progress": 10, 
            "message": "Loading and saving uploaded files"
        }
        
        # Получаем загруженные файлы
        form = await request.form
        req_files = await request.files
        
        # Сохраняем файлы
        files_dict = {}
        
        # Функция для асинхронного сохранения файлов
        async def save_uploaded_file(file_key, file_obj, base_prefix):
            if file_obj and file_obj.filename and file_obj.filename.endswith(".nii.gz"):
                fname = base_prefix + prefix + secure_filename(file_obj.filename)
                full_path = os.path.join(INPUT_DIR, fname)
                
                async with aiofiles.open(full_path, 'wb') as f:
                    content = file_obj.read()
                    if asyncio.iscoroutine(content):
                        content = await content
                    await f.write(content)
                return fname
            return None
        
        # Обрабатываем файлы по типам
        for file_key in req_files:
            file_obj = req_files[file_key]
            
            print(f"DEBUG: Processing file key: {file_key}")
            
            # Извлекаем тип из ключа: "file_t1" -> "t1", "file_t2fl" -> "t2fl"
            if "_" in file_key:
                ftype = file_key.split("_")[1]  # "file_t1" -> "t1"
            else:
                print(f"DEBUG: Could not determine file type for: {file_key} (no underscore)")
                continue
            
            # Определяем префикс для сохранения файла
            base_prefix = ftype + "_"
            
            # Сохраняем файл
            fname = await save_uploaded_file(file_key, file_obj, base_prefix)
            if fname:
                files_dict[ftype] = fname
                print(f"DEBUG: Mapped {file_key} -> {ftype}: {fname}")

        print(f"DEBUG: Final files mapping: {files_dict}")
        
        # Проверяем наличие всех необходимых файлов
        required_files = ['t1', 't1c', 't2fl', 't2']
        missing_files = [f for f in required_files if f not in files_dict]
        
        if missing_files:
            error_msg = f"Missing required files: {missing_files}"
            task_status[task_id] = {
                "status": "failed", 
                "progress": 0, 
                "error": error_msg
            }
            return jsonify({"task_id": task_id, "error": error_msg}), 400
        
        # Обновляем статус - файлы загружены
        task_status[task_id] = {
            "status": "processing", 
            "progress": 20, 
            "message": "Files uploaded successfully, starting segmentation"
        }
        
        # Запускаем асинхронную сегментацию
        
        # КРИТИЧЕСКИ ВАЖНО: запускаем задачу и сразу возвращаем task_id
        print(f"=== INFERENCE_ASYNC: Starting async task {task_id} ===")
        asyncio.create_task(async_segmentation_with_progress(task_id, model_desc, files_dict, prefix))
        
        print(f"=== INFERENCE_ASYNC: Returning task_id {task_id} ===")
        return jsonify({"task_id": task_id, "status": "processing"})
        
    except Exception as e:
        print(f"=== INFERENCE_ASYNC: ERROR creating task: {e} ===")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/test_async_simple', methods=['POST'])
async def test_async_simple():
    """Простой тест асинхронной обработки без файлов"""
    task_id = str(uuid.uuid4())
    task_status[task_id] = {
        "status": "processing", 
        "progress": 0, 
        "message": "Starting simple test"
    }
    
    # Запускаем простую асинхронную задачу
    async def simple_test_task():
        for i in range(0, 101, 20):
            await asyncio.sleep(1)  # Имитация работы
            task_status[task_id] = {
                "status": "processing", 
                "progress": i, 
                "message": f"Processing... {i}%"
            }
        
        task_status[task_id] = {
            "status": "completed", 
            "progress": 100, 
            "message": "Test completed successfully"
        }
    
    asyncio.create_task(simple_test_task())
    return jsonify({"task_id": task_id})

async def prepare_files_for_unet(files, prefix):
    import hashlib
    rnd_suffix = hashlib.sha1(os.urandom(512)).hexdigest()[0:10]
    rnd_str = prefix + rnd_suffix
    
    # Используем TMP_DIR вместо хардкоженного пути
    in_path = os.path.join(TMP_DIR, f"{rnd_str}_in")
    os.mkdir(in_path)
    out_path = os.path.join(TMP_DIR, f"{rnd_str}_out")
    os.mkdir(out_path)
    
    import shutil
    print(f"DEBUG: Moving files:")
    print(f"  t1: {files['t1']} -> {os.path.join(in_path, f'{prefix}0000.nii.gz')}")
    print(f"  t1c: {files['t1c']} -> {os.path.join(in_path, f'{prefix}0001.nii.gz')}")
    print(f"  t2: {files['t2']} -> {os.path.join(in_path, f'{prefix}0002.nii.gz')}")
    print(f"  t2fl: {files['t2fl']} -> {os.path.join(in_path, f'{prefix}0003.nii.gz')}")
    
    shutil.move(files["t1"], os.path.join(in_path, f"{prefix}0000.nii.gz"))
    shutil.move(files["t1c"], os.path.join(in_path, f"{prefix}0001.nii.gz"))
    shutil.move(files["t2"], os.path.join(in_path, f"{prefix}0002.nii.gz"))
    shutil.move(files["t2fl"], os.path.join(in_path, f"{prefix}0003.nii.gz"))
    
    return in_path, out_path

async def async_segmentation_with_progress(task_id, model_desc, files_dict, prefix):
    """
    Асинхронная сегментация с детальным логированием GPU.
    
    ДИАГНОСТИЧЕСКАЯ ВЕРСИЯ - расширенное логирование для отладки параллелизма GPU.
    """
    global is_processing
    
    print("=" * 80)
    print(f"🔵 TASK {task_id}: STARTING")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"   Prefix: {prefix}")
    print("=" * 80)
    
    # Получаем доступный GPU (ждём если все заняты)
    print(f"🔵 TASK {task_id}: Requesting GPU from pool...")
    print(f"   Available GPUs in queue: {available_gpus.qsize()}")
    
    loop = asyncio.get_event_loop()
    gpu_allocation_start = datetime.now()
    gpu_id = await loop.run_in_executor(None, get_available_gpu)
    gpu_allocation_time = (datetime.now() - gpu_allocation_start).total_seconds()
    
    print(f"✅ TASK {task_id}: GPU {gpu_id} ALLOCATED (waited {gpu_allocation_time:.2f}s)")
    print(f"   Remaining GPUs in pool: {available_gpus.qsize()}")

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(gpu_id)
    
    # Сохраняем оригинальные настройки
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print(f"   Original CUDA_VISIBLE_DEVICES: '{original_cuda_devices}'")
    
    try:
        # Устанавливаем этот GPU как единственный видимый для текущего процесса
        import torch
        torch.cuda.set_device(gpu_id)  
        print(f"TASK {task_id}: Set torch device to GPU {gpu_id}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"   Process PID: {os.getpid()}")
        print(f"   Thread ID: {threading.current_thread().ident}")
        
        # Проверяем, что PyTorch видит правильный GPU
        try:
            import torch
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            print(f"   PyTorch CUDA available: {cuda_available}")
            print(f"   PyTorch sees {device_count} GPU(s)")
            if cuda_available and device_count > 0:
                print(f"   Current device: {torch.cuda.current_device()}")
                print(f"   Device name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"   ⚠️  Could not check PyTorch GPU: {e}")
        
        task_status[task_id] = {
            "status": "processing",
            "progress": 15,
            "message": f"Using GPU {gpu_id}",
            "gpu_id": gpu_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Подготовка файлов
        print(f"📁 TASK {task_id}: Preparing files...")
        files_with_paths = {
            "t1": os.path.join(INPUT_DIR, files_dict["t1"]),
            "t1c": os.path.join(INPUT_DIR, files_dict["t1c"]),
            "t2fl": os.path.join(INPUT_DIR, files_dict["t2fl"]),
            "t2": os.path.join(INPUT_DIR, files_dict["t2"])
        }
        
        in_path, out_path = await prepare_files_for_unet(files_with_paths, prefix)
        print(f"   Input dir: {in_path}")
        print(f"   Output dir: {out_path}")
        
        task_status[task_id] = {
            "status": "processing",
            "progress": 45,
            "message": f"Running segmentation on GPU {gpu_id}",
            "gpu_id": gpu_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Запуск сегментации
        print(f"🚀 TASK {task_id}: Starting nnUNet inference on GPU {gpu_id}...")
        print(f"   Model: {model_desc.name}")

        # Start GPU monitoring
        gpu_monitor.start()
        print(f"   GPU monitoring started for GPU {gpu_id}")
        
        import inference as nnUNet_inference
        
        task_name = server_config.get('task_name', 'Task115_AllData5foldsMeta')
        
        inference_start = datetime.now()
        
        import torch
        
        def run_inference_on_gpu(gpu_id, in_path, out_path, use_tta, folds, task_name):
            """Wrapper чтобы установить GPU внутри worker потока"""
            import torch
            torch.cuda.set_device(gpu_id)
            print(f"Worker thread: using GPU {torch.cuda.current_device()}")
            import inference as nnUNet_inference
            return nnUNet_inference.predict_for_api(in_path, out_path, use_tta, folds, task_name)

        # Для вызова:
        loop = asyncio.get_event_loop()

        if model_desc.name.startswith("Unet+Folds+TTA"):
            func = partial(run_inference_on_gpu, gpu_id, in_path, out_path, True, (0, 1, 2, 3, 4), task_name)
            result = await asyncio.wait_for(loop.run_in_executor(executor, func), timeout=600)
        elif model_desc.name.startswith("Unet+Folds"):
            func = partial(run_inference_on_gpu, gpu_id, in_path, out_path, False, (0, 1, 2, 3, 4), task_name)
            result = await asyncio.wait_for(loop.run_in_executor(executor, func), timeout=600)
        else:
            func = partial(run_inference_on_gpu, gpu_id, in_path, out_path, False, (0,), task_name)
            result = await asyncio.wait_for(loop.run_in_executor(executor, func), timeout=600)



        
        inference_time = (datetime.now() - inference_start).total_seconds()
        print(f"⏱️  TASK {task_id}: Inference completed in {inference_time:.2f}s")

        # Stop GPU monitoring and collect metrics
        gpu_metrics = gpu_monitor.stop()
        print(f"📊 TASK {task_id}: GPU metrics collected")
        if gpu_metrics:
            print(f"   Utilization: avg={gpu_metrics.get('utilization_avg', 0):.1f}%, max={gpu_metrics.get('utilization_max', 0):.1f}%")
            print(f"   Memory: avg={gpu_metrics.get('memory_used_mb_avg', 0):.1f}MB, max={gpu_metrics.get('memory_used_mb_max', 0):.1f}MB")
            print(f"   Temperature: avg={gpu_metrics.get('temperature_avg', 0):.1f}°C, max={gpu_metrics.get('temperature_max', 0):.1f}°C")
        
        # Постобработка
        print(f"📦 TASK {task_id}: Post-processing...")
        str_dt = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        out_fname = prefix + str_dt + "_result.nii.gz"
        final_result_path = os.path.join(OUTPUT_DIR, out_fname)
        
        if result == "":
            result_files = glob.glob(f'{out_path}/*.nii.gz')
            if result_files:
                shutil.move(result_files[0], final_result_path)
                
                if os.path.exists(final_result_path):
                    task_status[task_id] = {
                        "status": "completed",
                        "progress": 100,
                        "message": f"Completed on GPU {gpu_id}",
                        "result_file": out_fname,
                        "download_url": f"/uploads/{out_fname}",
                        "gpu_id": gpu_id,
                        "inference_time": inference_time,
                        "gpu_metrics": gpu_metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    print("=" * 80)
                    print(f"✅ TASK {task_id}: COMPLETED SUCCESSFULLY")
                    print(f"   GPU: {gpu_id}")
                    print(f"   Inference time: {inference_time:.2f}s")
                    print(f"   Output: {out_fname}")
                    print("=" * 80)
                else:
                    raise Exception("Result file not found after move")
            else:
                raise Exception("No result files found")
        else:
            raise Exception(f"Segmentation failed: {result}")
    
    except Exception as e:
        print("=" * 80)
        print(f"❌ TASK {task_id}: FAILED")
        print(f"   GPU: {gpu_id}")
        print(f"   Error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "gpu_id": gpu_id,
            "timestamp": datetime.now().isoformat()
        }
    
    finally:
        pass 
        
        # Возвращаем GPU в пул
        release_gpu(gpu_id)

        print(f"♻️  TASK {task_id}: Released GPU {gpu_id} back to pool")
        print(f"   GPUs now in pool: {available_gpus.qsize()}")
        print("=" * 80)


# Дополнительные функции для мониторинга

def log_gpu_pool_state():
    """Выводит текущее состояние GPU pool"""
    print("\n" + "=" * 80)
    print("📊 GPU POOL STATE")
    print(f"   Available GPUs in queue: {available_gpus.qsize()}")
    print(f"   Active tasks: {len([t for t in task_status.values() if t.get('status') == 'processing'])}")
    print(f"   Completed tasks: {len([t for t in task_status.values() if t.get('status') == 'completed'])}")
    print(f"   Failed tasks: {len([t for t in task_status.values() if t.get('status') == 'failed'])}")
    print("=" * 80 + "\n")


def print_gpu_diagnostics():
    """Выводит диагностическую информацию о GPU при старте сервера"""
    print("\n" + "=" * 80)
    print("🔍 GPU DIAGNOSTICS")
    print("=" * 80)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Multi-processors: {props.multi_processor_count}")
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    
    print("=" * 80 + "\n")

        
if __name__ == '__main__':
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 5000)
    debug = server_config.get('debug', True)
    
    print(f"\n{'='*60}")
    print(f"Starting Segmentation Server")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Models: {server_config.get('nnunet_models', 'Unknown')}")
    print(f"Task: {server_config.get('task_name', 'Unknown')}")
    print(f"GPUs: {server_config.get('gpu_ids', [])}")
    print(f"{'='*60}\n")
    
    print_gpu_diagnostics()
    app.run(debug=debug, host=host, port=port)
