import os
import subprocess
import zipfile
import uuid
from datetime import datetime
import threading
import logging
from logging.handlers import RotatingFileHandler
import time # Для симуляции задержки или проверки статуса (не используется активно сейчас)
import shutil

import yaml
from pathlib import Path
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, send_from_directory, session)
from werkzeug.utils import secure_filename
import sys

# --- Конфигурация приложения ---
app = Flask(__name__)

# --- Глобальные переменные и структуры ---
active_runs = {} # Словарь для отслеживания запусков
active_runs_lock = threading.Lock() # Блокировка для безопасного доступа

# --- Настройка логгера для Flask ---
flask_logger = logging.getLogger('flask_webapp')
flask_logger.setLevel(logging.INFO) # INFO для Flask, DEBUG для пайплайна
flask_log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
webapp_log_dir = Path(__file__).parent / "logs_webapp"
webapp_log_dir.mkdir(parents=True, exist_ok=True)
flask_log_file = webapp_log_dir / "webapp.log"
file_handler_flask = RotatingFileHandler(flask_log_file, maxBytes=1024000, backupCount=10, encoding='utf-8')
file_handler_flask.setFormatter(flask_log_formatter)
file_handler_flask.setLevel(logging.INFO)
flask_logger.addHandler(file_handler_flask)
if not app.debug or os.environ.get("FLASK_ENV") == "development":
    console_handler_flask = logging.StreamHandler(sys.stdout)
    console_handler_flask.setFormatter(flask_log_formatter)
    console_handler_flask.setLevel(logging.INFO)
    flask_logger.addHandler(console_handler_flask)
flask_logger.info("Flask логгер настроен.")

# --- Загрузка конфигурации пайплайна ---
CONFIG_FILE_PATH = Path(__file__).parent.parent / "config/config.yaml"
PIPELINE_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" # Не используется напрямую, но для контекста
PIPELINE_RUN_SCRIPT = Path(__file__).parent.parent / "pipeline" / "run_pipeline.py"
pipeline_config = None
try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        pipeline_config = yaml.safe_load(f)
    flask_logger.info(f"Конфигурация пайплайна успешно загружена из {CONFIG_FILE_PATH}")
    UPLOAD_FOLDER_BASE_STR = pipeline_config.get('paths', {}).get('output_base_dir')
    if not UPLOAD_FOLDER_BASE_STR:
        flask_logger.critical("Ключ 'paths.output_base_dir' не найден в config.yaml!")
        raise KeyError("'paths.output_base_dir' не найден в конфиге")
    UPLOAD_FOLDER_BASE = Path(UPLOAD_FOLDER_BASE_STR)
    UPLOAD_FOLDER_BASE.mkdir(parents=True, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER_BASE.resolve())
    flask_logger.info(f"Папка для загрузок и результатов: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    flask_logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при загрузке config.yaml ({CONFIG_FILE_PATH}): {e}", exc_info=True)
    fallback_dir = Path(__file__).parent.parent / "PIPELINE_RESULTS_CONFIG_ERROR"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = str(fallback_dir.resolve())
    flask_logger.error(f"Используется запасной путь для результатов: {app.config['UPLOAD_FOLDER']}")

app.config['ALLOWED_EXTENSIONS'] = {'zip'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_pipeline_user_log(log_path: Path, last_n_lines=200) -> list[str]:
    """
    Читает лог-файл пайплайна и извлекает сообщения, предназначенные для пользователя.
    """
    user_log_messages = []
    user_log_prefix_in_file = "[Pipeline] PIPELINE_USER_MSG:" # Точный префикс из run_pipeline.py

    if not log_path:
        return ["[Путь к лог-файлу пайплайна не был определен в данных запуска]"]

    if log_path.exists() and log_path.is_file():
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

            if not all_lines:
                user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Лог пайплайна ({log_path.name}) пока пуст...")
                return user_log_messages

            for line in all_lines:
                stripped_line = line.strip()
                if user_log_prefix_in_file in stripped_line:
                    try:
                        # Извлекаем сообщение после временной метки и уровня лога
                        # Пример: 2025-05-13 10:00:00,123 - INFO - [Pipeline] PIPELINE_USER_MSG: Сообщение
                        message_part = stripped_line.split(user_log_prefix_in_file, 1)[1].strip()
                        time_part = stripped_line.split(" - ")[0] # 2025-05-13 10:00:00,123
                        # Отображаем только время HH:MM:SS
                        display_time = time_part.split(',')[0].split(' ')[-1] if ',' in time_part else time_part.split(' ')[-1]
                        user_log_messages.append(f"[{display_time}] {message_part}")
                    except IndexError:
                        # Если не удалось распарсить, добавляем как есть (без префикса пайплайна)
                        user_log_messages.append(stripped_line.split(" [Pipeline] ",1)[-1])
            
            if not user_log_messages and all_lines : # Если не нашли USER_MSG, но файл не пуст
                 user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Ожидание сообщений о прогрессе из пайплайна ({log_path.name})...")

        except Exception as e:
            flask_logger.error(f"Ошибка чтения/парсинга лог-файла {log_path}: {e}")
            user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [Ошибка чтения лога пайплайна: {e}]")
    else:
        user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Лог-файл пайплайна ({log_path.name}) еще не создан или не найден.")
    
    return user_log_messages[-last_n_lines:]


def find_reports(run_specific_dir: Path, run_id: str) -> list[dict]:
    """Ищет готовые отчеты в папке запуска."""
    reports = []
    if not pipeline_config: return reports
    subdirs = pipeline_config.get('paths', {}).get('subdirs', {})

    # 1. Отчет BIDS Validation
    validation_report_dir = run_specific_dir / subdirs.get('validation_reports', 'validation_results')
    bids_validator_report = validation_report_dir / "bids_validator_report.txt"
    if bids_validator_report.is_file(): # Проверяем, что это файл
        reports.append({
            "name": "Отчет BIDS Валидации",
            "url": url_for('download_file_from_run', run_id=run_id, subpath=str(bids_validator_report.relative_to(run_specific_dir)))
        })

    # 2. Отчеты Fast QC
    fast_qc_dir = run_specific_dir / subdirs.get('fast_qc_reports', 'bids_quality_metrics')
    if fast_qc_dir.is_dir(): # Проверяем, что это директория
        for report_file in fast_qc_dir.rglob("*_quality_report.txt"):
            if report_file.is_file():
                relative_report_path = report_file.relative_to(run_specific_dir)
                reports.append({
                    "name": f"FastQC: {report_file.stem.replace('_quality_report','')}",
                    "url": url_for('download_file_from_run', run_id=run_id, subpath=str(relative_report_path))
                })
    # ... (Аналогично для MRIQC и интерпретаций, если они будут локально или после копирования) ...
    
    # Ссылка на предобработанные данные (если они есть)
    preprocessed_dir_name = subdirs.get('preprocessed', 'preprocessed_data')
    preprocessed_path = run_specific_dir / preprocessed_dir_name
    if preprocessed_path.is_dir() and any(preprocessed_path.iterdir()):
        reports.append({
            "name": f"Предобработанные данные (NIfTI) в папке: '{preprocessed_dir_name}' (скачивание папки не поддерживается)",
            "url": "#" 
        })
    return reports


def monitor_pipeline_process(run_id: str, process: subprocess.Popen, pipeline_log_path: Path):
    """Отслеживает процесс пайплайна, логирует и обновляет статус."""
    flask_logger.info(f"Мониторинг пайплайна для run_id: {run_id} запущен в потоке {threading.get_ident()}.")
    stdout, stderr = process.communicate()

    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data: flask_logger.error(f"Данные для run_id {run_id} не найдены после завершения процесса."); return

        try:
            with open(pipeline_log_path, 'a', encoding='utf-8') as f:
                f.write("\n\n--- Pipeline Process STDOUT (from Popen) ---\n")
                f.write(stdout or "[No stdout from Popen]")
                f.write("\n\n--- Pipeline Process STDERR (from Popen) ---\n")
                f.write(stderr or "[No stderr from Popen]")
                f.write("\n--- End Pipeline Process Popen Output ---")
            flask_logger.debug(f"Stdout/Stderr пайплайна (Popen) для {run_id} записаны в {pipeline_log_path}")
        except Exception as e: flask_logger.error(f"Ошибка записи stdout/stderr (Popen) пайплайна {run_id}: {e}")

        final_status_pipeline = ""
        if process.returncode == 0:
            final_status_pipeline = 'Completed'
            run_data['can_run_mriqc'] = True
            flask_logger.info(f"Пайплайн для run_id: {run_id} успешно завершен.")
        else:
            final_status_pipeline = f'Error (Pipeline Code: {process.returncode})'
            flask_logger.error(f"Пайплайн для run_id: {run_id} завершился с ошибкой (код: {process.returncode}).")
        
        run_data['status_pipeline'] = final_status_pipeline
        run_data['process_pipeline'] = None
        run_data['thread_pipeline'] = None
        # Добавляем финальное сообщение в user_log (дублируем из parse_pipeline_log для надежности)
        user_log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Локальный пайплайн: {final_status_pipeline}"
        if not run_data['user_log'] or run_data['user_log'][-1] != user_log_entry:
             run_data['user_log'].append(user_log_entry)


@app.route('/')
def index():
    flask_logger.info(f"Запрос GET / от {request.remote_addr}")
    with active_runs_lock:
        sorted_runs = sorted(active_runs.items(), key=lambda item: item[1].get('start_time_iso', '0'), reverse=True)
    return render_template('index.html', active_runs=sorted_runs)


@app.route('/upload', methods=['POST'])
def upload_file():
    flask_logger.info(f"Запрос POST /upload от {request.remote_addr}")
    if 'dicom_archive' not in request.files:
        flash('Файл не был выбран.', 'error'); flask_logger.warning("Запрос /upload: Файл 'dicom_archive' отсутствует."); return redirect(url_for('index'))
    file = request.files['dicom_archive']
    if file.filename == '':
        flash('Файл не был выбран (пустое имя).', 'error'); flask_logger.warning("Запрос /upload: Имя файла пустое."); return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        flask_logger.info(f"Генерация run_id: {run_id} для файла {filename}")

        run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
        input_archive_dir = run_specific_dir / "input_archive"
        input_raw_data_dir = run_specific_dir / "input_raw_data"
        # Папка логов для этого запуска (создается пайплайном, но мы знаем путь)
        logs_subdir_name = pipeline_config.get('paths', {}).get('subdirs', {}).get('logs', 'logs')
        pipeline_log_path_for_run = run_specific_dir / logs_subdir_name / 'pipeline.log'


        try:
            flask_logger.debug(f"Создание директорий для run_id {run_id} в {run_specific_dir}")
            run_specific_dir.mkdir(parents=True, exist_ok=True)
            input_archive_dir.mkdir(exist_ok=True); input_raw_data_dir.mkdir(exist_ok=True)
            (run_specific_dir / logs_subdir_name).mkdir(exist_ok=True) # Создаем папку логов заранее
        except OSError as e:
            flask_logger.error(f"Не удалось создать директории для {run_id}: {e}", exc_info=True)
            flash(f"Ошибка сервера при создании директорий: {e}", "error"); return redirect(url_for('index'))

        archive_path = input_archive_dir / filename
        try:
            file.save(str(archive_path)); flask_logger.info(f"Архив '{filename}' сохранен в {archive_path}")
        except Exception as e:
            flask_logger.error(f"Не удалось сохранить архив {archive_path}: {e}", exc_info=True)
            flash(f"Ошибка сохранения архива: {e}", "error"); return redirect(url_for('index'))

        try:
            flask_logger.info(f"Распаковка {archive_path} в {input_raw_data_dir}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref: zip_ref.extractall(input_raw_data_dir)
            flask_logger.info(f"Архив успешно распакован.")
        except zipfile.BadZipFile:
            flask_logger.error(f"{archive_path} не ZIP-архив."); flash("Некорректный ZIP-архив.", "error"); return redirect(url_for('index'))
        except Exception as e:
            flask_logger.error(f"Ошибка распаковки {archive_path}: {e}", exc_info=True)
            flash(f"Ошибка распаковки архива: {e}", "error"); return redirect(url_for('index'))

        # Определяем effective_input_data_dir
        effective_input_data_dir = input_raw_data_dir
        items_in_raw = list(input_raw_data_dir.iterdir())
        if len(items_in_raw) == 1 and items_in_raw[0].is_dir():
            effective_input_data_dir = items_in_raw[0]
            flask_logger.info(f"Используется содержимое подпапки: {effective_input_data_dir}")
        
        console_log_level_pipeline = pipeline_config.get("logging", {}).get("pipeline_console_level", "INFO")


        cmd = [ sys.executable, str(PIPELINE_RUN_SCRIPT.resolve()),
                "--config", str(CONFIG_FILE_PATH.resolve()),
                "--run_id", run_id,
                "--input_data_dir", str(effective_input_data_dir.resolve()),
                "--output_base_dir", str(run_specific_dir.resolve()),
                "--console_log_level", console_log_level_pipeline
              ]
        flask_logger.info(f"Команда для пайплайна ({run_id}): {' '.join(cmd)}")

        try:
            project_root_dir = Path(__file__).parent.parent
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', cwd=str(project_root_dir))
            flask_logger.info(f"Пайплайн для {run_id} запущен (PID: {process.pid})")

            monitor_thread = threading.Thread(target=monitor_pipeline_process, args=(run_id, process, pipeline_log_path_for_run))
            monitor_thread.daemon = True; monitor_thread.start()

            with active_runs_lock:
                active_runs[run_id] = {
                    'status_pipeline': 'Queued', 'status_mriqc': 'Not Started',
                    'start_time_obj': datetime.now(), 'start_time_iso': datetime.now().isoformat(),
                    'start_time_display': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'pipeline_log_path': pipeline_log_path_for_run,
                    'user_log': [f"[{datetime.now().strftime('%H:%M:%S')}] Запуск {run_id} добавлен в очередь."],
                    'process_pipeline': process, 'thread_pipeline': monitor_thread,
                    'reports': [], 'can_run_mriqc': False, 'mriqc_requested': False
                }
            flash(f'Файл загружен. Обработка запущена (ID: {run_id})', 'success')
            return redirect(url_for('processing_status', run_id=run_id))
        except Exception as e:
            flask_logger.critical(f"Критическая ошибка запуска пайплайна {run_id}: {e}", exc_info=True)
            flash(f"Ошибка сервера при запуске обработки: {e}", "error")
            if run_specific_dir.exists(): 
                try: shutil.rmtree(run_specific_dir)
                except Exception as e_clean: flask_logger.error(f"Ошибка очистки {run_specific_dir}: {e_clean}")
            return redirect(url_for('index'))
    else:
        flash('Недопустимый тип файла. Только .zip.', 'error'); flask_logger.warning(f"Недопустимый тип файла: {file.filename if file else 'No file'}")
        return redirect(url_for('index'))


@app.route('/status/<run_id>')
def processing_status(run_id: str):
    flask_logger.info(f"Запрос GET /status/{run_id} от {request.remote_addr}")
    with active_runs_lock: run_info_copy = active_runs.get(run_id, {}).copy()
    if not run_info_copy:
        flash(f'Запуск {run_id} не найден.', 'error'); flask_logger.warning(f"Запуск {run_id} не найден."); return redirect(url_for('index'))
    return render_template('processing.html', run_id=run_id, run_info=run_info_copy)


@app.route('/api/status/<run_id>')
def api_status(run_id: str):
    flask_logger.debug(f"API запрос статуса для run_id: {run_id}")
    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data: flask_logger.warning(f"API: Запуск {run_id} не найден."); return jsonify({'error': 'Run ID not found'}), 404

        current_pipeline_status = run_data.get('status_pipeline', 'Unknown')
        pipeline_log_path = run_data.get('pipeline_log_path')

        if pipeline_log_path:
            # Обновляем user_log только если статус еще не финальный или лог не содержит финального сообщения
            # Это предотвращает перезапись финального сообщения, если оно уже было добавлено потоком мониторинга
            is_pipeline_final = "Completed" in current_pipeline_status or "Error" in current_pipeline_status
            final_log_msg_present = run_data['user_log'] and "Локальный пайплайн" in run_data['user_log'][-1]

            if not (is_pipeline_final and final_log_msg_present):
                 run_data['user_log'] = parse_pipeline_user_log(pipeline_log_path)
            elif not run_data['user_log']: # Если лог пуст, но статус финальный (маловероятно)
                 run_data['user_log'] = parse_pipeline_user_log(pipeline_log_path)

        else:
            if not run_data.get('user_log') or "Путь к логу не определен" not in run_data.get('user_log', [""])[-1]:
                 run_data.setdefault('user_log', []).append(f"[{datetime.now().strftime('%H:%M:%S')}] Путь к логу пайплайна не определен.")

        run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
        run_data['reports'] = find_reports(run_specific_dir, run_id)

        if current_pipeline_status == 'Completed' and not run_data.get('can_run_mriqc_checked'):
            bids_nifti_path = run_specific_dir / pipeline_config['paths']['subdirs'].get('bids_nifti', 'bids_data_nifti')
            if bids_nifti_path.is_dir() and any(bids_nifti_path.iterdir()): run_data['can_run_mriqc'] = True
            run_data['can_run_mriqc_checked'] = True # Помечаем, что проверили
        
        # Копируем данные для JSON ответа, чтобы избежать проблем с многопоточностью при доступе к словарю
        response_data = {
            'run_id': run_id,
            'status_pipeline': current_pipeline_status,
            'status_mriqc': run_data.get('status_mriqc', 'Not Started'),
            'user_log': run_data['user_log'],
            'reports': run_data['reports'],
            'can_run_mriqc': run_data.get('can_run_mriqc', False),
            'mriqc_requested': run_data.get('mriqc_requested', False)
        }
        return jsonify(response_data)

@app.route('/download/<run_id>/<path:subpath>')
def download_file_from_run(run_id: str, subpath: str):
    flask_logger.info(f"Запрос на скачивание: {run_id}/{subpath} от {request.remote_addr}")
    run_specific_dir_abs = (Path(app.config['UPLOAD_FOLDER']) / run_id).resolve()
    file_to_download_abs = (run_specific_dir_abs / subpath).resolve()
    if not str(file_to_download_abs).startswith(str(run_specific_dir_abs)):
        flask_logger.warning(f"Попытка доступа к файлу вне {run_specific_dir_abs}: {file_to_download_abs}")
        flash("Ошибка: Запрошен недопустимый путь.", "error"); return redirect(url_for('processing_status', run_id=run_id)), 403
    if not file_to_download_abs.is_file():
        flask_logger.error(f"Файл не найден: {file_to_download_abs}")
        flash(f"Файл '{subpath}' не найден для {run_id}.", "error"); return redirect(url_for('processing_status', run_id=run_id)), 404
    try:
        return send_from_directory(directory=str(file_to_download_abs.parent), path=str(file_to_download_abs.name), as_attachment=True)
    except Exception as e:
        flask_logger.exception(f"Ошибка отправки файла {file_to_download_abs}: {e}")
        flash(f"Ошибка сервера при скачивании.", "error"); return redirect(url_for('processing_status', run_id=run_id)), 500

if __name__ == '__main__':
    flask_logger.info(f"Запуск Flask dev server (host 0.0.0.0, port 5000, debug={app.debug})...")
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)