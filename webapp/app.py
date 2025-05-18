import os
import subprocess
import zipfile
import uuid
from datetime import datetime
import threading
import logging
from logging.handlers import RotatingFileHandler
import time # Для возможной имитации или проверок
import shutil # Для shutil.rmtree при очистке

import yaml
from pathlib import Path
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, send_from_directory, session, current_app) 
from werkzeug.utils import secure_filename
import sys

# --- Конфигурация приложения ---
app = Flask(__name__)

# --- Глобальные переменные и структуры ---
active_runs = {}
active_runs_lock = threading.Lock()

# --- Настройка логгера для Flask ---
flask_logger = logging.getLogger('flask_webapp')
flask_logger.setLevel(logging.INFO)
flask_log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d - %(funcName)s]' # Добавил funcName
)
webapp_log_dir = Path(__file__).parent / "logs_webapp"
try:
    webapp_log_dir.mkdir(parents=True, exist_ok=True)
    flask_log_file = webapp_log_dir / "webapp.log"
    file_handler_flask = RotatingFileHandler(flask_log_file, maxBytes=1024000, backupCount=10, encoding='utf-8')
    file_handler_flask.setFormatter(flask_log_formatter)
    file_handler_flask.setLevel(logging.INFO)
    flask_logger.addHandler(file_handler_flask)
except Exception as e_log_setup:
    print(f"CRITICAL: Failed to setup Flask file logging: {e_log_setup}", file=sys.stderr)

# Добавляем консольный обработчик всегда, но уровень может быть разным
console_handler_flask = logging.StreamHandler(sys.stdout)
console_handler_flask.setFormatter(flask_log_formatter)
if os.environ.get("FLASK_ENV") == "development" or app.debug:
    console_handler_flask.setLevel(logging.DEBUG)
else:
    console_handler_flask.setLevel(logging.INFO)
flask_logger.addHandler(console_handler_flask)
flask_logger.info("Flask логгер настроен.")


# --- Загрузка конфигурации пайплайна ---
CONFIG_FILE_PATH = Path(__file__).parent.parent / "config/config.yaml"
PIPELINE_RUN_SCRIPT = Path(__file__).parent.parent / "pipeline" / "run_pipeline.py"
pipeline_config = None
try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        pipeline_config = yaml.safe_load(f)
    flask_logger.info(f"Конфигурация пайплайна успешно загружена из {CONFIG_FILE_PATH}")
    # Сохраняем весь конфиг в app.config для доступа из потоков
    app.config['PIPELINE_CONFIG'] = pipeline_config
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
    fallback_dir = Path(__file__).parent.parent / "PIPELINE_RESULTS_CONFIG_ERROR_CRITICAL"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = str(fallback_dir.resolve())
    app.config['PIPELINE_CONFIG'] = {} # Пустой конфиг в случае ошибки
    flask_logger.error(f"Используется запасной путь для результатов: {app.config['UPLOAD_FOLDER']}")
    # Для рабочего сервера здесь должен быть sys.exit(1) или аналогичная остановка

app.config['ALLOWED_EXTENSIONS'] = {'zip'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PIPELINE_SCRIPTS_DIR'] = Path(__file__).parent.parent / "scripts" # Путь к папке scripts

def allowed_file(filename: str) -> bool:
    """Проверяет, имеет ли файл разрешенное расширение."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_pipeline_user_log(log_path: Path, last_n_lines=200) -> list[str]:
    """
    Читает лог-файл пайплайна и извлекает сообщения, предназначенные для пользователя,
    с префиксом "PIPELINE_USER_MSG:".
    """
    user_log_messages = []
    user_log_prefix_in_file = "[Pipeline] PIPELINE_USER_MSG:"

    if not log_path: return ["[Путь к лог-файлу пайплайна не определен]"]
    if not log_path.is_file():
        return [f"[{datetime.now().strftime('%H:%M:%S')}] Лог пайплайна ({log_path.name}) еще не создан или не найден."]

    try:
        with open(log_path, 'r', encoding='utf-8') as f: all_lines = f.readlines()
        if not all_lines:
            user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Лог пайплайна ({log_path.name}) пока пуст...")
            return user_log_messages

        for line in all_lines:
            stripped_line = line.strip()
            if user_log_prefix_in_file in stripped_line:
                try:
                    message_part = stripped_line.split(user_log_prefix_in_file, 1)[1].strip()
                    time_part = stripped_line.split(" - ")[0]
                    display_time = time_part.split(',')[0].split(' ')[-1] if ',' in time_part else time_part.split(' ')[-1]
                    user_log_messages.append(f"[{display_time}] {message_part}")
                except IndexError: user_log_messages.append(stripped_line.split(" [Pipeline] ",1)[-1])
        
        if not user_log_messages and all_lines:
             user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Ожидание сообщений о прогрессе...")
    except Exception as e:
        flask_logger.error(f"Ошибка чтения/парсинга лог-файла {log_path}: {e}")
        user_log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] [Ошибка чтения лога: {e}]")
    
    return user_log_messages[-last_n_lines:]

def find_reports(run_specific_dir: Path, run_id: str) -> list[dict]:
    """Ищет готовые отчеты и результаты в папке запуска."""
    reports = []
    p_config = app.config.get('PIPELINE_CONFIG', {}) # Используем app.config
    if not p_config: flask_logger.error("Конфиг пайплайна не загружен, не могу найти пути к отчетам."); return reports
    subdirs = p_config.get('paths', {}).get('subdirs', {})

    # Отчет BIDS Validation
    validation_report_dir = run_specific_dir / subdirs.get('validation_reports', 'validation_results')
    bids_validator_report = validation_report_dir / "bids_validator_report.txt"
    if bids_validator_report.is_file():
        reports.append({
            "name": "Отчет BIDS Валидации",
            "url": url_for('download_file_from_run', run_id=run_id, subpath=str(bids_validator_report.relative_to(run_specific_dir)))
        })

    # Отчеты Fast QC
    fast_qc_dir = run_specific_dir / subdirs.get('fast_qc_reports', 'bids_quality_metrics')
    if fast_qc_dir.is_dir():
        for report_file in fast_qc_dir.rglob("*_quality_report.txt"):
            if report_file.is_file():
                reports.append({
                    "name": f"FastQC: {report_file.stem.replace('_quality_report','')}",
                    "url": url_for('download_file_from_run', run_id=run_id, subpath=str(report_file.relative_to(run_specific_dir)))
                })

    # Отчеты MRIQC (HTML)
    mriqc_out_dir = run_specific_dir / subdirs.get('mriqc_output', 'mriqc_output')
    if mriqc_out_dir.is_dir():
        for html_report in list(mriqc_out_dir.glob("*.html")) + list(mriqc_out_dir.rglob("sub-*/**/*.html")):
             if html_report.is_file():
                display_name = f"MRIQC: {html_report.name}"
                if html_report.parent.name.startswith("sub-"): display_name = f"MRIQC {html_report.parent.name}: {html_report.name}"
                reports.append({
                    "name": display_name,
                    "url": url_for('download_file_from_run', run_id=run_id, subpath=str(html_report.relative_to(run_specific_dir)))
                })
    
    # Файлы интерпретации MRIQC
    mriqc_interpret_dir = run_specific_dir / subdirs.get('mriqc_interpret', 'mriqc_interpretation')
    if mriqc_interpret_dir.is_dir():
        for interpret_file in mriqc_interpret_dir.rglob("*_interpretation.txt"):
            if interpret_file.is_file():
                reports.append({
                    "name": f"Интерпретация MRIQC: {interpret_file.stem.replace('_interpretation','')}",
                    "url": url_for('download_file_from_run', run_id=run_id, subpath=str(interpret_file.relative_to(run_specific_dir)))
                })
    
    # Маски сегментации
    segmentation_dir_name = subdirs.get('segmentation_masks', 'segmentation_masks')
    segmentation_path = run_specific_dir / segmentation_dir_name
    if segmentation_path.is_dir():
        for mask_file in segmentation_path.rglob("*_segmask.nii.gz"): # Ищем по суффиксу
            if mask_file.is_file():
                try: display_name_prefix = f"{mask_file.parent.parent.name}_{mask_file.parent.name}" # sub-XXX_ses-YYY
                except: display_name_prefix = mask_file.stem.replace("_segmask","")
                reports.append({
                    "name": f"Маска сегментации: {display_name_prefix}",
                    "url": url_for('download_file_from_run', run_id=run_id, subpath=str(mask_file.relative_to(run_specific_dir)))
                })
    return reports


def monitor_pipeline_process(run_id: str, process: subprocess.Popen, pipeline_log_path: Path):
    """
    Отслеживает завершение процесса основного пайплайна в отдельном потоке,
    логирует его stdout/stderr и обновляет статус в active_runs.
    """
    flask_logger.info(f"Мониторинг основного пайплайна для run_id: {run_id} запущен (поток: {threading.get_ident()}).")
    stdout, stderr = process.communicate() # Блокирует до завершения процесса

    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data: flask_logger.error(f"[{run_id}] Данные не найдены в active_runs после завершения пайплайна."); return

        try:
            with open(pipeline_log_path, 'a', encoding='utf-8') as f: # Дозаписываем в лог
                f.write("\n\n--- Pipeline Process STDOUT (from Popen) ---\n")
                f.write(stdout or "[No stdout from Popen object for main pipeline]")
                f.write("\n\n--- Pipeline Process STDERR (from Popen) ---\n")
                f.write(stderr or "[No stderr from Popen object for main pipeline]")
                f.write("\n--- End Pipeline Process Popen Output ---")
            flask_logger.debug(f"[{run_id}] Stdout/Stderr пайплайна (Popen) записаны в {pipeline_log_path}")
        except Exception as e: flask_logger.error(f"[{run_id}] Ошибка записи stdout/stderr пайплайна (Popen): {e}")

        final_status_pipeline = ""
        user_log_msg_suffix = ""
        if process.returncode == 0:
            final_status_pipeline = 'Completed'
            # run_data['can_run_mriqc'] = True
            # run_data['can_run_segmentation'] = True # Теперь можно запускать и сегментацию
            flask_logger.info(f"[{run_id}] Основной пайплайна успешно завершен.")
        else:
            final_status_pipeline = f'Error (Pipeline Code: {process.returncode})'
            user_log_msg_suffix = f" Код ошибки: {process.returncode}."
            flask_logger.error(f"[{run_id}] Основной пайплайна завершился с ошибкой (код: {process.returncode}).")
        
        run_data['status_pipeline'] = final_status_pipeline
        run_data['process_pipeline'] = None
        run_data['thread_pipeline'] = None
        user_log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Локальный пайплайн: {final_status_pipeline}.{user_log_msg_suffix}"
        if not run_data['user_log'] or run_data['user_log'][-1] != user_log_entry: run_data['user_log'].append(user_log_entry)

def execute_remote_mriqc_task(flask_app_instance, run_id: str):
    """
    Выполняет задачу MRIQC на удаленном сервере.
    Эта функция предназначена для запуска в отдельном потоке.

    Args:
        flask_app_instance: Экземпляр текущего Flask-приложения.
        run_id: Идентификатор текущего запуска.
    """
    # Создаем контекст приложения внутри потока
    # Весь код, использующий flask_app_instance.config или .logger, должен быть внутри этого блока
    with flask_app_instance.app_context():
        config_from_app = flask_app_instance.config.get('PIPELINE_CONFIG', {})
        logger_from_app = flask_app_instance.logger # Это ваш настроенный flask_logger

        logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Запуск execute_remote_mriqc_task.")
        current_status_mriqc = 'MRIQC_Error_Unknown_Thread_Start' # Начальный статус на случай раннего выхода

        try:
            # Блокировка для безопасного доступа к active_runs
            with active_runs_lock:
                if run_id not in active_runs:
                    logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Ошибка: run_id не найден в active_runs.")
                    return # Выход из потока, если run_id недействителен
                run_data = active_runs[run_id]
                run_data['status_mriqc'] = 'MRIQC_Preparing'
                current_status_mriqc = run_data['status_mriqc']
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Статус MRIQC обновлен на: {current_status_mriqc}")

            # --- 0. Получение путей и параметров ---
            local_bids_nifti_dir_str = run_data.get('paths', {}).get('bids_nifti')
            local_mriqc_output_dir_str = run_data.get('paths', {}).get('mriqc_output')
            local_mriqc_interpretation_dir_str = run_data.get('paths', {}).get('mriqc_interpretation')

            if not all([local_bids_nifti_dir_str, local_mriqc_output_dir_str, local_mriqc_interpretation_dir_str]):
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Не определены все необходимые локальные пути в active_runs[run_id]['paths'].")
                current_status_mriqc = 'MRIQC_Error_LocalPath_Setup'
                with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
                return
            
            local_bids_nifti_dir = Path(local_bids_nifti_dir_str)
            # local_mriqc_output_dir = Path(local_mriqc_output_dir_str) # Эти пути понадобятся позже для копирования результатов
            # local_mriqc_interpretation_dir = Path(local_mriqc_interpretation_dir_str)

            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Локальная BIDS NIfTI: {local_bids_nifti_dir}")
            # logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Локальный MRIQC output: {local_mriqc_output_dir}")
            # logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Локальный MRIQC interpretation: {local_mriqc_interpretation_dir}")

            mriqc_step_config = config_from_app.get('steps', {}).get('mriqc', {})
            report_type = mriqc_step_config.get('report_type', 'participant')
            n_procs = mriqc_step_config.get('n_procs', 1)
            n_threads = mriqc_step_config.get('n_threads', 1)
            mem_gb = mriqc_step_config.get('mem_gb', 4)
            
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Параметры MRIQC: report_type={report_type}, n_procs={n_procs}, n_threads={n_threads}, mem_gb={mem_gb}")

            server_config = config_from_app.get('server_mriqc', {})
            ssh_user = server_config.get('ssh_user')
            ssh_host = server_config.get('ssh_host')
            ssh_key_path = server_config.get('ssh_key_path') # Может быть None
            remote_base_data_dir = server_config.get('remote_base_data_dir')
            remote_wrapper_script_path = server_config.get('remote_wrapper_script_path')
            mriqc_executable_path_on_server = server_config.get('mriqc_executable_path_on_server')
            
            monitoring_interval_sec = server_config.get('monitoring_interval_seconds', 60)
            monitoring_timeout_total_sec = server_config.get('monitoring_timeout_hours', 24) * 3600
            delete_remote_on_success = server_config.get('delete_remote_task_dir_on_success', True)
            delete_remote_on_error = server_config.get('delete_remote_task_dir_on_error', False)

            if not all([ssh_user, ssh_host, remote_base_data_dir, remote_wrapper_script_path, mriqc_executable_path_on_server]):
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Не все параметры server_mriqc заданы в конфигурации (ssh_user, ssh_host, remote_base_data_dir, remote_wrapper_script_path, mriqc_executable_path_on_server).")
                current_status_mriqc = 'MRIQC_Error_ServerConfig'
                with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
                return

            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Сервер: user={ssh_user}, host={ssh_host}, base_dir={remote_base_data_dir}, wrapper_script={remote_wrapper_script_path}")
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) MRIQC exec на сервере: {mriqc_executable_path_on_server}")

            # --- 1. Создание уникальной директории для задачи на сервере ---
            current_status_mriqc = 'MRIQC_Server_Dir_Creation'
            with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Статус MRIQC обновлен на: {current_status_mriqc}")

            # Используем os.path.join, хотя пути удаленные, для консистентности и читаемости.
            # На сервере это будут Unix-пути.
            remote_task_dir = os.path.join(remote_base_data_dir, run_id)
            remote_bids_nifti_dir = os.path.join(remote_task_dir, "bids_nifti")
            remote_mriqc_output_dir = os.path.join(remote_task_dir, "mriqc_output")
            remote_mriqc_interpretation_dir = os.path.join(remote_task_dir, "mriqc_interpretation")
            remote_server_job_log_dir = os.path.join(remote_task_dir, "logs_server_wrapper")
            remote_server_job_log_file = os.path.join(remote_server_job_log_dir, "server_mriqc_job.log")
            remote_nohup_log_file = os.path.join(remote_task_dir, "nohup_ssh_command.log")

            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Путь к задаче на сервере: {remote_task_dir}")
            
            mkdir_command_str = f"mkdir -p '{remote_task_dir}' '{remote_mriqc_output_dir}' '{remote_mriqc_interpretation_dir}' '{remote_server_job_log_dir}'"
            ssh_base_cmd_list = ["ssh"]
            if ssh_key_path: ssh_base_cmd_list.extend(["-i", ssh_key_path])
            ssh_base_cmd_list.extend([f"{ssh_user}@{ssh_host}", mkdir_command_str])

            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) ТЕСТ: Команда SSH (создание директорий): {' '.join(ssh_base_cmd_list)}")

            process_mkdir = subprocess.run(ssh_base_cmd_list, capture_output=True, text=True, check=False, encoding='utf-8')
            if process_mkdir.returncode != 0:
                err_msg = process_mkdir.stderr or process_mkdir.stdout or "Неизвестная ошибка mkdir"
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Ошибка создания директорий на сервере ({process_mkdir.returncode}): {err_msg.strip()}")
                current_status_mriqc = 'MRIQC_Error_Server_Mkdir'
                with active_runs_lock:
                    run_data['status_mriqc'] = current_status_mriqc
                    run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC Ошибка mkdir: {err_msg[:200].strip()}")
                return
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Директории на сервере успешно созданы.")
            
            # --- 2. Копирование BIDS NIfTI на сервер ---
            current_status_mriqc = 'MRIQC_Copying_To_Server'
            with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Статус MRIQC обновлен на: {current_status_mriqc}")
            
            # Убедимся, что локальная директория существует перед копированием
            if not local_bids_nifti_dir.is_dir():
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Локальная директория BIDS NIfTI не найдена: {local_bids_nifti_dir}")
                current_status_mriqc = 'MRIQC_Error_Local_Nifti_NotFound'
                with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
                return

            # scp -r /local/path/bids_nifti user@host:/remote/path/run_id/
            # Копируем содержимое local_bids_nifti_dir внутрь remote_bids_nifti_dir
            # Для scp -r важно, чтобы remote_bids_nifti_dir уже существовал (создан на шаге 1)
            scp_cmd_list = ["scp"]
            if ssh_key_path: scp_cmd_list.extend(["-i", ssh_key_path])
            scp_cmd_list.extend([
                "-r", 
                str(local_bids_nifti_dir),  # Это ваша локальная папка, например /.../bids_data_nifti
                f"{ssh_user}@{ssh_host}:{remote_bids_nifti_dir}" # Это целевой путь на сервере, включая желаемое имя папки, например /.../run_id/bids_nifti
            ])

            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) ТЕСТ: Команда SCP (копирование BIDS NIfTI): {' '.join(scp_cmd_list)}")

            process_scp_to = subprocess.run(scp_cmd_list, capture_output=True, text=True, check=False, encoding='utf-8')
            if process_scp_to.returncode != 0:
                err_msg = process_scp_to.stderr or process_scp_to.stdout or "Неизвестная ошибка scp"
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Ошибка копирования BIDS NIfTI на сервер ({process_scp_to.returncode}): {err_msg.strip()}")
                current_status_mriqc = 'MRIQC_Error_Upload'
                with active_runs_lock:
                    run_data['status_mriqc'] = current_status_mriqc
                    run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC Ошибка scp: {err_msg[:200].strip()}")
                return
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) BIDS NIfTI успешно скопированы на сервер.")

            # --- 3. Запуск скрипта-обертки MRIQC на сервере ---
            current_status_mriqc = 'MRIQC_Running_On_Server'
            with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Статус MRIQC обновлен на: {current_status_mriqc}")
            
            wrapper_args = [
                run_id, remote_bids_nifti_dir, remote_mriqc_output_dir,
                remote_mriqc_interpretation_dir, remote_server_job_log_file,
                str(report_type), str(n_procs), str(n_threads), str(mem_gb),
                mriqc_executable_path_on_server
            ]
            quoted_wrapper_args = [f"'{arg}'" for arg in wrapper_args]
            
            remote_exec_command_str = f"nohup bash {remote_wrapper_script_path} {' '.join(quoted_wrapper_args)} > '{remote_nohup_log_file}' 2>&1 &"
            
            ssh_exec_cmd_list = ["ssh"]
            if ssh_key_path: ssh_exec_cmd_list.extend(["-i", ssh_key_path])
            ssh_exec_cmd_list.extend([f"{ssh_user}@{ssh_host}", remote_exec_command_str])

            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) ТЕСТ: Команда SSH (запуск MRIQC): {' '.join(ssh_exec_cmd_list)}")

            process_ssh_exec = subprocess.run(ssh_exec_cmd_list, capture_output=True, text=True, check=False, encoding='utf-8')
            # nohup & обычно возвращает 0 немедленно, если команда принята системой.
            # Ошибку здесь можно поймать, если сама команда ssh не смогла подключиться или передать команду.
            if process_ssh_exec.returncode != 0:
                err_msg = process_ssh_exec.stderr or process_ssh_exec.stdout or "Неизвестная ошибка запуска скрипта на сервере"
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Ошибка запуска MRIQC скрипта на сервере ({process_ssh_exec.returncode}): {err_msg.strip()}")
                current_status_mriqc = 'MRIQC_Error_Server_Exec_Start'
                with active_runs_lock:
                    run_data['status_mriqc'] = current_status_mriqc
                    run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC Ошибка запуска: {err_msg[:200].strip()}")
                return
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Команда запуска MRIQC на сервере отправлена.")
            
            # --- 4. Мониторинг файлов .done / .error ---
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Начало мониторинга MRIQC на сервере.")
            current_status_mriqc = 'MRIQC_Monitoring_On_Server'
            with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc
            
            start_time_monitoring = time.time()
            mriqc_task_succeeded = False
            mriqc_task_failed_on_server = False
            
            remote_done_file = os.path.join(remote_mriqc_output_dir, ".done")
            remote_error_file = os.path.join(remote_mriqc_output_dir, ".error")

            while True:
                # Проверка таймаута
                elapsed_time = time.time() - start_time_monitoring
                if elapsed_time > monitoring_timeout_total_sec:
                    logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Таймаут мониторинга MRIQC ({monitoring_timeout_total_sec / 3600:.1f} часов).")
                    current_status_mriqc = 'MRIQC_Error_Timeout'
                    with active_runs_lock:
                        run_data['status_mriqc'] = current_status_mriqc
                        run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC: Таймаут ожидания завершения на сервере.")
                    # Здесь можно решить, нужно ли пытаться скопировать логи или удалить папку на сервере
                    break # Выход из цикла while

                # Проверка файла .done
                check_done_cmd_list = ["ssh"]
                if ssh_key_path: check_done_cmd_list.extend(["-i", ssh_key_path])
                # 'test -f /path/to/file' возвращает 0, если файл существует, и 1 (или другое не 0), если нет
                check_done_cmd_list.extend([f"{ssh_user}@{ssh_host}", f"test -f '{remote_done_file}'"])
                
                # logger_from_app.debug(f"[{run_id}] (MRIQC_Thread) Проверка .done: {' '.join(check_done_cmd_list)}")
                process_check_done = subprocess.run(check_done_cmd_list, capture_output=True, text=True, check=False)
                
                if process_check_done.returncode == 0:
                    logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Файл .done найден на сервере. MRIQC успешно завершен.")
                    mriqc_task_succeeded = True
                    break # Выход из цикла while

                # Проверка файла .error (только если .done не найден)
                check_error_cmd_list = ["ssh"]
                if ssh_key_path: check_error_cmd_list.extend(["-i", ssh_key_path])
                check_error_cmd_list.extend([f"{ssh_user}@{ssh_host}", f"test -f '{remote_error_file}'"])

                # logger_from_app.debug(f"[{run_id}] (MRIQC_Thread) Проверка .error: {' '.join(check_error_cmd_list)}")
                process_check_error = subprocess.run(check_error_cmd_list, capture_output=True, text=True, check=False)

                if process_check_error.returncode == 0:
                    logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Файл .error найден на сервере. Ошибка выполнения MRIQC.")
                    mriqc_task_failed_on_server = True
                    break # Выход из цикла while
                
                # Если ни один из файлов не найден и нет таймаута, продолжаем ждать
                logger_from_app.debug(f"[{run_id}] (MRIQC_Thread) Файлы .done/.error не найдены. Ожидание {monitoring_interval_sec} сек...")
                time.sleep(monitoring_interval_sec)
            
            # --- Конец цикла мониторинга ---

            if mriqc_task_succeeded:
                # --- 5. Копирование результатов обратно ---
                current_status_mriqc = 'MRIQC_Copying_From_Server'
                with active_runs_lock: run_data['status_mriqc'] = current_status_mriqc # Обновляем статус перед началом копирования
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Статус MRIQC обновлен на: {current_status_mriqc}")
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Начало копирования результатов MRIQC с сервера.")

                # --- Код для копирования mriqc_output ---
                # Целевая локальная директория: local_mriqc_output_dir_str (из run_data['paths']['mriqc_output'])
                # Источник на сервере: remote_mriqc_output_dir
                
                # Убедимся, что локальная целевая директория существует или создаем ее
                local_mriqc_output_path = Path(local_mriqc_output_dir_str)
                local_mriqc_output_path.mkdir(parents=True, exist_ok=True) # Создаем, если нет

                scp_output_cmd_list = ["scp"]
                if ssh_key_path: scp_output_cmd_list.extend(["-i", ssh_key_path])
                # Копируем содержимое remote_mriqc_output_dir в local_mriqc_output_path
                # scp -r user@host:/remote/path/to/mriqc_output/* /local/path/to/mriqc_output/
                # Чтобы скопировать все содержимое, включая подпапки, но без создания лишней вложенности:
                scp_output_cmd_list.extend([
                    "-r",
                    f"{ssh_user}@{ssh_host}:{remote_mriqc_output_dir}/.", # Источник с "/." копирует содержимое
                    str(local_mriqc_output_path) # Локальная целевая папка
                ])
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Команда SCP (копирование mriqc_output): {' '.join(scp_output_cmd_list)}")
                process_scp_output = subprocess.run(scp_output_cmd_list, capture_output=True, text=True, check=False, encoding='utf-8')
                if process_scp_output.returncode != 0:
                    err_msg = process_scp_output.stderr or process_scp_output.stdout or "Ошибка копирования mriqc_output"
                    logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Ошибка копирования mriqc_output с сервера ({process_scp_output.returncode}): {err_msg.strip()}")
                    # Решаем, что делать дальше: пометить ошибкой или продолжить с другими файлами?
                    # Пока просто логируем и продолжаем (можно установить флаг частичного успеха)
                else:
                    logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Папка mriqc_output успешно скопирована с сервера.")

                # --- Код для копирования mriqc_interpretation ---
                # Целевая локальная директория: local_mriqc_interpretation_dir_str
                # Источник на сервере: remote_mriqc_interpretation_dir
                local_mriqc_interpretation_path = Path(local_mriqc_interpretation_dir_str)
                local_mriqc_interpretation_path.mkdir(parents=True, exist_ok=True)

                scp_interpret_cmd_list = ["scp"]
                if ssh_key_path: scp_interpret_cmd_list.extend(["-i", ssh_key_path])
                scp_interpret_cmd_list.extend([
                    "-r",
                    f"{ssh_user}@{ssh_host}:{remote_mriqc_interpretation_dir}/.",
                    str(local_mriqc_interpretation_path)
                ])
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Команда SCP (копирование mriqc_interpretation): {' '.join(scp_interpret_cmd_list)}")
                process_scp_interpret = subprocess.run(scp_interpret_cmd_list, capture_output=True, text=True, check=False, encoding='utf-8')
                if process_scp_interpret.returncode != 0:
                    err_msg = process_scp_interpret.stderr or process_scp_interpret.stdout or "Ошибка копирования mriqc_interpretation"
                    logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Ошибка копирования mriqc_interpretation с сервера ({process_scp_interpret.returncode}): {err_msg.strip()}")
                else:
                    logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Папка mriqc_interpretation успешно скопирована с сервера.")

                # --- Копирование лог-файлов сервера ---
                # Локальная директория для логов пайплайна (run_data['paths']['pipeline_log'] - это файл, нужна папка)
                local_pipeline_log_path_obj = Path(run_data['paths'].get('pipeline_log'))
                local_logs_dir = local_pipeline_log_path_obj.parent if local_pipeline_log_path_obj else Path(run_data['paths']['base']) / "logs"
                local_logs_dir.mkdir(parents=True, exist_ok=True)

                # Копируем server_mriqc_job.log
                scp_server_job_log_cmd = ["scp"]
                if ssh_key_path: scp_server_job_log_cmd.extend(["-i", ssh_key_path])
                scp_server_job_log_cmd.extend([
                    f"{ssh_user}@{ssh_host}:{remote_server_job_log_file}",
                    str(local_logs_dir / f"server_mriqc_job_{run_id}.log") # Даем уникальное имя
                ])
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Команда SCP (копирование server_job_log): {' '.join(scp_server_job_log_cmd)}")
                process_scp_job_log = subprocess.run(scp_server_job_log_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
                if process_scp_job_log.returncode != 0: logger_from_app.warning(f"[{run_id}] (MRIQC_Thread) Не удалось скопировать server_mriqc_job.log")
                else: logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Файл server_mriqc_job.log скопирован.")

                # Копируем nohup_ssh_command.log
                scp_nohup_log_cmd = ["scp"]
                if ssh_key_path: scp_nohup_log_cmd.extend(["-i", ssh_key_path])
                scp_nohup_log_cmd.extend([
                    f"{ssh_user}@{ssh_host}:{remote_nohup_log_file}",
                    str(local_logs_dir / f"nohup_mriqc_ssh_{run_id}.log") # Уникальное имя
                ])
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Команда SCP (копирование nohup_log): {' '.join(scp_nohup_log_cmd)}")
                process_scp_nohup_log = subprocess.run(scp_nohup_log_cmd, capture_output=True, text=True, check=False, encoding='utf-8')
                if process_scp_nohup_log.returncode != 0: logger_from_app.warning(f"[{run_id}] (MRIQC_Thread) Не удалось скопировать nohup_ssh_command.log")
                else: logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Файл nohup_ssh_command.log скопирован.")
                    
                # После успешного копирования (или попытки копирования):
                current_status_mriqc = 'MRIQC_Completed'
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) MRIQC успешно завершен, результаты и логи (попытка) скопированы.")

            elif mriqc_task_failed_on_server:
                current_status_mriqc = 'MRIQC_Error_On_Server'
                logger_from_app.error(f"[{run_id}] (MRIQC_Thread) MRIQC завершился с ошибкой на сервере.")
                # Попытка скопировать логи сервера даже в случае ошибки
                local_pipeline_log_path_obj = Path(run_data['paths'].get('pipeline_log'))
                local_logs_dir = local_pipeline_log_path_obj.parent if local_pipeline_log_path_obj else Path(run_data['paths']['base']) / "logs"
                local_logs_dir.mkdir(parents=True, exist_ok=True)
                
                # Копируем server_mriqc_job.log
                scp_server_job_log_cmd_err = ["scp"]
                if ssh_key_path: scp_server_job_log_cmd_err.extend(["-i", ssh_key_path])
                scp_server_job_log_cmd_err.extend([
                    f"{ssh_user}@{ssh_host}:{remote_server_job_log_file}",
                    str(local_logs_dir / f"server_mriqc_job_ERROR_{run_id}.log")
                ])
                subprocess.run(scp_server_job_log_cmd_err, capture_output=True, text=True, check=False, encoding='utf-8') # Просто пытаемся
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Попытка копирования лога server_mriqc_job.log после ошибки.")

                # Копируем nohup_ssh_command.log
                scp_nohup_log_cmd_err = ["scp"]
                if ssh_key_path: scp_nohup_log_cmd_err.extend(["-i", ssh_key_path])
                scp_nohup_log_cmd_err.extend([
                    f"{ssh_user}@{ssh_host}:{remote_nohup_log_file}",
                    str(local_logs_dir / f"nohup_mriqc_ssh_ERROR_{run_id}.log")
                ])
                subprocess.run(scp_nohup_log_cmd_err, capture_output=True, text=True, check=False, encoding='utf-8') # Просто пытаемся
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Попытка копирования лога nohup_ssh_command.log после ошибки.")
                
            # else: current_status_mriqc будет 'MRIQC_Error_Timeout', если вышли по таймауту

            # Обновляем статус в active_runs
            with active_runs_lock:
                run_data['status_mriqc'] = current_status_mriqc
                run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC: {current_status_mriqc}.")


        # --- 6. Очистка на сервере (если настроено) ---
            should_delete_remote_dir = False
            if mriqc_task_succeeded and delete_remote_on_success:
                should_delete_remote_dir = True
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) MRIQC успешно завершен, настроено удаление данных на сервере.")
            elif (mriqc_task_failed_on_server or current_status_mriqc == 'MRIQC_Error_Timeout') and delete_remote_on_error:
                # Если была ошибка на сервере или таймаут, и настроено удаление при ошибке
                should_delete_remote_dir = True
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Ошибка/таймаут MRIQC, настроено удаление данных на сервере.")
            elif not mriqc_task_succeeded: # Ошибка или таймаут, но не настроено удаление при ошибке
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Ошибка/таймаут MRIQC, НЕ настроено удаление данных на сервере. Данные останутся для анализа.")


            if should_delete_remote_dir:
                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Попытка удаления временной директории на сервере: {remote_task_dir}")
                delete_cmd_list = ["ssh"]
                if ssh_key_path: delete_cmd_list.extend(["-i", ssh_key_path])
                delete_cmd_list.extend([f"{ssh_user}@{ssh_host}", f"rm -rf '{remote_task_dir}'"])

                logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Команда SSH (удаление директории на сервере): {' '.join(delete_cmd_list)}")
                process_delete = subprocess.run(delete_cmd_list, capture_output=True, text=True, check=False, encoding='utf-8')
                if process_delete.returncode != 0:
                    err_msg = process_delete.stderr or process_delete.stdout or "Ошибка удаления директории на сервере"
                    logger_from_app.warning(f"[{run_id}] (MRIQC_Thread) Не удалось удалить временную директорию на сервере ({process_delete.returncode}): {err_msg.strip()}")
                    # Это не критическая ошибка для статуса MRIQC, но нужно залогировать
                else:
                    logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Временная директория на сервере успешно удалена.")


        except Exception as e:
            # Этот блок перехватит любые другие непредвиденные ошибки в потоке
            current_status_mriqc = 'MRIQC_Error_Critical_Exception_In_Thread'
            logger_from_app.critical(f"[{run_id}] (MRIQC_Thread) КРИТИЧЕСКАЯ ОШИБКА в execute_remote_mriqc_task: {e}", exc_info=True)
            try: # Попытка обновить статус даже при ошибке
                with active_runs_lock:
                    if run_id in active_runs: # run_data может быть не определена, если ошибка до ее присвоения
                        active_runs[run_id]['status_mriqc'] = current_status_mriqc
                        active_runs[run_id].setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC: Крит. ошибка потока.")
            except Exception as e_lock:
                 logger_from_app.error(f"[{run_id}] (MRIQC_Thread) Не удалось обновить статус после крит. ошибки из-за: {e_lock}")
        finally:
            # Код, который должен выполниться в любом случае при завершении потока
            # Например, если нужно освободить какие-то ресурсы, специфичные для этого потока
            # Ссылку на сам поток ('thread_mriqc_trigger') лучше обнулять в основном потоке Flask
            # после проверки thread.is_alive() или thread.join().
            logger_from_app.info(f"[{run_id}] (MRIQC_Thread) Поток execute_remote_mriqc_task завершает работу со статусом: {current_status_mriqc}.")

# --- Маршруты Flask ---
@app.route('/')
def index():
    """Отображает главную страницу с формой загрузки и историей запусков."""
    flask_logger.info(f"Запрос GET / от {request.remote_addr}")
    with active_runs_lock:
        sorted_runs = sorted(active_runs.items(), key=lambda item: item[1].get('start_time_iso', '0'), reverse=True)
    return render_template('index.html', active_runs=sorted_runs)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обрабатывает загрузку ZIP-архива и запускает основной пайплайн."""
    flask_logger.info(f"Запрос POST /upload от {request.remote_addr}")
    # (Код проверки файла, генерации run_id, создания папок, сохранения и распаковки архива - как в предыдущем ответе)
    # ...
    if 'dicom_archive' not in request.files: flash('Файл не выбран.', 'error'); return redirect(url_for('index'))
    file = request.files['dicom_archive']
    if file.filename == '': flash('Файл не выбран.', 'error'); return redirect(url_for('index'))
    if not (file and allowed_file(file.filename)): flash('Недопустимый тип файла. Только .zip.', 'error'); return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    flask_logger.info(f"Генерация run_id: {run_id} для файла {filename}")

    run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
    input_archive_dir = run_specific_dir / "input_archive"
    input_raw_data_dir = run_specific_dir / "input_raw_data"

    # Используем app.config для доступа к pipeline_config
    p_config = app.config.get('PIPELINE_CONFIG', {})
    subdirs_config = p_config.get('paths', {}).get('subdirs', {})
    logs_subdir_name = subdirs_config.get('logs', 'logs')
    pipeline_log_path_for_run = run_specific_dir / logs_subdir_name / 'pipeline.log'

    # Определяем пути для MRIQC сразу
    mriqc_output_subdir_name = subdirs_config.get('mriqc_output', 'mriqc_output')
    mriqc_interpret_subdir_name = subdirs_config.get('mriqc_interpret', 'mriqc_interpretation')
    bids_nifti_subdir_name = subdirs_config.get('bids_nifti', 'bids_data_nifti')
    preprocessed_subdir_name = subdirs_config.get('preprocessed', 'preprocessed_data')

    local_paths_for_run = {
        'base': str(run_specific_dir.resolve()),
        'input_archive': str(input_archive_dir.resolve()),
        'input_raw': str(input_raw_data_dir.resolve()),
        'pipeline_log': pipeline_log_path_for_run, # Оставляем как Path объект для parse_pipeline_user_log
        'bids_nifti': str((run_specific_dir / bids_nifti_subdir_name).resolve()),
        # --- NEW: Добавляем путь к preprocessed_data ---
        'preprocessed': str((run_specific_dir / preprocessed_subdir_name).resolve()),
        'mriqc_output': str((run_specific_dir / mriqc_output_subdir_name).resolve()),
        'mriqc_interpretation': str((run_specific_dir / mriqc_interpret_subdir_name).resolve()),
        # Добавьте другие пути, если они нужны для других опциональных шагов или отчетов
        # Например, для dciodvfy_reports, dicom_metadata и т.д., если они нужны в 'paths'
        'dicom_checks': str((run_specific_dir / subdirs_config.get('dicom_checks', 'dciodvfy_reports')).resolve()),
        'dicom_meta': str((run_specific_dir / subdirs_config.get('dicom_meta', 'dicom_metadata')).resolve()),
        'validation_reports': str((run_specific_dir / subdirs_config.get('validation_reports', 'validation_results')).resolve()),
        'fast_qc_reports': str((run_specific_dir / subdirs_config.get('fast_qc_reports', 'bids_quality_metrics')).resolve()),
        'transforms': str((run_specific_dir / subdirs_config.get('transforms', 'transformations')).resolve()),
        'segmentation_masks': str((run_specific_dir / subdirs_config.get('segmentation_masks', 'segmentation_masks')).resolve()),
    }

    try:
        run_specific_dir.mkdir(parents=True, exist_ok=True)
        input_archive_dir.mkdir(exist_ok=True); input_raw_data_dir.mkdir(exist_ok=True)
        (run_specific_dir / logs_subdir_name).mkdir(parents=True, exist_ok=True)
        archive_path = input_archive_dir / filename
        file.save(str(archive_path)); flask_logger.info(f"Архив '{filename}' сохранен в {archive_path}")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref: zip_ref.extractall(input_raw_data_dir)
        flask_logger.info(f"Архив успешно распакован.")
    except Exception as e:
        flask_logger.error(f"Ошибка подготовки данных для {run_id}: {e}", exc_info=True)
        flash(f"Ошибка сервера при подготовке данных: {e}", "error"); return redirect(url_for('index'))

    effective_input_data_dir = input_raw_data_dir
    items_in_raw = list(input_raw_data_dir.iterdir())
    if len(items_in_raw) == 1 and items_in_raw[0].is_dir(): effective_input_data_dir = items_in_raw[0]
    
    console_log_lvl_pipe = p_config.get("logging", {}).get("pipeline_console_level", "INFO")

    cmd = [ sys.executable, str(PIPELINE_RUN_SCRIPT.resolve()),
            "--config", str(CONFIG_FILE_PATH.resolve()), "--run_id", run_id,
            "--input_data_dir", str(effective_input_data_dir.resolve()),
            "--output_base_dir", str(run_specific_dir.resolve()),
            "--console_log_level", console_log_lvl_pipe ]
    flask_logger.info(f"Команда для пайплайна ({run_id}): {' '.join(cmd)}")

    try:
        project_root_dir = Path(__file__).parent.parent
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', cwd=str(project_root_dir))
        flask_logger.info(f"Пайплайн для {run_id} запущен (PID: {process.pid})")
        monitor_thread = threading.Thread(target=monitor_pipeline_process, args=(run_id, process, pipeline_log_path_for_run))
        monitor_thread.daemon = True; monitor_thread.start()
        with active_runs_lock:
            active_runs[run_id] = {
                'status_pipeline': 'Queued', 'status_mriqc': 'Not Started', 'status_segmentation': 'Not Started',
                'start_time_obj': datetime.now(), 'start_time_iso': datetime.now().isoformat(),
                'start_time_display': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'pipeline_log_path': pipeline_log_path_for_run,
                'user_log': [f"[{datetime.now().strftime('%H:%M:%S')}] Запуск {run_id} добавлен в очередь."],
                'process_pipeline': process, 'thread_pipeline': monitor_thread,
                'reports': [], 
                'paths': local_paths_for_run,
                'can_run_mriqc': False, 
                'can_run_segmentation': False,
                'mriqc_requested': False, 
                'segmentation_requested': False,
                'thread_mriqc_trigger': None, 'thread_segmentation': None, # для хранения потоков опц. шагов
                'optional_steps_eligibility_checked': False # флаг для проверки готовности опц. шагов
            }
        flash(f'Файл загружен. Обработка запущена (ID: {run_id})', 'success')
        return redirect(url_for('processing_status', run_id=run_id))
    except Exception as e:
        flask_logger.critical(f"Ошибка запуска пайплайна {run_id}: {e}", exc_info=True)
        flash(f"Ошибка сервера при запуске обработки: {e}", "error")
        if run_specific_dir.exists(): 
            try: shutil.rmtree(run_specific_dir)
            except Exception as e_clean: flask_logger.error(f"Ошибка очистки {run_specific_dir}: {e_clean}")
        return redirect(url_for('index'))
    
@app.route('/download_file_from_run/<run_id>/<path:subpath>')
def download_file_from_run(run_id: str, subpath: str):
    """Позволяет скачивать файлы из директории конкретного запуска."""
    # Важно: subpath должен быть относительным путем внутри run_specific_dir
    flask_logger.info(f"Запрос на скачивание файла: run_id={run_id}, subpath={subpath}")
    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data:
            flask_logger.warning(f"Попытка скачать файл для несуществующего run_id: {run_id}")
            return "Run ID not found", 404

    # Используем базовый путь из сохраненных 'paths' этого run_id, если есть, или строим заново
    run_specific_dir_str = run_data.get('paths', {}).get('base', str(Path(app.config['UPLOAD_FOLDER']) / run_id))
    run_specific_dir = Path(run_specific_dir_str)
    
    # Предотвращение выхода за пределы директории run_id
    # Нормализуем subpath и проверяем, что он не пытается выйти за пределы
    full_path = (run_specific_dir / Path(subpath)).resolve()
    # Проверяем, что resolved path начинается с resolved run_specific_dir
    if not str(full_path).startswith(str(run_specific_dir.resolve())):
        flask_logger.error(f"Попытка доступа к файлу за пределами директории запуска: {subpath} для {run_id}")
        return "Path traversal attempt detected", 403 # Forbidden
    
    if not full_path.is_file():
        flask_logger.warning(f"Файл не найден для скачивания: {full_path}")
        return "File not found", 404

    try:
        # send_from_directory требует абсолютный путь к директории и имя файла
        return send_from_directory(str(full_path.parent), full_path.name, as_attachment=True)
    except Exception as e:
        flask_logger.error(f"Ошибка при отправке файла {full_path}: {e}")
        return "Error sending file", 500

@app.route('/status/<run_id>')
def processing_status(run_id: str):
    """Отображает страницу статуса для конкретного запуска."""
    flask_logger.info(f"Запрос GET /status/{run_id} от {request.remote_addr}")
    with active_runs_lock: run_info_copy = active_runs.get(run_id, {}).copy()
    if not run_info_copy:
        flash(f'Запуск {run_id} не найден.', 'error'); flask_logger.warning(f"Запуск {run_id} не найден."); return redirect(url_for('index'))
    return render_template('processing.html', run_id=run_id, run_info=run_info_copy)


@app.route('/api/status/<run_id>')
def api_status(run_id: str):
    """
    Возвращает JSON со статусом, логом и отчетами для AJAX-обновлений.
    Также проверяет и устанавливает флаги 'can_run_mriqc' и 'can_run_segmentation'.
    """
    flask_logger.debug(f"API запрос статуса для run_id: {run_id}")
    
    # Получаем конфигурацию пайплайна из app.config
    # Это безопаснее, так как app.config['PIPELINE_CONFIG'] инициализируется при старте приложения
    p_config = app.config.get('PIPELINE_CONFIG', {})
    if not p_config:
        # Эта ситуация не должна возникать, если приложение стартовало корректно,
        # но на всякий случай логируем и возвращаем ошибку.
        flask_logger.error(f"[{run_id}] Критическая ошибка: Конфигурация пайплайна не найдена в app.config.")
        return jsonify({'error': 'Server configuration error: Pipeline config not loaded.'}), 500

    with active_runs_lock:
        run_data = active_runs.get(run_id)

        if not run_data:
            flask_logger.warning(f"API: Запуск {run_id} не найден при запросе статуса.")
            return jsonify({'error': f"Run ID '{run_id}' не найден."}), 404

        # --- Обновление пользовательского лога из файла лога пайплайна ---
        current_pipeline_status = run_data.get('status_pipeline', 'Unknown')
        pipeline_log_path = run_data.get('pipeline_log_path') # Должен быть объектом Path

        # Определяем, завершен ли пайплайн, чтобы не парсить лог без надобности
        is_pipeline_final = "Completed" in current_pipeline_status or "Error" in current_pipeline_status
        # Проверяем, добавлено ли уже финальное сообщение о статусе пайплайна в user_log
        user_log_list = run_data.get('user_log', [])
        final_log_msg_present = any("Локальный пайплайн" in msg for msg in user_log_list)


        if pipeline_log_path and isinstance(pipeline_log_path, Path):
            if not (is_pipeline_final and final_log_msg_present):
                # Парсим лог, если пайплайн не завершен ИЛИ если завершен, но финальное сообщение еще не добавлено
                run_data['user_log'] = parse_pipeline_user_log(pipeline_log_path)
        elif not pipeline_log_path:
            # Если путь к логу не определен, добавляем сообщение об этом, если его еще нет
            no_log_path_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Путь к логу пайплайна не определен."
            if not any("Путь к логу пайплайна не определен" in msg for msg in user_log_list):
                 run_data.setdefault('user_log', []).append(no_log_path_msg)
        
        # --- Поиск готовых отчетов ---
        # Используем базовый путь из сохраненных 'paths' этого run_id
        run_specific_dir_str = run_data.get('paths', {}).get('base', str(Path(app.config['UPLOAD_FOLDER']) / run_id))
        run_specific_dir = Path(run_specific_dir_str)
        run_data['reports'] = find_reports(run_specific_dir, run_id)

        # --- Проверка и установка флагов для запуска опциональных шагов ---
        subdirs_cfg = p_config.get('paths', {}).get('subdirs', {})

        # Проверка для MRIQC: становится доступно, как только есть NIfTI файлы
        if not run_data.get('can_run_mriqc'): # Проверяем только если флаг еще не установлен
            bids_nifti_subdir_name = subdirs_cfg.get('bids_nifti', 'bids_data_nifti')
            # Путь к bids_nifti должен быть в run_data['paths']
            bids_nifti_path_str = run_data.get('paths', {}).get('bids_nifti')
            
            if bids_nifti_path_str:
                bids_nifti_path = Path(bids_nifti_path_str)
                if bids_nifti_path.is_dir() and any(f.name != '.DS_Store' for f in bids_nifti_path.iterdir() if f.is_file()): # Проверяем наличие файлов
                    run_data['can_run_mriqc'] = True
                    flask_logger.info(f"[{run_id}] Условие для запуска MRIQC выполнено (найдена непустая папка NIfTI: {bids_nifti_path}).")
                    run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] Данные NIfTI готовы, MRIQC можно запускать.")
            else:
                flask_logger.warning(f"[{run_id}] Путь 'bids_nifti' не найден в run_data['paths'] для проверки can_run_mriqc.")

        # Проверка для Сегментации:
        # Зависит от завершения основного пайплайна И наличия предобработанных данных.
        # (Можно изменить условие, если предобработка может завершиться раньше всего пайплайна)
        if not run_data.get('can_run_segmentation'): # Проверяем только если флаг еще не установлен
            if current_pipeline_status == 'Completed': # Основной пайплайн должен быть завершен
                preprocessed_subdir_name = subdirs_cfg.get('preprocessed', 'preprocessed_data')
                # Путь к preprocessed_data должен быть в run_data['paths']
                preprocessed_path_str = run_data.get('paths', {}).get('preprocessed') # Предполагаем, что такой ключ будет добавлен в 'paths'
                
                if preprocessed_path_str:
                    preprocessed_path = Path(preprocessed_path_str)
                    if preprocessed_path.is_dir() and any(f.name != '.DS_Store' for f in preprocessed_path.iterdir() if f.is_file()):
                        run_data['can_run_segmentation'] = True
                        flask_logger.info(f"[{run_id}] Условие для запуска Сегментации выполнено (пайплайн завершен, найдена папка preprocessed: {preprocessed_path}).")
                        run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] Данные для сегментации готовы.")
                else:
                    flask_logger.warning(f"[{run_id}] Путь 'preprocessed' не найден в run_data['paths'] для проверки can_run_segmentation.")
            # else: # Если пайплайн еще не завершен, то и сегментацию запускать нельзя (по текущей логике)
            #    pass

        # --- Сбор данных для ответа ---
        # Ключи, которые мы хотим всегда возвращать клиенту
        response_keys = [
            'status_pipeline', 'status_mriqc', 'status_segmentation',
            'user_log', 'reports', 'can_run_mriqc', 'can_run_segmentation',
            'mriqc_requested', 'segmentation_requested'
        ]
        response_data = {key: run_data.get(key) for key in response_keys}
        response_data['run_id'] = run_id # Всегда добавляем run_id для полноты

        return jsonify(response_data)

@app.route('/run_mriqc_remote/<run_id>', methods=['POST'])
def trigger_mriqc_on_server(run_id: str):
    """
    Обрабатывает запрос на запуск MRIQC на удаленном сервере для указанного run_id.
    Запускает execute_remote_mriqc_task в отдельном потоке.
    """
    flask_logger.info(f"Запрос POST /run_mriqc_remote/{run_id} от {request.remote_addr}")
    
    # Получаем текущий экземпляр Flask приложения.
    # 'app' здесь - это ваш глобальный экземпляр Flask, определенный как app = Flask(__name__)
    # Если вы хотите быть более явным или используете blueprints, можно использовать current_app._get_current_object()
    # но для простого приложения достаточно передать глобальный 'app'.
    current_flask_app = app 

    with active_runs_lock:
        run_data = active_runs.get(run_id)

        if not run_data: 
            flask_logger.warning(f"[{run_id}] Попытка запуска MRIQC для несуществующего run_id.")
            return jsonify({"error": f"Run ID '{run_id}' не найден."}), 404
        
        # Проверка, можно ли запускать MRIQC (например, готовы ли NIfTI файлы)
        if not run_data.get('can_run_mriqc', False):
            flask_logger.warning(f"[{run_id}] Попытка запуска MRIQC, но условия не выполнены (can_run_mriqc=False).")
            return jsonify({"error": "Условия для запуска MRIQC не выполнены (например, NIfTI файлы еще не готовы)."}), 400
        
        # Проверка, не активен ли уже поток MRIQC для этого запуска
        # 'thread_mriqc_trigger' хранит объект потока
        current_mriqc_thread_obj = run_data.get('thread_mriqc_trigger')
        if current_mriqc_thread_obj and current_mriqc_thread_obj.is_alive():
            flask_logger.info(f"[{run_id}] Попытка запуска MRIQC, но предыдущий поток MRIQC еще активен.")
            return jsonify({"message": "Задача MRIQC уже выполняется для этого запуска. Пожалуйста, подождите."}), 409 # 409 Conflict

        # Проверка, если MRIQC уже был успешно завершен (или успешно выполнен через заглушку)
        # Это предотвращает случайный повторный запуск уже выполненной задачи.
        # Если нужен механизм перезапуска, эту логику нужно будет изменить.
        mriqc_status = run_data.get('status_mriqc', 'Not Started')
        if mriqc_status.startswith('MRIQC_Completed') or mriqc_status.startswith('MRIQC_Placeholder_Success'):
            flask_logger.info(f"[{run_id}] Попытка запуска MRIQC, но он уже был успешно выполнен (статус: {mriqc_status}).")
            return jsonify({"message": f"MRIQC уже был успешно выполнен для этого запуска (статус: {mriqc_status}). Для перезапуска может потребоваться другая процедура."}), 200 # OK, но ничего не делаем

        # Если дошли сюда, можно запускать
        run_data['mriqc_requested'] = True # Флаг, что пользователь запросил запуск
        run_data['status_mriqc'] = 'MRIQC_Queued' # Статус "в очереди" перед фактическим запуском потока
        run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC: Запрос на запуск на сервере принят, постановка в очередь...")
        flask_logger.info(f"[{run_id}] MRIQC поставлен в очередь. Запуск потока execute_remote_mriqc_task.")

        # Создание и запуск потока для выполнения execute_remote_mriqc_task
        # Передаем экземпляр Flask-приложения (current_flask_app) и run_id
        mriqc_thread = threading.Thread(
            target=execute_remote_mriqc_task, 
            args=(current_flask_app, run_id,) # Передаем текущий экземпляр Flask app
        )
        mriqc_thread.daemon = True # Позволяет основному процессу завершиться, даже если поток еще работает (обычно для dev сервера)
                                   # В продакшене может потребоваться более аккуратное управление жизненным циклом потоков.
        
        run_data['thread_mriqc_trigger'] = mriqc_thread # Сохраняем объект потока в active_runs
        mriqc_thread.start() # Запускаем поток

    # Возвращаем ответ, что запрос принят
    # Клиент должен будет опрашивать /api/status/<run_id> для получения обновлений
    return jsonify({"message": f"Запрос на запуск MRIQC на сервере для run_id '{run_id}' принят и поставлен в очередь."}), 202 # 202 Accepted

@app.route('/trigger_mriqc_auto/<run_id>', methods=['POST'])
def trigger_mriqc_auto_from_pipeline(run_id: str):
    """
    Эндпоинт для автоматического запуска MRIQC из run_pipeline.py.
    Предполагается, что run_pipeline.py вызывает его, когда NIfTI файлы готовы
    и конфигурация разрешает автоматический запуск.
    """
    flask_logger.info(f"Автоматический триггер MRIQC для run_id: {run_id} получен от пайплайна.")
    
    current_flask_app = app # Получаем текущий экземпляр Flask

    with active_runs_lock:
        run_data = active_runs.get(run_id)

        if not run_data:
            flask_logger.error(f"[{run_id}] Авто-триггер MRIQC: Run ID не найден.")
            return jsonify({"error": f"Run ID '{run_id}' не найден."}), 404

        # Проверяем, не активен ли уже поток MRIQC для этого запуска
        current_mriqc_thread_obj = run_data.get('thread_mriqc_trigger')
        if current_mriqc_thread_obj and current_mriqc_thread_obj.is_alive():
            flask_logger.info(f"[{run_id}] Авто-триггер MRIQC: Задача MRIQC уже выполняется.")
            return jsonify({"message": "Задача MRIQC уже выполняется для этого запуска."}), 200 # OK, уже запущено

        # Проверка, если MRIQC уже был успешно завершен
        mriqc_status = run_data.get('status_mriqc', 'Not Started')
        if mriqc_status.startswith('MRIQC_Completed') or mriqc_status.startswith('MRIQC_Placeholder_Success'):
            flask_logger.info(f"[{run_id}] Авто-триггер MRIQC: Задача MRIQC уже была успешно выполнена (статус: {mriqc_status}).")
            return jsonify({"message": f"MRIQC уже был успешно выполнен для этого запуска (статус: {mriqc_status})."}), 200 # OK, уже сделано

        # В этом эндпоинте мы доверяем, что run_pipeline.py вызвал его в правильный момент,
        # то есть NIfTI файлы готовы. Флаг can_run_mriqc должен быть уже true,
        # или будет установлен true в /api/status при следующей проверке.
        # Если он еще false, execute_remote_mriqc_task может выйти, если проверка там строгая.
        # Для надежности, можно здесь еще раз установить can_run_mriqc, если NIfTI есть.
        # (логика проверки can_run_mriqc из /api/status может быть вынесена в отдельную функцию)

        # Убедимся, что 'paths' и необходимые пути существуют в run_data
        if not run_data.get('paths', {}).get('bids_nifti'):
            flask_logger.error(f"[{run_id}] Авто-триггер MRIQC: Отсутствуют необходимые пути в run_data (например, bids_nifti).")
            return jsonify({"error": "Внутренняя ошибка сервера: отсутствуют пути для запуска MRIQC."}), 500

        # Если все хорошо, ставим в очередь и запускаем
        run_data['mriqc_requested'] = True # Помечаем, что запуск был инициирован (хоть и автоматически)
        run_data['status_mriqc'] = 'MRIQC_Queued_Auto' # Новый статус для автозапуска
        run_data.setdefault('user_log',[]).append(f"[{datetime.now().strftime('%H:%M:%S')}] MRIQC: Автоматический запуск инициирован пайплайном.")
        flask_logger.info(f"[{run_id}] MRIQC (авто) поставлен в очередь. Запуск потока execute_remote_mriqc_task.")

        mriqc_thread = threading.Thread(
            target=execute_remote_mriqc_task,
            args=(current_flask_app, run_id,)
        )
        mriqc_thread.daemon = True
        run_data['thread_mriqc_trigger'] = mriqc_thread
        mriqc_thread.start()

    return jsonify({"message": f"Автоматический запуск MRIQC для run_id '{run_id}' инициирован."}), 202

# === РЕАЛИЗАЦИЯ запуска сегментации ===
def run_segmentation_for_all_subjects_thread_target(
    run_id: str,
    preprocessed_dir_str: str,
    segmentation_output_root_str: str,
    main_config_path_str: str, # Путь к основному config.yaml
    overall_segmentation_log_path_str: str,
    p_config: dict # Уже загруженный pipeline_config
):
    """
    Потоковая функция для запуска скрипта сегментации для всех субъектов/сессий.
    """
    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data: flask_logger.error(f"[{run_id}] (SegThread) Run data not found."); return
        run_data['status_segmentation'] = 'Segmentation_Running_Init'
        run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Начало AI сегментации...")

    flask_logger.info(f"[{run_id}] (SegThread) Начало выполнения всех задач сегментации.")
    # Отдельный логгер для этого макро-шага сегментации
    seg_overall_logger = logging.getLogger(f"seg_overall_{run_id}")
    seg_overall_logger.setLevel(logging.INFO) # Пишем INFO и выше в этот лог
    # Очищаем обработчики, если они были от предыдущего запуска с тем же run_id (маловероятно)
    if seg_overall_logger.hasHandlers(): seg_overall_logger.handlers.clear()
    try:
        fh_overall = logging.FileHandler(overall_segmentation_log_path_str, mode='a', encoding='utf-8')
        # Используем тот же форматтер, что и основной логгер пайплайна для консистентности
        # или можно создать свой, более специфичный для сегментации
        seg_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [SegmentationOverall] %(message)s')
        fh_overall.setFormatter(seg_formatter)
        seg_overall_logger.addHandler(fh_overall)
        seg_overall_logger.info("Логгер для общего процесса сегментации настроен.")
    except Exception as e_log:
        flask_logger.error(f"[{run_id}] (SegThread) Ошибка настройки логгера для сегментации: {e_log}")
        # Продолжаем без файлового лога для этого макро-шага, если не удалось

    preprocessed_dir = Path(preprocessed_dir_str)
    segmentation_output_root_dir = Path(segmentation_output_root_str)
    scripts_dir_path = app.config['PIPELINE_SCRIPTS_DIR']
    segmentation_script_abs_path = (scripts_dir_path / "segmentation.py").resolve()
    python_exec = sys.executable # Используем тот же Python, что и для Flask/пайплайна

    modality_map_config = p_config.get('steps', {}).get('segmentation', {}).get('modality_input_map')
    if not modality_map_config:
        msg = "Карта модальностей 'modality_input_map' не найдена в конфиге. Сегментация невозможна."
        flask_logger.error(f"[{run_id}] (SegThread) {msg}"); seg_overall_logger.error(msg)
        with active_runs_lock: run_data['status_segmentation'] = 'Segmentation_Error_Config'; run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Сегментация: {msg}")
        return

    # --- Логика поиска и группировки файлов (из run_pipeline.py, адаптированная) ---
    subject_session_files_grouped = {}
    seg_overall_logger.debug(f"Поиск файлов в {preprocessed_dir} для группировки...")
    for subj_dir in preprocessed_dir.glob("sub-*"):
        if not subj_dir.is_dir(): continue
        for ses_dir in subj_dir.glob("ses-*"):
            if not ses_dir.is_dir(): continue
            anat_dir = ses_dir / "anat"
            if not anat_dir.is_dir(): continue
            current_subj_ses_key = (subj_dir.name, ses_dir.name)
            if current_subj_ses_key not in subject_session_files_grouped: subject_session_files_grouped[current_subj_ses_key] = {}
            for mod_key_internal, mod_identifier_str in modality_map_config.items():
                found_mod_file = None
                # Ищем файл, который содержит идентификатор модальности
                # Усложняем поиск, чтобы он был более точным
                # Пример: sub-001_ses-001_T1w.nii.gz или sub-001_ses-001_desc-preproc_T1w.nii.gz
                # Идентификатор должен быть окружен '_' или быть в конце/начале значимой части имени
                for n_file in anat_dir.glob(f"*{mod_identifier_str}*.nii*"):
                    if n_file.is_file():
                        # Более точная проверка, чтобы отличить T1w от ce-T1w, если идентификаторы пересекаются
                        if mod_key_internal == 't1' and modality_map_config.get('t1c') and modality_map_config.get('t1c') in n_file.name:
                            continue
                        found_mod_file = n_file; break
                if found_mod_file:
                    if mod_key_internal not in subject_session_files_grouped[current_subj_ses_key]:
                        subject_session_files_grouped[current_subj_ses_key][mod_key_internal] = found_mod_file
                        seg_overall_logger.debug(f"Найден {mod_key_internal} для {current_subj_ses_key}: {found_mod_file.name}")
    # --- Конец логики поиска ---

    if not subject_session_files_grouped:
        msg = f"Не найдено сгруппированных по sub/ses файлов в {preprocessed_dir} для сегментации."
        flask_logger.warning(f"[{run_id}] (SegThread) {msg}"); seg_overall_logger.warning(msg)
        with active_runs_lock: run_data['status_segmentation'] = 'Segmentation_NoData'; run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Сегментация: {msg}")
        return

    total_sets_to_process = len(subject_session_files_grouped)
    completed_seg_sets = 0; failed_seg_sets = 0
    seg_overall_logger.info(f"Найдено {total_sets_to_process} наборов (субъект/сессия) для сегментации.")

    for i, ((subj_id, sess_id), modalities_map) in enumerate(subject_session_files_grouped.items()):
        client_id = f"{subj_id}_{sess_id}"
        current_step_user_log = f"Сегментация для {client_id} ({i+1}/{total_sets_to_process})"
        with active_runs_lock:
            run_data = active_runs.get(run_id)
            if not run_data: flask_logger.error(f"[{run_id}] (SegThread) Данные запуска исчезли во время итерации."); return
            run_data['status_segmentation'] = f'Segmentation_Running: {client_id}'
            run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {current_step_user_log}: Запуск...")
        seg_overall_logger.info(f"--- {current_step_user_log} ---")

        required_modalities = ['t1', 't1c', 't2', 'flair']
        input_files_cmd_dict = {}; all_modalities_found = True
        for req_mod in required_modalities:
            if req_mod not in modalities_map:
                msg = f"Отсутствует модальность '{req_mod}'. Пропуск."
                seg_overall_logger.warning(f"  {client_id}: {msg}")
                with active_runs_lock: run_data['user_log'][-1] += f" {msg}" # Добавляем к последней строке
                all_modalities_found = False; break
            input_files_cmd_dict[f"--input_{req_mod}"] = str(modalities_map[req_mod])
        if not all_modalities_found: failed_seg_sets += 1; continue

        base_name_for_out = Path(modalities_map['t1'].name).stem.replace('.nii','').replace(f"_{modality_map_config['t1']}", "")
        if base_name_for_out.endswith('_'): base_name_for_out = base_name_for_out[:-1]
        output_mask_fname = f"{base_name_for_out}_segmask.nii.gz"
        output_mask_subfolder = segmentation_output_root_dir / subj_id / sess_id / "anat"
        output_mask_subfolder.mkdir(parents=True, exist_ok=True)
        output_mask_full_path = output_mask_subfolder / output_mask_fname

        # Индивидуальный лог для каждого вызова скрипта segmentation.py
        logs_subdir_name = p_config.get('paths', {}).get('subdirs', {}).get('logs', 'logs')
        seg_script_log_dir_base = Path(app.config['UPLOAD_FOLDER']) / run_id / logs_subdir_name / "10_segmentation_individual_logs"
        seg_script_log_dir_base.mkdir(parents=True, exist_ok=True)
        seg_script_log_file = seg_script_log_dir_base / f"{base_name_for_out}_segmentation_script.log"

        cmd_seg = [ python_exec, str(segmentation_script_abs_path),
                    "--output_mask", str(output_mask_full_path),
                    "--config", main_config_path_str,
                    "--client_id", client_id,
                    "--log_file", str(seg_script_log_file),
                    "--console_log_level", "DEBUG" ] # Лог этого скрипта всегда DEBUG в файл
        for arg_name, file_p_str in input_files_cmd_dict.items(): cmd_seg.extend([arg_name, file_p_str])
        seg_overall_logger.debug(f"  Команда: {' '.join(cmd_seg)}")

        try:
            seg_process = subprocess.run(cmd_seg, capture_output=True, text=True, check=True, encoding='utf-8', env=os.environ)
            seg_overall_logger.info(f"  Сегментация для {client_id} успешно завершена.")
            if seg_process.stdout: seg_overall_logger.debug(f"    Stdout: {seg_process.stdout.strip()}")
            if seg_process.stderr: seg_overall_logger.warning(f"    Stderr: {seg_process.stderr.strip()}")
            completed_seg_sets += 1
            with active_runs_lock: run_data['user_log'][-1] = f"[{datetime.now().strftime('%H:%M:%S')}] {current_step_user_log}: Успешно."
        except subprocess.CalledProcessError as e_seg:
            failed_seg_sets += 1
            msg = f"Ошибка сегментации (код: {e_seg.returncode})."
            seg_overall_logger.error(f"  {client_id}: {msg}")
            if e_seg.stdout: seg_overall_logger.error(f"    Stdout: {e_seg.stdout.strip()}")
            if e_seg.stderr: seg_overall_logger.error(f"    Stderr: {e_seg.stderr.strip()}")
            with active_runs_lock: run_data['user_log'][-1] = f"[{datetime.now().strftime('%H:%M:%S')}] {current_step_user_log}: {msg}"
        except Exception as e_fatal_seg:
            failed_seg_sets += 1
            msg = f"Критическая ошибка запуска скрипта сегментации."
            seg_overall_logger.exception(f"  {client_id}: {msg}")
            with active_runs_lock: run_data['user_log'][-1] = f"[{datetime.now().strftime('%H:%M:%S')}] {current_step_user_log}: {msg}"


    final_seg_status = ""
    if failed_seg_sets > 0 and completed_seg_sets > 0: final_seg_status = f"Segmentation_Completed_With_Errors ({failed_seg_sets} failed)"
    elif failed_seg_sets > 0: final_seg_status = f"Segmentation_Error ({failed_seg_sets} failed)"
    elif completed_seg_sets > 0: final_seg_status = "Segmentation_Completed"
    elif total_sets_to_process == 0 : final_seg_status = "Segmentation_NoDataToProcess" # Если не было файлов для обработки
    else: final_seg_status = "Segmentation_AllSkipped_Or_UnknownError" # Если все пропустили или др.

    seg_overall_logger.info(f"Общий итог этапа сегментации для run_id {run_id}: {final_seg_status}")
    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if run_data:
            run_data['status_segmentation'] = final_seg_status
            run_data['thread_segmentation'] = None
            run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] AI сегментация: {final_seg_status}.")
    for h_seg in seg_overall_logger.handlers[:]: seg_overall_logger.removeHandler(h_seg); h_seg.close()
    flask_logger.info(f"[{run_id}] (SegThread) Завершение всех задач сегментации.")


@app.route('/run_segmentation/<run_id>', methods=['POST'])
def trigger_segmentation_locally(run_id: str):
    """Запускает процесс AI сегментации для указанного run_id."""
    flask_logger.info(f"Запрос POST /run_segmentation/{run_id} от {request.remote_addr}")
    p_config_seg = app.config.get('PIPELINE_CONFIG', {})
    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data: return jsonify({"error": "Run ID не найден"}), 404
        if run_data.get('status_pipeline') != 'Completed':
            return jsonify({"error": "Основной пайплайн предобработки еще не завершен."}), 400
        if not run_data.get('can_run_segmentation', False):
             return jsonify({"error": "Условия для запуска сегментации не выполнены (нет предобработанных данных)."}), 400
        if run_data.get('segmentation_requested', False) and \
           ("Running" in run_data.get('status_segmentation', "") or "Queued" in run_data.get('status_segmentation', "")):
            return jsonify({"message": "Сегментация уже запущена или в очереди."}), 202

        run_data['segmentation_requested'] = True
        run_data['status_segmentation'] = 'Segmentation_Queued'

        # Используем сохраненные пути и app.config
        run_specific_dir = Path(run_data.get('paths', {}).get('base', str(Path(app.config['UPLOAD_FOLDER']) / run_id)))
        subdirs_cfg_seg = p_config_seg.get('paths', {}).get('subdirs', {})
        preprocessed_dir_name = subdirs_cfg_seg.get('preprocessed', 'preprocessed_data')
        preprocessed_input_dir = run_specific_dir / preprocessed_dir_name
        segmentation_masks_dir_name = subdirs_cfg_seg.get('segmentation_masks', 'segmentation_masks')
        segmentation_output_dir = run_specific_dir / segmentation_masks_dir_name
        logs_subdir_name = subdirs_cfg_seg.get('logs', 'logs')
        segmentation_overall_log_path = run_specific_dir / logs_subdir_name / "10_segmentation_overall.log"

        seg_thread = threading.Thread(
            target=run_segmentation_for_all_subjects_thread_target,
            args=(run_id, str(preprocessed_input_dir.resolve()),
                  str(segmentation_output_dir.resolve()), str(CONFIG_FILE_PATH.resolve()), # Передаем путь к конфигу
                  str(segmentation_overall_log_path.resolve()), p_config_seg) # Передаем загруженный конфиг
        )
        seg_thread.daemon = True; seg_thread.start()
        run_data['thread_segmentation'] = seg_thread
        flask_logger.info(f"Поток для AI сегментации ({run_id}) запущен.")
        return jsonify({"message": "Запрос на AI сегментацию принят."}), 202


if __name__ == '__main__':
    flask_logger.info(f"Запуск Flask dev server (host 0.0.0.0, port 5001, debug={app.debug})...")
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)