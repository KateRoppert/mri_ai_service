import os
import subprocess
import zipfile
import uuid
from datetime import datetime
import threading
import logging
from logging.handlers import RotatingFileHandler
import time # Для имитации задержек и проверки статуса
import shutil

import yaml
from pathlib import Path
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, send_from_directory, session) # Добавили session
from werkzeug.utils import secure_filename
import sys # Для sys.executable

# --- Конфигурация приложения ---
app = Flask(__name__)

# --- Глобальные переменные и структуры ---
# { 'run_id': {
#       'status_pipeline': 'Queued' | 'Running - Step X' | 'Completed' | 'Error - Step X',
#       'status_mriqc': 'Not Started' | 'MRIQC_Requested' | 'MRIQC_Running' | 'MRIQC_Completed' | 'MRIQC_Error',
#       'start_time_obj': datetime_obj,
#       'start_time_iso': 'iso_string',
#       'start_time_display': 'display_string',
#       'pipeline_log_path': Path_obj,
#       'mriqc_error_log_path': Path_obj, # Лог ошибок самого MRIQC
#       'user_log': ['log_line_1', ...], # Упрощенный лог для пользователя
#       'process_pipeline': Popen_obj | None,
#       'thread_pipeline': Thread_obj | None,
#       'process_mriqc_trigger': Popen_obj | None, # Для scp/ssh команд
#       'thread_mriqc_trigger': Thread_obj | None,
#       'reports': [{'name': '...', 'url': '...'}],
#       'can_run_mriqc': False, # Становится True после завершения нужных шагов пайплайна
#       'mriqc_requested': False
# }}
active_runs = {}
# Используем блокировку для безопасного доступа к active_runs из разных потоков
active_runs_lock = threading.Lock()


# --- Настройка логгера для Flask ---
flask_logger = logging.getLogger('flask_webapp')
flask_logger.setLevel(logging.INFO) # Установим INFO для Flask логгера, DEBUG для пайплайна
flask_log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
# (Код setup_logging для flask_logger как в предыдущем ответе, создаем webapp/logs/webapp.log)
webapp_log_dir = Path(__file__).parent / "logs_webapp" # Отдельная папка для логов веб-приложения
webapp_log_dir.mkdir(parents=True, exist_ok=True)
flask_log_file = webapp_log_dir / "webapp.log"
file_handler = RotatingFileHandler(flask_log_file, maxBytes=1024000, backupCount=10, encoding='utf-8') # Увеличил размер
file_handler.setFormatter(flask_log_formatter)
file_handler.setLevel(logging.INFO)
flask_logger.addHandler(file_handler)
# Добавим вывод в консоль для отладки Flask
if not app.debug or os.environ.get("FLASK_ENV") == "development": # Более точная проверка для dev
    console_handler_flask = logging.StreamHandler(sys.stdout)
    console_handler_flask.setFormatter(flask_log_formatter)
    console_handler_flask.setLevel(logging.INFO) # INFO для консоли Flask
    flask_logger.addHandler(console_handler_flask)
flask_logger.info("Flask логгер настроен.")


# --- Загрузка конфигурации пайплайна ---
CONFIG_FILE_PATH = Path(__file__).parent.parent / "config/config.yaml"
pipeline_config = None
PIPELINE_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
PIPELINE_RUN_SCRIPT = Path(__file__).parent.parent / "pipeline" / "run_pipeline.py"

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
    # Устанавливаем запасной вариант, чтобы приложение могло хотя бы запуститься и показать ошибку
    fallback_dir = Path(__file__).parent.parent / "PIPELINE_RESULTS_CONFIG_ERROR"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = str(fallback_dir.resolve())
    flask_logger.error(f"Используется запасной путь для результатов: {app.config['UPLOAD_FOLDER']}")
    # В реальном приложении здесь можно было бы завершить работу Flask, если конфиг критичен
    # sys.exit(1)


app.config['ALLOWED_EXTENSIONS'] = {'zip'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # 500 MB
app.config['SECRET_KEY'] = os.urandom(24) # Для flash сообщений

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def parse_pipeline_log(log_path: Path, last_n_lines=50) -> list[str]:
    """Читает лог-файл пайплайна для отображения пользователю."""
    user_log_lines = []
    if not log_path: # Если путь к логу не определен
        return ["[Путь к лог-файлу пайплайна не определен]"]

    if log_path.exists() and log_path.is_file():
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                user_log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Лог-файл пайплайна пуст ({log_path.name}). Ожидание записей...")
                return user_log_lines

            # Простой вариант: берем все строки или последние N
            # Отфильтруем пустые строки для чистоты вывода
            processed_lines = [line.strip() for line in lines if line.strip()]

            # Можно оставить фильтрацию, если она нужна, но сделать ее менее строгой в начале
            # relevant_lines = [
            #     line for line in processed_lines
            #     if "[Pipeline]" in line or "--- Шаг" in line or "--- ОШИБКА" in line or "успешно завершен" in line or "Запуск пайплайна" in line
            # ]
            # if not relevant_lines and processed_lines: # Если фильтрация ничего не дала, но строки есть
            #     user_log_lines = processed_lines[-last_n_lines:] # Берем просто последние
            # else:
            #     user_log_lines = relevant_lines[-last_n_lines:]
            user_log_lines = processed_lines[-last_n_lines:] # Пока просто берем последние N непустых строк

            if not user_log_lines: # Если после всех обработок список пуст
                 user_log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] В логе пайплайна пока нет отображаемых сообщений.")

        except Exception as e:
            flask_logger.error(f"Ошибка чтения лог-файла {log_path}: {e}")
            user_log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] [Ошибка чтения лога: {e}]")
    else:
        user_log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Лог-файл пайплайна ({log_path.name}) еще не создан или не найден.")
    return user_log_lines

def find_reports(run_specific_dir: Path, run_id: str) -> list[dict]:
    """Ищет готовые отчеты в папке запуска."""
    reports = []
    if not pipeline_config: # Если конфиг не загружен
        return reports

    subdirs = pipeline_config.get('paths', {}).get('subdirs', {})

    # 1. Отчет BIDS Validation
    validation_report_dir = run_specific_dir / subdirs.get('validation_reports', 'validation_results')
    bids_validator_report = validation_report_dir / "bids_validator_report.txt"
    if bids_validator_report.exists():
        reports.append({
            "name": "Отчет BIDS Валидации",
            "url": url_for('download_file_from_run', run_id=run_id, subpath=str(bids_validator_report.relative_to(run_specific_dir)))
        })

    # 2. Отчеты Fast QC
    fast_qc_dir = run_specific_dir / subdirs.get('fast_qc_reports', 'bids_quality_metrics')
    if fast_qc_dir.exists():
        for report_file in fast_qc_dir.rglob("*_quality_report.txt"):
            relative_report_path = report_file.relative_to(run_specific_dir)
            reports.append({
                "name": f"FastQC: {report_file.stem.replace('_quality_report','')}",
                "url": url_for('download_file_from_run', run_id=run_id, subpath=str(relative_report_path))
            })

    # 3. Отчеты MRIQC (HTML)
    mriqc_out_dir = run_specific_dir / subdirs.get('mriqc_output', 'mriqc_output')
    if mriqc_out_dir.exists():
        # Ищем основные HTML отчеты
        for html_report in mriqc_out_dir.glob("*.html"): # Групповые отчеты
             reports.append({
                "name": f"MRIQC Отчет: {html_report.name}",
                "url": url_for('download_file_from_run', run_id=run_id, subpath=str(html_report.relative_to(run_specific_dir)))
            })
        # Индивидуальные отчеты могут быть в подпапках sub-XXX
        for subj_dir in mriqc_out_dir.glob("sub-*"):
            if subj_dir.is_dir():
                for html_report in subj_dir.rglob("*.html"): # Рекурсивный поиск HTML
                     reports.append({
                        "name": f"MRIQC {subj_dir.name}: {html_report.name}",
                        "url": url_for('download_file_from_run', run_id=run_id, subpath=str(html_report.relative_to(run_specific_dir)))
                    })

    # 4. Файлы интерпретации MRIQC
    mriqc_interpret_dir = run_specific_dir / subdirs.get('mriqc_interpret', 'mriqc_interpretation')
    if mriqc_interpret_dir.exists():
        for interpret_file in mriqc_interpret_dir.rglob("*_interpretation.txt"):
            relative_interpret_path = interpret_file.relative_to(run_specific_dir)
            reports.append({
                "name": f"Интерпретация MRIQC: {interpret_file.stem.replace('_interpretation','')}",
                "url": url_for('download_file_from_run', run_id=run_id, subpath=str(relative_interpret_path))
            })
    
    # 5. Ссылка на папку с предобработанными данными (если они есть)
    preprocessed_dir_name = subdirs.get('preprocessed', 'preprocessed_data')
    preprocessed_path = run_specific_dir / preprocessed_dir_name
    if preprocessed_path.exists() and any(preprocessed_path.iterdir()): # Если папка есть и не пуста
        # Прямое скачивание папки сложно, даем информацию
        # Можно реализовать архивацию и скачивание архива позже
        reports.append({
            "name": f"Предобработанные данные (NIfTI) находятся в папке: {preprocessed_dir_name}",
            "url": "#" # Заглушка, т.к. папку напрямую не скачать
        })


    return reports

def monitor_pipeline_process(run_id: str, process: subprocess.Popen, pipeline_log_path: Path):
    """
    Отслеживает завершение процесса пайплайна в отдельном потоке,
    логирует stdout/stderr и обновляет статус.
    """
    flask_logger.info(f"Мониторинг пайплайна для run_id: {run_id} запущен в потоке {threading.get_ident()}.")
    stdout, stderr = process.communicate() # Блокирует до завершения процесса

    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data:
            flask_logger.error(f"Данные для run_id {run_id} не найдены в active_runs после завершения процесса.")
            return

        # Записываем полный stdout/stderr пайплайна в его основной лог
        try:
            with open(pipeline_log_path, 'a', encoding='utf-8') as f:
                f.write("\n\n--- Pipeline Process STDOUT ---\n")
                f.write(stdout or "[No stdout]")
                f.write("\n\n--- Pipeline Process STDERR ---\n")
                f.write(stderr or "[No stderr]")
                f.write("\n--- End Pipeline Process Output ---")
            flask_logger.debug(f"Stdout/Stderr пайплайна для {run_id} записаны в {pipeline_log_path}")
        except Exception as e:
            flask_logger.error(f"Ошибка записи stdout/stderr пайплайна для {run_id} в лог: {e}")


        if process.returncode == 0:
            run_data['status_pipeline'] = 'Completed'
            run_data['can_run_mriqc'] = True # Теперь можно запускать MRIQC
            flask_logger.info(f"Пайплайн для run_id: {run_id} успешно завершен.")
        else:
            run_data['status_pipeline'] = f'Error (Code: {process.returncode})'
            flask_logger.error(f"Пайплайн для run_id: {run_id} завершился с ошибкой (код: {process.returncode}).")
        run_data['process_pipeline'] = None # Очищаем объект процесса
        run_data['thread_pipeline'] = None  # Очищаем объект потока
        # Добавляем финальное сообщение в user_log
        run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Локальный пайплайн: {run_data['status_pipeline']}")

# --- Маршруты ---
@app.route('/')
def index():
    """Отображает главную страницу с формой загрузки и историей запусков."""
    flask_logger.info(f"Запрос GET / от {request.remote_addr}")
    with active_runs_lock:
        # Сортируем копию элементов для безопасной итерации в шаблоне
        sorted_runs = sorted(
            active_runs.items(),
            key=lambda item: item[1].get('start_time_iso', '0'),
            reverse=True
        )
    return render_template('index.html', active_runs=sorted_runs)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Обрабатывает загрузку ZIP-архива, создает структуру папок для запуска
    и запускает основной пайплайн обработки в фоновом потоке.
    """
    flask_logger.info(f"Запрос POST /upload от {request.remote_addr}")
    if 'dicom_archive' not in request.files:
        flash('Файл не был выбран.', 'error')
        flask_logger.warning("Запрос /upload: Файл 'dicom_archive' отсутствует.")
        return redirect(url_for('index'))

    file = request.files['dicom_archive']
    if file.filename == '':
        flash('Файл не был выбран (пустое имя).', 'error')
        flask_logger.warning("Запрос /upload: Имя файла пустое.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        flask_logger.info(f"Генерация run_id: {run_id} для файла {filename}")

        run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
        input_archive_dir = run_specific_dir / "input_archive"
        input_raw_data_dir = run_specific_dir / "input_raw_data" # Это будет --input_data_dir для пайплайна
        logs_subdir_name = pipeline_config.get('paths', {}).get('subdirs', {}).get('logs', 'logs') # Из конфига
        pipeline_log_path_for_run = run_specific_dir / logs_subdir_name / 'pipeline.log'

        try:
            flask_logger.debug(f"Создание директорий для run_id {run_id} в {run_specific_dir}")
            run_specific_dir.mkdir(parents=True, exist_ok=True)
            input_archive_dir.mkdir(exist_ok=True)
            input_raw_data_dir.mkdir(exist_ok=True)
        except OSError as e:
            flask_logger.error(f"Не удалось создать директории для run_id {run_id}: {e}", exc_info=True)
            flash(f"Ошибка сервера при создании директорий для запуска: {e}", "error")
            return redirect(url_for('index'))

        # Сохраняем загруженный архив
        archive_path = input_archive_dir / filename
        try:
            file.save(str(archive_path))
            flask_logger.info(f"Архив '{filename}' сохранен в {archive_path}")
        except Exception as e:
            flask_logger.error(f"Не удалось сохранить архив {archive_path}: {e}", exc_info=True)
            flash(f"Ошибка сохранения загруженного архива: {e}", "error")
            return redirect(url_for('index'))

        # Распаковываем архив
        try:
            flask_logger.info(f"Распаковка архива {archive_path} в {input_raw_data_dir}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(input_raw_data_dir)
            flask_logger.info(f"Архив успешно распакован.")
        except zipfile.BadZipFile:
            flask_logger.error(f"Файл {archive_path} не является корректным ZIP-архивом.")
            flash("Загруженный файл не является корректным ZIP-архивом.", "error")
            return redirect(url_for('index'))
        except Exception as e:
            flask_logger.error(f"Ошибка при распаковке архива {archive_path}: {e}", exc_info=True)
            flash(f"Ошибка распаковки архива: {e}", "error")
            return redirect(url_for('index'))
        
        # --- Дополнительная логика для "поднятия" содержимого, если есть одна корневая папка ---
        # Эта переменная будет передана в пайплайн как --input_data_dir
        effective_input_data_dir = input_raw_data_dir

        items_in_raw_data = list(input_raw_data_dir.iterdir())
        if len(items_in_raw_data) == 1 and items_in_raw_data[0].is_dir():
            # Если внутри input_raw_data только одна папка, предполагаем, что это общая папка архива
            single_root_folder_in_zip = items_in_raw_data[0]
            flask_logger.info(
                f"Обнаружена одна корневая папка в архиве: {single_root_folder_in_zip.name}. "
                f"Содержимое этой папки будет использовано как входные данные."
            )
            effective_input_data_dir = single_root_folder_in_zip

        # Формируем команду для запуска пайплайна
        # Пайплайн сам создаст подпапки logs/ и results/ внутри run_specific_dir
        # на основе своего output_base_dir и run_id
        # Формируем команду для запуска пайплайна, используя effective_input_data_dir
        cmd = [
            sys.executable,
            str(PIPELINE_RUN_SCRIPT.resolve()),
            "--config", str(CONFIG_FILE_PATH.resolve()),
            "--run_id", run_id,
            "--input_data_dir", str(effective_input_data_dir.resolve()), # <<< ИСПОЛЬЗУЕМ ЭТОТ ПУТЬ
            "--output_base_dir", str(run_specific_dir.resolve()),
            "--console_log_level", "DEBUG"
        ]
        flask_logger.info(f"Команда для запуска пайплайна ({run_id}): {' '.join(cmd)}")

        # Запускаем пайплайн в отдельном потоке
        try:
            # stdout и stderr будут перенаправлены в Popen объект
            # cwd устанавливаем в корень проекта, чтобы относительные пути в пайплайне работали
            project_root_dir = Path(__file__).parent.parent
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', cwd=str(project_root_dir))
            flask_logger.info(f"Пайплайн для run_id: {run_id} запущен с PID: {process.pid}")

            pipeline_log_path = run_specific_dir / pipeline_config['paths']['subdirs']['logs'] / 'pipeline.log'

            # Запускаем поток для мониторинга
            monitor_thread = threading.Thread(target=monitor_pipeline_process, args=(run_id, process, pipeline_log_path))
            monitor_thread.daemon = True # Поток завершится, если основной процесс Flask упадет
            monitor_thread.start()

            # Сохраняем информацию о запуске
            with active_runs_lock:
                active_runs[run_id] = {
                    'status_pipeline': 'Queued',
                    'status_mriqc': 'Not Started',
                    'start_time_obj': datetime.now(),
                    'start_time_iso': datetime.now().isoformat(),
                    'start_time_display': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'pipeline_log_path': pipeline_log_path,
                    'user_log': [f"[{datetime.now().strftime('%H:%M:%S')}] Запуск {run_id} добавлен в очередь."],
                    'process_pipeline': process, # Сохраняем объект Popen
                    'thread_pipeline': monitor_thread, # Сохраняем объект Thread
                    'reports': [],
                    'can_run_mriqc': False,
                    'mriqc_requested': False,
                    'pipeline_log_path': pipeline_log_path_for_run # Сохраняем объект Path
                }
            flash(f'Файл успешно загружен. Обработка запущена с ID: {run_id}', 'success')
            return redirect(url_for('processing_status', run_id=run_id))

        except Exception as e:
            flask_logger.critical(f"Критическая ошибка при запуске пайплайна для run_id {run_id}: {e}", exc_info=True)
            flash(f"Ошибка сервера при запуске обработки: {e}", "error")
            # Попытка очистить созданные папки, если пайплайн не запустился
            if run_specific_dir.exists():
                try: shutil.rmtree(run_specific_dir)
                except Exception as e_clean: flask_logger.error(f"Ошибка очистки {run_specific_dir}: {e_clean}")
            return redirect(url_for('index'))
    else:
        flash('Недопустимый тип файла. Разрешены только .zip архивы.', 'error')
        flask_logger.warning(f"Попытка загрузить недопустимый тип файла: {file.filename if file else 'No file'}")
        return redirect(url_for('index'))


@app.route('/status/<run_id>')
def processing_status(run_id: str):
    """Отображает страницу статуса для конкретного запуска."""
    flask_logger.info(f"Запрос GET /status/{run_id} от {request.remote_addr}")
    with active_runs_lock:
        run_info = active_runs.get(run_id) # Получаем копию, чтобы избежать гонок состояний при передаче в шаблон
        if run_info:
            # Копируем для передачи в шаблон, чтобы избежать изменения во время рендеринга
            run_info_copy = run_info.copy()
            # Упрощенный лог уже должен быть в run_info_copy['user_log']
            # Отчеты также должны быть там
        else:
            run_info_copy = None

    if not run_info_copy:
        flash(f'Запуск с ID {run_id} не найден.', 'error')
        flask_logger.warning(f"Запуск {run_id} не найден в active_runs.")
        return redirect(url_for('index'))
    return render_template('processing.html', run_id=run_id, run_info=run_info_copy)


@app.route('/api/status/<run_id>')
def api_status(run_id: str):
    flask_logger.debug(f"API запрос статуса для run_id: {run_id}")
    with active_runs_lock:
        run_data = active_runs.get(run_id)
        if not run_data:
            flask_logger.warning(f"API: Запуск {run_id} не найден.")
            return jsonify({'error': 'Run ID not found'}), 404

        current_pipeline_status = run_data.get('status_pipeline', 'Unknown')
        current_mriqc_status = run_data.get('status_mriqc', 'Not Started')

        # Обновляем user_log из файла лога пайплайна
        pipeline_log_path = run_data.get('pipeline_log_path')
        if pipeline_log_path: # Проверяем, что путь вообще есть
            # Всегда пытаемся прочитать лог, parse_pipeline_log обработает отсутствие файла
            flask_logger.debug(f"API: Пытаюсь прочитать лог из: {run_data['pipeline_log_path']}")
            run_data['user_log'] = parse_pipeline_log(pipeline_log_path)
        else:
            # Если пути к логу нет в run_data, добавляем сообщение
            if not run_data.get('user_log') or "Путь к лог-файлу пайплайна не определен" not in run_data['user_log'][-1]:
                 run_data['user_log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] Путь к лог-файлу пайплайна не определен в данных запуска.")

        # ... (остальная логика обновления отчетов и статуса mriqc) ...
        run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
        run_data['reports'] = find_reports(run_specific_dir, run_id)

        if current_pipeline_status == 'Completed':
            bids_nifti_path = run_specific_dir / pipeline_config['paths']['subdirs'].get('bids_nifti', 'bids_data_nifti')
            if bids_nifti_path.exists() and any(bids_nifti_path.iterdir()):
                run_data['can_run_mriqc'] = True
        # else: # Не сбрасываем can_run_mriqc, если он уже был true
            # run_data['can_run_mriqc'] = False

        response_data = {
            'run_id': run_id,
            'status_pipeline': current_pipeline_status,
            'status_mriqc': current_mriqc_status,
            'user_log': run_data['user_log'],
            'reports': run_data['reports'],
            'can_run_mriqc': run_data.get('can_run_mriqc', False),
            'mriqc_requested': run_data.get('mriqc_requested', False)
        }
        return jsonify(response_data)


@app.route('/download/<run_id>/<path:subpath>')
def download_file_from_run(run_id: str, subpath: str):
    """
    Позволяет скачивать файлы из директории результатов конкретного запуска.
    `subpath` должен быть относительным путем внутри папки запуска.
    """
    flask_logger.info(f"Запрос на скачивание: run_id={run_id}, subpath={subpath} от {request.remote_addr}")
    run_specific_dir_abs = (Path(app.config['UPLOAD_FOLDER']) / run_id).resolve()

    # Формируем полный путь к файлу и проверяем безопасность
    # subpath может содержать '..' - Path.resolve() должен это обработать
    # Но лучше дополнительно проверить, что мы не выходим за пределы run_specific_dir_abs
    file_to_download_abs = (run_specific_dir_abs / subpath).resolve()

    if not str(file_to_download_abs).startswith(str(run_specific_dir_abs)):
        flask_logger.warning(f"Попытка несанкционированного доступа к файлу: {file_to_download_abs} (за пределами {run_specific_dir_abs})")
        flash("Ошибка: Запрошен недопустимый путь к файлу.", "error")
        return redirect(url_for('processing_status', run_id=run_id)), 403 # Forbidden

    if not file_to_download_abs.is_file():
        flask_logger.error(f"Файл не найден для скачивания: {file_to_download_abs}")
        flash(f"Файл '{subpath}' не найден для запуска {run_id}.", "error")
        return redirect(url_for('processing_status', run_id=run_id)), 404 # Not Found

    try:
        # send_from_directory требует путь к директории и имя файла
        return send_from_directory(
            directory=str(file_to_download_abs.parent),
            path=str(file_to_download_abs.name),
            as_attachment=True # Скачать как вложение
        )
    except Exception as e:
        flask_logger.exception(f"Ошибка при отправке файла {file_to_download_abs}: {e}")
        flash(f"Ошибка сервера при скачивании файла.", "error")
        return redirect(url_for('processing_status', run_id=run_id)), 500


if __name__ == '__main__':
    flask_logger.info(f"Запуск Flask development server на хосте 0.0.0.0, порт 5000 (debug={app.debug})...")
    # Включение threaded=True может быть полезно, если у вас есть фоновые задачи,
    # но будьте осторожны с разделяемыми ресурсами.
    # Для production используйте WSGI сервер (gunicorn, uWSGI).
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)