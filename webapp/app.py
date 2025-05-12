import os
import subprocess # Будет нужен позже для запуска пайплайна
import zipfile    # Будет нужен позже для распаковки
import uuid
from datetime import datetime
import threading  # Будет нужен позже для фоновых задач
import logging    # Для логгера Flask
from logging.handlers import RotatingFileHandler # Для ротации логов Flask

import yaml
from pathlib import Path
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, send_from_directory) # Добавили flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename # Для безопасных имен файлов

# --- Конфигурация приложения ---
# Создаем экземпляр Flask приложения
app = Flask(__name__)

# --- Глобальные переменные и структуры ---
# Словарь для отслеживания активных/прошлых запусков пайплайна.
# В реальном приложении это лучше хранить в базе данных или более надежном хранилище.
# Структура: { 'run_id_1': {'status': '...', 'start_time': ..., 'log_path': ..., ...}, ... }
active_runs = {}

# --- Настройка логгера для Flask ---
# Этот логгер будет специфичен для веб-приложения (запросы, ошибки Flask и т.д.),
# он отличается от логгера пайплайна.
flask_logger = logging.getLogger('flask_webapp')
flask_logger.setLevel(logging.DEBUG) # Устанавливаем уровень для логгера Flask
flask_log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# Создаем папку для логов веб-приложения, если ее нет
webapp_log_dir = Path(__file__).parent / "logs" # Например, webapp/logs/
webapp_log_dir.mkdir(parents=True, exist_ok=True)
flask_log_file = webapp_log_dir / "webapp.log"

# Файловый обработчик с ротацией
file_handler = RotatingFileHandler(flask_log_file, maxBytes=102400, backupCount=5, encoding='utf-8')
file_handler.setFormatter(flask_log_formatter)
file_handler.setLevel(logging.INFO) # В файл пишем INFO и выше для Flask
flask_logger.addHandler(file_handler)
if not app.debug: # Не добавляем в консоль, если debug=True, т.к. Flask сам выводит
    # Консольный обработчик для логгера Flask (опционально)
    console_handler_flask = logging.StreamHandler(sys.stdout)
    console_handler_flask.setFormatter(flask_log_formatter)
    console_handler_flask.setLevel(logging.INFO)
    flask_logger.addHandler(console_handler_flask)

flask_logger.info("Flask логгер настроен.")

# --- Загрузка конфигурации пайплайна ---
# Путь к config.yaml относительно текущего файла app.py
CONFIG_FILE_PATH = Path(__file__).parent.parent / "config.yaml"
pipeline_config = None
try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        pipeline_config = yaml.safe_load(f)
    flask_logger.info(f"Конфигурация пайплайна успешно загружена из {CONFIG_FILE_PATH}")

    # --- Конфигурация Flask на основе pipeline_config ---
    # Папка для загрузок и всех результатов запусков пайплайна
    # Эта папка должна быть доступна для записи веб-серверу
    UPLOAD_FOLDER_BASE_STR = pipeline_config.get('paths', {}).get('output_base_dir')
    if not UPLOAD_FOLDER_BASE_STR:
        flask_logger.critical("Ключ 'paths.output_base_dir' не найден в config.yaml!")
        # Устанавливаем запасной вариант и логируем критическую ошибку
        UPLOAD_FOLDER_BASE_STR = str(Path(__file__).parent.parent / "PIPELINE_RESULTS_FALLBACK")
        flask_logger.warning(f"Используется запасной путь для результатов: {UPLOAD_FOLDER_BASE_STR}")

    UPLOAD_FOLDER_BASE = Path(UPLOAD_FOLDER_BASE_STR)
    UPLOAD_FOLDER_BASE.mkdir(parents=True, exist_ok=True) # Создаем, если ее нет
    app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER_BASE.resolve())
    flask_logger.info(f"Папка для загрузок и результатов установлена в: {app.config['UPLOAD_FOLDER']}")

except FileNotFoundError:
    flask_logger.critical(f"Конфигурационный файл пайплайна не найден: {CONFIG_FILE_PATH}")
    # Устанавливаем запасной вариант или останавливаем приложение
    app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent.parent / "PIPELINE_RESULTS_ERROR_NO_CONFIG")
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    flask_logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Работа без config.yaml. Результаты будут в {app.config['UPLOAD_FOLDER']}")
except yaml.YAMLError as e:
    flask_logger.critical(f"Ошибка парсинга конфигурационного файла {CONFIG_FILE_PATH}: {e}")
    # Аналогично, запасной вариант или остановка
    app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent.parent / "PIPELINE_RESULTS_ERROR_YAML_CONFIG")
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    flask_logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Работа с невалидным config.yaml. Результаты будут в {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    flask_logger.critical(f"Непредвиденная ошибка при загрузке конфигурации: {e}", exc_info=True)
    app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent.parent / "PIPELINE_RESULTS_ERROR_UNKNOWN_CONFIG")
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    flask_logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Результаты будут в {app.config['UPLOAD_FOLDER']}")


# Разрешенные расширения для загружаемых файлов
app.config['ALLOWED_EXTENSIONS'] = {'zip'}
# Максимальный размер загружаемого файла (например, 500 MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
# Секретный ключ для Flask (нужен для flash сообщений и сессий)
# В реальном приложении его нужно генерировать и хранить безопаснее
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    """Проверяет, имеет ли файл разрешенное расширение."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# --- Маршруты ---
@app.route('/')
def index():
    """Отображает главную страницу с формой загрузки."""
    flask_logger.info(f"Запрос главной страницы {request.remote_addr}")
    # Передаем список активных запусков в шаблон для отображения истории
    # Сортируем по времени старта (если оно есть)
    sorted_runs = sorted(
        active_runs.items(),
        key=lambda item: item[1].get('start_time_iso', '0'), # Используем ISO строку для сортировки
        reverse=True
    )
    return render_template('index.html', active_runs=sorted_runs)


# --- Заглушки для маршрутов, которые будут реализованы позже ---
@app.route('/upload', methods=['POST'])
def upload_file():
    """Обрабатывает загрузку файла и запускает пайплайн."""
    flask_logger.info(f"Попытка загрузки файла от {request.remote_addr}")
    # Здесь будет логика из Шага 3.4
    # ...
    # Пример:
    if 'dicom_archive' not in request.files:
        flash('Файл не был выбран.', 'error')
        flask_logger.warning("Файл не был выбран для загрузки.")
        return redirect(url_for('index'))
    file = request.files['dicom_archive']
    if file.filename == '':
        flash('Файл не был выбран.', 'error')
        flask_logger.warning("Имя файла пустое.")
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) # Безопасное имя файла
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
        # ... (создание папок, сохранение, распаковка, запуск пайплайна) ...
        flask_logger.info(f"Файл '{filename}' загружен. Запускается обработка с run_id: {run_id}")
        # Добавляем в active_runs
        active_runs[run_id] = {
            'status': 'Queued', # Начальный статус
            'start_time_obj': datetime.now(),
            'start_time_iso': datetime.now().isoformat(),
            'start_time_display': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_log': [f"[{datetime.now().strftime('%H:%M:%S')}] Запуск {run_id} добавлен в очередь."],
            'process': None, # Объект Popen
            'thread': None,  # Объект Thread
            'reports': []    # Список доступных отчетов
        }
        # !!! ЗДЕСЬ ДОЛЖЕН БЫТЬ ЗАПУСК ПАЙПЛАЙНА В ОТДЕЛЬНОМ ПОТОКЕ !!!
        # Например:
        # pipeline_thread = threading.Thread(target=start_pipeline_processing, args=(run_id, run_specific_dir, filename, ...))
        # pipeline_thread.start()
        # active_runs[run_id]['thread'] = pipeline_thread

        flash(f'Файл успешно загружен. Обработка запущена с ID: {run_id}', 'success')
        return redirect(url_for('processing_status', run_id=run_id))
    else:
        flash('Недопустимый тип файла. Разрешены только .zip архивы.', 'error')
        flask_logger.warning(f"Попытка загрузить недопустимый тип файла: {file.filename}")
        return redirect(url_for('index'))


@app.route('/status/<run_id>')
def processing_status(run_id: str):
    """Отображает страницу статуса для конкретного запуска."""
    flask_logger.info(f"Запрос страницы статуса для run_id: {run_id} от {request.remote_addr}")
    run_info = active_runs.get(run_id)
    if not run_info:
        flash(f'Запуск с ID {run_id} не найден.', 'error')
        return redirect(url_for('index'))
    return render_template('processing.html', run_id=run_id, run_info=run_info)


@app.route('/api/status/<run_id>')
def api_status(run_id: str):
    """Возвращает JSON со статусом обработки для AJAX запросов."""
    flask_logger.debug(f"API запрос статуса для run_id: {run_id}")
    run_info = active_runs.get(run_id)
    if not run_info:
        return jsonify({'error': 'Run ID not found'}), 404
    # Здесь будет логика обновления статуса из Popen и парсинга логов
    # ...
    # Пример возвращаемых данных
    return jsonify({
        'run_id': run_id,
        'status': run_info.get('status', 'Unknown'),
        'user_log': run_info.get('user_log', []),
        'reports': run_info.get('reports', [])
    })

@app.route('/download/<run_id>/<path:subpath>')
def download_file_from_run(run_id: str, subpath: str):
    """Позволяет скачивать файлы из директории конкретного запуска."""
    flask_logger.info(f"Запрос на скачивание: run_id={run_id}, subpath={subpath}")
    run_specific_dir = Path(app.config['UPLOAD_FOLDER']) / run_id
    # Путь subpath должен быть относительным внутри run_specific_dir
    # Например, 'logs/pipeline.log' или 'results/bids_validation/bids_validator_report.txt'
    # Важно обеспечить безопасность, чтобы subpath не выходил за пределы run_specific_dir
    # send_from_directory делает это достаточно безопасно
    try:
        # Проверяем, что путь не выходит за пределы директории запуска
        full_path = (run_specific_dir / subpath).resolve()
        if not str(full_path).startswith(str(run_specific_dir.resolve())):
            flask_logger.warning(f"Попытка доступа к файлу вне директории запуска: {subpath}")
            return "Access denied", 403

        return send_from_directory(directory=run_specific_dir, path=subpath, as_attachment=True)
    except FileNotFoundError:
        flask_logger.error(f"Файл не найден для скачивания: {run_specific_dir / subpath}")
        flash(f"Файл {subpath} не найден для запуска {run_id}.", "error")
        return redirect(url_for('processing_status', run_id=run_id)) # Или 404


if __name__ == '__main__':
    # Запуск Flask development server
    # В реальном развертывании используется WSGI сервер (gunicorn, uWSGI)
    # Устанавливаем хост 0.0.0.0, чтобы был доступен извне Docker или по сети (если нужно)
    # debug=True - удобно для разработки, НО НЕ ИСПОЛЬЗУЙТЕ В ПРОДАШЕНЕ!
    flask_logger.info("Запуск Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)