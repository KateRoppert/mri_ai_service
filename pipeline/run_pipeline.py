# pipeline/run_pipeline.py

import argparse
import yaml
import subprocess
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid
import traceback
import requests

# --- Настройка основного логгера пайплайна ---
logger = logging.getLogger("Pipeline")
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Pipeline] %(message)s')
# Временный консольный обработчик для самых ранних сообщений
temp_console_handler = logging.StreamHandler(sys.stdout)
temp_console_handler.setFormatter(log_formatter)
temp_console_handler.setLevel(logging.INFO)
logger.addHandler(temp_console_handler)

# Префикс для сообщений, предназначенных для пользователя в веб-интерфейсе
USER_LOG_PREFIX = "PIPELINE_USER_MSG:"

def setup_pipeline_logging(log_file_path: Path, console_level_str: str):
    """Настраивает логгер для пайплайна."""
    global temp_console_handler
    if temp_console_handler and temp_console_handler in logger.handlers:
        logger.removeHandler(temp_console_handler)
    temp_console_handler = None

    ch = logging.StreamHandler(sys.stdout)
    try: console_level = getattr(logging, console_level_str.upper())
    except AttributeError: console_level = logging.INFO; logger.warning(f"Неверный уровень лога консоли '{console_level_str}'. Используется INFO.")
    ch.setLevel(console_level); ch.setFormatter(log_formatter); logger.addHandler(ch)

    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='w')
        fh.setLevel(logging.DEBUG); fh.setFormatter(log_formatter); logger.addHandler(fh)
        logger.debug(f"Логирование пайплайна в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование пайплайна в файл {log_file_path}: {e}")


def run_step(cmd: list, step_name: str, step_log_path: Path):
    """Запускает шаг пайплайна, логирует и проверяет результат."""
    # Сообщение для пользователя о старте шага
    logger.info(f"{USER_LOG_PREFIX} Запуск шага - {step_name}...")
    logger.info(f"--- Запуск шага (детально): {step_name} ---") # Для основного лога

    # Убедимся, что все аргументы команды - строки
    cmd_str_list = [str(c) for c in cmd]
    cmd_str_list.extend(["--log_file", str(step_log_path)])

    cmd_str_log = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd_str_list])
    logger.debug(f"Команда: {cmd_str_log}")

    try:
        result = subprocess.run(
            cmd_str_list, check=True, capture_output=True, text=True,
            env=os.environ, encoding='utf-8'
        )
        if result.stdout: logger.debug(f"Stdout шага '{step_name}':\n{result.stdout.strip()}")
        if result.stderr: logger.warning(f"Stderr шага '{step_name}' (успех):\n{result.stderr.strip()}")

        # Сообщение для пользователя об успехе
        logger.info(f"{USER_LOG_PREFIX} Успешно - {step_name}. Лог шага: {step_log_path.name}")
        logger.info(f"--- Шаг '{step_name}' успешно завершен. ---") # Для основного лога
        return True

    except subprocess.CalledProcessError as e:
        # Сообщение для пользователя об ошибке
        logger.error(f"{USER_LOG_PREFIX} ОШИБКА - {step_name}. Код: {e.returncode}. См. лог шага: {step_log_path.name}")
        logger.critical(f"--- ОШИБКА на шаге: {step_name} (Код возврата: {e.returncode}) ---") # Для основного лога
        if e.stdout: logger.error(f"Stdout шага '{step_name}' перед ошибкой:\n{e.stdout.strip()}")
        if e.stderr: logger.error(f"Stderr шага '{step_name}' (ошибка):\n{e.stderr.strip()}")
        else: logger.error("Stderr не содержит дополнительной информации об ошибке.")
        logger.critical(f"Прерывание пайплайна из-за ошибки на шаге '{step_name}'.")
        logger.info(f"Для деталей ошибки см. лог шага: {step_log_path}")
        return False
    except FileNotFoundError:
        logger.error(f"{USER_LOG_PREFIX} КРИТИЧЕСКАЯ ОШИБКА - {step_name}: Не найден Python или скрипт шага.")
        logger.critical(f"--- ОШИБКА на шаге: {step_name} ---")
        logger.critical(f"Не удалось найти Python ('{cmd_str_list[0]}') или скрипт ('{cmd_str_list[1]}').")
        return False
    except Exception as e:
        logger.error(f"{USER_LOG_PREFIX} КРИТИЧЕСКАЯ ОШИБКА - {step_name}: Непредвиденная ошибка Python.")
        logger.critical(f"--- НЕПРЕДВИДЕННАЯ ОШИБКА Python при запуске шага: {step_name} ---")
        logger.exception(e)
        return False


def main():
    """Основная функция запуска пайплайна."""
    parser = argparse.ArgumentParser(
        description="Запускает пайплайн обработки МРТ данных.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, type=str, help="Путь к YAML конфигу.")
    parser.add_argument("--run_id", type=str, default=None, help="Уникальный ID запуска.")
    parser.add_argument("--input_data_dir", type=str, default=None, help="Переопределить входные данные.")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Переопределить базовый выход.")
    parser.add_argument("--console_log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Уровень лога консоли.")
    args = parser.parse_args()

    # --- 1. Загрузка конфигурации ---
    try:
        config_path = Path(args.config).resolve()
        if not config_path.is_file(): raise FileNotFoundError(f"Конфиг не найден: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    except Exception as e:
        print(f"[CRITICAL ERROR] Load/parse config {args.config}: {e}", file=sys.stderr)
        logger.critical(f"Load/parse config {args.config}: {e}", exc_info=True) # Уже в консоль через temp_handler
        sys.exit(1)

    # --- 2. Определение ID запуска и ключевых путей ---
    run_id_to_use = args.run_id if args.run_id else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    try:
        output_base_dir_from_cli_or_config = Path(args.output_base_dir or config['paths']['output_base_dir']).resolve()
        raw_input_dir = Path(args.input_data_dir or config['paths']['raw_input_dir']).resolve()
        template_path = Path(config['paths']['template_path']).resolve()
        subdirs_config = config['paths']['subdirs']
        executables_config = config.get('executables', {})
        steps_config = config.get('steps', {})
    except KeyError as e:
        print(f"[CRITICAL ERROR] Missing key in 'paths' of {config_path}: {e}", file=sys.stderr)
        logger.critical(f"Missing key in 'paths' of {config_path}: {e}")
        sys.exit(1)

    if output_base_dir_from_cli_or_config.name == run_id_to_use:
        run_output_dir = output_base_dir_from_cli_or_config
        effective_output_base_for_log = output_base_dir_from_cli_or_config.parent
    else:
        run_output_dir = output_base_dir_from_cli_or_config / run_id_to_use
        effective_output_base_for_log = output_base_dir_from_cli_or_config

    # --- 3. Настройка основного логирования пайплайна ---
    logs_subdir_name = subdirs_config.get('logs', 'logs')
    pipeline_log_dir = run_output_dir / logs_subdir_name
    pipeline_log_file = pipeline_log_dir / 'pipeline.log'
    try:
        pipeline_log_dir.mkdir(parents=True, exist_ok=True)
        setup_pipeline_logging(pipeline_log_file, args.console_log_level)
    except OSError as e:
        logger.critical(f"Не удалось создать {pipeline_log_dir} или настроить логгер: {e}")
        print(f"CRITICAL ERROR: Create log dir {pipeline_log_dir} or setup logger: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"{USER_LOG_PREFIX} Начало обработки пайплайна (ID: {run_id_to_use})")
    logger.info("=" * 60)
    logger.info(f"Запуск пайплайна обработки МРТ")
    # ... (остальные info логи как раньше) ...
    logger.info(f"Run ID: {run_id_to_use}")
    logger.info(f"Конфигурационный файл: {config_path}")
    logger.info(f"Входные сырые данные: {raw_input_dir}")
    logger.info(f"Базовая выходная директория (эффективная): {effective_output_base_for_log}")
    logger.info(f"Директория этого запуска: {run_output_dir}")
    logger.info(f"Файл шаблона: {template_path}")
    logger.info(f"Лог пайплайна: {pipeline_log_file}")
    logger.info(f"Уровень лога консоли: {args.console_log_level.upper()}")
    logger.info("=" * 60)


    # --- 4. Проверка входных данных и создание структуры директорий ---
    if not raw_input_dir.is_dir(): logger.critical(f"{USER_LOG_PREFIX} ОШИБКА: Входная директория не найдена: {raw_input_dir}"); sys.exit(1)
    if not template_path.is_file(): logger.critical(f"{USER_LOG_PREFIX} ОШИБКА: Файл шаблона не найден: {template_path}"); sys.exit(1)

    sd = subdirs_config
    bids_dicom_dir = run_output_dir / sd['bids_dicom']
    dicom_checks_dir = run_output_dir / sd['dicom_checks']; dicom_meta_dir = run_output_dir / sd['dicom_meta']
    bids_nifti_dir = run_output_dir / sd['bids_nifti']; validation_reports_dir = run_output_dir / sd['validation_reports']
    fast_qc_reports_dir = run_output_dir / sd['fast_qc_reports']; mriqc_output_dir = run_output_dir / sd['mriqc_output']
    mriqc_interpret_dir = run_output_dir / sd['mriqc_interpret']; transforms_dir = run_output_dir / sd['transforms']
    preprocessed_dir = run_output_dir / sd['preprocessed']
    output_dirs_to_create = [ bids_dicom_dir, dicom_checks_dir, dicom_meta_dir, bids_nifti_dir,
                              validation_reports_dir, fast_qc_reports_dir, mriqc_output_dir,
                              mriqc_interpret_dir, transforms_dir, preprocessed_dir ]
    
    # === ДОБАВЛЕНИЕ ДИРЕКТОРИИ ДЛЯ СЕГМЕНТАЦИИ ===
    segmentation_masks_subdir_name = sd.get('segmentation_masks', 'segmentation_masks')
    segmentation_output_root_dir = run_output_dir / segmentation_masks_subdir_name
    # Создаем также подпапку для индивидуальных логов сегментации внутри основной папки логов пайплайна
    segmentation_individual_logs_dir = pipeline_log_dir / "10_segmentation_logs"


    output_dirs_to_create = [
        bids_dicom_dir, dicom_checks_dir, dicom_meta_dir, bids_nifti_dir,
        validation_reports_dir, fast_qc_reports_dir, mriqc_output_dir,
        mriqc_interpret_dir, transforms_dir, preprocessed_dir,
        segmentation_output_root_dir, # <<< Добавлена
        segmentation_individual_logs_dir # <<< Добавлена
    ]
    try:
        logger.info(f"{USER_LOG_PREFIX} Создание структуры выходных директорий...")
        for dir_path in output_dirs_to_create: dir_path.mkdir(parents=True, exist_ok=True); logger.debug(f"  Создана/проверена: {dir_path}")
        logger.info(f"{USER_LOG_PREFIX} Структура выходных директорий готова.")
    except OSError as e: logger.critical(f"{USER_LOG_PREFIX} ОШИБКА: Не удалось создать выходные директории: {e}"); sys.exit(1)

    # --- 5. Определение путей к скриптам и исполняемым файлам ---
    try:
        project_root = Path(__file__).resolve().parent.parent
        scripts_dir = project_root / 'scripts'
        if not scripts_dir.is_dir(): raise FileNotFoundError(f"Директория scripts не найдена: {scripts_dir}")
    except NameError:
        logger.warning("Не удалось определить project_root. Используется cwd для scripts/.")
        scripts_dir = Path.cwd() / 'scripts'
        if not scripts_dir.is_dir(): logger.critical(f"{USER_LOG_PREFIX} КРИТИЧЕСКАЯ ОШИБКА: Директория scripts {scripts_dir} не найдена."); sys.exit(1)

    dciodvfy_exec = executables_config.get('dciodvfy', 'dciodvfy')
    dcm2niix_exec = executables_config.get('dcm2niix', 'dcm2niix')
    bids_validator_exec = executables_config.get('bids_validator', 'bids-validator')
    mriqc_exec = executables_config.get('mriqc', 'mriqc')
    python_executable = sys.executable

    # --- 6. Последовательный запуск шагов пайплайна ---
    # Шаг 1: reorganize_folders.py
    step_name = "1_Reorganize_to_BIDS_DICOM"
    step_script = scripts_dir / "reorganize_folders.py"
    step_log = pipeline_log_dir / "01_reorganize_folders.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(raw_input_dir),
            "--output_dir", str(bids_dicom_dir),
            #"--console_log_level", args.console_log_level 
            ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # Шаг 2: dicom_standard_check.py
    step_name = "2_DICOM_Standard_Check"
    step_script = scripts_dir / "dicom_standard_check.py"
    step_log = pipeline_log_dir / "02_dicom_check.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_dicom_dir),
            "--output_dir", str(dicom_checks_dir),
            "--dciodvfy_path", dciodvfy_exec,
            #"--console_log_level", args.console_log_level 
            ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # Шаг 3: extract_metadata.py
    step_name = "3_Extract_DICOM_Metadata"
    step_script = scripts_dir / "extract_metadata.py"
    step_log = pipeline_log_dir / "03_extract_metadata.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_dicom_dir),
            "--output_dir", str(dicom_meta_dir),
            #"--console_log_level", args.console_log_level 
            ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # Шаг 4: convert_dicom_to_nifti.py
    step_name = "4_Convert_DICOM_to_NIfTI"
    step_script = scripts_dir / "convert_dicom_to_nifti.py"
    step_log = pipeline_log_dir / "04_convert_nifti.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_dicom_dir),
            "--output_dir", str(bids_nifti_dir),
            "--dcm2niix_path", dcm2niix_exec,
            #"--console_log_level", args.console_log_level 
            ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # Шаг 5: bids_validation.py
    step_name = "5_BIDS_Validation"
    step_script = scripts_dir / "bids_validation.py"
    step_log = pipeline_log_dir / "05_bids_validation.log"
    cmd = [ python_executable, str(step_script),
            "--bids_dir", str(bids_nifti_dir),
            "--output_dir", str(validation_reports_dir),
            "--validator_path", bids_validator_exec,
            #"--console_log_level", args.console_log_level 
            ]
    if not run_step(cmd, step_name, step_log):
        logger.warning(f"{USER_LOG_PREFIX} ВНИМАНИЕ: Скрипт BIDS валидации завершился с ошибкой, но пайплайн продолжит работу. Проверьте отчеты валидации.")
        # Не выходим из пайплайна, если сам скрипт упал, но логируем как проблему
        # sys.exit(1) # Раскомментировать, если ошибка валидации должна останавливать пайплайн
    else:
        # Проверка содержимого отчета валидатора, если нужно принимать решение
        validator_report_file = validation_reports_dir / "bids_validator_report.txt"
        if validator_report_file.exists():
            with open(validator_report_file, 'r', encoding='utf-8') as f_report:
                report_content_val = f_report.read()
            if "error" in report_content_val.lower() or "failed" in report_content_val.lower(): # Упрощенная проверка
                 logger.warning(f"{USER_LOG_PREFIX} ВНИМАНИЕ: BIDS валидация обнаружила ошибки! См. {validator_report_file.name}")
            elif "warning" in report_content_val.lower():
                 logger.info(f"{USER_LOG_PREFIX} BIDS валидация обнаружила предупреждения. См. {validator_report_file.name}")
            else:
                 logger.info(f"{USER_LOG_PREFIX} BIDS валидация не обнаружила критических ошибок.")

    # --- Автоматический запуск MRIQC на сервере, если настроено ---
    logger.info("--- Проверка необходимости автоматического запуска MRIQC на сервере ---")
    mriqc_step_cfg = steps_config.get('mriqc', {}) # Получаем конфигурацию шага mriqc
    
    should_run_mriqc_generally = mriqc_step_cfg.get('enabled', False)
    run_mriqc_on_server = mriqc_step_cfg.get('run_on_server', False)
    auto_trigger_mriqc = mriqc_step_cfg.get('run_on_server_auto_trigger', False)

    # Предполагаем, что Flask работает на порту 5001 локально
    # Этот URL лучше бы вынести в конфигурацию, если он может меняться,
    # но для простоты пока захардкодим.
    FLASK_API_BASE_URL = "http://127.0.0.1:5001" # Убедитесь, что порт совпадает с портом вашего Flask-приложения

    if should_run_mriqc_generally and run_mriqc_on_server and auto_trigger_mriqc:
        if bids_nifti_dir.is_dir() and any(bids_nifti_dir.iterdir()): # Проверка, что папка NIfTI не пуста
            logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} NIfTI файлы готовы. Инициирую автоматический запуск MRIQC на сервере...")
            
            trigger_url = f"{FLASK_API_BASE_URL}/trigger_mriqc_auto/{run_id_to_use}"
            
            try:
                logger.debug(f"Отправка POST-запроса на: {trigger_url}")
                response = requests.post(trigger_url, timeout=15) # Таймаут 15 секунд

                if response.status_code == 202: # Ожидаем 202 Accepted от Flask
                    logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} Запрос на автоматический запуск MRIQC успешно отправлен на Flask. "
                                f"Ответ: {response.status_code} - {response.json().get('message', '')}")
                    logger.info(f"--- Запрос на авто-запуск MRIQC для {run_id_to_use} принят Flask API. Пайплайн продолжит локальную обработку. ---")
                elif response.status_code == 200: # Flask может вернуть 200, если задача уже была запущена/выполнена
                     logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} Запрос на автоматический запуск MRIQC обработан Flask. "
                                f"Возможно, задача уже была запущена/выполнена. Ответ: {response.status_code} - {response.json().get('message', '')}")
                     logger.info(f"--- Запрос на авто-запуск MRIQC для {run_id_to_use} обработан Flask API (возможно, уже запущен). Пайплайн продолжит локальную обработку. ---")
                else:
                    logger.error(f"[{run_id_to_use}] {USER_LOG_PREFIX} ОШИБКА при отправке запроса на авто-запуск MRIQC. "
                                 f"Статус: {response.status_code}, Ответ: {response.text}")
                    logger.error(f"--- Ошибка от Flask API при попытке авто-запуска MRIQC для {run_id_to_use}. Статус: {response.status_code} ---")
            
            except requests.exceptions.ConnectionError as e_conn:
                logger.error(f"[{run_id_to_use}] {USER_LOG_PREFIX} ОШИБКА: Не удалось подключиться к Flask API ({trigger_url}) для авто-запуска MRIQC.")
                logger.error(f"--- Ошибка соединения с Flask API ({trigger_url}): {e_conn} ---")
            except requests.exceptions.Timeout as e_timeout:
                logger.error(f"[{run_id_to_use}] {USER_LOG_PREFIX} ОШИБКА: Таймаут при подключении к Flask API ({trigger_url}) для авто-запуска MRIQC.")
                logger.error(f"--- Таймаут соединения с Flask API ({trigger_url}): {e_timeout} ---")
            except Exception as e_req:
                logger.error(f"[{run_id_to_use}] {USER_LOG_PREFIX} ОШИБКА: Непредвиденная ошибка при отправке запроса на авто-запуск MRIQC.")
                logger.error(f"--- Непредвиденная ошибка при запросе к Flask API: {e_req} ---", exc_info=True)
        else:
            logger.warning(f"[{run_id_to_use}] {USER_LOG_PREFIX} Авто-запуск MRIQC на сервере настроен, но папка BIDS NIfTI ({bids_nifti_dir}) пуста или не существует. Пропуск триггера.")
            logger.warning(f"--- Пропуск авто-запуска MRIQC: папка {bids_nifti_dir} пуста или не найдена. ---")
    elif should_run_mriqc_generally and run_mriqc_on_server and not auto_trigger_mriqc:
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} MRIQC настроен для запуска на сервере, но автоматический триггер отключен. Запуск через веб-интерфейс.")
        logger.info(f"--- Автоматический запуск MRIQC на сервере отключен в конфигурации. ---")
    elif should_run_mriqc_generally and not run_mriqc_on_server:
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} MRIQC настроен для локального запуска (будет выполнен позже, если это Шаг 7).")
        logger.info(f"--- MRIQC настроен для локального запуска (Шаг 7). ---")
    else: # should_run_mriqc_generally is False
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} MRIQC отключен в конфигурации.")
        logger.info(f"--- MRIQC полностью отключен в конфигурации. ---")


    # Шаг 6: quality_metrics.py
    step_name = "6_Fast_Quality_Metrics"
    step_script = scripts_dir / "quality_metrics.py"
    step_log = pipeline_log_dir / "06_fast_qc.log"
    qc_config = steps_config.get('quality_metrics', {})
    anisotropy_thresh_qc = qc_config.get('anisotropy_thresh', 3.0)
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_nifti_dir),
            "--output_dir", str(fast_qc_reports_dir),
            "--anisotropy_thresh", str(anisotropy_thresh_qc),
            #"--console_log_level", args.console_log_level 
            ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # Шаг 7: mriqc_quality.py (Опциональный)
    step_name = "7_MRIQC_Analysis"
    mriqc_config = steps_config.get('mriqc', {})
    if mriqc_step_cfg.get('enabled', False) and not mriqc_step_cfg.get('run_on_server', False):
        logger.info(f"--- Запуск ЛОКАЛЬНОГО MRIQC (Шаг 7) ---")
        step_script = scripts_dir / "mriqc_quality.py"
        step_log = pipeline_log_dir / "07_mriqc.log"
        mriqc_error_log_name = mriqc_config.get("error_log_filename", "mriqc_run_errors.log")
        mriqc_error_log_path = pipeline_log_dir / mriqc_error_log_name
        cmd = [ python_executable, str(step_script),
                "--bids_dir", str(bids_nifti_dir),
                "--output_dir", str(mriqc_output_dir),
                "--report_type", mriqc_config.get('report_type', 'both'),
                "--mriqc_path", mriqc_exec,
                "--n_procs", str(mriqc_config.get('n_procs', 1)),
                "--n_threads", str(mriqc_config.get('n_threads', 1)),
                "--mem_gb", str(mriqc_config.get('mem_gb', 4)),
                "--error_log", str(mriqc_error_log_path),
                "--console_log_level", args.console_log_level ]
        if not run_step(cmd, step_name, step_log): sys.exit(1)
    elif mriqc_step_cfg.get('enabled', False) and mriqc_step_cfg.get('run_on_server', False):
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} Пропущен локальный шаг - {step_name}, так как MRIQC настроен для запуска на сервере.")
        logger.info(f"--- Шаг '{step_name}' (локальный MRIQC) пропущен, так как он запускается/запущен на сервере. ---")
    else: # mriqc_step_cfg.get('enabled', False) is False
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} Пропущен шаг - {step_name} (отключен в конфигурации).")
        logger.info(f"--- Шаг '{step_name}' (локальный MRIQC) пропущен (отключен). ---")

    # Шаг 8: mriqc_interpretation.py (Опциональный)
    step_name = "8_MRIQC_Interpretation"
    mriqc_interpret_config = steps_config.get('mriqc_interpretation', {})
    if mriqc_step_cfg.get('enabled', False) and \
       not mriqc_step_cfg.get('run_on_server', False) and \
       mriqc_interpret_config.get('enabled', False):
        logger.info(f"--- Запуск ЛОКАЛЬНОЙ интерпретации MRIQC (Шаг 8) ---") 
        if mriqc_output_dir.exists() and any(mriqc_output_dir.iterdir()):
            step_script = scripts_dir / "mriqc_interpretation.py"
            step_log = pipeline_log_dir / "08_mriqc_interpret.log"
            cmd = [ python_executable, str(step_script),
                    "--mriqc_dir", str(mriqc_output_dir),
                    "--output_dir", str(mriqc_interpret_dir),
                    #"--console_log_level", args.console_log_level 
                    ]
            if not run_step(cmd, step_name, step_log): sys.exit(1)
        else:
            logger.warning(f"{USER_LOG_PREFIX} Пропущен шаг - {step_name}: папка результатов MRIQC не найдена или пуста.")
            logger.warning(f"--- Шаг '{step_name}' пропущен: {mriqc_output_dir} не найдена/пуста. ---")
    elif mriqc_step_cfg.get('enabled', False) and mriqc_step_cfg.get('run_on_server', False):
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} Пропущен локальный шаг - {step_name}, так как MRIQC (и его интерпретация) обрабатывается на сервере.")
        logger.info(f"--- Шаг '{step_name}' (локальная интерпретация MRIQC) пропущен, обрабатывается на сервере. ---")
    else:
        logger.info(f"[{run_id_to_use}] {USER_LOG_PREFIX} Пропущен шаг - {step_name} (отключен в конфигурации).")
        logger.info(f"--- Шаг '{step_name}' (локальная интерпретация MRIQC) пропущен (отключен). ---")
        
    # Шаг 9: preprocessing.py
    step_name = "9_Preprocessing"
    step_script = scripts_dir / "preprocessing.py"
    step_log = pipeline_log_dir / "09_preprocessing.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_nifti_dir),
            "--output_prep_dir", str(preprocessed_dir),
            "--output_transform_dir", str(transforms_dir),
            "--template_path", str(template_path),
            "--config", str(config_path),
            "--console_log_level", args.console_log_level ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # Шаг 10: segmentation.py
    step_name_segmentation_base = "10_AI_Segmentation"
    segmentation_config = steps_config.get('segmentation', {}) # Параметры шага из конфига
    aiaa_server_url = executables_config.get('aiaa_server_url')   # URL сервера AIAA

    if segmentation_config.get('enabled', False) and aiaa_server_url:
        logger.info(f"{USER_LOG_PREFIX} Запуск этапа сегментации...")
        segmentation_script_path = scripts_dir / "segmentation.py" 
        modality_map_config_from_pipeline = segmentation_config.get('modality_input_map') 

        if not modality_map_config_from_pipeline:
            logger.error(
                f"{USER_LOG_PREFIX} ОШИБКА - {step_name_segmentation_base}: "
                f"Карта модальностей 'modality_input_map' не найдена в config -> steps -> segmentation. Пропуск сегментации."
            )
        else:
            subject_session_modality_files = {}
            logger.debug(f"Поиск предобработанных файлов для сегментации в: {preprocessed_dir}")
            for subj_dir in preprocessed_dir.glob("sub-*"):
                if not subj_dir.is_dir(): continue
                for ses_dir in subj_dir.glob("ses-*"):
                    if not ses_dir.is_dir(): continue
                    anat_dir = ses_dir / "anat"
                    if not anat_dir.is_dir(): continue
                    current_subj_ses_key = (subj_dir.name, ses_dir.name)
                    if current_subj_ses_key not in subject_session_modality_files:
                        subject_session_modality_files[current_subj_ses_key] = {}
                    
                    # Ключи, которые ожидает segmentation.py (t1, t1c, t2, flair)
                    expected_script_mod_keys = ['t1', 't1c', 't2', 'flair']
                    for script_mod_key in expected_script_mod_keys:
                        # Ищем соответствующий идентификатор из конфига пайплайна
                        # (например, для script_mod_key 't1c', ищем 'ce-gd_T1w' в modality_map_config_from_pipeline)
                        mod_identifier_str = modality_map_config_from_pipeline.get(script_mod_key)
                        if not mod_identifier_str:
                            logger.warning(f"Для {current_subj_ses_key}: идентификатор для ключа '{script_mod_key}' не найден в modality_input_map конфига. Эта модальность будет пропущена для segmentation.py.")
                            continue

                        found_mod_file = None
                        for n_file in anat_dir.glob(f"*{mod_identifier_str}*.nii*"): 
                            if n_file.is_file():
                                if script_mod_key == 't1': # Особая проверка для T1, чтобы не спутать с T1c
                                    t1c_identifier = modality_map_config_from_pipeline.get('t1c')
                                    if t1c_identifier and t1c_identifier in n_file.name:
                                        continue 
                                found_mod_file = n_file
                                break 
                        
                        if found_mod_file:
                            subject_session_modality_files[current_subj_ses_key][script_mod_key] = found_mod_file
                            logger.debug(f"  Найден файл для {subj_dir.name}/{ses_dir.name}: {script_mod_key} -> {found_mod_file.name}")
                        else:
                            logger.info(f"  Файл для модальности '{script_mod_key}' (идент: '{mod_identifier_str}') не найден в {anat_dir} для {subj_dir.name}/{ses_dir.name}. Будет передан как отсутствующий.")
                            # Оставляем отсутствующий ключ в subject_session_modality_files[current_subj_ses_key]
                            # или явно subject_session_modality_files[current_subj_ses_key][script_mod_key] = None
                            # Но удобнее, если его просто не будет, а segmentation.py ожидает args.input_t1 и т.д.
                            # которые будут None, если аргумент не передан.

            if not subject_session_modality_files:
                logger.warning(f"{USER_LOG_PREFIX} {step_name_segmentation_base}: Не найдено сгруппированных по sub/ses файлов в {preprocessed_dir}.")
            
            segmentation_errors_count = 0
            for (subj_id, ses_id), found_modalities_map in subject_session_modality_files.items():
                logger.info(f"--- Подготовка к сегментации для {subj_id}/{ses_id} ---")
                
                client_id_for_server = f"{subj_id}_{ses_id}"
                
                # Формируем пути для выходной маски и индивидуального лога
                # Пытаемся взять имя на основе T1, если он есть, иначе первый доступный или дефолт.
                base_name_source_file = None
                if 't1' in found_modalities_map:
                    base_name_source_file = found_modalities_map['t1']
                elif found_modalities_map: # Если хоть какие-то модальности есть
                    base_name_source_file = next(iter(found_modalities_map.values()))
                
                if base_name_source_file:
                    base_name_for_output = base_name_source_file.name.replace(".nii.gz", "").replace(".nii", "")
                    t1_identifier_from_config = modality_map_config_from_pipeline.get('t1', 'T1w') # Идентификатор T1 из конфига
                    # Удаляем основной идентификатор модальности (например, _T1w), чтобы имя было чище
                    base_name_for_output = base_name_for_output.replace(f"_{t1_identifier_from_config}", "")
                else: # Если ни одного файла не нашлось (хотя это маловероятно, если current_subj_ses_key существует)
                    base_name_for_output = f"{subj_id}_{ses_id}" # Дефолтное имя
                    logger.warning(f"Для {client_id_for_server} не найдено ни одного файла для формирования базового имени, используется дефолт.")

                output_mask_filename = f"{base_name_for_output}_segmask.nii.gz"
                output_mask_subfolder = segmentation_output_root_dir / subj_id / ses_id / "anat"
                output_mask_subfolder.mkdir(parents=True, exist_ok=True)
                output_mask_file_path = output_mask_subfolder / output_mask_filename

                seg_log_filename = f"{base_name_for_output}_segmentation.log"
                individual_seg_log_path = segmentation_individual_logs_dir / seg_log_filename

                # Формируем команду для запуска segmentation.py
                # Теперь передаем аргументы, только если файл найден.
                # segmentation.py должен быть готов к тому, что некоторые аргументы --input_* могут отсутствовать.
                cmd_segmentation = [
                    python_executable, str(segmentation_script_path),
                    "--output_mask", str(output_mask_file_path),
                    "--config", str(config_path),
                    "--client_id", client_id_for_server,
                    "--console_log_level", args.console_log_level
                ]
                
                # Добавляем аргументы для найденных модальностей
                if found_modalities_map.get('t1'):
                    cmd_segmentation.extend(["--input_t1", str(found_modalities_map['t1'])])
                if found_modalities_map.get('t1c'):
                    cmd_segmentation.extend(["--input_t1ce", str(found_modalities_map['t1c'])])
                if found_modalities_map.get('t2'):
                    cmd_segmentation.extend(["--input_t2", str(found_modalities_map['t2'])])
                if found_modalities_map.get('flair'):
                    cmd_segmentation.extend(["--input_flair", str(found_modalities_map['flair'])])

                # Проверяем, есть ли хотя бы один входной файл для segmentation.py
                # (segmentation.py сам должен это проверить и выйти, если ни одного нет, но можно и здесь)
                if not any(key in found_modalities_map for key in ['t1', 't1c', 't2', 'flair']):
                    logger.error(f"{USER_LOG_PREFIX} ОШИБКА - {step_name_segmentation_base} для {client_id_for_server}: "
                                 f"Не найдено ни одного из необходимых файлов модальностей (T1, T1c, T2, FLAIR). Пропуск сегментации.")
                    segmentation_errors_count += 1
                    continue

                step_name_current_seg = f"{step_name_segmentation_base} для {client_id_for_server}"
                if not run_step(cmd_segmentation, step_name_current_seg, individual_seg_log_path):
                    segmentation_errors_count += 1
                    logger.error(f"{USER_LOG_PREFIX} Ошибка сегментации для: {client_id_for_server}. См. лог: {individual_seg_log_path.name}")
            
            if segmentation_errors_count > 0:
                logger.warning(f"{USER_LOG_PREFIX} Завершено с {segmentation_errors_count} ошибками на этапе сегментации.")
            else:
                logger.info(f"{USER_LOG_PREFIX} Все наборы файлов (с хотя бы одной модальностью) успешно переданы на сегментацию.")
    else:
        logger.info(f"{USER_LOG_PREFIX} Пропущен шаг - {step_name_segmentation_base} (отключен или не настроен URL сервера).")
        logger.info(f"--- Шаг '{step_name_segmentation_base}' пропущен. ---")


    # --- Завершение пайплайна ---
    logger.info("=" * 60)
    logger.info(f"{USER_LOG_PREFIX} Успех - Пайплайн завершил все запущенные шаги.")
    logger.info("Пайплайн успешно завершил все запущенные шаги.") # Для основного лога
    logger.info(f"Результаты сохранены в: {run_output_dir.resolve()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    bootstrap_console_handler = logging.StreamHandler(sys.stdout)
    bootstrap_console_handler.setFormatter(log_formatter)
    bootstrap_console_handler.setLevel(logging.INFO)
    logger.addHandler(bootstrap_console_handler)
    try:
        main()
        if bootstrap_console_handler in logger.handlers : logger.removeHandler(bootstrap_console_handler)
        sys.exit(0)
    except SystemExit as e:
        if bootstrap_console_handler in logger.handlers : logger.removeHandler(bootstrap_console_handler)
        sys.exit(e.code)
    except Exception as e:
        print(f"[CRITICAL UNEXPECTED ERROR] Pipeline failed at top level.", file=sys.stderr)
        traceback.print_exc()
        if logger.hasHandlers() and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers if h != bootstrap_console_handler):
            logger.critical(f"Непредвиденная ошибка: {e}", exc_info=True)
        else: print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        if bootstrap_console_handler in logger.handlers : logger.removeHandler(bootstrap_console_handler)
        sys.exit(1)