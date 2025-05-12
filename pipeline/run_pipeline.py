import argparse
import yaml
import subprocess
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid
import traceback # Для вывода traceback в main

# --- Настройка основного логгера пайплайна ---
logger = logging.getLogger("Pipeline")
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Pipeline] %(message)s')
# Обработчики будут добавлены после определения пути к логу

def setup_pipeline_logging(log_file_path: Path, console_level: str = "INFO"):
    """Настраивает логгер для самого пайплайна."""
    if logger.hasHandlers(): logger.handlers.clear()
    # Консоль
    ch = logging.StreamHandler(sys.stdout)
    try: ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError: ch.setLevel(logging.INFO)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
    # Файл
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='w') # 'w' для перезаписи лога пайплайна
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
        logger.debug(f"Логирование пайплайна в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование пайплайна в файл {log_file_path}: {e}")
        print(f"ERROR: Failed to setup pipeline log file at {log_file_path}: {e}", file=sys.stderr)


def run_step(cmd: list, step_name: str, step_log_path: Path):
    """
    Запускает один шаг пайплайна как внешний процесс, проверяет результат и логирует.

    Args:
        cmd (list): Список команды и аргументов для subprocess.run.
        step_name (str): Название шага для логирования.
        step_log_path (Path): Путь к лог-файлу для этого шага.

    Returns:
        bool: True в случае успеха, False в случае ошибки.
    """
    logger.info(f"--- Запуск шага: {step_name} ---")
    # Убедимся, что путь к логу шага передан как строка
    cmd.extend(["--log_file", str(step_log_path)])
    # Формируем строку команды для логирования (экранируем пробелы, если нужно)
    cmd_str_log = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
    logger.debug(f"Команда: {cmd_str_log}")

    try:
        # Запускаем процесс
        result = subprocess.run(
            cmd,
            check=True,            # Выбросить CalledProcessError при ненулевом коде возврата
            capture_output=True,   # Захватить stdout и stderr
            text=True,             # Декодировать как текст
            env=os.environ,        # Передать текущее окружение
            encoding='utf-8'       # Явно указать кодировку
        )
        # Логируем stdout на уровне DEBUG (может быть много информации)
        if result.stdout:
            logger.debug(f"Stdout шага '{step_name}':\n{result.stdout.strip()}")
        # Логируем stderr как WARNING, если он есть при успешном завершении
        if result.stderr:
             logger.warning(f"Stderr шага '{step_name}' (при успешном коде возврата):\n{result.stderr.strip()}")
        logger.info(f"--- Шаг '{step_name}' успешно завершен. ---")
        return True

    except subprocess.CalledProcessError as e:
        logger.critical(f"--- ОШИБКА на шаге: {step_name} (Код возврата: {e.returncode}) ---")
        if e.stdout:
            logger.error(f"Stdout шага '{step_name}' перед ошибкой:\n{e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Stderr шага '{step_name}' (ошибка):\n{e.stderr.strip()}")
        else:
             logger.error("Stderr не содержит дополнительной информации об ошибке.")
        logger.critical(f"Прерывание выполнения пайплайна из-за ошибки на шаге '{step_name}'.")
        logger.info(f"Для деталей ошибки см. лог шага: {step_log_path}")
        return False # Сигнализируем об ошибке
    except FileNotFoundError:
        logger.critical(f"--- ОШИБКА на шаге: {step_name} ---")
        logger.critical(f"Не удалось найти Python ('{cmd[0]}') или скрипт шага ('{cmd[1]}').")
        logger.critical("Проверьте пути и переменную PATH.")
        return False
    except Exception as e:
        logger.critical(f"--- НЕПРЕДВИДЕННАЯ ОШИБКА Python при запуске шага: {step_name} ---")
        logger.exception(e) # Логируем полный traceback непредвиденной ошибки
        return False


def main():
    """Основная функция запуска пайплайна."""
    parser = argparse.ArgumentParser(
        description="Запускает пайплайн обработки МРТ данных согласно конфигурационному файлу.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Путь к конфигурационному файлу YAML."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Уникальный ID для этого запуска. Если не указан, генерируется автоматически (например, 'run_YYYYMMDD_HHMMSS_uuid')."
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default=None,
        help="Переопределить путь к входным сырым данным из конфиг. файла."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Переопределить базовую выходную директорию из конфиг. файла."
    )
    parser.add_argument(
        "--console_log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования для вывода в консоль."
    )
    # Можно добавить флаги для переопределения enabled статуса шагов
    # parser.add_argument("--force_run_mriqc", action="store_true", help="Принудительно запустить MRIQC, даже если disabled в конфиге.")
    # parser.add_argument("--skip_mriqc", action="store_true", help="Пропустить MRIQC, даже если enabled в конфиге.")

    args = parser.parse_args()

    # --- 1. Загрузка конфигурации ---
    try:
        config_path = Path(args.config).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load or parse config file {args.config}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Определение ID запуска и ключевых путей ---
    # Если run_id не передан, генерируем его
    run_id_to_use = args.run_id if args.run_id else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Получаем пути из конфига, с возможностью переопределения через CLI
    try:
        # output_base_dir_from_cli_or_config - это директория, *внутри* которой будет создана папка run_id_to_use,
        # ЛИБО это уже путь к папке конкретного запуска (если передано из Flask)
        output_base_dir_from_cli_or_config = Path(args.output_base_dir or config['paths']['output_base_dir']).resolve()
        raw_input_dir = Path(args.input_data_dir or config['paths']['raw_input_dir']).resolve()
        template_path = Path(config['paths']['template_path']).resolve()
        subdirs_config = config['paths']['subdirs']
        executables_config = config.get('executables', {})
        steps_config = config.get('steps', {})
    except KeyError as e: print(f"CRITICAL ERROR: Missing key in 'paths' of {config_path}: {e}", file=sys.stderr); sys.exit(1)


    # === ИЗМЕНЕННАЯ ЛОГИКА ФОРМИРОВАНИЯ run_output_dir ===
    # Если output_base_dir_from_cli_or_config уже заканчивается на run_id_to_use (случай вызова из Flask),
    # то это и есть наша run_output_dir.
    # Иначе (при ручном запуске), мы создаем run_output_dir внутри output_base_dir_from_cli_or_config.
    if output_base_dir_from_cli_or_config.name == run_id_to_use:
        run_output_dir = output_base_dir_from_cli_or_config
        # В этом случае, "истинный" output_base_dir (родитель папки запуска) - это output_base_dir_from_cli_or_config.parent
        # Но для логирования оставим output_base_dir_from_cli_or_config как "переданный базовый выход"
        effective_output_base_for_log = output_base_dir_from_cli_or_config.parent
    else:
        run_output_dir = output_base_dir_from_cli_or_config / run_id_to_use
        effective_output_base_for_log = output_base_dir_from_cli_or_config

    # --- 3. Настройка основного логирования пайплайна ---
    logs_subdir_name = subdirs_config.get('logs', 'logs')
    pipeline_log_dir = run_output_dir / logs_subdir_name # Логи теперь внутри run_output_dir
    pipeline_log_file = pipeline_log_dir / 'pipeline.log'
    try:
         pipeline_log_dir.mkdir(parents=True, exist_ok=True)
         setup_pipeline_logging(pipeline_log_file, args.console_log_level)
    except OSError as e: print(f"CRITICAL ERROR: Create log dir {pipeline_log_dir} or setup logger: {e}", file=sys.stderr); sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Запуск пайплайна обработки МРТ")
    logger.info(f"Run ID: {run_id_to_use}") # Используем актуальный run_id
    logger.info(f"Конфигурационный файл: {config_path}")
    logger.info(f"Входные сырые данные: {raw_input_dir}")
    logger.info(f"Базовая выходная директория (эффективная): {effective_output_base_for_log}")
    logger.info(f"Директория этого запуска: {run_output_dir}") # Это теперь всегда /.../run_ID
    logger.info(f"Файл шаблона: {template_path}")
    logger.info(f"Лог пайплайна: {pipeline_log_file}")
    logger.info("=" * 60)

    # --- 4. Проверка входных данных и создание структуры директорий ---
    if not raw_input_dir.is_dir(): logger.critical(f"Входная директория не найдена: {raw_input_dir}"); sys.exit(1)
    if not template_path.is_file(): logger.critical(f"Файл шаблона не найден: {template_path}"); sys.exit(1)

    # Все пути к поддиректориям строятся от run_output_dir
    sd = subdirs_config
    bids_dicom_dir = run_output_dir / sd['bids_dicom']
    dicom_checks_dir = run_output_dir / sd['dicom_checks']
    # ... (и так далее для всех остальных директорий из вашего предыдущего кода main()) ...
    dicom_meta_dir = run_output_dir / sd['dicom_meta']
    bids_nifti_dir = run_output_dir / sd['bids_nifti']
    validation_reports_dir = run_output_dir / sd['validation_reports']
    fast_qc_reports_dir = run_output_dir / sd['fast_qc_reports']
    mriqc_output_dir = run_output_dir / sd['mriqc_output']
    mriqc_interpret_dir = run_output_dir / sd['mriqc_interpret']
    transforms_dir = run_output_dir / sd['transforms']
    preprocessed_dir = run_output_dir / sd['preprocessed']

    output_dirs_to_create = [
        bids_dicom_dir, dicom_checks_dir, dicom_meta_dir, bids_nifti_dir,
        validation_reports_dir, fast_qc_reports_dir, mriqc_output_dir,
        mriqc_interpret_dir, transforms_dir, preprocessed_dir
    ]
    try:
        logger.info("Создание структуры выходных директорий...")
        for dir_path in output_dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"  Создана/проверена: {dir_path}")
        logger.info("Структура выходных директорий готова.")
    except OSError as e: logger.critical(f"Не удалось создать выходные директории: {e}"); sys.exit(1)
    
    # --- 5. Определение путей к скриптам и исполняемым файлам ---
    try:
        project_root = Path(__file__).resolve().parent.parent # Папка, где лежат pipeline/ и scripts/
        scripts_dir = project_root / 'scripts'
        if not scripts_dir.is_dir():
            raise FileNotFoundError(f"Директория со скриптами не найдена по ожидаемому пути: {scripts_dir}")
    except NameError: # Если __file__ не определен (например, при интерактивном запуске)
        logger.warning("Не удалось определить project_root через __file__. Используется текущая директория для поиска scripts/.")
        scripts_dir = Path.cwd() / 'scripts'
        if not scripts_dir.is_dir():
             logger.critical(f"Директория со скриптами {scripts_dir} не найдена.")
             sys.exit(1)


    # Получаем пути к исполняемым файлам
    dciodvfy_exec = executables_config.get('dciodvfy', 'dciodvfy')
    dcm2niix_exec = executables_config.get('dcm2niix', 'dcm2niix')
    bids_validator_exec = executables_config.get('bids_validator', 'bids-validator')
    mriqc_exec = executables_config.get('mriqc', 'mriqc')

    # --- 6. Последовательный запуск шагов пайплайна ---
    all_steps_successful = True
    python_executable = sys.executable # Используем тот же Python, что и для пайплайна

    # --- Шаг 1: reorganize_folders.py ---
    step_name = "1_Reorganize_to_BIDS_DICOM"
    step_script = scripts_dir / "reorganize_folders.py"
    step_log = pipeline_log_dir / "01_reorganize_folders.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(raw_input_dir),
            "--output_dir", str(bids_dicom_dir) ]
    if not run_step(cmd, step_name, step_log): sys.exit(1) # Выход при ошибке

    # --- Шаг 2: dicom_standard_check.py ---
    step_name = "2_DICOM_Standard_Check"
    step_script = scripts_dir / "dicom_standard_check.py"
    step_log = pipeline_log_dir / "02_dicom_check.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_dicom_dir),
            "--output_dir", str(dicom_checks_dir),
            "--dciodvfy_path", dciodvfy_exec ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # --- Шаг 3: extract_metadata.py ---
    step_name = "3_Extract_DICOM_Metadata"
    step_script = scripts_dir / "extract_metadata.py"
    step_log = pipeline_log_dir / "03_extract_metadata.log"
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_dicom_dir),
            "--output_dir", str(dicom_meta_dir) ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # --- Шаг 4: convert_dicom_to_nifti.py ---
    step_name = "4_Convert_DICOM_to_NIfTI"
    step_script = scripts_dir / "convert_dicom_to_nifti.py"
    step_log = pipeline_log_dir / "04_convert_nifti.log"
    # Убедитесь, что ваш convert_dicom_to_nifti.py ПРИНИМАЕТ --dcm2niix_path
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_dicom_dir),
            "--output_dir", str(bids_nifti_dir),
            "--dcm2niix_path", dcm2niix_exec # Передаем путь
          ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # --- Шаг 5: bids_validation.py ---
    step_name = "5_BIDS_Validation"
    step_script = scripts_dir / "bids_validation.py"
    step_log = pipeline_log_dir / "05_bids_validation.log"
    # Этот скрипт сохраняет свой лог и отчет валидатора в свою output_dir
    cmd = [ python_executable, str(step_script),
            "--bids_dir", str(bids_nifti_dir),
            "--output_dir", str(validation_reports_dir), # Указываем папку для его вывода
            "--validator_path", bids_validator_exec,
            # Передаем путь к основному логу пайплайна, если хотим дублировать туда
            # Но лучше оставить лог шага в его папке
            # "--log_file", str(step_log) # НЕ ПЕРЕДАЕМ, он создаст свой лог в validation_reports_dir
          ]
    # Запускаем, но НЕ выходим из пайплайна при ошибке валидации (returncode != 0)
    # run_step вернет False только при падении самого скрипта
    if not run_step(cmd, step_name, step_log):
         # Если сам скрипт валидации упал (не просто нашел ошибки BIDS)
         logger.critical(f"Критическая ошибка скрипта {step_name}. Прерывание пайплайна.")
         sys.exit(1)
    else:
         # Проверяем лог валидации или отчет на наличие ошибок/предупреждений
         # (можно добавить эту логику позже, если нужно останавливаться при невалидном BIDS)
         logger.info(f"Шаг {step_name} завершен. Проверьте отчет в {validation_reports_dir} на наличие ошибок/предупреждений BIDS.")


    # --- Шаг 6: quality_metrics_without_skull-strip.py ---
    step_name = "6_Fast_Quality_Metrics"
    step_script = scripts_dir / "quality_metrics.py"
    step_log = pipeline_log_dir / "06_fast_qc.log"
    qc_config = steps_config.get('quality_metrics', {})
    anisotropy_thresh_qc = qc_config.get('anisotropy_thresh', 3.0)
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_nifti_dir),
            "--output_dir", str(fast_qc_reports_dir),
            "--anisotropy_thresh", str(anisotropy_thresh_qc) ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)

    # --- Шаг 7: mriqc_quality.py (Опциональный) ---
    step_name = "7_MRIQC_Analysis"
    mriqc_config = steps_config.get('mriqc', {})
    if mriqc_config.get('enabled', False): # Проверяем флаг enabled
        step_script = scripts_dir / "mriqc_quality.py"
        step_log = pipeline_log_dir / "07_mriqc.log"
        # Лог ошибок MRIQC будет создан относительно output_dir скрипта mriqc_quality
        mriqc_error_log_name = mriqc_config.get("error_log_filename", "mriqc_run_errors.log") # Имя файла из конфига или дефолт
        mriqc_error_log_path = mriqc_output_dir / mriqc_error_log_name

        cmd = [ python_executable, str(step_script),
                "--bids_dir", str(bids_nifti_dir),
                "--output_dir", str(mriqc_output_dir),
                "--report_type", mriqc_config.get('report_type', 'both'),
                "--mriqc_path", mriqc_exec,
                "--n_procs", str(mriqc_config.get('n_procs', 1)),
                "--n_threads", str(mriqc_config.get('n_threads', 1)),
                "--mem_gb", str(mriqc_config.get('mem_gb', 4)),
                "--error_log", str(mriqc_error_log_path), # Передаем путь к логу ошибок
                # log_file основного скрипта передается через run_step
              ]
        # Если нужно передать список субъектов из конфига (пока не реализовано в примере конфига)
        # subjects_to_run = mriqc_config.get('subjects', None)
        # if subjects_to_run: cmd.extend(["--subjects"] + subjects_to_run)

        if not run_step(cmd, step_name, step_log): sys.exit(1)
    else:
        logger.info(f"--- Шаг '{step_name}' пропущен (отключен в конфигурации). ---")

    # --- Шаг 8: mriqc_interpretation.py (Опциональный) ---
    step_name = "8_MRIQC_Interpretation"
    mriqc_interpret_config = steps_config.get('mriqc_interpretation', {})
    # Запускаем, если он включен И предыдущий шаг MRIQC тоже был включен
    if mriqc_config.get('enabled', False) and mriqc_interpret_config.get('enabled', False):
        # Дополнительно проверим, существует ли папка с результатами MRIQC
        if mriqc_output_dir.exists() and any(mriqc_output_dir.iterdir()):
            step_script = scripts_dir / "mriqc_interpretation.py"
            step_log = pipeline_log_dir / "08_mriqc_interpret.log"
            cmd = [ python_executable, str(step_script),
                    "--mriqc_dir", str(mriqc_output_dir),
                    "--output_dir", str(mriqc_interpret_dir) ]
            if not run_step(cmd, step_name, step_log): sys.exit(1)
        else:
             logger.warning(f"--- Шаг '{step_name}' пропущен: папка результатов MRIQC пуста или не найдена ({mriqc_output_dir}). ---")
    else:
        logger.info(f"--- Шаг '{step_name}' пропущен (отключен в конфигурации). ---")

    # --- Шаг 9: preprocessing.py ---
    step_name = "9_Preprocessing"
    step_script = scripts_dir / "preprocessing.py"
    step_log = pipeline_log_dir / "09_preprocessing.log"
    # Параметры для этого шага будут прочитаны из конфига внутри самого скрипта
    cmd = [ python_executable, str(step_script),
            "--input_dir", str(bids_nifti_dir),
            "--output_prep_dir", str(preprocessed_dir),
            "--output_transform_dir", str(transforms_dir),
            "--template_path", str(template_path),
            "--config", str(config_path), # <<< Передаем путь к основному конфигу
            # Основной лог будет создан внутри скрипта (preprocessing_main.log)
            # Передаем уровень консоли для консистентности
            "--console_log_level", args.console_log_level
          ]
    if not run_step(cmd, step_name, step_log): sys.exit(1)


    # --- Завершение пайплайна ---
    logger.info("=" * 60)
    # if all_steps_successful: # Переменная all_steps_successful не используется из-за sys.exit(1)
    logger.info("Пайплайн успешно завершил все запущенные шаги.")
    # else:
    #     logger.error("Пайплайн завершился с ошибкой на одном из шагов.")
    logger.info(f"Результаты сохранены в директорию: {run_output_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0) # Успешный выход всего скрипта пайплайна
    except FileNotFoundError as e:
        # Эти ошибки должны ловиться внутри main, но перехват здесь для надежности
        print(f"[CRITICAL ERROR] Prerequisite file or directory not found: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
         print(f"[CRITICAL ERROR] Missing key in configuration file: {e}. Please check config.", file=sys.stderr)
         sys.exit(1)
    except ValueError as e: # Ловим ошибки загрузки/валидации конфига
         print(f"[CRITICAL ERROR] Invalid value or configuration error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        # Ловим любые другие ошибки на самом верхнем уровне
        print(f"[CRITICAL UNEXPECTED ERROR] Pipeline execution failed.", file=sys.stderr)
        # Выводим traceback в stderr для диагностики
        traceback.print_exc()
        # Попытка записать в лог, если он был настроен
        if logger.hasHandlers():
             logger.critical(f"Непредвиденная ошибка остановила пайплайн: {e}", exc_info=True)
        else: # Если логгер не настроен, пишем traceback в stderr
             print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)