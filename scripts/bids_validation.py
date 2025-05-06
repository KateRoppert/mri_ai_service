import os
import json
import subprocess
import argparse
import logging
import sys
import shutil
from pathlib import Path

# --- Глобальная настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path):
    """Настраивает вывод логов в консоль (INFO) и файл (DEBUG)."""
    if logger.hasHandlers():
        logger.handlers.clear()
    # Консоль
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Файл
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)

def create_dataset_description(bids_dir):
    """
    Создает файл dataset_description.json в указанной BIDS директории.
    Если файл уже существует, он будет перезаписан.
    """
    logger.info(f"Создание/обновление файла dataset_description.json в '{bids_dir}'")
    bids_path = Path(bids_dir)

    # Убедимся, что директория существует
    try:
        bids_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Не удалось создать директорию {bids_path}: {e}")
        raise # Передаем ошибку выше, т.к. файл создать не получится

    # Содержимое файла (можно сделать более настраиваемым в будущем)
    description = {
        "Name": "Brain Tumour MRI Dataset (Processed)",
        "BIDSVersion": "1.8.0",
        "License": "CC0",# CC0 - public domain dedication.
        "Authors": ["Your Name/Lab Name"],
        # "Acknowledgements": "...",
        # "HowToAcknowledge": "...",
        # "Funding": ["...", "..."],
        # "ReferencesAndLinks": ["...", "..."],
        "DatasetDOI": "10.1234/example.doi" # Замените на реальный DOI, если есть
        # Добавим информацию о генерации (опционально)
        # "GeneratedBy": [
        #     {
        #         "Name": "bids-pipeline",
        #         "Version": "1.0.0" # Укажите версию вашего пайплайна
        #     }
        # ]
    }
    logger.debug(f"Содержимое dataset_description.json:\n{json.dumps(description, indent=2)}")

    output_file = bids_path / "dataset_description.json"

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(description, f, indent=2, ensure_ascii=False)
        logger.info(f"Файл {output_file} успешно создан/обновлен.")
        return True # Возвращаем успех
    except OSError as e:
        logger.error(f"Не удалось записать файл {output_file}: {e}")
        return False # Возвращаем неудачу

def validate_bids(bids_dir, validator_path='bids-validator', report_path=None):
    """
    Запускает bids-validator для проверки BIDS структуры.

    Args:
        bids_dir (str): Путь к корневой директории BIDS датасета.
        validator_path (str): Путь к исполняемому файлу bids-validator.
        report_path (str, optional): Путь для сохранения полного отчета валидатора.

    Returns:
        bool: True, если валидация прошла без ошибок (код возврата 0), False в противном случае.
              Примечание: валидатор может вернуть ненулевой код и при наличии только предупреждений.
    """
    logger.info(f"Запуск BIDS валидации для директории: {bids_dir}")
    logger.info(f"Исполняемый файл валидатора: {validator_path}")

    bids_path = Path(bids_dir)

    # --- Проверки ---
    if not bids_path.is_dir():
        logger.error(f"Директория BIDS не найдена: {bids_path}")
        raise FileNotFoundError(f"Директория BIDS не найдена: {bids_path}")

    # Проверка наличия bids-validator
    validator_found_path = shutil.which(validator_path)
    if validator_found_path is None:
        logger.error(f"Исполняемый файл bids-validator не найден: '{validator_path}'.")
        logger.error("Пожалуйста, установите его глобально: npm install -g bids-validator")
        raise FileNotFoundError(f"bids-validator не найден: {validator_path}")
    else:
        logger.info(f"Исполняемый файл bids-validator найден: {validator_found_path}")
        validator_path = validator_found_path # Используем полный путь

    # --- Запуск валидатора ---
    cmd = [validator_path, str(bids_path)]
    logger.debug(f"Команда валидации: {' '.join(cmd)}")
    validation_successful = False
    report_content = ""

    try:
        # Запускаем валидатор. check=False - важно, т.к. ошибки валидации не должны крашить скрипт
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False, # Не выбрасывать исключение при ненулевом коде выхода
            timeout=600 # Таймаут 10 минут (валидация может быть долгой)
        )

        # Формируем отчет для лога и файла
        report_content += f"--- BIDS Validator Report for {bids_path} ---\n"
        report_content += f"Command: {' '.join(cmd)}\n"
        report_content += f"Return Code: {result.returncode}\n\n"
        report_content += "--- STDOUT & STDERR ---\n"
        # bids-validator обычно выводит все в stdout
        full_output = (result.stdout or "") + (result.stderr or "")
        if not full_output.strip():
            report_content += "[No output received]"
        else:
            report_content += full_output.strip()
        report_content += "\n--- End Report ---"

        # Логируем основной результат
        if result.returncode == 0:
            logger.info("BIDS валидация завершена успешно (код возврата 0).")
            # Ищем слово "Warning" в выводе на всякий случай
            if "warning" in full_output.lower():
                logger.warning("Валидация успешна, но обнаружены предупреждения (см. полный отчет/логи).")
            validation_successful = True
        else:
            logger.error(f"BIDS валидация завершилась с ошибками или предупреждениями (код возврата {result.returncode}).")
            # Можно попытаться определить, были ли это только предупреждения
            # (Эвристика: если вывод содержит 'warnings' но не 'errors')
            if "warning" in full_output.lower() and "error" not in full_output.lower():
                 logger.warning("Обнаружены только предупреждения BIDS валидации.")
            else:
                 logger.error("Обнаружены ошибки BIDS валидации (см. полный отчет/логи).")
            validation_successful = False # Считаем неудачей, если код не 0

        # Логируем полный вывод в DEBUG
        logger.debug(f"Полный вывод bids-validator:\n{report_content}")

    except FileNotFoundError:
         # Должно было быть поймано проверкой shutil.which, но на всякий случай
         logger.error(f"Критическая ошибка: Исполняемый файл валидатора не найден во время запуска: {validator_path}")
         raise # Передаем ошибку выше
    except subprocess.TimeoutExpired:
        logger.error(f"Ошибка: Время ожидания bids-validator истекло для директории {bids_path}.")
        report_content = f"Error: BIDS validation timed out for {bids_path}"
        validation_successful = False
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка при запуске bids-validator: {e}")
        report_content = f"Error: Unexpected error during BIDS validation: {e}"
        validation_successful = False

    # --- Сохранение отчета в файл (если указан путь) ---
    if report_path:
        report_file = Path(report_path)
        logger.debug(f"Сохранение отчета валидации в: {report_file}")
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Отчет BIDS валидации сохранен в: {report_file}")
        except OSError as e:
            logger.error(f"Не удалось записать файл отчета валидации {report_file}: {e}")

    return validation_successful


# --- Точка входа при запуске скрипта ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Создает dataset_description.json и запускает bids-validator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--bids_dir',
        required=True,
        help="Корневая директория BIDS датасета (например, bids_data_nifti)."
    )
    # Добавим output_dir для логов и отчета валидатора
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Директория для сохранения лог-файла и отчета валидации (например, папка этапа валидации)."
    )
    parser.add_argument(
        '--validator_path',
        default='bids-validator', # Искать в PATH по умолчанию
        help="Путь к исполняемому файлу bids-validator."
    )
    parser.add_argument(
        '--log_file',
        default=None,
        help="Путь к файлу для записи логов. Если не указан, будет создан 'bids_validation.log' внутри --output_dir."
    )

    args = parser.parse_args()

    # --- Настройка логирования ---
    log_file_path = args.log_file
    output_dir_path = args.output_dir
    if log_file_path is None:
        log_filename = 'bids_validation.log'
        try:
            if output_dir_path and not os.path.exists(output_dir_path):
                 os.makedirs(output_dir_path, exist_ok=True)
            log_file_path = os.path.join(output_dir_path or '.', log_filename)
        except OSError as e:
             log_file_path = log_filename
             print(f"Предупреждение: Не удалось использовать {output_dir_path} для лог-файла. Лог будет записан в {log_file_path}. Ошибка: {e}")

    setup_logging(log_file_path)

    # --- Основной блок выполнения ---
    validation_passed = False
    try:
        logger.info("="*50)
        logger.info(f"Запуск bids_validation.py")
        logger.info(f"  BIDS директория: {os.path.abspath(args.bids_dir)}")
        logger.info(f"  Output директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Путь bids-validator: {args.validator_path}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
        logger.info("="*50)

        # 1. Создание dataset_description.json
        success_dd = create_dataset_description(args.bids_dir)
        if not success_dd:
             # Решаем, является ли это критической ошибкой. Скорее да, т.к. валидация без него не пройдет.
             logger.error("Не удалось создать dataset_description.json. Прерывание работы.")
             sys.exit(1)

        # 2. Запуск валидации
        # Определяем путь для сохранения отчета валидации
        validation_report_file = os.path.join(args.output_dir, "bids_validator_report.txt")
        validation_passed = validate_bids(args.bids_dir, args.validator_path, validation_report_file)

        if validation_passed:
            logger.info("Скрипт успешно завершил работу. Валидация BIDS прошла успешно.")
            sys.exit(0) # Успешный выход
        else:
            # Если валидация не прошла (ошибки или предупреждения), логи уже выведены.
            # Считаем это штатным завершением скрипта, но сигнализируем пайплайну о проблемах с BIDS.
            # Можно вернуть специальный код или просто 0, но с предупреждением в логах.
            # Для простоты вернем 0, но пайплайн должен будет проверять логи/отчет.
            logger.warning("Скрипт завершил работу, но валидация BIDS обнаружила проблемы (см. лог/отчет).")
            sys.exit(0) # Все равно успешный выход скрипта, проблема в данных

    except FileNotFoundError as e:
        logger.error(f"Критическая ошибка: Файл или директория не найдены. {e}")
        sys.exit(1) # Выход с кодом ошибки
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Непредвиденная критическая ошибка: {e}")
        sys.exit(1)