import os
import pydicom
import json
import argparse
import logging
import sys
from pathlib import Path

# --- Глобальная настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path):
    """Настраивает вывод логов в консоль (INFO) и файл (DEBUG)."""
    if logger.hasHandlers():
        logger.handlers.clear()

    # Обработчик для консоли
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Обработчик для файла
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)


def clean_value(value):
    """Рекурсивно очищает значение элемента DICOM от бинарных данных."""
    if isinstance(value, list):
        # Если это список, очищаем каждый элемент списка
        return [clean_value(item) for item in value]
    elif isinstance(value, dict):
        # Если это вложенный словарь (например, внутри SQ), очищаем его
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, bytes):
        # Заменяем бинарные данные на заглушку
        return "BINARY_DATA_REMOVED"
    # Возвращаем значение как есть, если это не список, словарь или байты
    # (pydicom может возвращать специфичные типы данных, например, MultiValue)
    # Попробуем преобразовать к базовым типам, если возможно
    try:
        # Простая проверка на конвертируемость в JSON-совместимые типы
        json.dumps(value)
        return value
    except TypeError:
        logger.debug(f"Не удалось сериализовать значение типа {type(value)}, преобразуем в строку.")
        try:
            return str(value)
        except Exception:
             logger.warning(f"Не удалось преобразовать значение типа {type(value)} в строку.")
             return "UNSERIALIZABLE_DATA"


def clean_dicom_dict(meta_dict):
    """
    Очищает словарь, полученный из ds.to_json_dict(), удаляя бинарные данные.
    """
    cleaned_meta = {}
    for tag, element_dict in meta_dict.items():
        if isinstance(element_dict, dict) and 'vr' in element_dict:
            vr = element_dict.get('vr')
            value = element_dict.get('Value')

            if vr == 'SQ': # Обработка последовательностей
                 cleaned_value = [clean_dicom_dict(item) for item in value]
            elif value is not None:
                 cleaned_value = clean_value(value)
            else:
                 cleaned_value = None # Оставляем None, если значения не было

            cleaned_meta[tag] = {
                'vr': vr,
                'Value': cleaned_value
            }
            # Дополнительно можно логировать замену бинарных данных, если нужно
            # original_value = element_dict.get('Value')
            # if vr != 'SQ' and any(isinstance(v, bytes) for v in original_value):
            #     logger.debug(f"  Заменены бинарные данные в теге {tag}")

        else:
            # Если структура неожиданная, пытаемся очистить рекурсивно или пропускаем
            logger.warning(f"Неожиданная структура для тега {tag}. Попытка рекурсивной очистки.")
            cleaned_meta[tag] = clean_value(element_dict) # Пытаемся очистить как обычное значение

    return cleaned_meta


def extract_metadata(dicom_dir, output_dir):
    """
    Извлекает метаданные из DICOM файлов в dicom_dir, очищает от бинарных данных
    и сохраняет как JSON в output_dir, сохраняя структуру подпапок.
    """
    logger.info(f"Начало извлечения метаданных из '{dicom_dir}'. Результаты будут сохранены в '{output_dir}'.")

    # --- Проверки входных данных ---
    input_path = Path(dicom_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        logger.error(f"Входная директория не найдена: {dicom_dir}")
        raise FileNotFoundError(f"Входная директория не найдена: {dicom_dir}")

    # Создание корневой папки для метаданных (если нужно)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Директория для метаданных создана или уже существует: {output_path}")
    except OSError as e:
        logger.error(f"Не удалось создать директорию для метаданных {output_path}: {e}")
        raise # Критическая ошибка

    processed_files = 0
    error_files = 0

    # --- Обход входной директории ---
    for root, _, files in os.walk(dicom_dir):
        root_path = Path(root)
        logger.debug(f"Обработка директории: {root_path}")

        for file in files:
            # Пропускаем скрытые файлы
            if file.startswith('.'):
                continue

            # Обрабатываем только файлы с ожидаемым расширением (можно добавить .ima и др.)
            if not file.lower().endswith(".dcm"):
                 logger.debug(f"  Пропуск файла с неожидаемым расширением: {file}")
                 continue

            src_file_path = root_path / file
            # Относительный путь для сохранения структуры
            try:
                rel_path = src_file_path.relative_to(input_path)
            except ValueError:
                logger.error(f"  Не удалось вычислить относительный путь для {src_file_path} относительно {input_path}. Пропуск.")
                continue

            # Формируем путь для выходного JSON файла
            json_filename = src_file_path.stem + '_meta.json'
            json_file_path = output_path / rel_path.parent / json_filename

            logger.debug(f"  Обработка файла: {src_file_path} -> {json_file_path}")

            # Создаем поддиректории в выходной папке, если их нет
            try:
                json_file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"    Не удалось создать директорию {json_file_path.parent} для JSON файла: {e}. Пропуск файла {src_file_path}.")
                error_files += 1
                continue # Переходим к следующему файлу

            # --- Извлечение, очистка и сохранение метаданных ---
            try:
                # Читаем DICOM заголовок
                ds = pydicom.dcmread(str(src_file_path), stop_before_pixels=True) # pydicom < 2.1 требует str

                # Конвертируем в словарь
                meta_dict = ds.to_json_dict()

                # Очищаем от бинарных данных
                logger.debug(f"    Очистка метаданных от бинарных значений...")
                cleaned_meta = clean_dicom_dict(meta_dict)

                # Сохраняем в JSON
                logger.debug(f"    Сохранение метаданных в {json_file_path}...")
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_meta, f, indent=2, ensure_ascii=False)

                processed_files += 1
                logger.debug(f"    Метаданные для {file} успешно извлечены и сохранены.")

            except pydicom.errors.InvalidDicomError:
                 logger.warning(f"    Файл {src_file_path} не является валидным DICOM. Пропуск.")
                 # Не считаем это ошибкой скрипта, просто пропускаем файл
                 # error_files += 1 # Раскомментировать, если нужно считать это ошибкой
            except Exception as e:
                logger.error(f"    Ошибка при обработке файла {src_file_path}: {e}", exc_info=True)
                error_files += 1
                # Можно добавить запись файла с сообщением об ошибке:
                # try:
                #     with open(json_file_path, 'w', encoding='utf-8') as f:
                #         json.dump({"error": f"Failed to process DICOM file: {e}"}, f, indent=2)
                # except Exception as write_e:
                #     logger.error(f"      Не удалось записать файл ошибки {json_file_path}: {write_e}")


    logger.info(f"Извлечение метаданных завершено.")
    logger.info(f"Успешно обработано файлов: {processed_files}")
    if error_files > 0:
        logger.warning(f"Возникли ошибки при обработке {error_files} файлов (см. логи выше).")


# --- Точка входа при запуске скрипта ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Извлекает метаданные из DICOM файлов, очищает от бинарных данных и сохраняет в JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        help="Входная директория с данными в формате BIDS DICOM (например, bids_data_dicom)."
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Выходная директория для сохранения JSON файлов с метаданными (например, dicom_metadata)."
    )
    parser.add_argument(
        '--log_file',
        default=None,
        help="Путь к файлу для записи логов. Если не указан, будет создан файл 'extract_metadata.log' внутри --output_dir."
    )

    args = parser.parse_args()

    # --- Настройка логирования ---
    log_file_path = args.log_file
    if log_file_path is None:
        output_dir_path = args.output_dir
        log_filename = 'extract_metadata.log'
        try:
            if output_dir_path and not os.path.exists(output_dir_path):
                 os.makedirs(output_dir_path, exist_ok=True)
            log_file_path = os.path.join(output_dir_path or '.', log_filename)
        except OSError as e:
             log_file_path = log_filename
             print(f"Предупреждение: Не удалось использовать {output_dir_path} для лог-файла. Лог будет записан в {log_file_path}. Ошибка: {e}")

    setup_logging(log_file_path)

    # --- Основной блок выполнения ---
    try:
        logger.info("="*50)
        logger.info(f"Запуск extract_metadata.py")
        logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
        logger.info("="*50)

        extract_metadata(args.input_dir, args.output_dir)

        logger.info("Скрипт успешно завершил работу.")
        sys.exit(0) # Успешный выход

    except FileNotFoundError as e:
        logger.error(f"Критическая ошибка: Входная директория не найдена. {e}")
        sys.exit(1) # Выход с кодом ошибки
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Ловим все остальные непредвиденные ошибки
        logger.exception(f"Непредвиденная критическая ошибка: {e}")
        sys.exit(1)