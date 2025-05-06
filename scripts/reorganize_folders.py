import os
import shutil
import pydicom
import argparse
import logging
import sys
from collections import defaultdict

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

def is_dicom_file(file_path):
    """Проверяет, является ли файл валидным DICOM-файлом"""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        logger.debug(f"Файл {file_path} опознан как DICOM.")
        return True
    except pydicom.errors.InvalidDicomError:
        logger.debug(f"Файл {file_path} не является валидным DICOM.")
        return False
    except Exception as e:
        logger.warning(f"Ошибка при проверке файла {file_path} на DICOM: {e}")
        return False # Считаем его не-DICOM при любой ошибке чтения

def determine_modality(ds, file_path):
    """Автоматически определяет модальность используя комбинацию метаданных DICOM и анализа пути"""
    # Анализ DICOM тегов
    series_desc = ds.get('SeriesDescription', '').lower()
    protocol = ds.get('ProtocolName', '').lower()
    logger.debug(f"Определение модальности для {file_path}: SeriesDesc='{series_desc}', Protocol='{protocol}'")

    # Ключевые слова для модальностей (из исходного скрипта)
    modality_keywords = {
        't1c': ['t1c', 't1+c', 't1-ce', 't1contrast', 't1gd', 'contrast'],
        't1': ['t1', 't1w', 't1-weighted', 't1weighted', 'spgr', 'mprage'],
        't2fl': ['t2fl', 't2-flair', 'flair', 't2flair'],
        't2': ['t2', 't2w', 't2-weighted', 't2weighted', 'tse']
    }

    # Проверка тегов DICOM
    for modality, keys in modality_keywords.items():
        # проверяет наличие ЛЮБОГО ключа в SeriesDescription ИЛИ ProtocolName
        if any(key in series_desc for key in keys) or any(key in protocol for key in keys):
             logger.debug(f"Модальность '{modality}' определена по тегам DICOM.")
             return modality

    # Анализ пути к файлу (если по тегам не нашли)
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        part_lower = part.lower()
        for modality, keys in modality_keywords.items():
            if any(key in part_lower for key in keys):
                logger.debug(f"Модальность '{modality}' определена по пути к файлу ('{part_lower}').")
                return modality

    # Анализ параметров сканирования
    try:
        tr = float(ds.get('RepetitionTime', 0))
        te = float(ds.get('EchoTime', 0))
        logger.debug(f"Параметры сканирования: TR={tr}, TE={te}")
        if 300 < tr < 800 and te < 30:
             logger.debug("Модальность 't1' определена по TR/TE.")
             return 't1'
        elif 2000 < tr < 5000 and te > 80:
             logger.debug("Модальность 't2' определена по TR/TE.")
             return 't2'
    except Exception as e:
        logger.debug(f"Не удалось проанализировать параметры сканирования TR/TE: {e}")
        pass

    logger.warning(f"Не удалось определить модальность для файла: {file_path}")
    return 'unknown'

def organize_dicom_to_bids(input_dir, output_dir='bids_data_dicom'):
    """Организует DICOM файлы в формате BIDS."""
    logger.info(f"Начало организации данных из '{input_dir}' в '{output_dir}'.")

    # Проверка входной директории
    if not os.path.isdir(input_dir):
        logger.error(f"Входная директория не найдена: {input_dir}")
        # Выбрасываем ошибку, чтобы ее поймал основной блок
        raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")

    # Создаем корневую выходную директорию
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Выходная директория создана или уже существует: {output_dir}")
    except OSError as e:
        logger.error(f"Не удалось создать выходную директорию {output_dir}: {e}")
        raise # Передаем ошибку выше

    # Обрабатываем пациентов (первый уровень вложенности)
    patient_folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
    if not patient_folders:
        logger.warning(f"Во входной директории '{input_dir}' не найдено подпапок (пациентов).")
        return # Завершаем, если нет пациентов

    for patient_idx, patient_folder in enumerate(patient_folders, 1):
        patient_path = os.path.join(input_dir, patient_folder)
        sub_id = f"sub-{patient_idx:03d}"
        sub_path = os.path.join(output_dir, sub_id) # Папка субъекта в output_dir
        logger.info(f"Обработка пациента: {patient_folder} -> {sub_id}")

        # Обрабатываем сессии (второй уровень вложенности)
        session_folders = sorted([f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))])
        if not session_folders:
            logger.warning(f"  В папке пациента '{patient_folder}' не найдено подпапок (сессий).")
            continue # К следующему пациенту

        for session_idx, session_folder in enumerate(session_folders, 1):
            session_path = os.path.join(patient_path, session_folder)
            ses_id = f"ses-{session_idx:03d}"
            ses_bids_path = os.path.join(sub_path, ses_id, 'anat')
            logger.info(f"  Обработка сессии: {session_folder} -> {ses_id}")

            # Собираем все DICOM-файлы в сессии
            dcm_files = []
            logger.debug(f"  Поиск DICOM файлов в {session_path}...")
            for root, _, files in os.walk(session_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_dicom_file(file_path):
                        dcm_files.append(file_path)

            if not dcm_files:
                logger.warning(f"  В папке сессии '{session_folder}' не найдено DICOM файлов.")
                continue # К следующей сессии
            logger.info(f"  Найдено {len(dcm_files)} DICOM файлов.")

            # Группируем файлы по модальности
            modality_groups = defaultdict(list)
            files_with_unknown_modality = 0
            files_with_read_error = 0

            for file_path in dcm_files:
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    modality = determine_modality(ds, file_path) # 't1', 't1c', 't2', 't2fl' или 'unknown'

                    if modality == 'unknown':
                        # Логгирование происходит внутри determine_modality
                        files_with_unknown_modality += 1
                        continue # Не копируем неизвестные

                    modality_groups[modality].append(file_path)
                    logger.debug(f"  Файл {os.path.basename(file_path)} отнесен к модальности '{modality}'.")

                except Exception as e:
                    # Ловим ошибки чтения DICOM или определения модальности
                    logger.error(f"  Ошибка обработки файла {file_path}: {e}", exc_info=True)
                    files_with_read_error += 1
                    continue # Пропускаем файл с ошибкой

            if files_with_unknown_modality > 0:
                logger.warning(f"  Не удалось определить модальность для {files_with_unknown_modality} файлов.")
            if files_with_read_error > 0:
                 logger.error(f"  Произошли ошибки при чтении/обработке {files_with_read_error} файлов.")


            if not modality_groups:
                logger.warning(f"  Нет файлов с известной модальностью для копирования в сессии {ses_id}.")
                continue

            # Копируем файлы в BIDS-структуру
            logger.info(f"  Копирование файлов...")
            for modality, files_to_copy in modality_groups.items():
                # Создаем путь с папкой 'anat' и подпапкой модальности
                modality_path = os.path.join(ses_bids_path, modality)
                try:
                    os.makedirs(modality_path, exist_ok=True)
                    logger.debug(f"  Создана/проверена папка модальности: {modality_path}")
                except OSError as e:
                    logger.error(f"    Не удалось создать папку для модальности {modality}: {e}. Пропуск.")
                    continue # Пропускаем эту модальность

                logger.debug(f"  Копирование {len(files_to_copy)} файлов для '{modality}'...")
                for idx, src_file in enumerate(files_to_copy, 1):
                    # Имя файла как в оригинале: sub-XXX_ses-YYY_modality_ZZZ.dcm
                    dst_filename = f"{sub_id}_{ses_id}_{modality}_{idx:03d}.dcm"
                    dst_file = os.path.join(modality_path, dst_filename)
                    try:
                        shutil.copy(src_file, dst_file)
                        logger.debug(f"    Скопирован: {os.path.basename(src_file)} -> {dst_filename}")
                    except Exception as e:
                        logger.error(f"    Не удалось скопировать {src_file} в {dst_file}: {e}")

    logger.info("Организация данных завершена!")

if __name__ == '__main__':
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Организует DICOM файлы из входной директории в BIDS-подобную структуру.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Входная директория с сырыми данными (структура: patient/session/...).'
        )
    parser.add_argument(
        '--output_dir',
        default='bids_data_dicom',
        help='Корневая выходная директория для сохранения BIDS структуры.'
        )
    parser.add_argument(
        '--log_file',
        default=None,
        help='Путь к файлу для записи логов. Если не указан, будет создан файл "reorganize_folders.log" внутри --output_dir.'
        )

    args = parser.parse_args()

    # Определение пути к лог-файлу
    log_file_path = args.log_file
    if log_file_path is None:
        output_dir_path = args.output_dir
        log_filename = 'reorganize_folders.log'
        try:
            # Создаем output_dir заранее, если надо
            if output_dir_path and not os.path.exists(output_dir_path):
                 os.makedirs(output_dir_path, exist_ok=True)
            log_file_path = os.path.join(output_dir_path or '.', log_filename)
        except OSError as e:
             log_file_path = log_filename # Пишем в текущую папку при ошибке
             print(f"Предупреждение: Не удалось использовать {output_dir_path} для лог-файла. Лог будет записан в {log_file_path}. Ошибка: {e}")

    # Настройка логирования
    setup_logging(log_file_path)

    # Основной блок выполнения
    try:
        logger.info("="*50)
        logger.info(f"Запуск reorganize_folders.py")
        logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
        logger.info("="*50)

        # Вызов основной функции с аргументами из CLI
        organize_dicom_to_bids(args.input_dir, args.output_dir)

        logger.info("Скрипт успешно завершил работу.")
        sys.exit(0) # Успешный выход

    except FileNotFoundError as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1) # Выход с кодом ошибки
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Ловим все остальные непредвиденные ошибки
        logger.exception(f"Непредвиденная критическая ошибка: {e}")
        sys.exit(1)