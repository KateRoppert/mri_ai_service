import os
import subprocess
import shutil
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

def bids_modality_from_folder(folder_name):
    """Определяет BIDS-совместимое имя модальности по имени папки."""
    name = folder_name.lower()
    # Важно: Порядок проверки может иметь значение
    if 't1c' in name or 't1gd' in name or 'contrast' in name: # Добавлены варианты
        return 'ce-gd_T1w'
    elif 't2fl' in name or 'flair' in name: # Добавлены варианты
        return 'FLAIR'
    elif 't1' in name: # Проверяем T1 после T1c
        return 'T1w'
    elif 't2' in name:
        return 'T2w'
    else:
        logger.warning(f"Не удалось определить стандартную BIDS модальность для папки '{folder_name}'")
        return None # Возвращаем None, если не удалось определить

# --- Основная функция конвертации ---
def convert_bids_dicom_to_nifti(bids_dicom_dir, bids_nifti_dir, dcm2niix_path='/home/roppert/abin/dcm2niix_lnx/dcm2niix'):
    """
    Конвертирует DICOM файлы из bids_dicom_dir в NIfTI в bids_nifti_dir,
    используя dcm2niix и переименовывая файлы в BIDS формат.
    """
    logger.info(f"Начало конвертации DICOM в NIfTI.")
    logger.info(f"  Источник (DICOM): {bids_dicom_dir}")
    logger.info(f"  Назначение (NIfTI): {bids_nifti_dir}")
    logger.info(f"  Исполняемый файл dcm2niix: {dcm2niix_path}")

    # --- Проверки входных данных и утилиты ---
    input_path = Path(bids_dicom_dir)
    output_path_root = Path(bids_nifti_dir)

    if not input_path.is_dir():
        logger.error(f"Входная директория DICOM не найдена: {input_path}")
        raise FileNotFoundError(f"Входная директория DICOM не найдена: {input_path}")

    # Проверка наличия dcm2niix
    dcm2niix_found_path = shutil.which(dcm2niix_path)
    if dcm2niix_found_path is None:
        logger.error(f"Исполняемый файл dcm2niix не найден: '{dcm2niix_path}'. Убедитесь, что он установлен и путь указан верно или находится в системном PATH.")
        raise FileNotFoundError(f"dcm2niix не найден: {dcm2niix_path}")
    else:
        logger.info(f"Исполняемый файл dcm2niix найден: {dcm2niix_found_path}")
        dcm2niix_path = dcm2niix_found_path # Используем полный путь

    # Создание корневой папки NIfTI
    try:
        output_path_root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Корневая директория NIfTI создана или уже существует: {output_path_root}")
    except OSError as e:
        logger.error(f"Не удалось создать корневую директорию NIfTI {output_path_root}: {e}")
        raise # Критическая ошибка

    converted_count = 0
    skipped_count = 0
    error_count = 0

    # --- Обход входной директории DICOM ---
    # Ищем папки с DICOM файлами на нужном уровне вложенности
    for root, dirs, files in os.walk(bids_dicom_dir):
        root_path = Path(root)
        dicom_files = [f for f in files if f.lower().endswith('.dcm')]

        # Нас интересуют только папки, непосредственно содержащие DICOM файлы
        if not dicom_files:
            logger.debug(f"Пропуск директории без .dcm файлов: {root_path}")
            continue

        logger.info(f"Найдена папка с DICOM файлами: {root_path}")

        # Проверка структуры пути относительно входной директории
        try:
            rel_path = root_path.relative_to(input_path)
            path_parts = rel_path.parts # Кортеж частей пути: ('sub-001', 'ses-001', 'anat', 't1')
        except ValueError:
             logger.warning(f"Не удалось определить относительный путь для {root_path} относительно {input_path}. Пропуск.")
             skipped_count += 1
             continue

        # Ожидаем структуру: sub-XXX/ses-XXX/anat/<modality_folder>
        if not (len(path_parts) == 4 and path_parts[-2] == 'anat'):
            logger.warning(f"Структура пути не соответствует ожидаемой 'sub-*/ses-*/anat/modality': {rel_path}. Пропуск.")
            skipped_count += 1
            continue

        sub_id = path_parts[0]
        ses_id = path_parts[1]
        modality_folder = path_parts[-1] # Имя папки ('t1', 't1c', ...)

        # Определяем BIDS имя модальности для имени файла NIfTI
        bids_modality_name = bids_modality_from_folder(modality_folder)
        if bids_modality_name is None:
            logger.warning(f"Не удалось определить BIDS модальность для папки {modality_folder} в {rel_path}. Пропуск.")
            skipped_count += 1
            continue

        # Определяем путь для выходных NIfTI файлов
        output_nifti_subdir = output_path_root / sub_id / ses_id / 'anat'
        try:
            output_nifti_subdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"  Не удалось создать выходную поддиректорию {output_nifti_subdir}: {e}. Пропуск конвертации для {root_path}.")
            error_count += 1
            continue # Пропускаем эту папку DICOM

        # Имя для временных файлов dcm2niix перед переименованием
        # Используем уникальный временный префикс, чтобы избежать конфликтов, если dcm2niix вызывается несколько раз для одной выходной папки (хотя здесь это маловероятно)
        temp_prefix = f"tmp_{sub_id}_{ses_id}_{modality_folder}"

        # --- Формирование и запуск команды dcm2niix ---
        cmd = [
            dcm2niix_path,
            '-z', 'y',             # Включить сжатие gzip (.nii.gz)
            '-o', str(output_nifti_subdir), # Указать выходную папку
            '-f', temp_prefix,     # Задать формат имени временного файла
            str(root_path)         # Указать входную папку с DICOM файлами
        ]
        logger.info(f"  Конвертация: {root_path} -> {output_nifti_subdir}")
        logger.debug(f"    Команда: {' '.join(cmd)}")

        try:
            # Запускаем dcm2niix, не прерываемся на ошибках, проверяем результат позже
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False) # Таймаут 5 минут

            # Логируем вывод dcm2niix
            if result.stdout:
                 logger.debug(f"    dcm2niix stdout:\n{result.stdout.strip()}")
            if result.stderr:
                 # dcm2niix часто пишет полезную информацию или предупреждения в stderr
                 logger.debug(f"    dcm2niix stderr:\n{result.stderr.strip()}")

            # Проверяем код возврата
            if result.returncode != 0:
                logger.error(f"    Ошибка при конвертации {root_path}. dcm2niix завершился с кодом {result.returncode}.")
                # Логируем stderr как ошибку, если он есть
                if result.stderr:
                     logger.error(f"    dcm2niix stderr (ошибка):\n{result.stderr.strip()}")
                error_count += 1
                continue # Переходим к следующей папке DICOM

            logger.debug(f"    Конвертация для {root_path} успешно завершена (код возврата 0).")

        except subprocess.TimeoutExpired:
            logger.error(f"  Ошибка: Время ожидания dcm2niix истекло для папки {root_path}.")
            error_count += 1
            continue
        except Exception as e:
            logger.error(f"  Непредвиденная ошибка при запуске dcm2niix для {root_path}: {e}", exc_info=True)
            error_count += 1
            continue

        # --- Переименование выходных файлов ---
        # Формируем ожидаемые временные и финальные имена
        # dcm2niix мог создать несколько файлов (например, _e1, _e2), ищем основной
        final_bids_base = f"{sub_id}_{ses_id}_{bids_modality_name}"
        renamed_nii = False
        renamed_json = False

        logger.debug(f"  Переименование файлов с префиксом '{temp_prefix}' в '{final_bids_base}'...")
        # Ищем файлы, созданные dcm2niix с нашим временным префиксом
        try:
            for temp_file_path in output_nifti_subdir.glob(f"{temp_prefix}*.nii.gz"):
                # Обычно создается один .nii.gz, переименовываем его
                final_nii_path = output_nifti_subdir / f"{final_bids_base}.nii.gz"
                logger.info(f"    Переименование NIfTI: {temp_file_path.name} -> {final_nii_path.name}")
                shutil.move(str(temp_file_path), str(final_nii_path))
                renamed_nii = True
                # Если файлов несколько, переименуется только первый найденный.
                # Для сложных случаев (фазы, эхо) может потребоваться более сложная логика.
                break # Переименовали основной, выходим из цикла поиска .nii.gz
            if not renamed_nii:
                 logger.warning(f"  Не найден временный .nii.gz файл с префиксом '{temp_prefix}' в {output_nifti_subdir} после конвертации {root_path}")

            # Ищем и переименовываем соответствующий JSON sidecar
            for temp_file_path in output_nifti_subdir.glob(f"{temp_prefix}*.json"):
                 final_json_path = output_nifti_subdir / f"{final_bids_base}.json"
                 logger.info(f"    Переименование JSON: {temp_file_path.name} -> {final_json_path.name}")
                 shutil.move(str(temp_file_path), str(final_json_path))
                 renamed_json = True
                 break # Переименовали, выходим
            # Отсутствие JSON - это не всегда ошибка, но стоит отметить
            if not renamed_json and renamed_nii: # Если был NIfTI, но нет JSON
                 logger.debug(f"  JSON sidecar для '{final_bids_base}' не найден или не переименован.")

            if renamed_nii: # Считаем успешным, если основной файл переименован
                 converted_count += 1

        except OSError as e:
            logger.error(f"  Ошибка при переименовании файлов для {final_bids_base} в {output_nifti_subdir}: {e}")
            error_count += 1
        except Exception as e:
             logger.error(f"  Непредвиденная ошибка при переименовании файлов для {final_bids_base}: {e}", exc_info=True)
             error_count += 1


    logger.info("="*50)
    logger.info("Конвертация DICOM в NIfTI завершена.")
    logger.info(f"  Успешно конвертировано папок: {converted_count}")
    logger.info(f"  Пропущено папок (не DICOM/структура): {skipped_count}")
    logger.info(f"  Возникло ошибок: {error_count}")
    logger.info("="*50)


# --- Точка входа при запуске скрипта ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Конвертирует DICOM файлы из BIDS-подобной структуры в NIfTI формат с помощью dcm2niix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        help="Входная директория с данными BIDS DICOM (например, bids_data_dicom)."
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Выходная директория для сохранения NIfTI файлов (например, bids_data_nifti)."
    )
    parser.add_argument(
        '--dcm2niix_path',
        default='/home/roppert/abin/dcm2niix_lnx/dcm2niix',
        help="Путь к исполняемому файлу dcm2niix."
    )
    parser.add_argument(
        '--log_file',
        default=None,
        help="Путь к файлу для записи логов. Если не указан, будет создан 'convert_dicom_to_nifti.log' внутри --output_dir."
    )

    args = parser.parse_args()

    # --- Настройка логирования ---
    log_file_path = args.log_file
    if log_file_path is None:
        output_dir_path = args.output_dir
        log_filename = 'convert_dicom_to_nifti.log'
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
        logger.info(f"Запуск convert_dicom_to_nifti.py")
        logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Путь dcm2niix (указанный): {args.dcm2niix_path}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
        logger.info("="*50)

        convert_bids_dicom_to_nifti(args.input_dir, args.output_dir, args.dcm2niix_path)

        logger.info("Скрипт успешно завершил работу.")
        sys.exit(0) # Успешный выход

    except FileNotFoundError as e:
        # Ошибка отсутствия входной директории или dcm2niix
        logger.error(f"Критическая ошибка: Файл или директория не найдены. {e}")
        sys.exit(1) # Выход с кодом ошибки
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Непредвиденная критическая ошибка: {e}")
        sys.exit(1)