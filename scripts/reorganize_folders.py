# import os
# import shutil
# import pydicom
# import argparse
# import logging
# import sys
# from collections import defaultdict

# # --- Глобальная настройка логгера ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# def setup_logging(log_file_path):
#     """Настраивает вывод логов в консоль (INFO) и файл (DEBUG)."""
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # Обработчик для консоли
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.INFO)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     # Обработчик для файла
#     try:
#         log_dir = os.path.dirname(log_file_path)
#         if log_dir:
#             os.makedirs(log_dir, exist_ok=True)
#         fh = logging.FileHandler(log_file_path)
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#         logger.debug(f"Логирование в файл настроено: {log_file_path}")
#     except Exception as e:
#         logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)

# def is_dicom_file(file_path):
#     """Проверяет, является ли файл валидным DICOM-файлом"""
#     try:
#         pydicom.dcmread(file_path, stop_before_pixels=True)
#         logger.debug(f"Файл {file_path} опознан как DICOM.")
#         return True
#     except pydicom.errors.InvalidDicomError:
#         logger.debug(f"Файл {file_path} не является валидным DICOM.")
#         return False
#     except Exception as e:
#         logger.warning(f"Ошибка при проверке файла {file_path} на DICOM: {e}")
#         return False # Считаем его не-DICOM при любой ошибке чтения

# def determine_modality(ds, file_path):
#     """Автоматически определяет модальность используя комбинацию метаданных DICOM и анализа пути"""
#     # Анализ DICOM тегов
#     series_desc = ds.get('SeriesDescription', '').lower()
#     protocol = ds.get('ProtocolName', '').lower()
#     logger.debug(f"Определение модальности для {file_path}: SeriesDesc='{series_desc}', Protocol='{protocol}'")

#     # Ключевые слова для модальностей (из исходного скрипта)
#     modality_keywords = {
#         't1c': ['t1c', 't1+c', 't1-ce', 't1contrast', 't1gd', 'contrast'],
#         't1': ['t1', 't1w', 't1-weighted', 't1weighted', 'spgr', 'mprage'],
#         't2fl': ['t2fl', 't2-flair', 'flair', 't2flair'],
#         't2': ['t2', 't2w', 't2-weighted', 't2weighted', 'tse']
#     }

#     # Проверка тегов DICOM
#     for modality, keys in modality_keywords.items():
#         # проверяет наличие ЛЮБОГО ключа в SeriesDescription ИЛИ ProtocolName
#         if any(key in series_desc for key in keys) or any(key in protocol for key in keys):
#              logger.debug(f"Модальность '{modality}' определена по тегам DICOM.")
#              return modality

#     # Анализ пути к файлу (если по тегам не нашли)
#     path_parts = os.path.normpath(file_path).split(os.sep)
#     for part in reversed(path_parts):
#         part_lower = part.lower()
#         for modality, keys in modality_keywords.items():
#             if any(key in part_lower for key in keys):
#                 logger.debug(f"Модальность '{modality}' определена по пути к файлу ('{part_lower}').")
#                 return modality

#     # Анализ параметров сканирования
#     try:
#         tr = float(ds.get('RepetitionTime', 0))
#         te = float(ds.get('EchoTime', 0))
#         logger.debug(f"Параметры сканирования: TR={tr}, TE={te}")
#         if 300 < tr < 800 and te < 30:
#              logger.debug("Модальность 't1' определена по TR/TE.")
#              return 't1'
#         elif 2000 < tr < 5000 and te > 80:
#              logger.debug("Модальность 't2' определена по TR/TE.")
#              return 't2'
#     except Exception as e:
#         logger.debug(f"Не удалось проанализировать параметры сканирования TR/TE: {e}")
#         pass

#     logger.warning(f"Не удалось определить модальность для файла: {file_path}")
#     return 'unknown'

# def organize_dicom_to_bids(input_dir, output_dir='bids_data_dicom'):
#     """Организует DICOM файлы в формате BIDS."""
#     logger.info(f"Начало организации данных из '{input_dir}' в '{output_dir}'.")

#     # Проверка входной директории
#     if not os.path.isdir(input_dir):
#         logger.error(f"Входная директория не найдена: {input_dir}")
#         # Выбрасываем ошибку, чтобы ее поймал основной блок
#         raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")

#     # Создаем корневую выходную директорию
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         logger.debug(f"Выходная директория создана или уже существует: {output_dir}")
#     except OSError as e:
#         logger.error(f"Не удалось создать выходную директорию {output_dir}: {e}")
#         raise # Передаем ошибку выше

#     # Обрабатываем пациентов (первый уровень вложенности)
#     patient_folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
#     if not patient_folders:
#         logger.warning(f"Во входной директории '{input_dir}' не найдено подпапок (пациентов).")
#         return # Завершаем, если нет пациентов

#     for patient_idx, patient_folder in enumerate(patient_folders, 1):
#         patient_path = os.path.join(input_dir, patient_folder)
#         sub_id = f"sub-{patient_idx:03d}"
#         sub_path = os.path.join(output_dir, sub_id) # Папка субъекта в output_dir
#         logger.info(f"Обработка пациента: {patient_folder} -> {sub_id}")

#         # Обрабатываем сессии (второй уровень вложенности)
#         session_folders = sorted([f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))])
#         if not session_folders:
#             logger.warning(f"  В папке пациента '{patient_folder}' не найдено подпапок (сессий).")
#             continue # К следующему пациенту

#         for session_idx, session_folder in enumerate(session_folders, 1):
#             session_path = os.path.join(patient_path, session_folder)
#             ses_id = f"ses-{session_idx:03d}"
#             ses_bids_path = os.path.join(sub_path, ses_id, 'anat')
#             logger.info(f"  Обработка сессии: {session_folder} -> {ses_id}")

#             # Собираем все DICOM-файлы в сессии
#             dcm_files = []
#             logger.debug(f"  Поиск DICOM файлов в {session_path}...")
#             for root, _, files in os.walk(session_path):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     if is_dicom_file(file_path):
#                         dcm_files.append(file_path)

#             if not dcm_files:
#                 logger.warning(f"  В папке сессии '{session_folder}' не найдено DICOM файлов.")
#                 continue # К следующей сессии
#             logger.info(f"  Найдено {len(dcm_files)} DICOM файлов.")

#             # Группируем файлы по модальности
#             modality_groups = defaultdict(list)
#             files_with_unknown_modality = 0
#             files_with_read_error = 0

#             for file_path in dcm_files:
#                 try:
#                     ds = pydicom.dcmread(file_path, stop_before_pixels=True)
#                     modality = determine_modality(ds, file_path) # 't1', 't1c', 't2', 't2fl' или 'unknown'

#                     if modality == 'unknown':
#                         # Логгирование происходит внутри determine_modality
#                         files_with_unknown_modality += 1
#                         continue # Не копируем неизвестные

#                     modality_groups[modality].append(file_path)
#                     logger.debug(f"  Файл {os.path.basename(file_path)} отнесен к модальности '{modality}'.")

#                 except Exception as e:
#                     # Ловим ошибки чтения DICOM или определения модальности
#                     logger.error(f"  Ошибка обработки файла {file_path}: {e}", exc_info=True)
#                     files_with_read_error += 1
#                     continue # Пропускаем файл с ошибкой

#             if files_with_unknown_modality > 0:
#                 logger.warning(f"  Не удалось определить модальность для {files_with_unknown_modality} файлов.")
#             if files_with_read_error > 0:
#                  logger.error(f"  Произошли ошибки при чтении/обработке {files_with_read_error} файлов.")


#             if not modality_groups:
#                 logger.warning(f"  Нет файлов с известной модальностью для копирования в сессии {ses_id}.")
#                 continue

#             # Копируем файлы в BIDS-структуру
#             logger.info(f"  Копирование файлов...")
#             for modality, files_to_copy in modality_groups.items():
#                 # Создаем путь с папкой 'anat' и подпапкой модальности
#                 modality_path = os.path.join(ses_bids_path, modality)
#                 try:
#                     os.makedirs(modality_path, exist_ok=True)
#                     logger.debug(f"  Создана/проверена папка модальности: {modality_path}")
#                 except OSError as e:
#                     logger.error(f"    Не удалось создать папку для модальности {modality}: {e}. Пропуск.")
#                     continue # Пропускаем эту модальность

#                 logger.debug(f"  Копирование {len(files_to_copy)} файлов для '{modality}'...")
#                 for idx, src_file in enumerate(files_to_copy, 1):
#                     # Имя файла как в оригинале: sub-XXX_ses-YYY_modality_ZZZ.dcm
#                     dst_filename = f"{sub_id}_{ses_id}_{modality}_{idx:03d}.dcm"
#                     dst_file = os.path.join(modality_path, dst_filename)
#                     try:
#                         shutil.copy(src_file, dst_file)
#                         logger.debug(f"    Скопирован: {os.path.basename(src_file)} -> {dst_filename}")
#                     except Exception as e:
#                         logger.error(f"    Не удалось скопировать {src_file} в {dst_file}: {e}")

#     logger.info("Организация данных завершена!")

# if __name__ == '__main__':
#     # Настройка парсера аргументов командной строки
#     parser = argparse.ArgumentParser(
#         description='Организует DICOM файлы из входной директории в BIDS-подобную структуру.',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#         )
#     parser.add_argument(
#         '--input_dir',
#         required=True,
#         help='Входная директория с сырыми данными (структура: patient/session/...).'
#         )
#     parser.add_argument(
#         '--output_dir',
#         default='bids_data_dicom',
#         help='Корневая выходная директория для сохранения BIDS структуры.'
#         )
#     parser.add_argument(
#         '--log_file',
#         default=None,
#         help='Путь к файлу для записи логов. Если не указан, будет создан файл "reorganize_folders.log" внутри --output_dir.'
#         )

#     args = parser.parse_args()

#     # Определение пути к лог-файлу
#     log_file_path = args.log_file
#     if log_file_path is None:
#         output_dir_path = args.output_dir
#         log_filename = 'reorganize_folders.log'
#         try:
#             # Создаем output_dir заранее, если надо
#             if output_dir_path and not os.path.exists(output_dir_path):
#                  os.makedirs(output_dir_path, exist_ok=True)
#             log_file_path = os.path.join(output_dir_path or '.', log_filename)
#         except OSError as e:
#              log_file_path = log_filename # Пишем в текущую папку при ошибке
#              print(f"Предупреждение: Не удалось использовать {output_dir_path} для лог-файла. Лог будет записан в {log_file_path}. Ошибка: {e}")

#     # Настройка логирования
#     setup_logging(log_file_path)

#     # Основной блок выполнения
#     try:
#         logger.info("="*50)
#         logger.info(f"Запуск reorganize_folders.py")
#         logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
#         logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
#         logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
#         logger.info("="*50)

#         # Вызов основной функции с аргументами из CLI
#         organize_dicom_to_bids(args.input_dir, args.output_dir)

#         logger.info("Скрипт успешно завершил работу.")
#         sys.exit(0) # Успешный выход

#     except FileNotFoundError as e:
#         logger.error(f"Критическая ошибка: {e}")
#         sys.exit(1) # Выход с кодом ошибки
#     except OSError as e:
#         logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
#         sys.exit(1)
#     except Exception as e:
#         # Ловим все остальные непредвиденные ошибки
#         logger.exception(f"Непредвиденная критическая ошибка: {e}")
#         sys.exit(1)

import os
import shutil
import pydicom
import pydicom.dataelem # Нужно для isinstance
import argparse
import logging
import sys
from collections import defaultdict

# --- Глобальная настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path):
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
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
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except pydicom.errors.InvalidDicomError:
        return False
    except Exception as e:
        logger.warning(f"Ошибка при проверке файла {file_path} на DICOM: {e}")
        return False

def get_dicom_value(ds, tag_tuple_or_keyword, default=None):
    """
    Безопасно извлекает значение тега.
    Если ds.get() неожиданно возвращает DataElement, извлекает .value.
    Для строк возвращает lower(), для списков тоже.
    """
    try:
        val = ds.get(tag_tuple_or_keyword, default)

        if val is default: # Тега нет, или ds.get() вернул переданный default
            return default

        # Ключевое исправление: если ds.get() вернул DataElement, извлекаем .value
        if isinstance(val, pydicom.dataelem.DataElement):
            # Это не должно происходить согласно документации pydicom для ds.get(),
            # но ошибка указывает, что это возможно.
            logger.warning(
                f"Tag {tag_tuple_or_keyword}: ds.get() unexpectedly returned a DataElement. "
                f"Extracting .value. Original DataElement: {val}"
            )
            val = val.value
            if val is None: # Если .value пустое
                return default # Возвращаем исходный default

        if val is None: # После извлечения .value, оно могло стать None
            return default

        if isinstance(val, str):
            return val.lower()
        if isinstance(val, (pydicom.multival.MultiValue, list)):
            processed_list = []
            for v_item in val:
                if isinstance(v_item, str):
                    processed_list.append(v_item.lower())
                else:
                    processed_list.append(v_item)
            return processed_list
        # Для чисел (int, float, DSfloat, IS и т.д.)
        return val
    except Exception as e:
        logger.error(f"Исключение в get_dicom_value для {tag_tuple_or_keyword}: {e}", exc_info=True)
        return default


def find_keyword_in_text(text_to_search, keywords_list):
    if not text_to_search:
        return None
    for kw in keywords_list:
        if kw in text_to_search:
            return kw
    return None

def determine_modality(ds, file_path):
    modality_tag_val = get_dicom_value(ds, (0x0008, 0x0060), "")
    series_desc_val = get_dicom_value(ds, (0x0008, 0x103E), "")
    protocol_name_val = get_dicom_value(ds, (0x0018, 0x1030), "")
    
    series_desc_str = " ".join(series_desc_val) if isinstance(series_desc_val, list) else str(series_desc_val or "")
    protocol_name_str = " ".join(protocol_name_val) if isinstance(protocol_name_val, list) else str(protocol_name_val or "")
    combined_text = f"{series_desc_str} {protocol_name_str}"

    scan_seq_val = get_dicom_value(ds, (0x0018, 0x0020), [])
    seq_var_val = get_dicom_value(ds, (0x0018, 0x0021), [])
    image_type_val = get_dicom_value(ds, (0x0008, 0x0008), [])
    
    tr_val = get_dicom_value(ds, (0x0018, 0x0080))
    te_val = get_dicom_value(ds, (0x0018, 0x0081))
    ti_val = get_dicom_value(ds, (0x0018, 0x0082))
    
    contrast_agent_val = get_dicom_value(ds, (0x0018, 0x0010), "")

    logger.debug(f"Определение модальности для {os.path.basename(file_path)}:")
    logger.debug(f"  DICOM Tags:")
    logger.debug(f"    Modality (0008,0060): '{modality_tag_val}'")
    logger.debug(f"    SeriesDescription (0008,103E): '{series_desc_str}'")
    logger.debug(f"    ProtocolName (0018,1030): '{protocol_name_str}'")
    logger.debug(f"    ScanningSequence (0018,0020): {scan_seq_val}")
    logger.debug(f"    SequenceVariant (0018,0021): {seq_var_val}")
    logger.debug(f"    ImageType (0008,0008): {image_type_val}")
    logger.debug(f"    RepetitionTime (0018,0080): {tr_val} (type: {type(tr_val)})")
    logger.debug(f"    EchoTime (0018,0081): {te_val} (type: {type(te_val)})")
    logger.debug(f"    InversionTime (0018,0082): {ti_val} (type: {type(ti_val)})")
    logger.debug(f"    ContrastBolusAgent (0018,0010): '{contrast_agent_val}'")

    if modality_tag_val and modality_tag_val != 'mr':
        logger.info(f"  Тег Modality (0008,0060) для {os.path.basename(file_path)} имеет значение '{modality_tag_val}', а не 'mr'. Продолжаем анализ по другим тегам.")
    elif not modality_tag_val:
        logger.info(f"  Тег Modality (0008,0060) для {os.path.basename(file_path)} пуст или отсутствует. Продолжаем анализ по другим тегам.")

    flair_kws = ['flair', 't2fl', 'fluid attenuated inversion recovery', 'ir_fse', 'darkfluid']
    t1_kws = ['t1', 't1w', 't1 weighted', 'spgr', 'mprage', 'tfl', 'bravo']
    t1c_kws = ['t1c', 't1+c', 't1-ce', 't1contrast', 't1 gd', 'contrast', 'gad', 'postcontrast', 'ce-t1', 't1 post', 'gd t1', 'mdc', 'with contrast', '+gd', 'post gd']
    t2_kws = ['t2', 't2w', 't2 weighted', 'tse', 'fse', 't2 tse', 't2 fse', 'haste']

    # 1. Определение T2-FLAIR (t2fl)
    if ti_val is not None:
        try:
            ti_float = float(ti_val)
            if ti_float > 1500:
                reason = f"по TI (0018,0082) = {ti_float}"
                tr_float, te_float = None, None
                if tr_val is not None:
                    try: tr_float = float(tr_val)
                    except (ValueError, TypeError): pass
                if te_val is not None:
                    try: te_float = float(te_val)
                    except (ValueError, TypeError): pass

                if tr_float is not None and te_float is not None:
                    if tr_float > 4000 and te_float > 70:
                        reason += f" и классическим TR/TE (TR={tr_float}, TE={te_float})"
                    else:
                        reason += f" (TR={tr_float}, TE={te_float} не классические для FLAIR, но TI решающий)"
                logger.debug(f"Модальность 't2fl' определена {reason}.")
                return 't2fl'
        except (ValueError, TypeError) as e_conv: # Ловим и TypeError
            logger.debug(f"  Не удалось конвертировать TI (0018,0082)='{ti_val}' (тип: {type(ti_val)}) в float: {e_conv}")

    # ... (остальная логика с аналогичными исправлениями для float(tr_val) и float(te_val)) ...
    # Пример для TR/TE в T1:
    # if tr_val is not None and te_val is not None:
    #     try:
    #         tr_float = float(tr_val)
    #         te_float = float(te_val)
    #         if tr_float < 1200 and te_float < 30:
    #             # ...
    #     except (ValueError, TypeError) as e_conv:
    #         logger.debug(f"  Не удалось конвертировать TR/TE ('{tr_val}', '{te_val}') в float: {e_conv}")

    # Чтобы не повторять try-except для float() много раз, можно сделать вспомогательную функцию
    def safe_float(value, tag_name_for_log="value"):
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError) as e_conv:
            logger.debug(f"  Не удалось конвертировать {tag_name_for_log}='{value}' (тип: {type(value)}) в float: {e_conv}")
            return None

    # Используем safe_float в логике:
    ti_float = safe_float(ti_val, "TI (0018,0082)")
    tr_float = safe_float(tr_val, "TR (0018,0080)")
    te_float = safe_float(te_val, "TE (0018,0081)")

    # 1. Определение T2-FLAIR (t2fl)
    if ti_float is not None and ti_float > 1500:
        reason = f"по TI (0018,0082) = {ti_float}"
        if tr_float is not None and te_float is not None:
            if tr_float > 4000 and te_float > 70:
                reason += f" и классическим TR/TE (TR={tr_float}, TE={te_float})"
            else:
                reason += f" (TR={tr_float}, TE={te_float} не классические для FLAIR, но TI решающий)"
        logger.debug(f"Модальность 't2fl' определена {reason}.")
        return 't2fl'

    matched_flair_kw = find_keyword_in_text(combined_text, flair_kws)
    if matched_flair_kw:
        is_ir_scan_seq = any(val == 'ir' for val in scan_seq_val)
        is_ir_img_type = any(val == 'ir' for val in image_type_val)
        reason_kw = f"по ключевому слову '{matched_flair_kw}' в SeriesDescription/ProtocolName"
        
        if tr_float is not None and te_float is not None and tr_float > 4000 and te_float > 70:
            logger.debug(f"Модальность 't2fl' определена {reason_kw} и классическим TR/TE (TR={tr_float}, TE={te_float}).")
            return 't2fl'
        if is_ir_scan_seq:
            logger.debug(f"Модальность 't2fl' определена {reason_kw} и ScanningSequence (0018,0020) содержит 'ir'.")
            return 't2fl'
        if is_ir_img_type:
            logger.debug(f"Модальность 't2fl' определена {reason_kw} и ImageType (0008,0008) содержит 'ir'.")
            return 't2fl'
        
    # 2. Определение T1-contrast (t1c)
    contrast_reason_parts = []
    if contrast_agent_val and contrast_agent_val != "none": 
        contrast_reason_parts.append(f"ContrastBolusAgent (0018,0010)='{contrast_agent_val}'")
    
    matched_t1c_kw = find_keyword_in_text(combined_text, t1c_kws)
    if matched_t1c_kw:
        contrast_reason_parts.append(f"ключевому слову контраста '{matched_t1c_kw}' в SeriesDescription/ProtocolName")

    if contrast_reason_parts:
        full_contrast_reason = " и ".join(contrast_reason_parts)
        t1_confirmation_reason = ""
        if tr_float is not None and te_float is not None and tr_float < 1200 and te_float < 30:
            t1_confirmation_reason = f"TR/TE (TR={tr_float}, TE={te_float})"
        
        if not t1_confirmation_reason:
            if any(val == 'sp' for val in seq_var_val):
                t1_confirmation_reason = "SequenceVariant (0018,0021) содержит 'sp'"
            elif any(val == 'mp' for val in seq_var_val):
                 t1_confirmation_reason = "SequenceVariant (0018,0021) содержит 'mp'"
        
        if not t1_confirmation_reason:
            matched_t1_kw_for_t1c = find_keyword_in_text(combined_text, t1_kws)
            if matched_t1_kw_for_t1c:
                 t1_confirmation_reason = f"ключевому слову T1 '{matched_t1_kw_for_t1c}'"
        
        if t1_confirmation_reason:
            if not matched_flair_kw:
                logger.debug(f"Модальность 't1c' определена по {full_contrast_reason}, подтверждено как T1-подобное по {t1_confirmation_reason}.")
                return 't1c'
            else:
                logger.debug(f"  Потенциальный T1c (по {full_contrast_reason}), но также найден FLAIR keyword '{matched_flair_kw}'. Приоритет FLAIR, если TI высокий или нет других признаков T1.")

    # 3. Определение T1-weighted (t1) (без контраста, не FLAIR)
    if any(val == 'mp' for val in seq_var_val):
        if ti_float is not None and 700 < ti_float < 1300: 
            if not contrast_reason_parts:
                logger.debug(f"Модальность 't1' определена как MPRAGE по SequenceVariant (0018,0021) содержит 'mp' и TI (0018,0082)={ti_float}.")
                return 't1'
    
    if any(val == 'sp' for val in seq_var_val):
        reason_sp = "SequenceVariant (0018,0021) содержит 'sp' (SPGR/FLASH)"
        if not contrast_reason_parts:
            if tr_float is not None and te_float is not None and tr_float < 1200 and te_float < 30:
                logger.debug(f"Модальность 't1' определена {reason_sp} и TR/TE (TR={tr_float}, TE={te_float}).")
                return 't1'
            else: 
                matched_t1_kw_for_sp = find_keyword_in_text(combined_text, t1_kws)
                if matched_t1_kw_for_sp:
                    logger.debug(f"Модальность 't1' определена {reason_sp} и ключевому слову T1 '{matched_t1_kw_for_sp}'.")
                    return 't1'
    
    if tr_float is not None and te_float is not None and tr_float < 1000 and te_float < 30:
        if not contrast_reason_parts: 
            logger.debug(f"Модальность 't1' определена по TR (0018,0080)={tr_float} и TE (0018,0081)={te_float}.")
            return 't1'

    matched_t1_kw = find_keyword_in_text(combined_text, t1_kws)
    if matched_t1_kw:
        if not matched_flair_kw and not contrast_reason_parts:
            logger.debug(f"Модальность 't1' определена по ключевому слову '{matched_t1_kw}' в SeriesDescription/ProtocolName.")
            return 't1'

    # 4. Определение T2-weighted (t2) (не FLAIR)
    if any(val == 'se' for val in scan_seq_val) and any(val == 'sk' for val in seq_var_val):
        reason_tse = "ScanningSequence (0018,0020) содержит 'se' и SequenceVariant (0018,0021) содержит 'sk' (TSE/FSE)"
        if tr_float is not None and te_float is not None and tr_float > 2000 and te_float > 70:
            if not matched_flair_kw: 
                logger.debug(f"Модальность 't2' определена {reason_tse} и TR/TE (TR={tr_float}, TE={te_float}).")
                return 't2'
        else: 
            matched_t2_kw_for_tse = find_keyword_in_text(combined_text, t2_kws)
            if matched_t2_kw_for_tse and not matched_flair_kw:
                logger.debug(f"Модальность 't2' определена {reason_tse} и ключевому слову T2 '{matched_t2_kw_for_tse}'.")
                return 't2'
                
    if tr_float is not None and te_float is not None and tr_float > 2000 and te_float > 70:
        if not matched_flair_kw: 
            logger.debug(f"Модальность 't2' определена по TR (0018,0080)={tr_float} и TE (0018,0081)={te_float}.")
            return 't2'
    
    matched_t2_kw = find_keyword_in_text(combined_text, t2_kws)
    if matched_t2_kw:
        if not matched_flair_kw:
            logger.debug(f"Модальность 't2' определена по ключевому слову '{matched_t2_kw}' в SeriesDescription/ProtocolName.")
            return 't2'

    if matched_flair_kw: # Fallback для FLAIR по ключевому слову
        logger.debug(f"Модальность 't2fl' определена (как fallback) по ключевому слову '{matched_flair_kw}' в SeriesDescription/ProtocolName, т.к. другие модальности не подошли.")
        return 't2fl'
             
    path_parts = os.path.normpath(file_path).split(os.sep)
    logger.debug(f"  Анализ пути файла для определения модальности: {file_path}")
    for part in reversed(path_parts):
        part_lower = part.lower() 
        kw_path = find_keyword_in_text(part_lower, t1c_kws)
        if kw_path:
            logger.debug(f"Модальность 't1c' определена по пути к файлу: часть '{part_lower}' содержит ключевое слово '{kw_path}'.")
            return 't1c'
        kw_path = find_keyword_in_text(part_lower, flair_kws)
        if kw_path:
            logger.debug(f"Модальность 't2fl' определена по пути к файлу: часть '{part_lower}' содержит ключевое слово '{kw_path}'.")
            return 't2fl'
        kw_path = find_keyword_in_text(part_lower, t1_kws) 
        if kw_path:
            logger.debug(f"Модальность 't1' определена по пути к файлу: часть '{part_lower}' содержит ключевое слово '{kw_path}'.")
            return 't1'
        kw_path = find_keyword_in_text(part_lower, t2_kws) 
        if kw_path:
            logger.debug(f"Модальность 't2' определена по пути к файлу: часть '{part_lower}' содержит ключевое слово '{kw_path}'.")
            return 't2'

    logger.warning(f"Не удалось определить модальность для файла: {os.path.basename(file_path)} (путь: {file_path}) с использованием всех правил.")
    return 'unknown'


def organize_dicom_to_bids(input_dir, output_dir='bids_data_dicom'):
    logger.info(f"Начало организации данных из '{input_dir}' в '{output_dir}'.")
    if not os.path.isdir(input_dir):
        logger.error(f"Входная директория не найдена: {input_dir}")
        raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Не удалось создать выходную директорию {output_dir}: {e}")
        raise

    patient_folders = sorted([f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))])
    if not patient_folders:
        logger.warning(f"Во входной директории '{input_dir}' не найдено подпапок (пациентов).")
        return

    for patient_idx, patient_folder in enumerate(patient_folders, 1):
        patient_path = os.path.join(input_dir, patient_folder)
        sub_id = f"sub-{patient_idx:03d}"
        sub_path = os.path.join(output_dir, sub_id)
        logger.info(f"Обработка пациента: {patient_folder} -> {sub_id}")

        session_folders = sorted([f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))])
        if not session_folders:
            logger.warning(f"  В папке пациента '{patient_folder}' не найдено подпапок (сессий).")
            continue

        for session_idx, session_folder in enumerate(session_folders, 1):
            session_path_full = os.path.join(patient_path, session_folder)
            ses_id = f"ses-{session_idx:03d}"
            ses_bids_path = os.path.join(sub_path, ses_id, 'anat') 
            logger.info(f"  Обработка сессии: {session_folder} -> {ses_id}")

            dcm_files_in_session = []
            for root, _, files in os.walk(session_path_full):
                for file in files:
                    file_path_full = os.path.join(root, file)
                    if is_dicom_file(file_path_full): 
                        dcm_files_in_session.append(file_path_full)
            
            if not dcm_files_in_session:
                logger.warning(f"  В папке сессии '{session_folder}' ({session_path_full}) не найдено DICOM файлов.")
                continue
            logger.info(f"  Найдено {len(dcm_files_in_session)} DICOM файлов в сессии.")

            modality_groups = defaultdict(list)
            files_with_unknown_modality = 0
            files_with_read_error = 0
            processed_series_for_modality_decision = {}

            for file_path_item in dcm_files_in_session:
                try:
                    ds = pydicom.dcmread(file_path_item, stop_before_pixels=True)
                    series_uid = ds.get("SeriesInstanceUID", None)
                    modality = 'unknown'
                    
                    if series_uid and series_uid in processed_series_for_modality_decision:
                        modality = processed_series_for_modality_decision[series_uid]
                        # logger.debug(f"  Используется ранее определенная модальность '{modality}' для серии {series_uid} (файл {os.path.basename(file_path_item)}).")
                    else:
                        modality = determine_modality(ds, file_path_item)
                        if series_uid and modality != 'unknown':
                            processed_series_for_modality_decision[series_uid] = modality

                    if modality == 'unknown':
                        files_with_unknown_modality +=1
                        continue 
                    modality_groups[modality].append(file_path_item)
                except Exception as e:
                    # Включаем exc_info=True для полного трейсбека
                    logger.error(f"  Ошибка обработки файла {file_path_item}: {e}", exc_info=True) 
                    files_with_read_error += 1
                    continue
            
            if files_with_unknown_modality > 0:
                 logger.warning(f"  Не удалось определить модальность для {files_with_unknown_modality} файлов в сессии {ses_id} ({session_folder}).")
            if files_with_read_error > 0:
                 logger.error(f"  Произошли ошибки при чтении/обработке {files_with_read_error} файлов в сессии {ses_id} ({session_folder}).")

            if not modality_groups:
                logger.warning(f"  Нет файлов с известной модальностью для копирования в сессии {ses_id} ({session_folder}).")
                continue

            logger.info(f"  Копирование файлов для сессии {ses_id} ({session_folder})...")
            for modality_key, files_to_copy_list in modality_groups.items():
                modality_target_dir = os.path.join(ses_bids_path, modality_key) 
                try:
                    os.makedirs(modality_target_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"    Не удалось создать папку {modality_target_dir}: {e}. Пропуск модальности {modality_key}.")
                    continue

                # logger.debug(f"  Копирование {len(files_to_copy_list)} файлов для '{modality_key}' в {modality_target_dir}...")
                for idx, src_file_path in enumerate(files_to_copy_list, 1):
                    dst_filename = f"{sub_id}_{ses_id}_{modality_key}_{idx:03d}.dcm"
                    dst_file_path = os.path.join(modality_target_dir, dst_filename)
                    try:
                        shutil.copy(src_file_path, dst_file_path)
                    except Exception as e:
                        logger.error(f"    Не удалось скопировать {src_file_path} в {dst_file_path}: {e}")
    logger.info("Организация данных завершена!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Организует DICOM файлы из входной директории в BIDS-подобную структуру на основе DICOM тегов.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', required=True, help='Входная директория.')
    parser.add_argument('--output_dir', default='bids_data_dicom_fixed', help='Выходная директория.')
    parser.add_argument('--log_file', default=None, help='Файл логов.')
    args = parser.parse_args()

    log_file_path = args.log_file
    if log_file_path is None:
        output_dir_path_for_log = args.output_dir
        log_filename = 'reorganize_folders_fixed.log' 
        try:
            if output_dir_path_for_log and not os.path.exists(output_dir_path_for_log):
                 os.makedirs(output_dir_path_for_log, exist_ok=True)
            log_file_path = os.path.join(output_dir_path_for_log or '.', log_filename)
        except OSError as e:
             log_file_path = log_filename 
             print(f"Предупреждение: Не удалось использовать {output_dir_path_for_log} для лог-файла. Лог будет записан в {log_file_path}. Ошибка: {e}")
    setup_logging(log_file_path)
    
    try:
        logger.info("="*50)
        logger.info(f"Запуск скрипта (исправленная версия)")
        logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
        logger.info("="*50)
        organize_dicom_to_bids(args.input_dir, args.output_dir)
        logger.info("Скрипт успешно завершил работу.")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Непредвиденная критическая ошибка")
        sys.exit(1)