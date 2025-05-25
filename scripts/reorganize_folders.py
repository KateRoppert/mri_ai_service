import os
import shutil
import pydicom
import pydicom.dataelem # Для isinstance DataElement
import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime # Для сортировки сессий

# --- Глобальная настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        if log_dir and not os.path.exists(log_dir): # Создаем директорию, если ее нет
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w') # 'w' для перезаписи лога при каждом запуске
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        # Если не удалось настроить файловый логгер, выводим ошибку в консоль
        # и продолжаем только с консольным логгером.
        logger.removeHandler(ch) # Удаляем предыдущий консольный, чтобы не дублировать
        ch_err = logging.StreamHandler(sys.stdout)
        ch_err.setLevel(logging.ERROR) # Показываем только ошибки в этом случае
        ch_err.setFormatter(formatter)
        logger.addHandler(ch_err)
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}. Логирование будет только в консоль.", exc_info=False)


def is_dicom_file(file_path):
    """Проверяет, является ли файл валидным DICOM-файлом"""
    try:
        # stop_before_pixels=True значительно ускоряет чтение, т.к. нам не нужны пиксельные данные
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except pydicom.errors.InvalidDicomError:
        logger.debug(f"Файл {file_path} не является валидным DICOM.")
        return False
    except Exception as e: # Ловим другие возможные ошибки при чтении файла
        logger.warning(f"Ошибка при проверке файла {file_path} на DICOM: {e}")
        return False

def get_dicom_value(ds, tag_tuple_or_keyword, default=None):
    """
    Безопасно извлекает значение тега из DICOM датасета.
    - Если ds.get() возвращает DataElement, извлекает .value.
    - Для строк возвращает их в нижнем регистре.
    - Для MultiValue (списков DICOM) возвращает список строк в нижнем регистре или оригинальных значений.
    - Для числовых и других типов возвращает значение как есть.
    """
    try:
        val = ds.get(tag_tuple_or_keyword, default)

        if val is default: # Тега нет, или ds.get() вернул переданный default
            return default

        # Если ds.get() вернул DataElement (хотя обычно не должен для .get)
        if isinstance(val, pydicom.dataelem.DataElement):
            logger.debug(
                f"Tag {tag_tuple_or_keyword}: ds.get() вернул DataElement. Извлечение .value. DataElement: {val}"
            )
            val = val.value
            if val is None:
                return default

        if val is None:
            return default

        if isinstance(val, str):
            return val.strip().lower() # Удаляем пробелы по краям и приводим к нижнему регистру
        if isinstance(val, (pydicom.multival.MultiValue, list)):
            processed_list = []
            for v_item in val:
                if isinstance(v_item, str):
                    processed_list.append(v_item.strip().lower())
                elif v_item is not None: # Пропускаем None значения в списке
                    processed_list.append(v_item)
            return processed_list
        return val # Для чисел (DSfloat, IS, etc.) и других типов
    except Exception as e:
        logger.error(f"Исключение в get_dicom_value для тега {tag_tuple_or_keyword} в файле (Series: {ds.get('SeriesInstanceUID', 'N/A')}): {e}", exc_info=False)
        return default

def safe_float(value, tag_name_for_log="value"):
    """Безопасно конвертирует значение в float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug(f"  Не удалось конвертировать {tag_name_for_log}='{value}' (тип: {type(value)}) в float.")
        return None

def find_keyword_in_text(text_to_search, keywords_list):
    """Ищет любое из ключевых слов в тексте. Возвращает первое найденное или None."""
    if not text_to_search or not keywords_list:
        return None
    # Приводим text_to_search к строке и нижнему регистру один раз
    text_to_search_lower = str(text_to_search).lower()
    for kw in keywords_list:
        if kw.lower() in text_to_search_lower: # Ключевые слова тоже приводим к lower для надежности
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



def sort_dicom_files_in_series(dicom_files_paths):
    """Сортирует список путей к DICOM файлам по InstanceNumber."""
    sorted_files = []
    for f_path in dicom_files_paths:
        try:
            ds_slice = pydicom.dcmread(f_path, stop_before_pixels=True, specific_tags=[(0x0020,0x0013)]) # Читаем только InstanceNumber
            instance_number = get_dicom_value(ds_slice, (0x0020,0x0013))
            if instance_number is not None:
                try:
                    instance_number = int(instance_number)
                except ValueError:
                    logger.warning(f"Не удалось конвертировать InstanceNumber '{instance_number}' в int для файла {f_path}. Используем имя файла для сортировки этого элемента.")
                    instance_number = f_path # Fallback для этого файла
            else: # Если InstanceNumber отсутствует
                logger.warning(f"InstanceNumber отсутствует в файле {f_path}. Используем имя файла для сортировки этого элемента.")
                instance_number = f_path # Fallback для этого файла
            sorted_files.append((instance_number, f_path))
        except Exception as e:
            logger.error(f"Ошибка чтения InstanceNumber из файла {f_path}: {e}. Файл будет в конце или отсортирован по имени.")
            sorted_files.append((float('inf'), f_path)) # Помещаем файлы с ошибками в конец

    # Сортируем: сначала по числовому InstanceNumber, затем по пути файла (если InstanceNumber был строкой/одинаковый)
    sorted_files.sort(key=lambda x: (isinstance(x[0], str), x[0]))
    return [f_path for _, f_path in sorted_files]


def organize_dicom_to_bids(input_dir, output_dir='bids_data_dicom_universal', action_type='copy'):
    """Организует DICOM файлы из любой структуры в BIDS на основе DICOM тегов."""
    logger.info(f"Начало организации данных из '{input_dir}' в '{output_dir}'. Действие: {action_type.upper()}")

    if not os.path.isdir(input_dir):
        logger.error(f"Входная директория не найдена: {input_dir}")
        raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Не удалось создать корневую выходную директорию {output_dir}: {e}")
        raise

    # --- Фаза 1: Глобальное сканирование и сбор информации ---
    logger.info("Фаза 1: Сканирование DICOM файлов и сбор метаданных...")
    # Структура: patient_orig_id -> study_orig_uid -> series_orig_uid -> {info}
    collected_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    dicom_file_count = 0
    processed_file_count = 0

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            dicom_file_count += 1
            if dicom_file_count % 500 == 0:
                logger.info(f"  Просканировано файлов: {dicom_file_count}...")

            if is_dicom_file(file_path):
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    
                    pat_id = get_dicom_value(ds, (0x0010, 0x0020), "UNKNOWN_PATIENT_ID")
                    study_uid = get_dicom_value(ds, (0x0020, 0x000D), "UNKNOWN_STUDY_UID")
                    series_uid = get_dicom_value(ds, (0x0020, 0x000E), "UNKNOWN_SERIES_UID")

                    if any(val.startswith("UNKNOWN_") for val in [pat_id, study_uid, series_uid]):
                        logger.warning(f"Пропущен файл {file_path}: отсутствует PatientID, StudyUID или SeriesUID.")
                        continue
                    
                    # Сохраняем информацию о серии
                    series_data = collected_data[pat_id][study_uid][series_uid]
                    if 'files' not in series_data:
                        series_data['files'] = []
                        series_data['first_ds'] = ds # Сохраняем первый датасет для определения модальности
                        
                        study_date_str = get_dicom_value(ds, (0x0008,0x0020), "00000000")
                        study_time_str = get_dicom_value(ds, (0x0008,0x0030), "000000.000000").split('.')[0] # Берем только HHMMSS
                        try:
                            series_data['study_datetime'] = datetime.strptime(f"{study_date_str}{study_time_str}", "%Y%m%d%H%M%S")
                        except ValueError:
                            logger.warning(f"Некорректная дата/время ({study_date_str}/{study_time_str}) для {study_uid}. Используется начало эпохи.")
                            series_data['study_datetime'] = datetime.min

                        series_num_val = get_dicom_value(ds, (0x0020,0x0011))
                        try:
                            series_data['series_number_int'] = int(series_num_val) if series_num_val is not None else float('inf')
                        except ValueError:
                            series_data['series_number_int'] = float('inf') # Если не число, будет последним при сортировке
                            logger.warning(f"SeriesNumber '{series_num_val}' для серии {series_uid} не является числом.")
                    
                    series_data['files'].append(file_path)
                    processed_file_count +=1

                except Exception as e:
                    logger.error(f"Ошибка обработки DICOM файла {file_path} на этапе сбора: {e}", exc_info=False)
    
    logger.info(f"Фаза 1 завершена. Всего просканировано файлов: {dicom_file_count}. Обработано DICOM файлов: {processed_file_count}.")
    if not collected_data:
        logger.warning("Не найдено валидных DICOM данных для организации.")
        return

    # --- Фаза 2: Формирование BIDS структуры и копирование файлов ---
    logger.info("Фаза 2: Формирование BIDS структуры и копирование файлов...")
    
    # Создание BIDS ID для пациентов
    sorted_original_patient_ids = sorted(list(collected_data.keys()))
    patient_bids_map = {orig_id: f"sub-{i+1:03d}" for i, orig_id in enumerate(sorted_original_patient_ids)}

    for orig_pat_id, studies_data in collected_data.items():
        bids_sub_id = patient_bids_map[orig_pat_id]
        logger.info(f"Обработка пациента: {orig_pat_id} -> {bids_sub_id}")

        # Создание BIDS ID для сессий (сортировка по дате/времени)
        sorted_original_study_uids = sorted(
            studies_data.keys(),
            # key=lambda suid: studies_data[suid].get(next(iter(studies_data[suid])), {}).get('study_datetime', datetime.min)
            # Ключ для сортировки сессий: берем study_datetime из первой серии этой сессии
            key=lambda suid: studies_data[suid][next(iter(studies_data[suid]))]['study_datetime']

        )
        session_bids_map = {orig_id: f"ses-{i+1:03d}" for i, orig_id in enumerate(sorted_original_study_uids)}

        for orig_study_uid, series_collection in studies_data.items():
            bids_ses_id = session_bids_map[orig_study_uid]
            logger.info(f"  Обработка сессии: {orig_study_uid} -> {bids_ses_id}")

            bids_anat_path = os.path.join(output_dir, bids_sub_id, bids_ses_id, 'anat')

            # Группировка серий по модальности для определения run-номеров
            modality_to_series_runs = defaultdict(list) # {'t1w': [(series_num_int, series_uid), ...]}
            
            for orig_series_uid, series_info in series_collection.items():
                first_ds_for_modality = series_info['first_ds']
                # Используем первый файл серии для логгинга в determine_modality и для fallback по пути
                modality_label = determine_modality(first_ds_for_modality, series_info['files'][0]) 
                
                if modality_label == 'unknown':
                    logger.warning(f"    Пропуск серии {orig_series_uid}: не удалось определить модальность.")
                    continue
                
                modality_to_series_runs[modality_label].append(
                    (series_info['series_number_int'], orig_series_uid)
                )
            
            if not modality_to_series_runs:
                logger.warning(f"    Нет серий с известной модальностью для сессии {bids_ses_id}.")
                continue

            # Копирование файлов с присвоением run-номеров
            for modality_label, run_candidates in modality_to_series_runs.items():
                # Сортируем серии внутри одной модальности по их SeriesNumber
                sorted_run_candidates = sorted(run_candidates, key=lambda x: x[0])
                
                bids_modality_dir = os.path.join(bids_anat_path, modality_label)
                try:
                    os.makedirs(bids_modality_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"    Не удалось создать директорию {bids_modality_dir}: {e}. Пропуск модальности {modality_label}.")
                    continue

                for run_idx, (_series_num_val, orig_series_uid_for_run) in enumerate(sorted_run_candidates, 1):
                    files_to_copy = series_collection[orig_series_uid_for_run]['files']
                    
                    # Сортируем файлы внутри серии по InstanceNumber
                    sorted_files_for_run = sort_dicom_files_in_series(files_to_copy)

                    logger.info(f"    Копирование {len(sorted_files_for_run)} файлов для {modality_label}"
                                f"{f'_run-{run_idx:02d}' if len(sorted_run_candidates) > 1 else ''} (Серия UID: {orig_series_uid_for_run})")

                    for slice_idx, src_file_path in enumerate(sorted_files_for_run, 1):
                        if len(sorted_run_candidates) > 1: # Если больше одной серии этой модальности
                            bids_filename = f"{bids_sub_id}_{bids_ses_id}_run-{run_idx:02d}_{modality_label}_{slice_idx:03d}.dcm"
                        else:
                            bids_filename = f"{bids_sub_id}_{bids_ses_id}_{modality_label}_{slice_idx:03d}.dcm"
                        
                        dst_file_path = os.path.join(bids_modality_dir, bids_filename)
                        try:
                            if action_type == 'move':
                                shutil.move(src_file_path, dst_file_path)
                                # logger.debug(f"      Перемещен: ...")
                            else: # По умолчанию 'copy'
                                shutil.copy(src_file_path, dst_file_path)
                                # logger.debug(f"      Скопирован: ...")
                        except Exception as e:
                            logger.error(f"      Не удалось {action_type} {src_file_path} в {dst_file_path}: {e}")
    
    logger.info("Организация данных в BIDS формат завершена!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Организует DICOM файлы из входной директории в BIDS-структуру на основе DICOM тегов.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Входная директория с DICOM файлами (любой вложенности).'
    )
    parser.add_argument(
        '--output_dir',
        default='bids_data_universal', # Изменено имя по умолчанию
        help='Корневая выходная директория для сохранения BIDS структуры.'
    )
    parser.add_argument(
        '--log_file',
        default=None,
        help='Путь к файлу для записи логов. Если не указан, будет создан файл "dicom_to_bids.log" внутри --output_dir.'
    )
    parser.add_argument(
        '--action', 
        type=str,
        default='copy',
        choices=['copy', 'move'],
        help='Действие с файлами: "copy" для копирования, "move" для перемещения.'
    )

    args = parser.parse_args()

    log_file_path_arg = args.log_file
    if log_file_path_arg is None:
        # Создаем output_dir заранее, если его нет, чтобы положить туда лог
        # Это делается также в setup_logging, но здесь для определения пути
        if args.output_dir and not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir, exist_ok=True)
            except OSError as e:
                print(f"Предупреждение: Не удалось создать выходную директорию {args.output_dir} для лог-файла. Ошибка: {e}")
                # Лог будет в текущей директории, если output_dir создать не удалось
                log_file_path_arg = 'dicom_to_bids.log'

        if log_file_path_arg is None: # Если все еще None (т.е. output_dir был создан или существовал)
             log_file_path_arg = os.path.join(args.output_dir or '.', 'dicom_to_bids.log')

    setup_logging(log_file_path_arg) # Настройка логгирования

    try:
        logger.info("="*60)
        logger.info(f"Запуск скрипта DICOM в BIDS конвертера")
        logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path_arg)}")
        logger.info("="*60)

        organize_dicom_to_bids(args.input_dir, args.output_dir, action_type=args.action)

        logger.info("Скрипт успешно завершил работу.")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Критическая ошибка: Файл или директория не найдены. {e}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Непредвиденная критическая ошибка:") # exc_info=True по умолчанию для logger.exception
        sys.exit(1)