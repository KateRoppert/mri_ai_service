# import argparse
# import logging
# import sys
# from pathlib import Path
# import yaml
# import os
# import requests # Для HTTP запросов
# import json # Для обработки JSON ответов или параметров

# # --- Настройка логгера ---
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s')

# def setup_segmentation_logging(log_file_path_str: str, console_level_str: str = "INFO"):
#     """Настраивает логгер для этого скрипта."""
#     if logger.hasHandlers(): logger.handlers.clear()
#     ch = logging.StreamHandler(sys.stdout)
#     try: ch.setLevel(getattr(logging, console_level_str.upper()))
#     except AttributeError: ch.setLevel(logging.INFO); print(f"WARN: Invalid console log level '{console_level_str}'. Using INFO.")
#     ch.setFormatter(formatter); logger.addHandler(ch)
#     try:
#         log_dir = os.path.dirname(log_file_path_str)
#         if log_dir: os.makedirs(log_dir, exist_ok=True)
#         fh = logging.FileHandler(log_file_path_str, encoding='utf-8', mode='a')
#         fh.setLevel(logging.DEBUG); fh.setFormatter(formatter); logger.addHandler(fh)
#         logger.debug(f"Логирование segmentation.py в файл: {log_file_path_str}")
#     except Exception as e: logger.error(f"Не удалось настроить файловый лог {log_file_path_str}: {e}")


# def run_simple_server_segmentation(
#     input_t1_path: Path,
#     input_t1ce_path: Path,
#     input_t2_path: Path,
#     input_flair_path: Path,
#     output_mask_path: Path,
#     server_url: str,
#     model_name: str,
#     client_id: str # Например, "sub-001_ses-001"
# ) -> bool:
#     """
#     Отправляет 4 NIfTI файла на simple_server.py для сегментации и сохраняет результат.
#     Использует эндпоинт /v1/inference.

#     Args:
#         input_t1_path: Путь к T1 NIfTI.
#         input_t1ce_path: Путь к T1c NIfTI.
#         input_t2_path: Путь к T2 NIfTI.
#         input_flair_path: Путь к T2-FLAIR NIfTI.
#         output_mask_path: Путь для сохранения маски.
#         server_url (str): URL сервера (например, "http://172.16.71.222:5000").
#         model_name (str): Имя модели на сервере (например, "Unet").
#         client_id (str): Идентификатор клиента/запроса.

#     Returns:
#         bool: True при успехе, False при ошибке.
#     """
#     logger.info(f"Запуск сегментации для клиента: {client_id}")
#     logger.info(f"  Сервер: {server_url}, Модель: {model_name}")
#     logger.info(f"  T1: {input_t1_path.name}")
#     logger.info(f"  T1c: {input_t1ce_path.name}")
#     logger.info(f"  T2: {input_t2_path.name}")
#     logger.info(f"  FLAIR: {input_flair_path.name}")
#     logger.info(f"  Выходная маска: {output_mask_path}")

#     # Формируем URL с query параметрами
#     inference_url = f"{server_url}/v1/inference?net={model_name}&client_id={client_id}"
#     logger.debug(f"URL запроса: {inference_url}")

#     # Подготовка файлов для multipart/form-data
#     # Ключи должны быть такими, чтобы сервер мог извлечь 't1', 't1c', 't2', 't2flair'
#     # simple_server.py делает: ftype = file_key_from_request.split("_")[0]
#     # Значит, ключи должны быть: "t1_...", "t1c_...", "t2_...", "t2flair_..."
#     files_to_send = {}
#     opened_files = [] # Список для отслеживания открытых файлов, чтобы закрыть их

#     try:
#         # Открываем все файлы в бинарном режиме для чтения
#         f_t1 = open(input_t1_path, 'rb'); opened_files.append(f_t1)
#         f_t1c = open(input_t1ce_path, 'rb'); opened_files.append(f_t1c)
#         f_t2 = open(input_t2_path, 'rb'); opened_files.append(f_t2)
#         f_flair = open(input_flair_path, 'rb'); opened_files.append(f_flair)

#         # Ключи должны позволить серверу извлечь 't1', 't1c', 't2', 't2flair'
#         # Сервер использует file_key.split("_")[0]
#         files_to_send = {
#             f't1_{input_t1_path.name}': (input_t1_path.name, f_t1, 'application/octet-stream'),
#             f't1c_{input_t1ce_path.name}': (input_t1ce_path.name, f_t1c, 'application/octet-stream'),
#             f't2_{input_t2_path.name}': (input_t2_path.name, f_t2, 'application/octet-stream'),
#             # Для FLAIR, сервер может ожидать 't2flair' или просто 'flair' как ftype
#             # В simple_server.py prepare_files_for_unet использует files["t2flair"]
#             # Значит, ftype должен быть "t2flair"
#             f't2flair_{input_flair_path.name}': (input_flair_path.name, f_flair, 'application/octet-stream')
#         }
#         logger.debug(f"Подготовлены файлы для отправки с ключами: {list(files_to_send.keys())}")

#         logger.info("Отправка запроса на сегментацию на сервер...")
#         response = requests.post(inference_url, files=files_to_send, timeout=1200) # Таймаут 20 минут

#         response.raise_for_status() # Проверка на HTTP ошибки 4xx/5xx

#         # Сервер должен вернуть файл маски в теле ответа
#         if response.headers.get('Content-Type') == 'application/octet-stream' or \
#            response.headers.get('Content-Type') == 'application/x-nifti' or \
#            response.headers.get('Content-Type') == 'application/gzip': # Часто .nii.gz как gzip
#             logger.info("Получен ответ с файлом сегментации от сервера.")
#             output_mask_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(output_mask_path, 'wb') as f_mask:
#                 f_mask.write(response.content)
#             logger.info(f"Маска сегментации успешно сохранена: {output_mask_path}")
#             return True
#         else:
#             logger.error(f"Неожиданный Content-Type ответа сервера: {response.headers.get('Content-Type')}")
#             logger.error(f"Тело ответа (первые 500 символов): {response.text[:500]}")
#             return False

#     except requests.exceptions.Timeout:
#         logger.error(f"Ошибка: Таймаут при обращении к серверу {server_url}.")
#         return False
#     except requests.exceptions.ConnectionError:
#         logger.error(f"Ошибка: Не удалось подключиться к серверу {server_url}.")
#         return False
#     except requests.exceptions.HTTPError as e_http:
#         logger.error(f"Ошибка HTTP от сервера: {e_http.response.status_code} {e_http.response.reason}")
#         try: logger.error(f"  Тело ответа сервера: {e_http.response.json()}")
#         except: logger.error(f"  Тело ответа сервера (не JSON): {e_http.response.text[:500]}")
#         return False
#     except FileNotFoundError as e_fnf: # Если один из входных файлов не найден локально
#         logger.error(f"Ошибка: Входной NIfTI файл не найден: {e_fnf.filename}")
#         return False
#     except Exception as e:
#         logger.exception(f"Непредвиденная ошибка при выполнении сегментации: {e}")
#         return False
#     finally:
#         for f in opened_files:
#             if not f.closed:
#                 f.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Запускает сегментацию МРТ, отправляя 4 модальности на сервер.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("--input_t1", required=True, help="Путь к предобработанному T1 NIfTI.")
#     parser.add_argument("--input_t1ce", required=True, help="Путь к предобработанному T1c NIfTI.")
#     parser.add_argument("--input_t2", required=True, help="Путь к предобработанному T2 NIfTI.")
#     parser.add_argument("--input_flair", required=True, help="Путь к предобработанному T2-FLAIR NIfTI.")
#     parser.add_argument("--output_mask", required=True, help="Путь для сохранения выходной маски.")
#     parser.add_argument("--config", required=True, help="Путь к YAML конфигу пайплайна.")
#     parser.add_argument("--client_id", required=True, help="Идентификатор клиента/запуска (например, sub-001_ses-001).")
#     parser.add_argument("--log_file", default=None, help="Лог-файл. По умолч.: <output_mask_stem>_segmentation.log")
#     parser.add_argument("--console_log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Уровень лога консоли.")
#     args = parser.parse_args()

#     # --- Настройка логирования ---
#     log_file_p_arg = args.log_file
#     output_mask_p_obj = Path(args.output_mask)
#     if log_file_p_arg is None:
#         log_filename_default = output_mask_p_obj.stem.replace("_segmask","") + "_segmentation.log"
#         log_p_final_obj = output_mask_p_obj.parent / log_filename_default
#     else:
#         log_p_final_obj = Path(log_file_p_arg)
#     try: log_p_final_obj.parent.mkdir(parents=True, exist_ok=True)
#     except OSError as e_mkdir: print(f"Ошибка создания папки для лога {log_p_final_obj.parent}: {e_mkdir}")
#     setup_segmentation_logging(str(log_p_final_obj), args.console_log_level)

#     # --- Чтение параметров из конфига ---
#     try:
#         config_p_obj = Path(args.config)
#         if not config_p_obj.is_file(): raise FileNotFoundError(f"Конфиг не найден: {config_p_obj}")
#         with open(config_p_obj, 'r', encoding='utf-8') as f: pipeline_cfg_main = yaml.safe_load(f)
        
#         aiaa_srv_url = pipeline_cfg_main.get('executables', {}).get('aiaa_server_url')
#         seg_step_cfg = pipeline_cfg_main.get('steps', {}).get('segmentation', {})
#         model_nm_cfg = seg_step_cfg.get('model_name')

#         if not aiaa_srv_url: raise ValueError("'executables.aiaa_server_url' не найден в конфиге.")
#         if not model_nm_cfg: raise ValueError("'steps.segmentation.model_name' не найден в конфиге.")
#     except Exception as e_cfg: logger.critical(f"Ошибка загрузки/парсинга конфига {args.config}: {e_cfg}", exc_info=True); sys.exit(1)

#     # --- Основной блок ---
#     try:
#         logger.info("=" * 50); logger.info(f"Запуск segmentation.py")
#         logger.info(f"  Input T1: {Path(args.input_t1).resolve()}"); logger.info(f"  Input T1c: {Path(args.input_t1ce).resolve()}")
#         logger.info(f"  Input T2: {Path(args.input_t2).resolve()}"); logger.info(f"  Input FLAIR: {Path(args.input_flair).resolve()}")
#         logger.info(f"  Output Mask: {output_mask_p_obj.resolve()}")
#         logger.info(f"  AIAA Server: {aiaa_srv_url}"); logger.info(f"  Model: {model_nm_cfg}"); logger.info(f"  Client ID: {args.client_id}")
#         logger.info(f"  Log: {log_p_final_obj.resolve()}"); logger.info("=" * 50)

#         success = run_simple_server_segmentation(
#             Path(args.input_t1), Path(args.input_t1ce), Path(args.input_t2), Path(args.input_flair),
#             output_mask_p_obj, aiaa_srv_url, model_nm_cfg, args.client_id
#         )
#         if success: logger.info("Сегментация успешно завершена."); sys.exit(0)
#         else: logger.error("Сегментация завершилась с ошибкой."); sys.exit(1)
#     except FileNotFoundError as e: logger.error(f"Крит. ошибка: Файл не найден. {e}"); sys.exit(1)
#     except Exception as e: logger.exception(f"Непредвиденная крит. ошибка: {e}"); sys.exit(1)




import argparse
import logging
import sys
from pathlib import Path
import yaml
import os
import requests # Для HTTP запросов
import json # Для обработки JSON ответов или параметров

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s')

def setup_segmentation_logging(log_file_path_str: str, console_level_str: str = "INFO"):
    """Настраивает логгер для этого скрипта."""
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    try: ch.setLevel(getattr(logging, console_level_str.upper()))
    except AttributeError: ch.setLevel(logging.INFO); print(f"WARN: Invalid console log level '{console_level_str}'. Using INFO.")
    ch.setFormatter(formatter); logger.addHandler(ch)
    try:
        log_dir = os.path.dirname(log_file_path_str)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path_str, encoding='utf-8', mode='a') # mode 'a' для дозаписи в существующий лог пайплайна
        fh.setLevel(logging.DEBUG); fh.setFormatter(formatter); logger.addHandler(fh)
        logger.debug(f"Логирование segmentation.py в файл: {log_file_path_str}")
    except Exception as e: logger.error(f"Не удалось настроить файловый лог {log_file_path_str}: {e}")


def determine_files_for_server(
    input_t1_path: Path | None,
    input_t1ce_path: Path | None,
    input_t2_path: Path | None,
    input_flair_path: Path | None
) -> dict[str, Path] | None:
    """
    Определяет, какие файлы отправить на сервер, заполняя недостающие по приоритету.
    Возвращает словарь с ключами 't1', 't1c', 't2', 'flair' и путями к файлам,
    либо None, если ни одного входного файла не предоставлено.
    """
    server_inputs = {
        "t1": input_t1_path,
        "t1c": input_t1ce_path,
        "t2": input_t2_path,
        "flair": input_flair_path
    }
    logger.debug(f"Получены пути для сегментации: {server_inputs}")

    # Список доступных файлов в порядке приоритета (t1c > t1 > t2 > flair)
    priority_order = ['t1c', 't1', 't2', 'flair']
    available_files_prioritized = []
    for mod_key in priority_order:
        if server_inputs[mod_key] and server_inputs[mod_key].is_file():
            available_files_prioritized.append(server_inputs[mod_key])
        elif server_inputs[mod_key]: # Путь указан, но файла нет
             logger.warning(f"Указанный файл для модальности '{mod_key}' не найден: {server_inputs[mod_key]}")
             server_inputs[mod_key] = None # Считаем его отсутствующим
        # Если server_inputs[mod_key] изначально None, ничего не делаем

    if not available_files_prioritized:
        logger.error("Не предоставлено ни одного валидного входного файла для сегментации.")
        return None

    logger.info(f"Доступные файлы для сегментации (в порядке приоритета использования): "
                f"{[p.name for p in available_files_prioritized]}")

    # Заполняем недостающие слоты
    # Сервер ожидает ключи 't1', 't1c', 't2', 't2flair' (для ключей файлов)
    # Мы будем использовать 't1', 't1c', 't2', 'flair' для удобства здесь
    final_files_for_server = {}

    # Сначала присваиваем то, что есть
    final_files_for_server['t1'] = server_inputs['t1']
    final_files_for_server['t1c'] = server_inputs['t1c']
    final_files_for_server['t2'] = server_inputs['t2']
    final_files_for_server['flair'] = server_inputs['flair'] # Серверный ключ будет 't2flair'

    # Заполняем None самым приоритетным из доступных
    fallback_file = available_files_prioritized[0]
    
    if final_files_for_server['t1'] is None:
        final_files_for_server['t1'] = fallback_file
        logger.info(f"  Слот T1 заполнен файлом: {fallback_file.name}")
    if final_files_for_server['t1c'] is None:
        final_files_for_server['t1c'] = fallback_file
        logger.info(f"  Слот T1c заполнен файлом: {fallback_file.name}")
    if final_files_for_server['t2'] is None:
        final_files_for_server['t2'] = fallback_file
        logger.info(f"  Слот T2 заполнен файлом: {fallback_file.name}")
    if final_files_for_server['flair'] is None:
        final_files_for_server['flair'] = fallback_file
        logger.info(f"  Слот FLAIR заполнен файлом: {fallback_file.name}")
        
    return final_files_for_server


def run_simple_server_segmentation(
    files_for_server: dict[str, Path], # Словарь с ключами 't1', 't1c', 't2', 'flair'
    output_mask_path: Path,
    server_url: str,
    model_name: str,
    client_id: str
) -> bool:
    """
    Отправляет 4 NIfTI файла (возможно, с дубликатами) на simple_server.py для сегментации.
    """
    input_t1_path = files_for_server['t1']
    input_t1ce_path = files_for_server['t1c']
    input_t2_path = files_for_server['t2']
    input_flair_path = files_for_server['flair'] # На сервере это будет t2flair

    logger.info(f"Запуск сегментации для клиента: {client_id} с подготовленными файлами:")
    logger.info(f"  Сервер: {server_url}, Модель: {model_name}")
    logger.info(f"  Отправляемый T1: {input_t1_path.name}")
    logger.info(f"  Отправляемый T1c: {input_t1ce_path.name}")
    logger.info(f"  Отправляемый T2: {input_t2_path.name}")
    logger.info(f"  Отправляемый FLAIR (как t2flair): {input_flair_path.name}")
    logger.info(f"  Выходная маска: {output_mask_path}")

    inference_url = f"{server_url}/v1/inference?net={model_name}&client_id={client_id}"
    logger.debug(f"URL запроса: {inference_url}")

    files_to_send_multipart = {}
    opened_files_handles = []

    try:
        f_t1 = open(input_t1_path, 'rb'); opened_files_handles.append(f_t1)
        f_t1c = open(input_t1ce_path, 'rb'); opened_files_handles.append(f_t1c)
        f_t2 = open(input_t2_path, 'rb'); opened_files_handles.append(f_t2)
        f_flair = open(input_flair_path, 'rb'); opened_files_handles.append(f_flair)

        # Ключи для сервера, как и раньше
        files_to_send_multipart = {
            f't1_{input_t1_path.name}': (input_t1_path.name, f_t1, 'application/octet-stream'),
            f't1c_{input_t1ce_path.name}': (input_t1ce_path.name, f_t1c, 'application/octet-stream'),
            f't2_{input_t2_path.name}': (input_t2_path.name, f_t2, 'application/octet-stream'),
            f't2flair_{input_flair_path.name}': (input_flair_path.name, f_flair, 'application/octet-stream')
        }
        logger.debug(f"Подготовлены файлы для отправки с ключами: {list(files_to_send_multipart.keys())}")

        logger.info("Отправка запроса на сегментацию на сервер...")
        response = requests.post(inference_url, files=files_to_send_multipart, timeout=1200)
        response.raise_for_status()

        if response.headers.get('Content-Type') == 'application/octet-stream' or \
           response.headers.get('Content-Type') == 'application/x-nifti' or \
           response.headers.get('Content-Type') == 'application/gzip':
            logger.info("Получен ответ с файлом сегментации от сервера.")
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_mask_path, 'wb') as f_mask:
                f_mask.write(response.content)
            logger.info(f"Маска сегментации успешно сохранена: {output_mask_path}")
            return True
        else:
            logger.error(f"Неожиданный Content-Type ответа сервера: {response.headers.get('Content-Type')}")
            logger.error(f"Тело ответа (первые 500 символов): {response.text[:500]}")
            return False

    except requests.exceptions.Timeout:
        logger.error(f"Ошибка: Таймаут при обращении к серверу {server_url}.")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Ошибка: Не удалось подключиться к серверу {server_url}.")
        return False
    except requests.exceptions.HTTPError as e_http:
        logger.error(f"Ошибка HTTP от сервера: {e_http.response.status_code} {e_http.response.reason}")
        try: logger.error(f"  Тело ответа сервера: {e_http.response.json()}")
        except: logger.error(f"  Тело ответа сервера (не JSON): {e_http.response.text[:500]}")
        return False
    except FileNotFoundError as e_fnf:
        logger.error(f"Ошибка: Входной NIfTI файл не найден при попытке его открыть: {e_fnf.filename}")
        return False
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка при выполнении сегментации: {e}")
        return False
    finally:
        for f_handle in opened_files_handles:
            if not f_handle.closed:
                f_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запускает сегментацию МРТ, отправляя 4 модальности (или их заменители) на сервер.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Делаем аргументы необязательными
    parser.add_argument("--input_t1", default=None, required=False, help="Путь к предобработанному T1 NIfTI.")
    parser.add_argument("--input_t1ce", default=None, required=False, help="Путь к предобработанному T1c NIfTI.")
    parser.add_argument("--input_t2", default=None, required=False, help="Путь к предобработанному T2 NIfTI.")
    parser.add_argument("--input_flair", default=None, required=False, help="Путь к предобработанному T2-FLAIR NIfTI.")
    
    parser.add_argument("--output_mask", required=True, help="Путь для сохранения выходной маски.")
    parser.add_argument("--config", required=True, help="Путь к YAML конфигу пайплайна.")
    parser.add_argument("--client_id", required=True, help="Идентификатор клиента/запуска.")
    parser.add_argument("--log_file", default=None, help="Лог-файл.")
    parser.add_argument("--console_log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Уровень лога консоли.")
    args = parser.parse_args()

    log_file_p_arg = args.log_file
    output_mask_p_obj = Path(args.output_mask)
    if log_file_p_arg is None:
        log_filename_default = output_mask_p_obj.stem.replace("_segmask","") + "_segmentation.log"
        # Помещаем лог рядом с маской, если не указано иное
        log_p_final_obj = output_mask_p_obj.parent / log_filename_default
    else:
        log_p_final_obj = Path(log_file_p_arg)
    try: log_p_final_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e_mkdir: print(f"Ошибка создания папки для лога {log_p_final_obj.parent}: {e_mkdir}")
    setup_segmentation_logging(str(log_p_final_obj), args.console_log_level)

    try:
        config_p_obj = Path(args.config)
        if not config_p_obj.is_file(): raise FileNotFoundError(f"Конфиг не найден: {config_p_obj}")
        with open(config_p_obj, 'r', encoding='utf-8') as f: pipeline_cfg_main = yaml.safe_load(f)
        
        aiaa_srv_url = pipeline_cfg_main.get('executables', {}).get('aiaa_server_url')
        seg_step_cfg = pipeline_cfg_main.get('steps', {}).get('segmentation', {})
        model_nm_cfg = seg_step_cfg.get('model_name')

        if not aiaa_srv_url: raise ValueError("'executables.aiaa_server_url' не найден в конфиге.")
        if not model_nm_cfg: raise ValueError("'steps.segmentation.model_name' не найден в конфиге.")
    except Exception as e_cfg: logger.critical(f"Ошибка загрузки/парсинга конфига {args.config}: {e_cfg}", exc_info=True); sys.exit(1)

    # --- Основной блок ---
    try:
        logger.info("=" * 50); logger.info(f"Запуск segmentation.py (с выбором файлов)")
        
        # Преобразуем пути из аргументов в Path или None
        input_t1_p = Path(args.input_t1) if args.input_t1 else None
        input_t1ce_p = Path(args.input_t1ce) if args.input_t1ce else None
        input_t2_p = Path(args.input_t2) if args.input_t2 else None
        input_flair_p = Path(args.input_flair) if args.input_flair else None

        logger.info(f"  Заданный Input T1: {input_t1_p.resolve() if input_t1_p else 'Не указан'}")
        logger.info(f"  Заданный Input T1c: {input_t1ce_p.resolve() if input_t1ce_p else 'Не указан'}")
        logger.info(f"  Заданный Input T2: {input_t2_p.resolve() if input_t2_p else 'Не указан'}")
        logger.info(f"  Заданный Input FLAIR: {input_flair_p.resolve() if input_flair_p else 'Не указан'}")
        
        files_to_process = determine_files_for_server(
            input_t1_p, input_t1ce_p, input_t2_p, input_flair_p
        )

        if files_to_process is None:
            logger.error("Не удалось определить набор файлов для отправки на сервер. Завершение.")
            sys.exit(1)

        logger.info(f"  Output Mask: {output_mask_p_obj.resolve()}")
        logger.info(f"  AIAA Server: {aiaa_srv_url}"); logger.info(f"  Model: {model_nm_cfg}"); logger.info(f"  Client ID: {args.client_id}")
        logger.info(f"  Log: {log_p_final_obj.resolve()}"); logger.info("=" * 50)

        success = run_simple_server_segmentation(
            files_to_process,
            output_mask_p_obj, aiaa_srv_url, model_nm_cfg, args.client_id
        )
        if success: logger.info("Сегментация успешно завершена."); sys.exit(0)
        else: logger.error("Сегментация завершилась с ошибкой."); sys.exit(1)
        
    except FileNotFoundError as e: logger.error(f"Крит. ошибка: Файл не найден. {e}"); sys.exit(1)
    except Exception as e: logger.exception(f"Непредвиденная крит. ошибка: {e}"); sys.exit(1)