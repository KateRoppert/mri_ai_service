import os
import json
from pathlib import Path
import argparse
import logging 
import sys 

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path: str):
    """
    Настраивает вывод логов в консоль (INFO) и файл (DEBUG).
    """
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
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)

# --- Пороги для интерпретации метрик ---
# Эти пороги являются эвристическими и могут требовать настройки
# в зависимости от конкретных данных и протоколов сканирования.
THRESHOLDS_MRIQC = {
    'cjv': {'high_thresh': 1.0, 'interpretation': "Коэффициент вариации между тканями (CJV)"},
    'cnr': {'low_thresh': 1.0, 'interpretation': "Контраст между серым и белым веществом (CNR)"},
    'efc': {'high_thresh': 0.5, 'interpretation': "Энтропия Фурье-коэффициентов (EFC)"}, # EFC из MRIQC обычно ниже, чем EFC из предыдущего скрипта
    'inu_med': {'low_thresh': 0.4, 'high_thresh': 1.5, 'interpretation': "Неоднородность интенсивности (INU Median)"},
    'snr_total': {'low_thresh': 5.0, 'interpretation': "Отношение сигнал/шум (SNR Total)"}
    # Можно добавить другие метрики из JSON отчетов MRIQC, если они там есть
}

def interpret_mriqc_metrics(metrics: dict, source_json_filename: str) -> tuple[str, str]:
    """
    Генерирует текстовую интерпретацию метрик из отчета MRIQC.

    Args:
        metrics (dict): Словарь с метриками, загруженный из JSON файла MRIQC.
        source_json_filename (str): Имя исходного JSON файла (для заголовка отчета).

    Returns:
        tuple[str, str]: (Текст интерпретации, Итоговый вердикт качества)
    """
    lines = [f"Интерпретация результатов MRIQC для файла: {source_json_filename}\n"]
    overall_quality = "Хорошее" # Оптимистичное предположение
    issues_found = []

    # --- Интерпретация отдельных метрик ---
    cjv = metrics.get("cjv")
    if cjv is not None:
        thresh = THRESHOLDS_MRIQC['cjv']
        lines.append(f"- {thresh['interpretation']}: {cjv:.3f}")
        if cjv > thresh['high_thresh']:
            lines.append(f"  - Вердикт: Высокий. Может указывать на шум или артефакты движения.")
            issues_found.append("Высокий CJV")
        else:
            lines.append(f"  - Вердикт: В норме. Сигнал и шум сбалансированы.")

    cnr = metrics.get("cnr") # MRIQC может не всегда считать эту метрику для всех типов сканов
    if cnr is not None:
        thresh = THRESHOLDS_MRIQC['cnr']
        lines.append(f"- {thresh['interpretation']}: {cnr:.3f}")
        if cnr < thresh['low_thresh']:
            lines.append(f"  - Вердикт: Низкий. Может быть трудно различить серое и белое вещество.")
            issues_found.append("Низкий CNR")
        else:
            lines.append(f"  - Вердикт: Приемлемый контраст.")

    efc = metrics.get("efc")
    if efc is not None:
        thresh = THRESHOLDS_MRIQC['efc']
        lines.append(f"- {thresh['interpretation']}: {efc:.3f}")
        # В MRIQC EFC обычно ниже. Высокие значения (>0.5, но <1) могут указывать на артефакты.
        # Очень низкие (<0.3-0.4) могут быть проблемой для некоторых типов анализа.
        # Здесь простая эвристика.
        if efc > thresh['high_thresh']: # Пример порога
            lines.append(f"  - Вердикт: Повышенный. Может указывать на наличие артефактов или шума.")
            # issues_found.append("Повышенный EFC") # Решаем, считать ли это проблемой для вердикта
        elif efc < 0.35: # Пример нижнего порога
            lines.append(f"  - Вердикт: Низкий. Может указывать на излишнее сглаживание или низкую сложность сигнала.")
            # issues_found.append("Низкий EFC")
        else:
            lines.append(f"  - Вердикт: В пределах ожидаемого диапазона.")

    inu_med = metrics.get("inu_med") # Медианное значение INU
    if inu_med is not None:
        thresh = THRESHOLDS_MRIQC['inu_med']
        lines.append(f"- {thresh['interpretation']}: {inu_med:.3f}")
        if not (thresh['low_thresh'] <= inu_med <= thresh['high_thresh']):
            lines.append(f"  - Вердикт: Выходит за пределы нормы. Возможна значительная неоднородность интенсивности.")
            issues_found.append("Проблемы с INU")
        else:
            lines.append(f"  - Вердикт: Равномерность интенсивности в норме.")
    
    # В MRIQC SNR обычно разделен по тканям (snr_csf, snr_gm, snr_wm)
    # snr_total из вашего скрипта, возможно, относится к другой метрике.
    # Посмотрим на `rpve_csf`, `rpve_gm`, `rpve_wm` или `snr_csf` и т.д.
    # Для примера, возьмем snr_wm, если он есть
    snr_wm = metrics.get("snr_wm")
    if snr_wm is not None:
        # Используем пороги от snr_total для примера, но они могут быть нерелевантны для snr_wm
        thresh = THRESHOLDS_MRIQC['snr_total']
        lines.append(f"- SNR для белого вещества (snr_wm): {snr_wm:.3f}")
        if snr_wm < thresh['low_thresh']: # Примерный порог
            lines.append(f"  - Вердикт: Низкое SNR для белого вещества. Изображение может быть шумным.")
            issues_found.append("Низкий SNR (WM)")
        else:
            lines.append(f"  - ВердиKT: Хорошее качество сигнала для белого вещества.")
    elif "snr_total" in metrics: # Если есть ваша метрика snr_total
        snr_total_val = metrics["snr_total"]
        thresh = THRESHOLDS_MRIQC['snr_total']
        lines.append(f"- Общее SNR (snr_total): {snr_total_val:.3f}")
        if snr_total_val < thresh['low_thresh']:
            lines.append(f"  - Вердикт: Низкое. Изображение может быть шумным.")
            issues_found.append("Низкий SNR (Total)")
        else:
            lines.append(f"  - Вердикт: Хорошее качество сигнала.")


    # Проверка предупреждений из секции provenance (если есть)
    # Пример структуры: metrics.get("provenance", {}).get("settings", {}).get("warnings", {})
    # В реальном MRIQC JSON структура может быть сложнее. Нужно смотреть конкретный файл.
    # Упрощенный вариант из вашего скрипта:
    warnings_mriqc = metrics.get("provenance", {}).get("warnings", {})
    if isinstance(warnings_mriqc, dict) and warnings_mriqc: # Убедимся, что это словарь и не пустой
        lines.append("\n- Предупреждения от MRIQC (provenance/warnings):")
        for key, value in warnings_mriqc.items():
            if value: # Если значение предупреждения True или не пустое
                warning_text = str(value) if not isinstance(value, bool) else "Присутствует"
                lines.append(f"  - {key.replace('_', ' ').capitalize()}: {warning_text}")
                issues_found.append(f"Предупреждение MRIQC: {key}")
    elif isinstance(metrics.get("warnings"), list) and metrics["warnings"]: # Альтернативная структура
        lines.append("\n- Предупреждения от MRIQC (корень/warnings):")
        for warn_item in metrics["warnings"]:
            if isinstance(warn_item, dict) and warn_item.get("message"):
                 lines.append(f"  - {warn_item.get('message')} (Уровень: {warn_item.get('level', 'N/A')})")
                 issues_found.append(f"Предупреждение MRIQC: {warn_item.get('message')[:30]}...")
            else:
                 lines.append(f"  - {str(warn_item)}")
                 issues_found.append("Общее предупреждение MRIQC")


    # --- Итоговый вывод ---
    lines.append("\nИтоговый вывод по качеству:")
    if issues_found:
        overall_quality = "Приемлемое / Требует внимания"
        lines.append(f"⚠️ Обнаружены следующие моменты, требующие внимания ({len(issues_found)}):")
        for issue in issues_found:
            lines.append(f"  - {issue}")
        if len(issues_found) >= 2 or "Низкий SNR" in " ".join(issues_found) or "Высокий CJV" in " ".join(issues_found):
            overall_quality = "Низкое / Рекомендуется проверка"
            lines.append("  Общая оценка: Качество изображения вызывает опасения. Рекомендуется детальная проверка специалистом.")
        else:
            lines.append("  Общая оценка: Качество изображения в целом приемлемое, но есть замечания.")
    else:
        lines.append("✅ По данным автоматическим метрикам, качество изображения в целом хорошее.")

    return "\n".join(lines), overall_quality


def process_mriqc_jsons(mriqc_output_dir_str: str, interpretation_output_dir_str: str):
    """
    Обходит директорию с результатами MRIQC, читает JSON файлы с метриками,
    генерирует и сохраняет текстовые интерпретации.

    Args:
        mriqc_output_dir_str (str): Путь к директории mriqc_output.
        interpretation_output_dir_str (str): Путь для сохранения файлов интерпретаций.
    """
    mriqc_output_dir = Path(mriqc_output_dir_str)
    interpretation_output_dir = Path(interpretation_output_dir_str)

    # --- Проверки ---
    if not mriqc_output_dir.is_dir():
        logger.error(f"Директория с результатами MRIQC не найдена: {mriqc_output_dir}")
        raise FileNotFoundError(f"Директория MRIQC не найдена: {mriqc_output_dir}")

    try:
        interpretation_output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Директория для интерпретаций создана/проверена: {interpretation_output_dir}")
    except OSError as e:
        logger.error(f"Не удалось создать директорию для интерпретаций {interpretation_output_dir}: {e}")
        raise

    processed_files = 0
    error_files = 0

    logger.info(f"Поиск JSON файлов с метриками MRIQC в: {mriqc_output_dir}")
    # Ищем все JSON файлы, но исключаем dataset_description.json и групповые отчеты (group_bold.json, group_T1w.json)
    # Ориентируемся на индивидуальные отчеты, которые обычно имеют вид sub-XXX_ses-YYY_..._bold.json или ..._T1w.json
    for json_filepath in mriqc_output_dir.rglob('*.json'):
        if json_filepath.name.startswith("dataset_description") or \
           json_filepath.name.startswith("group_"):
            logger.debug(f"Пропуск файла: {json_filepath.name}")
            continue

        logger.info(f"Обработка файла: {json_filepath.relative_to(mriqc_output_dir)}")
        try:
            with open(json_filepath, "r", encoding='utf-8') as f:
                metrics = json.load(f)

            # Генерируем интерпретацию
            interpretation_text, quality_verdict = interpret_mriqc_metrics(metrics, json_filepath.name)
            logger.debug(f"  Сгенерирована интерпретация, вердикт: {quality_verdict}")

            # Сохраняем результат
            # Имя выходного файла: sub-XXX_ses-YYY_..._bold_interpretation.txt
            output_filename = json_filepath.stem + "_interpretation.txt"
            # Сохраняем в ту же структуру подпапок, что и в mriqc_output, но в interpretation_output_dir
            relative_dir = json_filepath.parent.relative_to(mriqc_output_dir)
            output_interpretation_subdir = interpretation_output_dir / relative_dir
            output_interpretation_subdir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_interpretation_subdir / output_filename

            logger.debug(f"  Сохранение интерпретации в: {output_file_path}")
            with open(output_file_path, "w", encoding='utf-8') as out_f:
                out_f.write(interpretation_text)
            processed_files += 1

        except json.JSONDecodeError as e:
            logger.error(f"  Ошибка декодирования JSON в файле {json_filepath.name}: {e}")
            error_files += 1
        except Exception as e:
            logger.exception(f"  Непредвиденная ошибка при обработке файла {json_filepath.name}: {e}")
            error_files += 1

    logger.info("-" * 50)
    logger.info("Интерпретация результатов MRIQC завершена.")
    logger.info(f"  Успешно обработано JSON файлов: {processed_files}")
    if error_files > 0:
        logger.warning(f"  Возникли ошибки при обработке {error_files} JSON файлов.")
    logger.info(f"  Файлы интерпретаций сохранены в: {interpretation_output_dir.resolve()}")
    logger.info("-" * 50)
    return True


# --- Точка входа при запуске скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерирует текстовые интерпретации из JSON отчетов MRIQC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mriqc_dir",
        required=True,
        help="Путь к директории с результатами MRIQC (например, 'mriqc_output')."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Путь к директории для сохранения файлов с интерпретациями (например, 'mriqc_interpretation')."
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Путь к файлу для записи логов. Если не указан, "
             "будет создан 'mriqc_interpretation.log' внутри --output_dir."
    )

    args = parser.parse_args()

    # --- Настройка логирования ---
    log_file_path_main = args.log_file
    output_dir_path_main = args.output_dir # Лог пойдет в папку с результатами этого скрипта
    default_main_log_filename = 'mriqc_interpretation.log'

    if log_file_path_main is None:
        try:
            if output_dir_path_main and not os.path.exists(output_dir_path_main):
                 os.makedirs(output_dir_path_main, exist_ok=True)
            log_file_path_final = os.path.join(output_dir_path_main or '.', default_main_log_filename)
        except OSError as e:
             log_file_path_final = default_main_log_filename
             print(f"Предупреждение: Не удалось использовать {output_dir_path_main} для лог-файла. "
                   f"Лог будет записан в {log_file_path_final}. Ошибка: {e}")
    else:
        log_file_path_final = log_file_path_main

    setup_logging(log_file_path_final)

    # --- Основной блок выполнения ---
    try:
        logger.info("=" * 50)
        logger.info(f"Запуск mriqc_interpretation.py")
        logger.info(f"  Директория с отчетами MRIQC: {os.path.abspath(args.mriqc_dir)}")
        logger.info(f"  Выходная директория для интерпретаций: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Лог-файл скрипта: {os.path.abspath(log_file_path_final)}")
        logger.info("=" * 50)

        success = process_mriqc_jsons(args.mriqc_dir, args.output_dir)

        if success: # process_mriqc_jsons всегда возвращает True, если сам скрипт не упал
            logger.info("Скрипт mriqc_interpretation.py успешно завершил работу.")
            sys.exit(0)
        # else: # Эта ветка не будет достигнута при текущей логике
        #     logger.error("Скрипт mriqc_interpretation.py завершился с ошибкой.")
        #     sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"Критическая ошибка: Директория MRIQC не найдена. {e}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Непредвиденная критическая ошибка на верхнем уровне: {e}")
        sys.exit(1)