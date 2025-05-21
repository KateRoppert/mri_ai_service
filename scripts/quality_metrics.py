import os
import nibabel as nib
import numpy as np # Убедитесь, что numpy импортирован как np
from scipy.stats import entropy
from pathlib import Path
import argparse
from datetime import datetime
import warnings
import logging
import sys

# --- Константы и Параметры ---
CORNER_SIZE = 10
FOREGROUND_THRESHOLD_FACTOR = 2.5
ANISOTROPY_THRESHOLD = 3.0 # Глобальная переменная, будет обновлена из args

THRESHOLDS = {
    'fber': {'poor': 10, 'acceptable': 25},
    'efc': {'poor': 0.42, 'acceptable': 0.55},
    'noise_std': {'good': 10, 'acceptable': 25}, # Меньше = лучше
    'snr': {'poor': 8, 'acceptable': 20}
}

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path: str):
    """Настраивает вывод логов в консоль (INFO) и файл (DEBUG)."""
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(formatter); logger.addHandler(ch)
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG); fh.setFormatter(formatter); logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)

# --- Функции расчета метрик (без изменений в логике, кроме get_background_stats) ---

def get_background_stats(data: np.ndarray, corner_size: int = CORNER_SIZE) -> tuple[float | None, float | None]:
    """Оценивает фон по углам изображения."""
    corners = []; dims = data.shape
    if len(dims) != 3: logger.warning(f"Ожидались 3D, получено {len(dims)}D."); return None, None
    cs = [min(corner_size, d // 2) for d in dims]
    if any(c == 0 for c in cs): logger.warning("Изображение мало для углов."); return None, None
    try:
        # Извлечение углов...
        corners.append(data[ :cs[0],  :cs[1],  :cs[2]].flatten())
        corners.append(data[-cs[0]:,  :cs[1],  :cs[2]].flatten())
        corners.append(data[ :cs[0], -cs[1]:,  :cs[2]].flatten())
        corners.append(data[ :cs[0],  :cs[1], -cs[2]:].flatten())
        corners.append(data[-cs[0]:, -cs[1]:,  :cs[2]].flatten())
        corners.append(data[-cs[0]:,  :cs[1], -cs[2]:].flatten())
        corners.append(data[ :cs[0], -cs[1]:, -cs[2]:].flatten())
        corners.append(data[-cs[0]:, -cs[1]:, -cs[2]:].flatten())
        background_voxels = np.concatenate(corners)
        background_voxels = background_voxels[background_voxels != 0] # Оригинальная фильтрация
        if background_voxels.size > 50:
            mean_bg = np.mean(background_voxels); std_bg = np.std(background_voxels)
            if std_bg < 1e-6: std_bg = 1e-6; logger.warning("Ст. откл. фона близко к нулю.")
            # Сравнение с std всего изображения (как было в оригинале)
            if np.any(data) and std_bg > 0.2 * np.std(data):
                 logger.warning(f"Ст. откл. фона ({std_bg:.2f}) необычно высокое (общее std: {np.std(data):.2f}).")
            logger.debug(f"Оценка фона: Mean={mean_bg:.3f}, Std={std_bg:.3f}")
            return mean_bg, std_bg
        else: logger.warning(f"Недостаточно вокселей фона (>50): {background_voxels.size}."); return None, None
    except IndexError: logger.error("Ошибка индексации при извлечении углов.", exc_info=False); return None, None
    except Exception as e: logger.error(f"Неизвестная ошибка при оценке фона: {e}", exc_info=True); return None, None

def calculate_iqms(data: np.ndarray) -> dict:
    """Рассчитывает метрики качества изображения (IQMs)."""
    if data.ndim == 4:
        if data.shape[3] > 1: logger.info("4D данные: используется первый том.")
        data = data[..., 0]
    elif data.ndim != 3: msg = f'Неподдерживаемая размерность {data.ndim}D'; logger.error(msg); return {'error': msg}
    if np.all(data == 0): msg = 'Нулевое изображение'; logger.error(msg); return {'error': msg}
    if np.max(data) <= np.min(data): msg = 'Константное изображение'; logger.error(msg); return {'error': msg}

    metrics = {}; logger.debug("Расчет IQM: Оценка фона...")
    bg_mean, bg_std = get_background_stats(data)
    if bg_std is None or bg_std < 1e-6:
        logger.warning("Фон не оценен, используется запасной метод порога FG.")
        metrics['noise_std'] = None; p1 = np.percentile(data, 1)
        low_vox = data[data < np.percentile(data, 5)]; bg_std_fallback = np.std(low_vox) if low_vox.size > 1 else 1e-6
        threshold = p1 + FOREGROUND_THRESHOLD_FACTOR * bg_std_fallback if bg_std_fallback > 1e-6 else np.percentile(data, 10)
    else: metrics['noise_std'] = float(bg_std); threshold = bg_mean + FOREGROUND_THRESHOLD_FACTOR * bg_std
    logger.debug(f"Порог FG = {threshold:.3f}")
    foreground_mask = data > threshold; foreground_voxels = data[foreground_mask]
    if foreground_voxels.size < 100:
        warning_msg = f'Мало ({foreground_voxels.size}) вокселей ПП.'; logger.warning(f"Расчет IQM: {warning_msg}")
        metrics.update({'foreground_voxels_count': int(foreground_voxels.size),
                        'foreground_mean': float(np.mean(foreground_voxels)) if foreground_voxels.size > 0 else 0.0,
                        'foreground_median': float(np.median(foreground_voxels)) if foreground_voxels.size > 0 else 0.0,
                        'foreground_std': float(np.std(foreground_voxels)) if foreground_voxels.size > 1 else 0.0,
                        'fber': None, 'efc': None, 'snr': None, 'warning': warning_msg})
        return metrics
    metrics['foreground_voxels_count'] = int(foreground_voxels.size)
    fg_mean = float(np.mean(foreground_voxels)); fg_median = float(np.median(foreground_voxels)); fg_std = float(np.std(foreground_voxels))
    metrics['foreground_mean'] = fg_mean; metrics['foreground_median'] = fg_median; metrics['foreground_std'] = fg_std
    if fg_std < 1e-6: warning_msg = 'Низкая вариация ПП.'; logger.warning(f"Расчет IQM: {warning_msg}"); metrics['warning'] = metrics.get('warning', '') + ' ' + warning_msg
    metrics['fber'] = float(fg_median / metrics['noise_std']) if metrics.get('noise_std') and fg_median else None
    metrics['efc'] = 0.0
    try:
        if foreground_voxels.size > 0:
            max_intensity_fg = np.max(foreground_voxels)
            if max_intensity_fg > threshold and foreground_voxels.size > 1:
                fg_voxels_gt_zero = foreground_voxels[foreground_voxels > 1e-9]
                if fg_voxels_gt_zero.size > 1:
                    normalized = fg_voxels_gt_zero / max_intensity_fg
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)
                         entropy_val = float(entropy(normalized, base=2))
                    max_entropy = np.log2(fg_voxels_gt_zero.size)
                    metrics['efc'] = float(entropy_val / max_entropy) if max_entropy > 0 else 0.0
    except Exception as e: logger.error(f"Расчет IQM: Ошибка EFC: {e}", exc_info=True); metrics['efc'] = None
    metrics['snr'] = float(fg_mean / metrics['noise_std']) if metrics.get('noise_std') and fg_mean else None
    logger.debug(f"Рассчитанные метрики: {metrics}")
    return metrics

# === ИЗМЕНЕНИЯ В ЭТОЙ ФУНКЦИИ ===
def check_voxel_anisotropy(nifti_image: nib.Nifti1Image) -> tuple[tuple[float, ...] | None, float | None, bool]:
    """
    Проверяет анизотропию вокселей на основе размеров вокселей из заголовка NIfTI.
    """
    global ANISOTROPY_THRESHOLD
    try:
        zooms = nifti_image.header.get_zooms()
        if len(zooms) < 3:
            logger.warning("Анизотропия: Не удалось получить размеры для 3D.")
            return None, None, False

        voxel_sizes_orig = zooms[:3]

        # --- ИСПРАВЛЕННАЯ ПРОВЕРКА ТИПА ---
        # Проверяем, что это числа (включая типы numpy) и они положительные
        if not all(isinstance(v, (int, float, np.integer, np.floating)) and v > 1e-9 for v in voxel_sizes_orig):
             logger.warning(f"Анизотропия: Некорректные или нулевые размеры вокселей в заголовке: {voxel_sizes_orig}.")
             return tuple(voxel_sizes_orig), None, False # Возвращаем как есть
        # --- КОНЕЦ ИСПРАВЛЕННОЙ ПРОВЕРКИ ---

        # Теперь можем безопасно конвертировать и считать
        voxel_sizes_float = tuple(float(v) for v in voxel_sizes_orig)
        max_res, min_res = np.max(voxel_sizes_float), np.min(voxel_sizes_float)
        anisotropy_ratio = float(max_res / min_res) if min_res > 1e-9 else float('inf')
        is_anisotropic = anisotropy_ratio > ANISOTROPY_THRESHOLD

        logger.debug(
            f"Анизотропия: Размеры={voxel_sizes_float}, Отношение={anisotropy_ratio:.2f}, "
            f"Порог={ANISOTROPY_THRESHOLD}, Анизотропный={is_anisotropic}"
        )
        if is_anisotropic:
            logger.warning(f"Высокая анизотропия: {voxel_sizes_float} (ratio: {anisotropy_ratio:.2f})")

        return voxel_sizes_float, anisotropy_ratio, is_anisotropic
    except Exception as e:
        logger.error(f"Ошибка при чтении размеров вокселей: {e}", exc_info=False)
        return None, None, False

def interpret_metric(name: str, value, thresholds: dict) -> tuple[str, str]:
    """
    Интерпретирует метрику, возвращает текст и уровень.
    """
    if value is None: return "N/A", "na"
    level = "info"; interpretation_text = ""
    if isinstance(value, (float, np.floating)): formatted_value = f"{value:.3f}"
    elif isinstance(value, (int, np.integer)): formatted_value = f"{value}"
    elif isinstance(value, (tuple, list)): formatted_value = ", ".join([f"{v:.2f}" for v in value])
    else: formatted_value = str(value)
    if name in thresholds:
        limits = thresholds[name]
        if name in ['fber', 'efc', 'snr']:
            if value < limits['poor']: level = "poor"; interpretation_text = f"{formatted_value} (Низкое)"
            elif value < limits['acceptable']: level = "acceptable"; interpretation_text = f"{formatted_value} (Приемлемо)"
            else: level = "good"; interpretation_text = f"{formatted_value} (Хорошо)"
        elif name in ['noise_std']:
             if value < limits['good']: level = "good"; interpretation_text = f"{formatted_value} (Низкий)"
             elif value < limits['acceptable']: level = "acceptable"; interpretation_text = f"{formatted_value} (Умеренный)"
             else: level = "poor"; interpretation_text = f"{formatted_value} (Высокий)"
        else: interpretation_text = formatted_value
    else: interpretation_text = formatted_value
    return interpretation_text, level

def generate_report(metrics: dict | None, input_filepath: Path, voxel_info: tuple) -> tuple[str, str]:
    """
    Создает текстовый отчет о качестве, включая вердикт с учетом анизотропии и N/A.
    """
    voxel_sizes, anisotropy_ratio, is_anisotropic = voxel_info
    report_lines = []; quality_levels = {}; interpretations = {}; possible_issues = []

    # --- Шапка отчета ---
    report_lines.append("=" * 60)
    report_lines.append(f"Отчет о качестве МРТ изображения (Быстрая оценка)")
    report_lines.append(f"Файл: {input_filepath}")
    report_lines.append(f"Время генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Определения метрик ---
    metric_definitions = {
        'noise_std': "Шум фона (Ст.Откл.)", 'snr': "Сигнал/Шум (SNR)",
        'fber': "Контраст ПП/Фон (FBER)", 'efc': "Концентрация сигнала (EFC)",
        'foreground_voxels_count': "Воксели ПП (кол-во)", 'foreground_mean': "Средняя интенсивность ПП",
        'foreground_median': "Медианная интенсивность ПП", 'foreground_std': "Ст.Откл. интенсивности ПП"
    }
    key_metrics_intensity = ['noise_std', 'snr', 'fber', 'efc']
    overall_quality = "Требуется ручная оценка"

    # --- Анализ метрик интенсивности ---
    if metrics is None or 'error' in metrics:
        error_msg = metrics.get('error', 'Неизвестная ошибка') if isinstance(metrics, dict) else 'Ошибка загрузки/обработки'
        possible_issues.append(f"Ошибка расчета метрик интенсивности: {error_msg}")
        overall_quality = "ОШИБКА РАСЧЕТА"
    else:
        for name in metric_definitions:
            value = metrics.get(name)
            interpretation, level = interpret_metric(name, value, THRESHOLDS)
            quality_levels[name] = level; interpretations[name] = interpretation
        if quality_levels.get('noise_std') == 'poor': possible_issues.append("Высокий уровень шума.")
        if quality_levels.get('snr') == 'poor': possible_issues.append("Низкое SNR.")
        if quality_levels.get('fber') == 'poor': possible_issues.append("Низкий контраст ПП/Фон.")
        if quality_levels.get('efc') == 'poor': possible_issues.append("Низкая концентрация сигнала (EFC).")
        if 'warning' in metrics and metrics['warning']: possible_issues.append(f"Предупреждение при расчете: {metrics['warning']}")

    # --- Учет анизотропии ---
    if is_anisotropic: # Проверяем флаг, возвращенный check_voxel_anisotropy
        anisotropy_issue = (f"Высокая анизотропия вокселей (ratio={anisotropy_ratio:.2f} > "
                            f"{ANISOTROPY_THRESHOLD:.1f}). Качество может быть снижено.")
        possible_issues.append(anisotropy_issue)

    # --- Определение вердикта ---
    if overall_quality != "ОШИБКА РАСЧЕТА":
        num_poor = list(quality_levels.values()).count('poor')
        num_acceptable = list(quality_levels.values()).count('acceptable')
        num_good = list(quality_levels.values()).count('good')
        key_metrics_na_count = sum(1 for m in key_metrics_intensity if quality_levels.get(m) == 'na')

        if num_poor > 0 or is_anisotropic or key_metrics_na_count > 0:
            overall_quality = "Низкое / Требует проверки"
            if key_metrics_na_count > 0:
                 na_issue = f"Не удалось рассчитать {key_metrics_na_count} ключевых метрик(и) интенсивности (N/A)."
                 if na_issue not in possible_issues: possible_issues.append(na_issue)
        elif num_acceptable > 0:
            overall_quality = "Приемлемое"
        elif num_good > 0:
            overall_quality = "Хорошее"

    # --- Формирование текста отчета ---
    report_lines.append("=" * 60)
    report_lines.append(f"*** ИТОГОВЫЙ ВЕРДИКТ: {overall_quality} ***")
    if possible_issues:
        report_lines.append("  Возможные проблемы и комментарии:")
        for issue in possible_issues: report_lines.append(f"  - {issue}")
    elif overall_quality not in ["ОШИБКА РАСЧЕТА", "Требуется ручная оценка"]:
        report_lines.append("  - Серьезных проблем по рассчитанным метрикам и геометрии не выявлено.")
    report_lines.append("=" * 60)

    report_lines.append("\nДетализация отчета:"); report_lines.append("-" * 60)

    # Геометрия - Вывод информации об анизотропии
    report_lines.append("-- Геометрия --")
    if voxel_sizes:
        report_lines.append(f"  Размеры вокселя (мм): {', '.join([f'{v:.2f}' for v in voxel_sizes])}")
        if anisotropy_ratio is not None:
             report_lines.append(f"  Отношение анизотропии: {anisotropy_ratio:.2f}")
             if is_anisotropic: # Если флаг True, выводим предупреждение
                 report_lines.append(f"  ПРЕДУПРЕЖДЕНИЕ: Высокая анизотропия!")
    else:
        report_lines.append("  Размеры вокселя: Не удалось прочитать.")

    # Метрики Интенсивности
    report_lines.append("\n-- Метрики Интенсивности --")
    if metrics is not None and 'error' not in metrics:
        for name, description in metric_definitions.items():
             interpretation_text = interpretations.get(name, "N/A")
             report_lines.append(f"  {description:<25}: {interpretation_text}")
    elif metrics is not None and 'error' in metrics:
         report_lines.append(f"  Ошибка при расчете метрик: {metrics['error']}")
    else:
         report_lines.append("  Метрики интенсивности не были рассчитаны.")

    # Важные примечания
    report_lines.append("\n" + "=" * 60)
    report_lines.append("ВАЖНО:")
    report_lines.append("- Оценка качества произведена без точной сегментации мозга.")
    report_lines.append("- Пороги для интерпретации являются ОРИЕНТИРОВОЧНЫМИ.")
    report_lines.append("- Отчет не заменяет визуальный контроль качества специалистом.")
    report_lines.append("=" * 60)

    return "\n".join(report_lines), overall_quality

def process_nifti_file(nifti_file: Path, output_dir: Path, input_root_dir: Path) -> bool:
    """Обрабатывает один NIfTI файл."""
    relative_path = nifti_file.relative_to(input_root_dir)
    logger.info(f"Обработка файла: {relative_path}")
    metrics = None; voxel_info = (None, None, False); quality_level = "Ошибка"; success = False
    try:
        logger.debug(f"Загрузка заголовка: {nifti_file}"); img = nib.load(nifti_file)
        logger.debug("Проверка анизотропии..."); voxel_info = check_voxel_anisotropy(img)
        logger.debug("Загрузка данных..."); data = img.get_fdata(dtype=np.float32)
        logger.debug("Расчет метрик..."); metrics = calculate_iqms(data)
        logger.debug("Генерация отчета...")
        report_text, quality_level = generate_report(metrics, relative_path, voxel_info)

        output_report_dir = output_dir / relative_path.parent
        output_report_dir.mkdir(parents=True, exist_ok=True)
        report_filename_base = relative_path.name
        if report_filename_base.endswith(".nii.gz"): report_filename = report_filename_base[:-7] + "_quality_report.txt"
        elif report_filename_base.endswith(".nii"): report_filename = report_filename_base[:-4] + "_quality_report.txt"
        else: report_filename = report_filename_base + "_quality_report.txt"
        output_report_path = output_report_dir / report_filename
        logger.debug(f"Сохранение отчета в {output_report_path}")
        with open(output_report_path, 'w', encoding='utf-8') as f: f.write(report_text)
        logger.info(f"Отчет сохранен: {output_report_path.relative_to(output_dir)} (Оценка: {quality_level})")
        success = True

    except (FileNotFoundError, nib.filebasedimages.ImageFileError) as e: logger.error(f"Ошибка загрузки NIfTI {relative_path}: {e}", exc_info=False)
    except MemoryError: logger.error(f"Ошибка памяти при обработке {relative_path}.", exc_info=False)
    except ValueError as e: logger.error(f"Ошибка значения при обработке {relative_path}: {e}", exc_info=True)
    except Exception as e: logger.exception(f"Непредвиденная ошибка при обработке {relative_path}: {e}")

    if not success:
        try:
            error_type_name = type(e).__name__ if 'e' in locals() and isinstance(e, Exception) else "UnknownError"
            error_msg = str(e) if 'e' in locals() and isinstance(e, Exception) else "Неизвестная ошибка"
            error_report_text, _ = generate_report({'error': f"{error_type_name}: {error_msg}"}, relative_path, voxel_info)
            output_report_dir = output_dir / relative_path.parent
            output_report_dir.mkdir(parents=True, exist_ok=True)
            report_filename_base = relative_path.name
            if report_filename_base.endswith(".nii.gz"): error_filename = report_filename_base[:-7] + "_quality_report_ERROR.txt"
            elif report_filename_base.endswith(".nii"): error_filename = report_filename_base[:-4] + "_quality_report_ERROR.txt"
            else: error_filename = report_filename_base + "_quality_report_ERROR.txt"
            output_error_report_path = output_report_dir / error_filename
            with open(output_error_report_path, 'w', encoding='utf-8') as f: f.write(error_report_text)
            logger.info(f"Отчет об ошибке сохранен: {output_error_report_path.relative_to(output_dir)}")
        except Exception as report_err: logger.error(f"Не удалось сохранить отчет об ошибке для {relative_path}: {report_err}")

    return success


def run_quality_check_pipeline(input_dir_str: str, output_dir_str: str, anisotropy_thresh_val: float):
    """Основная функция для запуска конвейера оценки качества."""
    global ANISOTROPY_THRESHOLD; ANISOTROPY_THRESHOLD = anisotropy_thresh_val
    input_path = Path(input_dir_str); output_path = Path(output_dir_str)
    if not input_path.is_dir(): logger.error(f"Входная директория не найдена: {input_path}"); raise FileNotFoundError(f"Входная директория не найдена: {input_path}")
    try: output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger.error(f"Не удалось создать выходную директорию {output_path}: {e}"); raise

    logger.info(f"Начало обработки: {input_path.resolve()}"); logger.info(f"Выход: {output_path.resolve()}"); logger.info(f"Порог анизотропии: {ANISOTROPY_THRESHOLD}")
    nifti_files_found = 0; reports_generated = 0; errors_encountered = 0

    for nifti_file in input_path.rglob('*.nii*'):
        if nifti_file.is_file() and (nifti_file.name.endswith('.nii') or nifti_file.name.endswith('.nii.gz')):
             nifti_files_found += 1
             if process_nifti_file(nifti_file, output_path, input_path): reports_generated += 1
             else: errors_encountered += 1
        else: logger.debug(f"Пропуск: {nifti_file}")

    logger.info("-" * 50); logger.info("Обработка завершена.")
    if nifti_files_found == 0: logger.warning("NIfTI файлы не найдены.")
    else: logger.info(f"Найдено NIfTI: {nifti_files_found}"); logger.info(f"Отчетов сгенерировано: {reports_generated}"); logger.info(f"Ошибок обработки файлов: {errors_encountered}")
    logger.info(f"Отчеты в: {output_path.resolve()}"); logger.info(f"Лог там же или указан."); logger.info("-" * 50)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Рассчитывает базовые метрики качества МРТ, генерирует отчеты и лог.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Входная директория NIfTI.")
    parser.add_argument("--output_dir", required=True, help="Выходная директория отчетов/логов.")
    parser.add_argument("--anisotropy_thresh", type=float, default=ANISOTROPY_THRESHOLD, help="Порог анизотропии.")
    parser.add_argument("--log_file", default=None, help="Путь к лог-файлу.")
    args = parser.parse_args()

    # --- Настройка логирования ---
    log_file_path = args.log_file
    if log_file_path is None:
        log_filename = 'quality_metrics.log'
        try:
            if args.output_dir and not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
            log_file_path = os.path.join(args.output_dir or '.', log_filename)
        except OSError as e: log_file_path = log_filename; print(f"Предупреждение: Не удалось создать/исп. {args.output_dir} для лога...")
    else: log_file_path = args.log_file
    setup_logging(log_file_path)

    # --- Основной блок ---
    try:
        logger.info("="*50); logger.info(f"Запуск quality_metrics_without_skull-strip.py")
        logger.info(f"  Input: {os.path.abspath(args.input_dir)}"); logger.info(f"  Output: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Anisotropy Thresh: {args.anisotropy_thresh}"); logger.info(f"  Log: {os.path.abspath(log_file_path)}"); logger.info("="*50)

        success = run_quality_check_pipeline(args.input_dir, args.output_dir, args.anisotropy_thresh)

        if success: logger.info("Скрипт успешно завершил работу."); sys.exit(0)
        else: logger.error("Скрипт завершился с ошибкой."); sys.exit(1) # Не должно достигаться

    except FileNotFoundError as e: logger.error(f"Критическая ошибка: Файл/директория не найдены. {e}"); sys.exit(1)
    except OSError as e: logger.error(f"Критическая ошибка ФС: {e}", exc_info=True); sys.exit(1)
    except Exception as e: logger.exception(f"Непредвиденная критическая ошибка: {e}"); sys.exit(1)