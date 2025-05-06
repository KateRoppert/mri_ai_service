import os
import nibabel as nib
import numpy as np
from scipy.stats import entropy
from pathlib import Path
import argparse
from datetime import datetime
import warnings
import logging
import logging.handlers

# --- Константы и Параметры ---
CORNER_SIZE = 10
FOREGROUND_THRESHOLD_FACTOR = 2.5
ANISOTROPY_THRESHOLD = 3.0

# Эвристические пороги для интерпретации (ТРЕБУЮТ НАСТРОЙКИ!)
THRESHOLDS = {
    'fber': {'poor': 10, 'acceptable': 25},
    'efc': {'poor': 0.42, 'acceptable': 0.55},
    'noise_std': {'good': 10, 'acceptable': 25},
    'snr': {'poor': 8, 'acceptable': 20}
}

# --- Функции расчета метрик (без изменений) ---
def get_background_stats(data, corner_size=CORNER_SIZE):
    """Оценивает фон, анализируя значения в углах изображения."""
    # ... (код функции без изменений) ...
    corners = []
    dims = data.shape
    if len(dims) != 3:
        logging.warning(f"Ожидались 3D данные, получено {len(dims)}D. Оценка фона может быть неточной.")
        if len(dims) < 3:
             return None, None

    cs = [min(corner_size, d // 2) for d in dims]
    if any(c == 0 for c in cs):
         logging.warning("Размер изображения слишком мал для взятия углов. Пропуск оценки фона.")
         return None, None

    try:
        corners.append(data[ :cs[0],  :cs[1],  :cs[2]].flatten())
        corners.append(data[-cs[0]:,  :cs[1],  :cs[2]].flatten())
        corners.append(data[ :cs[0], -cs[1]:,  :cs[2]].flatten())
        corners.append(data[ :cs[0],  :cs[1], -cs[2]:].flatten())
        corners.append(data[-cs[0]:, -cs[1]:,  :cs[2]].flatten())
        corners.append(data[-cs[0]:,  :cs[1], -cs[2]:].flatten())
        corners.append(data[ :cs[0], -cs[1]:, -cs[2]:].flatten())
        corners.append(data[-cs[0]:, -cs[1]:, -cs[2]:].flatten())

        background_voxels = np.concatenate(corners)
        background_voxels = background_voxels[background_voxels != 0]

        if background_voxels.size > 50:
            mean_bg = np.mean(background_voxels)
            std_bg = np.std(background_voxels)
            if std_bg < 1e-6:
                 std_bg = 1e-6
                 logging.warning("Стандартное отклонение фона близко к нулю.")
            if std_bg > 0.2 * np.std(data):
                 logging.warning(f"Стандартное отклонение фона ({std_bg:.2f}) необычно высокое...") # Сокращено для примера

            return mean_bg, std_bg
        else:
            logging.warning(f"Не удалось извлечь достаточное количество ненулевых вокселей фона...") # Сокращено
            return None, None
    except IndexError:
        logging.error(f"Ошибка индексации при извлечении углов...", exc_info=False) # Убрал traceback для краткости
        return None, None
    except Exception as e:
        logging.error(f"Неизвестная ошибка при оценке фона: {e}", exc_info=False) # Убрал traceback
        return None, None


def calculate_iqms(data):
    """Рассчитывает метрики качества для данных изображения."""
    # ... (код функции без изменений) ...
    if data.ndim == 4:
        if data.shape[3] > 1:
            logging.info("Обнаружены 4D данные, используется первый том.")
        data = data[..., 0]
    elif data.ndim != 3:
        return {'error': f'Неподдерживаемая размерность {data.ndim}D'}

    if np.all(data == 0): return {'error': 'Нулевое изображение'}
    if np.max(data) <= np.min(data): return {'error': 'Константное изображение'}

    metrics = {}
    bg_mean, bg_std = get_background_stats(data)
    if bg_std is None or bg_std < 1e-6:
        metrics['noise_std'] = None
        bg_mean_fallback = np.percentile(data, 1)
        bg_std_fallback = np.std(data[data < np.percentile(data, 5)])
        threshold = bg_mean_fallback + FOREGROUND_THRESHOLD_FACTOR * bg_std_fallback if bg_std_fallback > 1e-6 else np.percentile(data, 10)
    else:
        metrics['noise_std'] = float(bg_std)
        threshold = bg_mean + FOREGROUND_THRESHOLD_FACTOR * bg_std

    foreground_mask = data > threshold
    foreground_voxels = data[foreground_mask]

    if foreground_voxels.size < 100:
        metrics.update({ # Сокращенная запись
            'foreground_voxels_count': int(foreground_voxels.size),
            'foreground_mean': float(np.mean(foreground_voxels)) if foreground_voxels.size > 0 else 0.0,
            'foreground_median': float(np.median(foreground_voxels)) if foreground_voxels.size > 0 else 0.0,
            'foreground_std': float(np.std(foreground_voxels)) if foreground_voxels.size > 1 else 0.0,
            'fber': None, 'efc': None, 'snr': None, 'warning': 'Мало вокселей ПП'
        })
        return metrics

    metrics['foreground_voxels_count'] = int(foreground_voxels.size)
    fg_mean = float(np.mean(foreground_voxels))
    fg_median = float(np.median(foreground_voxels))
    fg_std = float(np.std(foreground_voxels))
    metrics['foreground_mean'] = fg_mean
    metrics['foreground_median'] = fg_median
    metrics['foreground_std'] = fg_std

    if fg_std < 1e-6: metrics['warning'] = metrics.get('warning', '') + ' Низкая вариация ПП.'

    # FBER
    metrics['fber'] = float(fg_median / metrics['noise_std']) if metrics.get('noise_std') and fg_median else None
    # EFC
    max_intensity_fg = np.max(foreground_voxels) if foreground_voxels.size > 0 else 0
    if max_intensity_fg > threshold and foreground_voxels.size > 1:
        fg_voxels_gt_zero = foreground_voxels[foreground_voxels > 1e-9]
        if fg_voxels_gt_zero.size > 1:
            normalized_intensities_mriqc = foreground_voxels / max_intensity_fg
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 entropy_val = -np.sum((normalized_intensities_mriqc + 1e-9) * np.log(normalized_intensities_mriqc + 1e-9))
            max_entropy = np.log(foreground_voxels.size)
            metrics['efc'] = float(entropy_val / max_entropy) if max_entropy > 0 else 0.0
        else: metrics['efc'] = 0.0
    else: metrics['efc'] = 0.0
    # SNR
    metrics['snr'] = float(fg_mean / metrics['noise_std']) if metrics.get('noise_std') and fg_mean else None

    return metrics

# --- Функция проверки анизотропии (без изменений) ---
def check_voxel_anisotropy(nifti_image):
    """Проверяет анизотропию вокселей по заголовку NIfTI."""
    # ... (код функции без изменений) ...
    try:
        zooms = nifti_image.header.get_zooms()
        if len(zooms) < 3:
             logging.warning("Не удалось получить размеры для 3 пространственных измерений.")
             return None, None, False

        voxel_sizes = zooms[:3]
        if not all(v > 1e-6 for v in voxel_sizes):
             logging.warning(f"Некорректные размеры вокселей: {voxel_sizes}.")
             return tuple(voxel_sizes), None, False

        max_res = np.max(voxel_sizes)
        min_res = np.min(voxel_sizes)
        anisotropy_ratio = max_res / min_res
        is_anisotropic = anisotropy_ratio > ANISOTROPY_THRESHOLD

        if is_anisotropic:
             logging.warning(f"Высокая анизотропия вокселей: {voxel_sizes} (ratio: {anisotropy_ratio:.2f})")

        return tuple(voxel_sizes), float(anisotropy_ratio), is_anisotropic
    except Exception as e:
        logging.error(f"Ошибка при чтении размеров вокселей: {e}", exc_info=False)
        return None, None, False

# --- Функция интерпретации метрики (без изменений) ---
def interpret_metric(name, value, thresholds):
    """Дает текстовую интерпретацию значения метрики и оценку."""
    # ... (код функции без изменений) ...
    if value is None: return "N/A", "na"
    if name in thresholds:
        limits = thresholds[name]
        if name in ['fber', 'efc', 'snr']: # Больше = лучше
            if value < limits['poor']: level, interp = "poor", f"{value:.3f} (Низкое)"
            elif value < limits['acceptable']: level, interp = "acceptable", f"{value:.3f} (Приемлемо)"
            else: level, interp = "good", f"{value:.3f} (Хорошо)"
        elif name in ['noise_std']: # Меньше = лучше
             if value < limits['good']: level, interp = "good", f"{value:.3f} (Низкий)"
             elif value < limits['acceptable']: level, interp = "acceptable", f"{value:.3f} (Умеренный)"
             else: level, interp = "poor", f"{value:.3f} (Высокий)"
        else: level, interp = "info", f"{value:.3f}" if isinstance(value, float) else f"{value}"
    else:
        level = "info"
        if isinstance(value, (int, float)): interp = f"{value:.3f}" if isinstance(value, float) else f"{value}"
        elif isinstance(value, (tuple, list)): interp = ", ".join([f"{v:.2f}" for v in value])
        else: interp = str(value)
    return interp, level

# --- Функция генерации отчета (ОБНОВЛЕНА) ---

def generate_report(metrics, input_filepath, voxel_info):
    """Создает текстовый отчет с вердиктом в начале."""
    voxel_sizes, anisotropy_ratio, is_anisotropic = voxel_info

    report_lines = []
    # --- Шапка отчета ---
    report_lines.append("=" * 60)
    report_lines.append(f"Отчет о качестве МРТ изображения")
    report_lines.append(f"Файл: {input_filepath}")
    report_lines.append(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Предварительный расчет вердикта и проблем ---
    overall_quality = "Не определено" # По умолчанию
    possible_issues = []
    quality_levels = {}
    interpretations = {}
    metric_definitions = { # Определим здесь для использования ниже
        'noise_std': "Шум фона (Ст.Откл. в углах)",
        'snr': "Сигнал/Шум (Средний сигнал ПП / Шум фона)",
        'fber': "Контраст ПереднийПлан/Фон (Медиана ПП / Шум фона)",
        'efc': "Концентрация сигнала (Entropy Focus Criterion)",
        'foreground_voxels_count': "Кол-во вокселей переднего плана (оценка)",
        'foreground_mean': "Средняя интенсивность ПП",
        'foreground_median': "Медианная интенсивность ПП",
        'foreground_std': "Ст.Откл. интенсивности ПП"
    }

    if metrics is None or 'error' in metrics:
        overall_quality = "Ошибка"
        error_msg = metrics.get('error', 'Неизвестная ошибка.') if isinstance(metrics, dict) else 'Неизвестная ошибка.'
        possible_issues = [f"Ошибка расчета метрик интенсивности: {error_msg}"]
    else:
        # Рассчитываем уровни качества метрик интенсивности
        for name, description in metric_definitions.items():
            if name in metrics:
                value = metrics[name]
                interpretation, level = interpret_metric(name, value, THRESHOLDS)
                quality_levels[name] = level
                interpretations[name] = interpretation # Сохраняем для раздела деталей

        num_poor = list(quality_levels.values()).count('poor')
        num_acceptable = list(quality_levels.values()).count('acceptable')
        num_good = list(quality_levels.values()).count('good')
        num_na = list(quality_levels.values()).count('na')

        # Определяем возможные проблемы
        if is_anisotropic:
             possible_issues.append(f"Высокая анизотропия вокселей (отношение > {ANISOTROPY_THRESHOLD:.1f}). Риск 'размазанного' изображения.")
        if 'warning' in metrics: # Добавляем предупреждения из calculate_iqms
             possible_issues.append(f"Предупреждение при расчете: {metrics['warning']}")
        if quality_levels.get('noise_std') == 'poor': possible_issues.append("Высокий уровень шума.")
        if quality_levels.get('snr') == 'poor': possible_issues.append("Низкое отношение Сигнал/Шум.")
        if quality_levels.get('fber') == 'poor': possible_issues.append("Низкий контраст ПереднийПлан/Фон.")
        if quality_levels.get('efc') == 'poor': possible_issues.append("Низкая концентрация сигнала (возможны артефакты движения/размытие/неоднородность).")

        # Определяем оценку качества по интенсивности
        overall_quality_intensity = "Не оценено (интенс.)"
        if num_poor > 0: overall_quality_intensity = "Низкое (интенс.)"
        elif num_acceptable > 0 or num_na > (len(THRESHOLDS) - num_poor - num_good) : overall_quality_intensity = "Приемлемое (интенс.)"
        elif num_good > 0: overall_quality_intensity = "Хорошее (интенс.)"

        # Итоговая оценка с учетом анизотропии
        if is_anisotropic:
             if "Хорошее" in overall_quality_intensity: overall_quality = "Приемлемое / Низкое разрешение по одной оси"
             elif "Низкое" in overall_quality_intensity: overall_quality = "Низкое / Проблемы с разрешением и интенсивностью"
             else: overall_quality = "Приемлемое / Низкое разрешение по одной оси" # Если приемлемо или N/A по интенсивности
        else: # Если анизотропии нет
             if "Хорошее" in overall_quality_intensity: overall_quality = "Хорошее качество"
             elif "Приемлемое" in overall_quality_intensity: overall_quality = "Приемлемое качество"
             elif "Низкое" in overall_quality_intensity: overall_quality = "Низкое качество / Требует проверки"
             else: overall_quality = "Требуется ручная оценка" # Если не было хороших/приемлемых/плохих

    # --- Добавляем ВЕРДИКТ в начало отчета ---
    report_lines.append("=" * 60)
    report_lines.append(f"*** ИТОГОВЫЙ ВЕРДИКТ: {overall_quality} ***")
    report_lines.append("=" * 60)

    # --- Теперь добавляем остальные детали ---
    report_lines.append("\n" + "-" * 60)
    report_lines.append("Детализация отчета:")
    report_lines.append("-" * 60)

    # Раздел Геометрии
    report_lines.append("\n-- Геометрия --")
    if voxel_sizes:
        report_lines.append(f"- Размеры вокселя (X, Y, Z), мм: {', '.join([f'{v:.2f}' for v in voxel_sizes])}")
        if anisotropy_ratio is not None:
             report_lines.append(f"- Отношение анизотропии (макс/мин): {anisotropy_ratio:.2f}")
             if is_anisotropic:
                  report_lines.append(f"  - ПРЕДУПРЕЖДЕНИЕ: Высокая анизотропия (>{ANISOTROPY_THRESHOLD}).")
    else:
        report_lines.append("- Размеры вокселя: Не удалось прочитать из заголовка.")

    # Раздел Метрик Интенсивности (только если не было ошибки)
    if metrics is not None and 'error' not in metrics:
        report_lines.append("\n-- Метрики Интенсивности --")
        for name, description in metric_definitions.items():
             interpretation_text = interpretations.get(name, None)
             if interpretation_text is not None:
                 report_lines.append(f"- {description:<35}: {interpretation_text}")
             # Опционально: показать N/A для метрик, которые должны были быть, но не рассчитались
             elif name in THRESHOLDS or name == 'noise_std': # Показать N/A для основных расчетных метрик
                  report_lines.append(f"- {description:<35}: N/A (не рассчитано)")


    # Раздел Возможных Проблем
    report_lines.append("\n-- Сводка возможных проблем --")
    if possible_issues:
        for issue in possible_issues:
            report_lines.append(f"- {issue}")
    elif overall_quality not in ["Ошибка", "Требуется ручная оценка"]:
         report_lines.append("- Серьезных проблем по рассчитанным метрикам и геометрии не выявлено.")
    else:
         report_lines.append("- См. причину ошибки или проведите ручную оценку.")


    # Важные примечания
    report_lines.append("\n" + "=" * 60)
    report_lines.append("ВАЖНО:")
    report_lines.append("- Данные метрики рассчитаны БЕЗ точного выделения мозга.")
    report_lines.append("- Пороги для интерпретации являются ОРИЕНТИРОВОЧНЫМИ и ЗАВИСЯТ от протокола.")
    report_lines.append("- Проверка анизотропии помогает выявить проблемы с разрешением по осям.")
    report_lines.append("- Этот отчет НЕ ЗАМЕНЯЕТ визуальный контроль качества специалистом.")
    report_lines.append("=" * 60)

    return "\n".join(report_lines), overall_quality # Возвращаем вердикт для логирования

# --- Настройка логирования (без изменений) ---
def setup_logging(log_dir):
    """Настраивает логирование в консоль и файл."""
    # ... (код функции без изменений) ...
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"quality_check_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Очищаем предыдущие обработчики
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Логирование настроено. Лог файл: {log_filename}")

# --- Основная функция обработки (без изменений) ---
def process_bids_directory(input_dir, output_dir):
    """Обходит директорию, рассчитывает метрики, сохраняет отчеты и лог."""
    # ... (код функции без изменений) ...
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"ОШИБКА: Входная директория не найдена: {input_dir}")
        return

    try: setup_logging(output_path)
    except Exception as e:
         print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить логирование в {output_path}: {e}")
         return

    logging.info(f"Начало обработки директории: {input_path}")
    logging.info(f"Выходная директория для отчетов и логов: {output_path}")
    logging.info(f"Порог анизотропии для предупреждения: {ANISOTROPY_THRESHOLD}")

    nifti_files_found = 0
    reports_generated = 0
    errors_encountered = 0

    for nifti_file in input_path.rglob('*.nii*'):
        if nifti_file.is_file() and nifti_file.suffix in ['.nii', '.gz']:
            if nifti_file.suffix == '.gz' and nifti_file.with_suffix('').suffix != '.nii':
                 logging.debug(f"Пропуск файла: {nifti_file}")
                 continue

            nifti_files_found += 1
            logging.info(f"Обработка файла: {nifti_file.relative_to(input_path)}")

            metrics = None; voxel_info = (None, None, False); quality_level = "Ошибка"

            try:
                img = nib.load(nifti_file)
                voxel_info = check_voxel_anisotropy(img)
                data = img.get_fdata(dtype=np.float32)
                metrics = calculate_iqms(data)
                report_text, quality_level = generate_report(metrics, nifti_file, voxel_info)

                # Сохранение отчета
                relative_path = nifti_file.relative_to(input_path)
                output_report_dir = output_path / relative_path.parent
                output_report_dir.mkdir(parents=True, exist_ok=True)
                report_filename_base = relative_path.name
                if report_filename_base.endswith(".nii.gz"): report_filename = report_filename_base[:-7] + "_quality_report.txt"
                elif report_filename_base.endswith(".nii"): report_filename = report_filename_base[:-4] + "_quality_report.txt"
                else: report_filename = report_filename_base + "_quality_report.txt"
                output_report_path = output_report_dir / report_filename

                with open(output_report_path, 'w', encoding='utf-8') as f: f.write(report_text)
                logging.info(f"Отчет сохранен: {output_report_path.relative_to(output_path)} (Оценка: {quality_level})")
                reports_generated += 1

            except (FileNotFoundError, nib.filebasedimages.ImageFileError, MemoryError, ValueError) as e:
                 logging.error(f"Ошибка ({type(e).__name__}) при обработке файла {nifti_file}: {e}", exc_info=False)
                 errors_encountered += 1
                 # Попытка создать отчет об ошибке
                 try:
                     error_report, _ = generate_report({'error': f'Ошибка {type(e).__name__}: {e}'}, nifti_file, voxel_info)
                     # ... (код сохранения отчета об ошибке, аналогично успешному) ...
                     # Важно: имя файла может быть таким же, он просто будет содержать сообщение об ошибке
                     relative_path = nifti_file.relative_to(input_path)
                     output_report_dir = output_path / relative_path.parent
                     output_report_dir.mkdir(parents=True, exist_ok=True)
                     report_filename_base = relative_path.name
                     if report_filename_base.endswith(".nii.gz"): report_filename = report_filename_base[:-7] + "_quality_report_ERROR.txt" # Добавим ERROR
                     elif report_filename_base.endswith(".nii"): report_filename = report_filename_base[:-4] + "_quality_report_ERROR.txt"
                     else: report_filename = report_filename_base + "_quality_report_ERROR.txt"
                     output_report_path = output_report_dir / report_filename
                     with open(output_report_path, 'w', encoding='utf-8') as f: f.write(error_report)
                     logging.info(f"Отчет об ошибке сохранен: {output_report_path.relative_to(output_path)}")
                 except Exception as report_err:
                      logging.error(f"Не удалось даже создать отчет об ошибке для {nifti_file}: {report_err}")
            except Exception as e:
                logging.error(f"Непредвиденная ошибка при обработке файла {nifti_file}: {e}", exc_info=True) # Оставляем traceback для неизвестных ошибок
                errors_encountered += 1
                 # Аналогичная попытка создать отчет об ошибке

    logging.info("-" * 30)
    logging.info("Обработка завершена.")
    logging.info(f"Найдено NIfTI файлов: {nifti_files_found}")
    logging.info(f"Сгенерировано отчетов: {reports_generated}")
    logging.info(f"Встречено ошибок: {errors_encountered}")
    logging.info(f"Отчеты сохранены в директорию: {output_path.resolve()}")
    logging.info(f"Полный лог сохранен в файл в той же директории.")
    logging.info("-" * 30)

# --- Точка входа (без изменений) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Рассчитывает метрики качества МРТ (включая анизотропию), генерирует отчеты с вердиктом в начале и лог.")
    parser.add_argument("input_dir", help="Путь к входной директории с NIfTI файлами.")
    parser.add_argument("output_dir", help="Путь к выходной директории для сохранения отчетов и лог-файла.")
    parser.add_argument("--anisotropy_thresh", type=float, default=ANISOTROPY_THRESHOLD,
                        help=f"Порог отношения макс/мин размера вокселя для флага высокой анизотропии (по умолчанию: {ANISOTROPY_THRESHOLD})")

    args = parser.parse_args()
    ANISOTROPY_THRESHOLD = args.anisotropy_thresh
    process_bids_directory(args.input_dir, args.output_dir)