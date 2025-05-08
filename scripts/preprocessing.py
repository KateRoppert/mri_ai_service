import os
import sys 
import glob
import shutil
import json
import logging
import tempfile
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np 

import yaml  # Для чтения конфига
import SimpleITK as sitk
from nipype.interfaces.fsl import BET, FSLCommand
from nipype.interfaces.base import Undefined # Для проверки вывода Nipype
import ants # Для ANTsPy

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s')

# Глобальные переменные
fsl_present = None
ANISOTROPY_THRESHOLD = 3.0 # Значение по умолчанию

def setup_main_logging(log_file_path: str, console_level: str = "INFO"):
    """
    Настраивает основной логгер скрипта: вывод в консоль и в указанный файл.
    """
    if logger.hasHandlers():
        logger.handlers.clear()

    # Консольный обработчик
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        ch.setLevel(logging.INFO)
        print(f"Предупреждение: Неверный уровень логирования '{console_level}'. Используется INFO.")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файловый обработчик
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # Дозапись в лог
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Основное логирование в файл настроено: {log_file_path}")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить основной файловый лог {log_file_path}: {e}")
        # Не выходим, т.к. логгер консоли может работать

def check_fsl_availability():
    """
    Проверяет доступность FSL (команды 'bet') и устанавливает глобальную переменную fsl_present.
    """
    global fsl_present
    if fsl_present is not None: # Проверяем только один раз
        return fsl_present

    try:
        FSLCommand.check_fsl() # Пытаемся через Nipype
        logger.info("Проверка FSL: Найден через Nipype.")
        fsl_present = True
    except Exception as e:
        logger.warning(f"Проверка FSL: Nipype не нашел FSL: {e}")
        logger.info("Проверка FSL: Попытка найти 'bet' в системном PATH...")
        try:
            # Пытаемся через subprocess 'which'
            result = subprocess.run(['which', 'bet'], check=True, capture_output=True, text=True)
            logger.info(f"Проверка FSL: 'bet' найден в PATH: {result.stdout.strip()}")
            fsl_present = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "Проверка FSL: Команда 'bet' НЕ найдена в PATH. "
                "Шаг удаления черепа (skull stripping) не будет выполнен."
            )
            fsl_present = False
    return fsl_present

def create_output_paths(
    input_nifti_path_str: str,
    input_root_str: str,
    output_root_prep_str: str,
    output_root_tfm_str: str
) -> tuple[Path, Path]:
    """
    Создает и возвращает пути для конечного предобработанного файла
    и директории для хранения трансформаций, сохраняя относительную структуру.

    Args:
        input_nifti_path_str: Путь к исходному NIfTI файлу.
        input_root_str: Корневая директория входных данных.
        output_root_prep_str: Корневая директория для предобработанных файлов.
        output_root_tfm_str: Корневая директория для трансформаций.

    Returns:
        tuple[Path, Path]: (Путь к финальному файлу, Путь к директории трансформаций).

    Raises:
        OSError: Если не удалось создать выходные директории.
    """
    input_path = Path(input_nifti_path_str)
    input_root = Path(input_root_str)
    output_root_prep = Path(output_root_prep_str)
    output_root_tfm = Path(output_root_tfm_str)

    try:
        # Определяем относительный путь файла внутри входной директории
        relative_path = input_path.relative_to(input_root)
    except ValueError:
        # Если файл не находится внутри input_root, используем только имя файла
        logger.warning(
            f"Не удалось определить относительный путь для {input_path} от {input_root}. "
            f"Результаты будут сохранены в корне выходных директорий."
        )
        relative_path = Path(input_path.name)

    # Формируем путь к финальному предобработанному файлу
    final_prep_path = output_root_prep / relative_path
    # Формируем путь к директории трансформаций для этого файла
    transform_dir_name = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    transform_dir = output_root_tfm / relative_path.parent / transform_dir_name

    # Создаем необходимые директории
    try:
        final_prep_path.parent.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Ошибка создания выходных директорий для {input_path.name}: {e}")
        raise # Передаем ошибку выше

    logger.debug(f"Путь к предобработанному файлу: {final_prep_path}")
    logger.debug(f"Директория для трансформаций: {transform_dir}")
    return final_prep_path, transform_dir


def original_bias_field_correction(
    input_img_path_str: str,
    out_path_str: str,
    params: dict
) -> sitk.Image:
    """
    Выполняет N4 коррекцию поля смещения с использованием SimpleITK и параметров из конфига.

    Args:
        input_img_path_str: Путь к входному изображению.
        out_path_str: Путь для сохранения скорректированного изображения.
        params (dict): Словарь с параметрами для N4 (например, 'sitk_shrinkFactor').

    Returns:
        sitk.Image: Логарифм поля смещения.
    """
    step_name = "N4 Bias Field Correction"
    logger.info(f"  {step_name}: {Path(input_img_path_str).name} -> {Path(out_path_str).name}")

    # Чтение входного изображения
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)

    # Создание маски головы по порогу Li
    head_mask = sitk.LiThreshold(raw_img_sitk, 0, 1)

    # Получение параметров из конфига или использование значений по умолчанию
    shrinkFactor = params.get('sitk_shrinkFactor', 4)
    # По умолчанию 50 итераций на каждом уровне масштабирования
    n_iterations = params.get('sitk_numberOfIterations', [50] * shrinkFactor)
    conv_thresh = params.get('sitk_convergenceThreshold', 0.0)

    # Валидация и подгонка длины списка итераций
    if len(n_iterations) != shrinkFactor:
        logger.warning(
            f"    {step_name}: Длина sitk_numberOfIterations ({len(n_iterations)}) "
            f"не совпадает с shrinkFactor ({shrinkFactor}). Используется адаптированный список."
        )
        # Просто повторяем или обрезаем список до нужной длины
        n_iterations = (n_iterations * shrinkFactor)[:shrinkFactor]

    logger.debug(f"    {step_name} параметры: shrinkFactor={shrinkFactor}, iterations={n_iterations}, convergence={conv_thresh}")

    # Уменьшение изображения и маски
    inputImage_shrink = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage_shrink = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())

    # Настройка и выполнение N4 корректора
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrector.SetMaximumNumberOfIterations(n_iterations)
    if conv_thresh > 0: # Устанавливаем порог сходимости, только если он задан
        bias_corrector.SetConvergenceThreshold(conv_thresh)

    logger.debug("    Выполнение N4 коррекции на уменьшенном изображении...")
    _ = bias_corrector.Execute(inputImage_shrink, maskImage_shrink) # Результат не используется напрямую

    logger.debug("    Получение логарифма поля смещения для полного разрешения...")
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)

    logger.debug("    Применение коррекции к изображению полного разрешения...")
    # Добавляем малое число для стабильности деления
    corrected_image_full_resolution = raw_img_sitk / (sitk.Exp(log_bias_field) + 1e-7)

    # Сохранение результата
    sitk.WriteImage(corrected_image_full_resolution, out_path_str)
    logger.debug("    N4 коррекция успешно завершена.")

    return log_bias_field


def original_intensity_normalization(
    input_img_path_str: str,
    out_path_str: str,
    template_img_path_str: str,
    params: dict # Параметры из конфига (в этой реализации не используются)
):
    """
    Выполняет нормализацию интенсивности методом сопоставления гистограмм с шаблоном.

    Args:
        input_img_path_str: Путь к входному изображению.
        out_path_str: Путь для сохранения нормализованного изображения.
        template_img_path_str: Путь к файлу шаблона.
        params (dict): Словарь с параметрами шага (для логирования).
    """
    step_name = "Intensity Normalization (Histogram Matching)"
    logger.info(f"  {step_name}: {Path(input_img_path_str).name} -> {Path(out_path_str).name}")
    logger.debug(f"    Используемый шаблон: {template_img_path_str}")
    logger.debug(f"    Параметры из конфига для этого шага: {params}") # Логируем полученные параметры

    # Чтение изображений
    template_img_sitk = sitk.ReadImage(template_img_path_str, sitk.sitkFloat32)
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)

    # Выполнение сопоставления гистограмм
    logger.debug("    Выполнение sitk.HistogramMatching...")
    transformed = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)

    # Сохранение результата
    sitk.WriteImage(transformed, out_path_str)
    logger.debug("    Нормализация интенсивности завершена.")


def original_registration(
    input_img_path_str: str,
    out_path_str: str,
    template_img_path_str: str,
    transforms_prefix_str: str,
    params: dict # Параметры регистрации из конфига
) -> tuple[list[str], list[str]]:
    """
    Выполняет регистрацию изображения на шаблон с использованием ANTsPy.

    Args:
        input_img_path_str: Путь к движущемуся изображению (которое регистрируем).
        out_path_str: Путь для сохранения зарегистрированного изображения.
        template_img_path_str: Путь к фиксированному изображению (шаблону).
        transforms_prefix_str: Префикс для сохранения файлов трансформаций ANTs.
        params (dict): Словарь с параметрами для ants.registration (например, 'ants_transform_type').

    Returns:
        tuple[list[str], list[str]]: Списки путей к файлам прямых и обратных трансформаций.
    """
    step_name = "ANTs Registration"
    logger.info(f"  {step_name}: {Path(input_img_path_str).name} -> {Path(out_path_str).name}")

    # Получаем параметры из конфига или используем значения по умолчанию
    transform_type = params.get('ants_transform_type', 'SyN')
    # Можно добавить чтение других параметров ANTs из `params`, если они заданы в YAML
    # ants_metric = params.get('ants_metric', 'MI') # Пример

    logger.debug(f"    Тип трансформации: {transform_type}")
    # logger.debug(f"    Метрика: {ants_metric}") # Пример
    logger.debug(f"    Префикс выходных файлов трансформации: {transforms_prefix_str}")

    # Чтение изображений с помощью ANTsPy
    template_img_ants = ants.image_read(template_img_path_str)
    raw_img_ants = ants.image_read(input_img_path_str)

    logger.debug(f"    Выполнение ants.registration (тип: {transform_type}). Это может занять время...")
    # Выполнение регистрации
    transformation = ants.registration(
        fixed=template_img_ants,
        moving=raw_img_ants,
        type_of_transform=transform_type,
        outprefix=transforms_prefix_str, # ANTs добавит сюда суффиксы для файлов
        verbose=False  # Отключаем подробный вывод ANTs в консоль
        # Здесь можно передать другие параметры через **kwargs, если они извлечены из params
        # syn_metric=ants_metric, ...
    )

    # Сохранение зарегистрированного изображения
    registered_img_ants = transformation['warpedmovout']
    registered_img_ants.to_file(out_path_str)
    logger.debug("    ANTs регистрация завершена.")

    # Возвращаем списки путей к файлам трансформаций
    return transformation.get('fwdtransforms', []), transformation.get('invtransforms', [])


def original_skull_stripping(
    input_img_path_str: str,
    out_file_base_str: str,
    params: dict # Параметры BET из конфига
) -> tuple[str | None, str | None]:
    """
    Выполняет удаление черепа с помощью FSL BET, используя параметры из конфига.
    Запрашивает выход в формате NIFTI_GZ.

    Args:
        input_img_path_str: Путь к входному NIfTI файлу.
        out_file_base_str: Базовый путь и имя для выходных файлов BET (без расширения).
        params (dict): Словарь с параметрами для BET (например, 'bet_fractional_intensity_threshold').

    Returns:
        tuple[str | None, str | None]: (Путь к файлу с удаленным черепом, Путь к файлу маски).
                                      Возвращает (None, None) при ошибке.
    """
    global fsl_present
    if not fsl_present: # Проверяем доступность FSL
         logger.error("  Skull Stripping: FSL 'bet' команда не найдена. Шаг пропускается.")
         return None, None

    step_name = "Skull Stripping (FSL BET)"
    logger.info(f"  {step_name}: {Path(input_img_path_str).name} -> {out_file_base_str}.nii.gz")

    # Получаем параметры BET из конфига или используем значения по умолчанию
    frac_thresh = params.get('bet_fractional_intensity_threshold', 0.5)
    robust = params.get('bet_robust', True)
    additional_opts = params.get('bet_options', "") # Дополнительные опции строкой

    logger.debug(f"    Параметры BET: frac={frac_thresh}, robust={robust}, options='{additional_opts}'")

    # Настройка интерфейса Nipype BET
    bet = BET()
    bet.inputs.in_file = input_img_path_str
    bet.inputs.out_file = out_file_base_str # BET добавит .nii.gz сам
    bet.inputs.frac = frac_thresh
    bet.inputs.mask = True      # Всегда запрашиваем маску
    bet.inputs.robust = robust
    bet.inputs.output_type = 'NIFTI_GZ' # Явно указываем желаемый формат
    if additional_opts: # Добавляем кастомные аргументы, если они есть
        bet.inputs.args = additional_opts

    try:
        logger.debug(f"    Выполняемая команда BET: {bet.cmdline}")
        result = bet.run() # Запускаем BET через Nipype

        # --- Проверка выходных файлов ---
        stripped_path_str = result.outputs.out_file
        mask_path_str = result.outputs.mask_file

        expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
        if stripped_path_str is Undefined or not Path(stripped_path_str).exists():
            if expected_stripped_path.exists():
                logger.warning(f"    Вывод BET (stripped) некорректен/отсутствует. Используется ожидаемый путь: {expected_stripped_path}")
                stripped_path_str = str(expected_stripped_path)
            else:
                logger.error(f"    BET не создал основной выходной файл: {expected_stripped_path}")
                # Логируем stderr, если он доступен в runtime
                if hasattr(result, 'runtime') and result.runtime and hasattr(result.runtime, 'stderr') and result.runtime.stderr:
                     logger.error(f"    BET stderr:\n{result.runtime.stderr.strip()}")
                return None, None # Критическая ошибка шага

        expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
        if mask_path_str is Undefined or not Path(mask_path_str).exists():
            if expected_mask_path.exists():
                logger.warning(f"    Вывод BET (mask) некорректен/отсутствует. Используется ожидаемый путь: {expected_mask_path}")
                mask_path_str = str(expected_mask_path)
            else:
                # Если маска не создалась, это предупреждение, но шаг мог удасться
                logger.warning(f"    BET не создал файл маски по ожидаемому пути: {expected_mask_path}")
                mask_path_str = None # Явно указываем, что маска не найдена

        logger.debug("    Удаление черепа (BET) успешно завершено.")
        return stripped_path_str, mask_path_str

    except Exception as e:
         # Ловим ошибки выполнения интерфейса Nipype
         logger.error(f"    Ошибка при выполнении Nipype BET интерфейса: {e}", exc_info=True)
         # Проверяем, не создались ли файлы несмотря на ошибку интерфейса
         expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
         expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
         if expected_stripped_path.exists():
              logger.warning(
                  "    Интерфейс Nipype BET завершился с ошибкой, но выходной файл "
                  f"(stripped) найден: {expected_stripped_path}. Попытка продолжить."
              )
              mask_found_after_error = str(expected_mask_path) if expected_mask_path.exists() else None
              return str(expected_stripped_path), mask_found_after_error
         else:
              logger.error("    Выходные файлы BET не найдены после ошибки интерфейса Nipype.")
              return None, None


def save_parameters(params_dict: dict, output_path: Path):
    """
    Сохраняет словарь параметров в JSON файл, преобразуя Path и NumPy типы.
    """
    try:
        # Убедимся, что директория для JSON существует
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Создаем копию словаря для сериализации
        params_serializable = {}
        for key, value in params_dict.items():
             # Преобразуем Path в строки с абсолютными путями
             if isinstance(value, Path):
                 params_serializable[key] = str(value.resolve())
             # Преобразуем списки путей/строк
             elif isinstance(value, list) and value and isinstance(value[0], (Path, str)):
                 params_serializable[key] = [str(Path(p).resolve()) for p in value]
             # Преобразуем числовые типы NumPy в стандартные Python типы
             elif isinstance(value, (np.integer)):
                 params_serializable[key] = int(value)
             elif isinstance(value, (np.floating)):
                 params_serializable[key] = float(value)
             elif isinstance(value, np.ndarray): # Преобразуем массивы NumPy в списки
                 params_serializable[key] = value.tolist()
             elif isinstance(value, bool) or isinstance(value, (int, float, str, dict, list)) or value is None:
                 params_serializable[key] = value # Оставляем стандартные типы JSON
             else:
                 # Для неизвестных типов пытаемся преобразовать в строку
                 logger.warning(f"Не удалось сериализовать параметр '{key}' типа {type(value)}. Преобразование в строку.")
                 params_serializable[key] = str(value)

        # Записываем JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params_serializable, f, indent=4, ensure_ascii=False)
        logger.debug(f"Параметры шага сохранены: {output_path}")

    except Exception as e:
        logger.error(f"Не удалось сохранить параметры в {output_path}: {e}", exc_info=True)

# --- Функции-обертки для шагов (теперь принимают параметры) ---

def run_intensity_normalization_and_save(
    current_input_path: Path, step_output_path: Path, template_path: Path,
    transform_dir: Path, step_params: dict
) -> tuple[dict, bool]:
    """Обертка для нормализации интенсивности."""
    step_name = "1_intensity_normalization"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    # Записываем параметры, которые *будут* использованы (включая путь к шаблону)
    run_params = {**step_params, "template_file": str(template_path.resolve())}
    try:
        original_intensity_normalization(
            str(current_input_path), str(step_output_path), str(template_path), step_params
        )
        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        run_params["error"] = str(e)
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, False

def run_bias_field_correction_and_save(
    current_input_path: Path, step_output_path: Path,
    transform_dir: Path, step_params: dict
) -> tuple[dict, bool]:
    """Обертка для N4 коррекции."""
    step_name = "2_bias_field_correction"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    run_params = {**step_params, "output_log_bias_field_path": None, "output_bias_field_path": None}
    log_bias_field_path = transform_dir / "log_bias_field.nii.gz"
    bias_field_path = transform_dir / "bias_field.nii.gz"
    try:
        log_bias_field_sitk = original_bias_field_correction(
            str(current_input_path), str(step_output_path), step_params # Передаем параметры
        )
        # Сохраняем поля смещения
        sitk.WriteImage(log_bias_field_sitk, str(log_bias_field_path))
        run_params["output_log_bias_field_path"] = str(log_bias_field_path.resolve())
        bias_field_sitk = sitk.Exp(log_bias_field_sitk)
        sitk.WriteImage(bias_field_sitk, str(bias_field_path))
        run_params["output_bias_field_path"] = str(bias_field_path.resolve())
        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        run_params["error"] = str(e)
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, False

def run_registration_and_save(
    current_input_path: Path, step_output_path: Path, template_path: Path,
    transform_dir: Path, transform_prefix: str, step_params: dict
) -> tuple[dict, bool]:
    """Обертка для регистрации ANTsPy."""
    step_name = "3_registration"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    ants_output_prefix_str = str(transform_dir / transform_prefix)
    run_params = {**step_params, "template_file": str(template_path.resolve()),
                  "output_prefix": ants_output_prefix_str,
                  "forward_transforms_paths": [], "inverse_transforms_paths": []}
    try:
        fwd_tforms, inv_tforms = original_registration(
            str(current_input_path), str(step_output_path), str(template_path),
            ants_output_prefix_str, step_params # Передаем параметры
        )
        run_params["forward_transforms_paths"] = [str(Path(p).resolve()) for p in fwd_tforms]
        run_params["inverse_transforms_paths"] = [str(Path(p).resolve()) for p in inv_tforms]
        if not run_params["forward_transforms_paths"]:
            logger.warning(f"  Не найдены файлы прямых трансформаций ANTs (префикс: {ants_output_prefix_str})")
        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        run_params["error"] = str(e)
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, False

def run_skull_stripping_and_save(
    current_input_path: Path, final_output_file_path: Path,
    transform_dir: Path, transform_prefix: str, step_params: dict
) -> tuple[dict, bool]:
    """Обертка для удаления черепа."""
    step_name = "4_skull_stripping"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    run_params = {**step_params, "output_mask_path": None, "output_stripped_path": None}
    bet_temp_output_base_str = str(transform_dir / f"{transform_prefix}_bet_temp")
    final_mask_path = transform_dir / f"{transform_prefix}_brain_mask.nii.gz"
    try:
        # Передаем параметры в функцию
        stripped_path_str, mask_path_str = original_skull_stripping(
            str(current_input_path), bet_temp_output_base_str, step_params
        )
        if stripped_path_str is None:
            raise RuntimeError("FSL BET не создал файл с удаленным черепом.")

        generated_stripped_path = Path(stripped_path_str)
        final_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"  Перемещение {generated_stripped_path} -> {final_output_file_path}")
        shutil.move(str(generated_stripped_path), str(final_output_file_path))
        run_params["output_stripped_path"] = str(final_output_file_path.resolve())

        if mask_path_str:
            generated_mask_path = Path(mask_path_str)
            if generated_mask_path.exists():
                logger.debug(f"  Перемещение маски {generated_mask_path} -> {final_mask_path}")
                shutil.move(str(generated_mask_path), str(final_mask_path))
                run_params["output_mask_path"] = str(final_mask_path.resolve())
            else: logger.warning(f"  Файл маски {generated_mask_path} не найден после BET.")
        else: logger.warning("  BET не вернул путь к файлу маски.")

        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        return run_params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        run_params["error"] = str(e); save_parameters(run_params, transform_dir / f"{step_name}_params.json")
        # Очистка временных файлов BET
        bet_temp_stripped = Path(bet_temp_output_base_str + ".nii.gz")
        bet_temp_mask = Path(bet_temp_output_base_str + "_mask.nii.gz")
        if bet_temp_stripped.exists(): 
            try: os.remove(bet_temp_stripped); logger.debug(f"Удален временный файл: {bet_temp_stripped}") 
            except OSError: pass
        if bet_temp_mask.exists(): 
            try: os.remove(bet_temp_mask); logger.debug(f"Удален временный файл: {bet_temp_mask}") 
            except OSError: pass
        return run_params, False

# === Основная функция пайплайна (теперь читает конфиг) ===
def run_preprocessing_pipeline(
    input_root_str: str,
    output_root_prep_str: str,
    output_root_tfm_str: str,
    template_path_str: str,
    config_path_str: str
):
    """
    Запускает полный конвейер предобработки для NIfTI файлов,
    читая параметры из конфигурационного файла.
    """
    input_root = Path(input_root_str); output_root_prep = Path(output_root_prep_str)
    output_root_tfm = Path(output_root_tfm_str); template_path = Path(template_path_str)
    config_path = Path(config_path_str)

    # --- Загрузка конфигурации ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
        logger.info(f"Конфигурация для шагов предобработки загружена из: {config_path}")
        preprocessing_params = config.get('steps', {}).get('preprocessing', {})
        keep_intermediate = preprocessing_params.get('keep_intermediate_files', False)
        # Получаем словари параметров для каждого под-шага (или пустые, если не заданы)
        norm_params_config = preprocessing_params.get('intensity_normalization', {})
        bias_params_config = preprocessing_params.get('bias_field_correction', {})
        reg_params_config = preprocessing_params.get('registration', {})
        strip_params_config = preprocessing_params.get('skull_stripping', {})
    except Exception as e:
        logger.critical(f"Не удалось загрузить/разобрать конфигурацию {config_path}: {e}")
        raise ValueError(f"Ошибка загрузки конфига: {e}") from e

    # --- Проверки входных путей и утилит ---
    if not input_root.is_dir(): raise FileNotFoundError(f"Входная директория не найдена: {input_root}")
    if not template_path.is_file(): raise FileNotFoundError(f"Файл шаблона не найден: {template_path}")
    check_fsl_availability() # Проверяем доступность FSL

    # Создание выходных директорий
    try: output_root_prep.mkdir(parents=True, exist_ok=True); output_root_tfm.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger.error(f"Не удалось создать выходные директории: {e}"); raise

    logger.info(f"Начало предобработки NIfTI из: {input_root.resolve()}")
    logger.info(f"  Предобработанные ->: {output_root_prep.resolve()}")
    logger.info(f"  Трансформации ->: {output_root_tfm.resolve()}")
    logger.info(f"  Шаблон: {template_path.resolve()}")
    logger.info(f"  Сохранять промежуточные: {'Да' if keep_intermediate else 'Нет'}")

    # --- Поиск NIfTI файлов ---
    patterns = [ str(input_root / p) for p in ['sub-*/ses-*/anat/*.nii.gz', 'sub-*/ses-*/anat/*.nii',
                                                'sub-*/anat/*.nii.gz', 'sub-*/anat/*.nii'] ]
    found_files_paths = set()
    for pattern in patterns: found_files_paths.update(glob.glob(pattern, recursive=True))
    nii_files_to_process = sorted(list(found_files_paths))
    if not nii_files_to_process: logger.warning(f"NIfTI файлы не найдены в {input_root}"); return True
    logger.info(f"Найдено {len(nii_files_to_process)} NIfTI файлов для обработки.")

    processed_count = 0; error_count = 0

    # --- Цикл обработки файлов ---
    for input_nifti_file_str in nii_files_to_process:
        input_nifti_path = Path(input_nifti_file_str)
        logger.info(f"--- Начало обработки файла: {input_nifti_path.relative_to(input_root)} ---")

        overall_processing_params = {"input_file": str(input_nifti_path.resolve()), "processing_timestamp": datetime.now().isoformat(), "steps_parameters": {}}
        file_processing_failed = False; per_file_log_handler = None; temp_processing_dir = None

        try:
            final_preprocessed_file_path, individual_transform_dir = create_output_paths(
                input_nifti_file_str, str(input_root), str(output_root_prep), str(output_root_tfm)
            )
            nifti_file_stem = input_nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
            individual_log_file_path = individual_transform_dir / f"{nifti_file_stem}_processing_log.txt"
            overall_params_json_path = individual_transform_dir / f"{nifti_file_stem}_processing_summary.json"

            # Настройка индивидуального логгера
            per_file_log_handler = logging.FileHandler(individual_log_file_path, mode='w', encoding='utf-8')
            per_file_log_handler.setFormatter(formatter); per_file_log_handler.setLevel(logging.DEBUG)
            logger.addHandler(per_file_log_handler)
            logger.info(f"Лог для файла будет сохранен в: {individual_log_file_path}")

            temp_processing_dir = Path(tempfile.mkdtemp(prefix=f"{nifti_file_stem}_temp_", dir=individual_transform_dir))
            logger.debug(f"Создана временная директория: {temp_processing_dir}")

            current_step_input_path = input_nifti_path
            all_steps_succeeded = True

            # Шаг 1: Нормализация
            step_output_path = temp_processing_dir / f"{nifti_file_stem}_norm.nii.gz"
            params, success = run_intensity_normalization_and_save(
                current_step_input_path, step_output_path, template_path, individual_transform_dir, norm_params_config
            )
            overall_processing_params["steps_parameters"]["1_intensity_normalization"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка нормализации интенсивности.")
            current_step_input_path = step_output_path

            # Шаг 2: Коррекция поля смещения
            step_output_path = temp_processing_dir / f"{nifti_file_stem}_norm_biascorr.nii.gz"
            params, success = run_bias_field_correction_and_save(
                current_step_input_path, step_output_path, individual_transform_dir, bias_params_config
            )
            overall_processing_params["steps_parameters"]["2_bias_field_correction"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка коррекции поля смещения.")
            current_step_input_path = step_output_path

            # Шаг 3: Регистрация
            step_output_path = temp_processing_dir / f"{nifti_file_stem}_norm_biascorr_reg.nii.gz"
            ants_tfm_prefix = f"{nifti_file_stem}_reg"
            params, success = run_registration_and_save(
                current_step_input_path, step_output_path, template_path, individual_transform_dir, ants_tfm_prefix, reg_params_config
            )
            overall_processing_params["steps_parameters"]["3_registration"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка регистрации.")
            current_step_input_path = step_output_path

            # Шаг 4: Удаление черепа
            skullstrip_transform_prefix = f"{nifti_file_stem}_strip"
            params, success = run_skull_stripping_and_save(
                current_step_input_path, final_preprocessed_file_path, individual_transform_dir, skullstrip_transform_prefix, strip_params_config
            )
            overall_processing_params["steps_parameters"]["4_skull_stripping"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка удаления черепа.")

            processed_count += 1
            logger.info(f"--- Успешно завершена обработка файла: {input_nifti_path.relative_to(input_root)} ---")

        except Exception as e:
            file_processing_failed = True; error_count += 1
            logger.error(f"--- ОШИБКА при обработке файла: {input_nifti_path.relative_to(input_root)} ---")
            logger.error(f"Сообщение: {e}", exc_info=False)
            if per_file_log_handler: logger.debug("Полный traceback ошибки:", exc_info=True)

        finally:
            # Сохраняем общий JSON с параметрами этого файла
            overall_processing_params["processing_status"] = "success" if not file_processing_failed else "failed"
            if file_processing_failed and 'e' in locals() and isinstance(e, Exception): overall_processing_params["error_message"] = str(e)
            save_parameters(overall_processing_params, overall_params_json_path)
            # Закрываем и удаляем обработчик лога файла
            if per_file_log_handler: logger.removeHandler(per_file_log_handler); per_file_log_handler.close()
            # Удаляем временную директорию, если нужно
            if temp_processing_dir and temp_processing_dir.exists():
                if not keep_intermediate and not file_processing_failed:
                    try: shutil.rmtree(temp_processing_dir); logger.debug(f"Временная директория {temp_processing_dir} удалена.")
                    except Exception as e_rm: logger.warning(f"Не удалось удалить {temp_processing_dir}: {e_rm}")
                else: logger.info(f"Промежуточные файлы сохранены в: {temp_processing_dir.resolve()}")
            logger.info("-" * 30)

    # --- Итоговая статистика ---
    logger.info("=" * 50); logger.info(f"Сводка по предобработке:")
    logger.info(f"  Всего найдено NIfTI: {len(nii_files_to_process)}"); logger.info(f"  Успешно обработано: {processed_count}")
    logger.info(f"  Файлов с ошибками: {error_count}"); logger.info("=" * 50)
    return True

# --- Точка входа ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Выполняет конвейер предобработки NIfTI МРТ изображений.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, type=str, help="Входная директория BIDS NIfTI.")
    parser.add_argument("--output_prep_dir", required=True, type=str, help="Выходная директория для предобработанных файлов.")
    parser.add_argument("--output_transform_dir", required=True, type=str, help="Выходная директория для трансформаций.")
    parser.add_argument("--template_path", required=True, type=str, help="Путь к файлу шаблона МРТ.")
    parser.add_argument("--config", required=True, type=str, help="Путь к YAML конфиг. файлу пайплайна.")
    parser.add_argument("--main_log_file", default=None, type=str, help="Путь к основному лог-файлу. По умолч.: 'preprocessing_main.log' в output_transform_dir.")
    parser.add_argument("--console_log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Уровень лога консоли.")

    args = parser.parse_args()

    # --- Настройка логирования ---
    main_log_path_arg = args.main_log_file
    output_tfm_dir_arg = args.output_transform_dir
    default_log_name = 'preprocessing_main.log'
    if main_log_path_arg is None:
        try:
            if output_tfm_dir_arg and not os.path.exists(output_tfm_dir_arg): os.makedirs(output_tfm_dir_arg, exist_ok=True)
            main_log_path_final = os.path.join(output_tfm_dir_arg or '.', default_log_name)
        except OSError as e: main_log_path_final = default_log_name; print(f"Предупреждение: Не удалось исп. {output_tfm_dir_arg} для лога...")
    else: main_log_path_final = main_log_path_arg
    setup_main_logging(main_log_path_final, args.console_log_level)

    # --- Запуск пайплайна ---
    try:
        logger.info("=" * 50); logger.info(f"Запуск скрипта: preprocessing.py")
        logger.info(f"  Input: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Output (Preprocessed): {os.path.abspath(args.output_prep_dir)}")
        logger.info(f"  Output (Transforms): {os.path.abspath(args.output_transform_dir)}")
        logger.info(f"  Template: {os.path.abspath(args.template_path)}")
        logger.info(f"  Config File: {os.path.abspath(args.config)}")
        # keep_intermediate читается из конфига внутри функции
        logger.info(f"  Основной лог: {os.path.abspath(main_log_path_final)}")
        logger.info(f"  Уровень лога консоли: {args.console_log_level}")
        logger.info("=" * 50)

        success = run_preprocessing_pipeline(
            args.input_dir,
            args.output_prep_dir,
            args.output_transform_dir,
            args.template_path,
            args.config # Передаем путь к конфигу
        )

        if success: logger.info("Скрипт предобработки успешно завершил работу."); sys.exit(0)
        else: logger.error("Скрипт предобработки завершился с ошибкой."); sys.exit(1) # Не должно достигаться

    except FileNotFoundError as e: logger.error(f"Критическая ошибка: Файл/директория не найдены. {e}"); sys.exit(1)
    except OSError as e: logger.error(f"Критическая ошибка ФС: {e}", exc_info=True); sys.exit(1)
    except (KeyError, ValueError) as e: logger.error(f"Критическая ошибка: Ошибка в конфигурационном файле ({args.config}): {e}"); sys.exit(1)
    except Exception as e: logger.exception(f"Непредвиденная критическая ошибка: {e}"); sys.exit(1)