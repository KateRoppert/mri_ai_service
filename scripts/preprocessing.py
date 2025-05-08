import os
import glob
import shutil
import json
import logging
import tempfile
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import yaml # <<< Добавлен импорт для чтения конфига
import SimpleITK as sitk
from nipype.interfaces.fsl import BET, FSLCommand
from nipype.interfaces.base import Undefined
import ants

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s')

# Глобальные переменные
fsl_present = None
ANISOTROPY_THRESHOLD = 3.0 # Значение по умолчанию, может быть переопределено в будущем

def setup_main_logging(log_file_path: str, console_level: str = "INFO"):
    """Настраивает основной логгер скрипта."""
    if logger.hasHandlers(): logger.handlers.clear()
    # Консоль
    ch = logging.StreamHandler(sys.stdout)
    try: ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError: ch.setLevel(logging.INFO); print(f"WARN: Invalid console log level '{console_level}'. Using INFO.")
    ch.setFormatter(formatter); logger.addHandler(ch)
    # Файл
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setLevel(logging.DEBUG); fh.setFormatter(formatter); logger.addHandler(fh)
        logger.debug(f"Основное логирование в файл настроено: {log_file_path}")
    except Exception as e: print(f"CRITICAL ERROR: Failed to setup main log file {log_file_path}: {e}"); sys.exit(1) # Выход, если лог не настроить

def check_fsl_availability():
    """Проверяет доступность FSL."""
    global fsl_present
    if fsl_present is not None: return fsl_present
    try:
        FSLCommand.check_fsl(); logger.info("Проверка FSL: Найден через Nipype."); fsl_present = True
    except Exception:
        logger.warning("Проверка FSL: Nipype не нашел FSL. Проверка PATH...");
        try: result = subprocess.run(['which', 'bet'], check=True, capture_output=True, text=True)
             logger.info(f"Проверка FSL: 'bet' найден в PATH: {result.stdout.strip()}"); fsl_present = True
        except: logger.error("Проверка FSL: 'bet' НЕ НАЙДЕН в PATH. Удаление черепа не будет выполнено."); fsl_present = False
    return fsl_present

def create_output_paths(
    input_nifti_path_str: str, input_root_str: str,
    output_root_prep_str: str, output_root_tfm_str: str
) -> tuple[Path, Path]:
    """Создает выходные пути и директории."""
    input_path = Path(input_nifti_path_str); input_root = Path(input_root_str)
    output_root_prep = Path(output_root_prep_str); output_root_tfm = Path(output_root_tfm_str)
    try: relative_path = input_path.relative_to(input_root)
    except ValueError: logger.warning(f"Не удалось опред. отн. путь {input_path} от {input_root}."); relative_path = Path(input_path.name)
    final_prep_path = output_root_prep / relative_path
    transform_dir_name = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    transform_dir = output_root_tfm / relative_path.parent / transform_dir_name
    try:
        final_prep_path.parent.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger.error(f"Ошибка создания выходных директорий для {input_path.name}: {e}"); raise
    logger.debug(f"Путь к предобработанному файлу: {final_prep_path}")
    logger.debug(f"Директория для трансформаций: {transform_dir}")
    return final_prep_path, transform_dir

# === Функции шагов предобработки (теперь принимают словарь параметров) ===

def original_bias_field_correction(
    input_img_path_str: str,
    out_path_str: str,
    params: dict # <<< Параметры N4 из конфига
) -> sitk.Image:
    """Выполняет N4 коррекцию поля смещения."""
    logger.info(f"  N4 Bias Field Correction: {Path(input_img_path_str).name} -> {Path(out_path_str).name}")
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)
    head_mask = sitk.LiThreshold(raw_img_sitk, 0, 1) # Маска по порогу Li

    # Используем параметры из конфига или значения по умолчанию
    shrinkFactor = params.get('sitk_shrinkFactor', 4)
    n_iterations = params.get('sitk_numberOfIterations', [50] * shrinkFactor) # По умолчанию 50 итераций на каждом уровне
    conv_thresh = params.get('sitk_convergenceThreshold', 0.0)

    # Убедимся, что n_iterations имеет правильную длину
    if len(n_iterations) != shrinkFactor:
        logger.warning(f"Длина sitk_numberOfIterations ({len(n_iterations)}) не совпадает с shrinkFactor ({shrinkFactor}). Используется усеченный/дополненный список.")
        n_iterations = (n_iterations * shrinkFactor)[:shrinkFactor] # Простой способ подогнать длину

    logger.debug(f"    Параметры N4: shrinkFactor={shrinkFactor}, iterations={n_iterations}, convergence={conv_thresh}")

    inputImage_shrink = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage_shrink = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrector.SetMaximumNumberOfIterations(n_iterations)
    if conv_thresh > 0: # Устанавливаем порог, только если он задан
        bias_corrector.SetConvergenceThreshold(conv_thresh)

    logger.debug("    Выполнение N4...")
    corrected_shrink = bias_corrector.Execute(inputImage_shrink, maskImage_shrink)
    logger.debug("    Получение логарифма поля смещения...")
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
    logger.debug("    Применение коррекции...")
    corrected_image_full_resolution = raw_img_sitk / (sitk.Exp(log_bias_field) + 1e-7)
    sitk.WriteImage(corrected_image_full_resolution, out_path_str)
    logger.debug("    N4 коррекция завершена.")
    return log_bias_field

def original_intensity_normalization(
    input_img_path_str: str,
    out_path_str: str,
    template_img_path_str: str,
    params: dict # <<< Параметры нормализации из конфига (пока не используются, но передаются)
):
    """Выполняет нормализацию интенсивности методом сопоставления гистограмм."""
    logger.info(f"  Intensity Normalization (Histogram Matching): {Path(input_img_path_str).name} -> {Path(out_path_str).name}")
    logger.debug(f"    Используемый шаблон: {template_img_path_str}")
    logger.debug(f"    Параметры из конфига (intensity_normalization): {params}") # Логируем параметры

    template_img_sitk = sitk.ReadImage(template_img_path_str, sitk.sitkFloat32)
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)
    logger.debug("    Выполнение сопоставления гистограмм...")
    transformed = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)
    sitk.WriteImage(transformed, out_path_str)
    logger.debug("    Нормализация интенсивности завершена.")

def original_registration(
    input_img_path_str: str,
    out_path_str: str,
    template_img_path_str: str,
    transforms_prefix_str: str,
    params: dict # <<< Параметры регистрации из конфига
) -> tuple[list[str], list[str]]:
    """Выполняет регистрацию с использованием ANTsPy."""
    logger.info(f"  ANTs Registration: {Path(input_img_path_str).name} -> {Path(out_path_str).name}")

    # Используем параметры из конфига или значения по умолчанию
    transform_type = params.get('ants_transform_type', 'SyN')
    # Другие параметры ANTs (примеры, можно добавить больше по необходимости)
    # registration_params = {
    #     'metric': params.get('ants_metric', 'MI'),
    #     'iterations': params.get('ants_iterations', (100, 70, 50, 20)), # Пример формата для ANTsPy
    #     'reg_iterations': params.get('ants_reg_iterations', None),
    #     'smoothing_sigmas': params.get('ants_smoothing_sigmas', (3, 2, 1, 0)), # Пример формата
    #     'sigma_units': ['vox'] * 4, # Пример
    #     'shrink_factors': params.get('ants_shrink_factors', (8, 4, 2, 1)), # Пример
    #     'grad_step': params.get('ants_grad_step', 0.1),
    #     # ... и т.д.
    # }
    # # Удаляем None значения, чтобы не передавать их в ants.registration
    # registration_params = {k: v for k, v in registration_params.items() if v is not None}

    logger.debug(f"    Тип трансформации: {transform_type}")
    # logger.debug(f"    Дополнительные параметры ANTs: {registration_params}")
    logger.debug(f"    Префикс выходных файлов трансформации: {transforms_prefix_str}")

    template_img_ants = ants.image_read(template_img_path_str)
    raw_img_ants = ants.image_read(input_img_path_str)

    logger.debug(f"    Выполнение ants.registration (тип: {transform_type})...")
    transformation = ants.registration(
        fixed=template_img_ants,
        moving=raw_img_ants,
        type_of_transform=transform_type,
        outprefix=transforms_prefix_str,
        verbose=False, # Отключаем встроенный вывод ANTsPy
        # **registration_params # Передаем остальные параметры
    )
    registered_img_ants = transformation['warpedmovout']
    registered_img_ants.to_file(out_path_str)
    logger.debug("    ANTs регистрация завершена.")
    return transformation.get('fwdtransforms', []), transformation.get('invtransforms', [])


def original_skull_stripping(
    input_img_path_str: str,
    out_file_base_str: str,
    params: dict # <<< Параметры BET из конфига
) -> tuple[str | None, str | None]:
    """Выполняет удаление черепа с помощью FSL BET, используя параметры из конфига."""
    global fsl_present
    if not fsl_present:
         logger.error("  Skull Stripping: FSL 'bet' не найден. Шаг пропускается.")
         return None, None

    logger.info(f"  Skull Stripping (FSL BET): {Path(input_img_path_str).name} -> {out_file_base_str}.nii.gz")

    # Используем параметры из конфига или значения по умолчанию
    frac_thresh = params.get('bet_fractional_intensity_threshold', 0.5)
    robust = params.get('bet_robust', True)
    additional_opts = params.get('bet_options', "") # Дополнительные опции строкой

    logger.debug(f"    Параметры BET: frac={frac_thresh}, robust={robust}, options='{additional_opts}'")

    bet = BET()
    bet.inputs.in_file = input_img_path_str
    bet.inputs.out_file = out_file_base_str
    bet.inputs.frac = frac_thresh
    bet.inputs.mask = True
    bet.inputs.robust = robust
    bet.inputs.output_type = 'NIFTI_GZ'
    if additional_opts: # Если заданы доп. опции
        bet.inputs.args = additional_opts

    try:
        logger.debug(f"    Команда BET: {bet.cmdline}")
        result = bet.run()
        stripped_path_str = result.outputs.out_file
        mask_path_str = result.outputs.mask_file

        # Проверка существования файлов (как и раньше)
        expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
        if stripped_path_str is Undefined or not Path(stripped_path_str).exists():
            if expected_stripped_path.exists():
                logger.warning(f"    Вывод BET (stripped) некорректен. Используется: {expected_stripped_path}")
                stripped_path_str = str(expected_stripped_path)
            else:
                logger.error(f"    BET не создал файл удаленного черепа: {expected_stripped_path}")
                if hasattr(result, 'runtime') and result.runtime and result.runtime.stderr: logger.error(f"    BET stderr:\n{result.runtime.stderr.strip()}")
                return None, None

        expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
        if mask_path_str is Undefined or not Path(mask_path_str).exists():
            if expected_mask_path.exists():
                logger.warning(f"    Вывод BET (mask) некорректен. Используется: {expected_mask_path}")
                mask_path_str = str(expected_mask_path)
            else:
                logger.error(f"    BET не создал файл маски: {expected_mask_path}")
                mask_path_str = None

        logger.debug("    Удаление черепа (BET) завершено.")
        return stripped_path_str, mask_path_str

    except Exception as e:
         logger.error(f"    Ошибка при выполнении Nipype BET: {e}", exc_info=True)
         expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
         expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
         if expected_stripped_path.exists():
              logger.warning("    Nipype BET упал, но выходной файл найден.")
              mask_found = str(expected_mask_path) if expected_mask_path.exists() else None
              return str(expected_stripped_path), mask_found
         else:
              logger.error("    Выходные файлы BET не найдены после ошибки Nipype.")
              return None, None

def save_parameters(params_dict: dict, output_path: Path):
    """Сохраняет словарь параметров в JSON файл."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params_serializable = {}
        for key, value in params_dict.items():
             if isinstance(value, Path): params_serializable[key] = str(value.resolve())
             elif isinstance(value, list) and value and isinstance(value[0], (Path, str)):
                 params_serializable[key] = [str(Path(p).resolve()) for p in value]
             # Добавим обработку numpy типов, которые не сериализуются напрямую
             elif isinstance(value, (np.integer, np.floating)): params_serializable[key] = value.item()
             elif isinstance(value, np.ndarray): params_serializable[key] = value.tolist() # Преобразуем массивы в списки
             else: params_serializable[key] = value
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params_serializable, f, indent=4, ensure_ascii=False)
        logger.debug(f"Параметры шага сохранены: {output_path}")
    except Exception as e:
        logger.error(f"Не удалось сохранить параметры в {output_path}: {e}", exc_info=True)

# === Функции-обертки (теперь принимают словарь параметров шага) ===

def run_intensity_normalization_and_save(
    current_input_path: Path, step_output_path: Path, template_path: Path,
    transform_dir: Path, step_params: dict # <<< Параметры шага
) -> tuple[dict, bool]:
    """Обертка для нормализации интенсивности."""
    step_name = "1_intensity_normalization"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    # Сохраняем параметры, с которыми будет запущен шаг
    run_params = {**step_params, "template_file": str(template_path.resolve())}
    try:
        # Передаем параметры в основную функцию
        original_intensity_normalization(str(current_input_path), str(step_output_path), str(template_path), step_params)
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
    transform_dir: Path, step_params: dict # <<< Параметры шага
) -> tuple[dict, bool]:
    """Обертка для N4 коррекции."""
    step_name = "2_bias_field_correction"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    # Сохраняем параметры + пути к выходам
    run_params = {**step_params, "output_log_bias_field_path": None, "output_bias_field_path": None}
    log_bias_field_path = transform_dir / "log_bias_field.nii.gz"
    bias_field_path = transform_dir / "bias_field.nii.gz"
    try:
        # Передаем параметры в основную функцию
        log_bias_field_sitk = original_bias_field_correction(str(current_input_path), str(step_output_path), step_params)
        sitk.WriteImage(log_bias_field_sitk, str(log_bias_field_path)); run_params["output_log_bias_field_path"] = str(log_bias_field_path.resolve())
        bias_field_sitk = sitk.Exp(log_bias_field_sitk); sitk.WriteImage(bias_field_sitk, str(bias_field_path)); run_params["output_bias_field_path"] = str(bias_field_path.resolve())
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
    transform_dir: Path, transform_prefix: str, step_params: dict # <<< Параметры шага
) -> tuple[dict, bool]:
    """Обертка для регистрации ANTsPy."""
    step_name = "3_registration"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    ants_output_prefix_str = str(transform_dir / transform_prefix)
    # Сохраняем параметры + пути
    run_params = {**step_params, "template_file": str(template_path.resolve()), "output_prefix": ants_output_prefix_str, "forward_transforms_paths": [], "inverse_transforms_paths": []}
    try:
        # Передаем параметры в основную функцию
        fwd_tforms, inv_tforms = original_registration(
            str(current_input_path), str(step_output_path), str(template_path), ants_output_prefix_str, step_params
        )
        run_params["forward_transforms_paths"] = [str(Path(p).resolve()) for p in fwd_tforms]
        run_params["inverse_transforms_paths"] = [str(Path(p).resolve()) for p in inv_tforms]
        if not run_params["forward_transforms_paths"]: logger.warning(f"  Не найдены файлы прямых трансформаций ANTs (префикс: {ants_output_prefix_str})")
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
    transform_dir: Path, transform_prefix: str, step_params: dict # <<< Параметры шага
) -> tuple[dict, bool]:
    """Обертка для удаления черепа."""
    step_name = "4_skull_stripping"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    # Сохраняем параметры + пути
    run_params = {**step_params, "output_mask_path": None, "output_stripped_path": None}
    bet_temp_output_base_str = str(transform_dir / f"{transform_prefix}_bet_temp")
    final_mask_path = transform_dir / f"{transform_prefix}_brain_mask.nii.gz"
    try:
        # Передаем параметры в основную функцию
        stripped_path_str, mask_path_str = original_skull_stripping(str(current_input_path), bet_temp_output_base_str, step_params)
        if stripped_path_str is None: raise RuntimeError("FSL BET не создал файл с удаленным черепом.")

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
        # Очистка временных файлов BET при ошибке
        bet_temp_stripped = Path(bet_temp_output_base_str + ".nii.gz")
        bet_temp_mask = Path(bet_temp_output_base_str + "_mask.nii.gz")
        if bet_temp_stripped.exists(): try: os.remove(bet_temp_stripped); logger.debug(f"Удален временный файл: {bet_temp_stripped}") except OSError: pass
        if bet_temp_mask.exists(): try: os.remove(bet_temp_mask); logger.debug(f"Удален временный файл: {bet_temp_mask}") except OSError: pass
        return run_params, False

# === Основная функция пайплайна (теперь читает конфиг) ===
def run_preprocessing_pipeline(
    input_root_str: str,
    output_root_prep_str: str,
    output_root_tfm_str: str,
    template_path_str: str,
    config_path_str: str # <<< Путь к основному конфигу пайплайна
):
    """
    Запускает полный конвейер предобработки, читая параметры из конфиг. файла.
    """
    input_root = Path(input_root_str)
    output_root_prep = Path(output_root_prep_str)
    output_root_tfm = Path(output_root_tfm_str)
    template_path = Path(template_path_str)
    config_path = Path(config_path_str)

    # --- Загрузка конфигурации ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Конфигурация для шагов предобработки загружена из: {config_path}")
        # Получаем параметры конкретно для шага preprocessing
        preprocessing_params = config.get('steps', {}).get('preprocessing', {})
        keep_intermediate = preprocessing_params.get('keep_intermediate_files', False)
        # Получаем параметры для под-шагов
        norm_params_config = preprocessing_params.get('intensity_normalization', {})
        bias_params_config = preprocessing_params.get('bias_field_correction', {})
        reg_params_config = preprocessing_params.get('registration', {})
        strip_params_config = preprocessing_params.get('skull_stripping', {})

    except Exception as e:
        logger.critical(f"Не удалось загрузить или разобрать конфигурационный файл {config_path}: {e}")
        raise ValueError(f"Ошибка загрузки конфига: {e}") from e # Поднимаем ошибку выше

    # --- Проверки ---
    if not input_root.is_dir(): raise FileNotFoundError(f"Входная директория не найдена: {input_root}")
    if not template_path.is_file(): raise FileNotFoundError(f"Файл шаблона не найден: {template_path}")
    check_fsl_availability()

    try:
        output_root_prep.mkdir(parents=True, exist_ok=True)
        output_root_tfm.mkdir(parents=True, exist_ok=True)
    except OSError as e: logger.error(f"Не удалось создать выходные директории: {e}"); raise

    logger.info(f"Начало предобработки NIfTI файлов из: {input_root.resolve()}")
    logger.info(f"  Предобработанные файлы ->: {output_root_prep.resolve()}")
    logger.info(f"  Трансформации ->: {output_root_tfm.resolve()}")
    logger.info(f"  Шаблон: {template_path.resolve()}")
    logger.info(f"  Сохранять промежуточные файлы: {'Да' if keep_intermediate else 'Нет'}")

    # --- Поиск файлов ---
    patterns = [ str(input_root / 'sub-*/ses-*/anat/*.nii.gz'), str(input_root / 'sub-*/ses-*/anat/*.nii'),
                 str(input_root / 'sub-*/anat/*.nii.gz'), str(input_root / 'sub-*/anat/*.nii') ]
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
            params, success = run_intensity_normalization_and_save(current_step_input_path, step_output_path, template_path, individual_transform_dir, norm_params_config) # Передаем параметры
            overall_processing_params["steps_parameters"]["1_intensity_normalization"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге нормализации интенсивности.")
            current_step_input_path = step_output_path

            # Шаг 2: Коррекция поля смещения
            step_output_path = temp_processing_dir / f"{nifti_file_stem}_norm_biascorr.nii.gz"
            params, success = run_bias_field_correction_and_save(current_step_input_path, step_output_path, individual_transform_dir, bias_params_config) # Передаем параметры
            overall_processing_params["steps_parameters"]["2_bias_field_correction"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге коррекции поля смещения.")
            current_step_input_path = step_output_path

            # Шаг 3: Регистрация
            step_output_path = temp_processing_dir / f"{nifti_file_stem}_norm_biascorr_reg.nii.gz"
            ants_tfm_prefix = f"{nifti_file_stem}_reg"
            params, success = run_registration_and_save(current_step_input_path, step_output_path, template_path, individual_transform_dir, ants_tfm_prefix, reg_params_config) # Передаем параметры
            overall_processing_params["steps_parameters"]["3_registration"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге регистрации.")
            current_step_input_path = step_output_path

            # Шаг 4: Удаление черепа
            skullstrip_transform_prefix = f"{nifti_file_stem}_strip"
            params, success = run_skull_stripping_and_save(current_step_input_path, final_preprocessed_file_path, individual_transform_dir, skullstrip_transform_prefix, strip_params_config) # Передаем параметры
            overall_processing_params["steps_parameters"]["4_skull_stripping"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге удаления черепа.")

            processed_count += 1
            logger.info(f"--- Успешно завершена обработка файла: {input_nifti_path.relative_to(input_root)} ---")

        except Exception as e:
            file_processing_failed = True; error_count += 1
            logger.error(f"--- ОШИБКА при обработке файла: {input_nifti_path.relative_to(input_root)} ---")
            logger.error(f"Сообщение: {e}", exc_info=False)
            if per_file_log_handler: logger.debug("Полный traceback ошибки:", exc_info=True)

        finally:
            overall_processing_params["processing_status"] = "success" if not file_processing_failed else "failed"
            if file_processing_failed and 'e' in locals() and isinstance(e, Exception): overall_processing_params["error_message"] = str(e)
            save_parameters(overall_processing_params, overall_params_json_path)
            if per_file_log_handler: logger.removeHandler(per_file_log_handler); per_file_log_handler.close()
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

# --- Точка входа при запуске скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Выполняет конвейер предобработки NIfTI МРТ изображений.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Основные пути передаются через CLI
    parser.add_argument("--input_dir", required=True, type=str, help="Входная директория BIDS NIfTI.")
    parser.add_argument("--output_prep_dir", required=True, type=str, help="Выходная директория для предобработанных файлов.")
    parser.add_argument("--output_transform_dir", required=True, type=str, help="Выходная директория для трансформаций и параметров.")
    parser.add_argument("--template_path", required=True, type=str, help="Путь к файлу шаблона МРТ.")
    # Путь к конфигу для получения параметров шагов
    parser.add_argument("--config", required=True, type=str, help="Путь к основному конфигурационному файлу YAML пайплайна.")
    # Основной лог и уровень консоли
    parser.add_argument("--main_log_file", default=None, type=str, help="Путь к основному лог-файлу скрипта. По умолчанию: 'preprocessing_main.log' в output_transform_dir.")
    parser.add_argument("--console_log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Уровень логирования для консоли.")
    # Флаг сохранения промежуточных файлов (читается из конфига, но можно переопределить)
    # parser.add_argument("--keep_intermediate_files", action='store_true', help="Сохранять промежуточные файлы.")

    args = parser.parse_args()

    # --- Настройка основного логирования ---
    main_log_path_arg = args.main_log_file
    output_tfm_dir_arg = args.output_transform_dir # Используем папку трансформаций для основного лога по умолчанию
    default_log_name = 'preprocessing_main.log'

    if main_log_path_arg is None:
        try:
            if output_tfm_dir_arg and not os.path.exists(output_tfm_dir_arg): os.makedirs(output_tfm_dir_arg, exist_ok=True)
            main_log_path_final = os.path.join(output_tfm_dir_arg or '.', default_log_name)
        except OSError as e:
            main_log_path_final = default_log_name
            print(f"Предупреждение: Не удалось исп. {output_tfm_dir_arg} для основного лога...")
    else:
        main_log_path_final = main_log_path_arg

    setup_main_logging(main_log_path_final, args.console_log_level)

    # --- Основной блок выполнения ---
    try:
        logger.info("=" * 50)
        logger.info(f"Запуск скрипта: preprocessing.py")
        logger.info(f"  Input: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Output (Preprocessed): {os.path.abspath(args.output_prep_dir)}")
        logger.info(f"  Output (Transforms): {os.path.abspath(args.output_transform_dir)}")
        logger.info(f"  Template: {os.path.abspath(args.template_path)}")
        logger.info(f"  Config File: {os.path.abspath(args.config)}")
        # keep_intermediate будет прочитан из конфига внутри run_preprocessing_pipeline
        logger.info(f"  Основной лог: {os.path.abspath(main_log_path_final)}")
        logger.info(f"  Уровень лога консоли: {args.console_log_level}")
        logger.info("=" * 50)

        # Вызываем основную функцию, передавая путь к конфигу
        success = run_preprocessing_pipeline(
            args.input_dir,
            args.output_prep_dir,
            args.output_transform_dir,
            args.template_path,
            args.config # Передаем путь к конфигу
            # keep_intermediate теперь читается из конфига внутри функции
        )

        if success: logger.info("Скрипт предобработки успешно завершил работу."); sys.exit(0)
        else: logger.error("Скрипт предобработки завершился с ошибкой."); sys.exit(1)

    except FileNotFoundError as e: logger.error(f"Критическая ошибка: Файл/директория не найдены. {e}"); sys.exit(1)
    except OSError as e: logger.error(f"Критическая ошибка ФС: {e}", exc_info=True); sys.exit(1)
    except KeyError as e: logger.error(f"Критическая ошибка: Отсутствует ключ в конфигурационном файле ({args.config}): {e}."); sys.exit(1)
    except ValueError as e: logger.error(f"Критическая ошибка: Неверное значение или ошибка в конфиг. файле ({args.config}): {e}"); sys.exit(1)
    except Exception as e: logger.exception(f"Непредвиденная критическая ошибка: {e}"); sys.exit(1)