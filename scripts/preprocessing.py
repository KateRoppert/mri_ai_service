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
from datetime import datetime # Не используется явно, но может быть полезно в будущем

import SimpleITK as sitk
from nipype.interfaces.fsl import BET, FSLCommand
from nipype.interfaces.base import Undefined # Для проверки вывода Nipype
import ants # Для ANTsPy

# --- Настройка логгера ---
# Логгер будет настроен функцией setup_logging
logger = logging.getLogger(__name__) # Логгер для этого модуля
logger.setLevel(logging.DEBUG)      # Устанавливаем базовый уровень
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s')

# Переменная для статуса FSL
fsl_present = None

def setup_main_logging(log_file_path: str, console_level: str = "INFO"):
    """
    Настраивает основной логгер скрипта: вывод в консоль и в указанный файл.
    Очищает предыдущие обработчики основного логгера.
    """
    # Очищаем обработчики только у логгера этого модуля, чтобы не затронуть другие
    if logger.hasHandlers():
        logger.handlers.clear()

    # Консольный обработчик
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        ch.setLevel(logging.INFO) # Уровень по умолчанию, если указан неверный
        print(f"Предупреждение: Неверный уровень логирования для консоли '{console_level}'. Используется INFO.")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файловый обработчик
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir: # Если указан путь с директорией
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # mode='a' для дозаписи
        fh.setLevel(logging.DEBUG) # В файл пишем все, начиная с DEBUG
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Основное логирование в файл настроено: {log_file_path}")
    except Exception as e:
        # Используем print, так как файловый логгер мог не настроиться
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить основной файловый лог {log_file_path}: {e}")
        # Можно решить, прерывать ли выполнение, если основной лог не создан
        # sys.exit(1)


def check_fsl_availability():
    """Проверяет доступность FSL и устанавливает глобальную переменную fsl_present."""
    global fsl_present
    if fsl_present is not None: # Если уже проверяли
        return fsl_present

    try:
        FSLCommand.check_fsl() # Метод Nipype
        logger.info("Проверка FSL: FSL найден с помощью Nipype.")
        fsl_present = True
    except Exception as e:
        logger.warning(f"Проверка FSL: Nipype не смог автоматически обнаружить FSL: {e}")
        logger.info("Проверка FSL: Попытка найти 'bet' через системный PATH...")
        try:
            # Дополнительная проверка через subprocess, если Nipype не нашел
            result = subprocess.run(['which', 'bet'], check=True, capture_output=True, text=True)
            logger.info(f"Проверка FSL: Команда 'bet' найдена в PATH: {result.stdout.strip()}")
            fsl_present = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Проверка FSL: Команда 'bet' НЕ найдена в PATH. Шаг удаления черепа (skull stripping) не будет выполнен.")
            fsl_present = False
    return fsl_present


# --- Оригинальные функции предобработки с добавленным логированием и Path ---

def create_output_paths(
    input_nifti_path_str: str,
    input_root_str: str,
    output_root_prep_str: str,
    output_root_tfm_str: str
) -> tuple[Path, Path]:
    """
    Создает пути для выходных файлов и директорий трансформаций, сохраняя структуру.
    """
    input_path = Path(input_nifti_path_str)
    input_root = Path(input_root_str)
    output_root_prep = Path(output_root_prep_str)
    output_root_tfm = Path(output_root_tfm_str)

    try:
        relative_path = input_path.relative_to(input_root)
    except ValueError:
        # Если input_path не внутри input_root, используем только имя файла
        logger.warning(
            f"Не удалось определить относительный путь для {input_path} от {input_root}. "
            f"Результаты будут сохранены на верхнем уровне выходной директории."
        )
        relative_path = Path(input_path.name)

    final_prep_path = output_root_prep / relative_path
    # Имя директории для трансформаций создается из имени файла без расширений
    transform_dir_name = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    transform_dir = output_root_tfm / relative_path.parent / transform_dir_name

    # Создаем необходимые директории
    try:
        final_prep_path.parent.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Ошибка создания выходных директорий для {input_path.name}: {e}")
        raise # Передаем ошибку выше, т.к. это критично

    logger.debug(f"Путь к предобработанному файлу: {final_prep_path}")
    logger.debug(f"Директория для трансформаций: {transform_dir}")
    return final_prep_path, transform_dir


def original_bias_field_correction(input_img_path_str: str, out_path_str: str) -> sitk.Image:
    """Выполняет N4 коррекцию поля смещения с использованием SimpleITK."""
    logger.info(f"  N4 Bias Field Correction: {Path(input_img_path_str).name} -> {Path(out_path_str).name}")
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)
    # Маска головы для N4 (простое пороговое значение)
    head_mask = sitk.LiThreshold(raw_img_sitk, 0, 1)

    shrinkFactor = 4 # Как в оригинале
    inputImage_shrink = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage_shrink = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Установим некоторые параметры по умолчанию, если они не заданы (для стабильности)
    bias_corrector.SetMaximumNumberOfIterations([50] * shrinkFactor) # Пример

    logger.debug("    Выполнение N4 коррекции на уменьшенном изображении...")
    corrected_shrink = bias_corrector.Execute(inputImage_shrink, maskImage_shrink)

    logger.debug("    Получение логарифма поля смещения...")
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)

    logger.debug("    Применение коррекции к полноразмерному изображению...")
    # Добавляем малое число в знаменатель для избежания деления на ноль
    corrected_image_full_resolution = raw_img_sitk / (sitk.Exp(log_bias_field) + 1e-7)
    sitk.WriteImage(corrected_image_full_resolution, out_path_str)
    logger.debug("    N4 коррекция завершена.")
    return log_bias_field


def original_intensity_normalization(input_img_path_str: str, out_path_str: str, template_img_path_str: str):
    """Выполняет нормализацию интенсивности методом сопоставления гистограмм."""
    logger.info(f"  Intensity Normalization (Histogram Matching): {Path(input_img_path_str).name} -> {Path(out_path_str).name}")
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
    transforms_prefix_str: str
) -> tuple[list[str], list[str]]:
    """Выполняет регистрацию с использованием ANTsPy."""
    logger.info(f"  ANTs Registration (SyN): {Path(input_img_path_str).name} -> {Path(out_path_str).name}")
    template_img_ants = ants.image_read(template_img_path_str)
    raw_img_ants = ants.image_read(input_img_path_str)

    logger.debug(f"    Выполнение ANTsPy registration (type_of_transform='SyN', outprefix='{transforms_prefix_str}')...")
    # verbose=False, чтобы ANTs не писал много в консоль (логгер справится)
    transformation = ants.registration(
        fixed=template_img_ants,
        moving=raw_img_ants,
        type_of_transform='SyN',
        outprefix=transforms_prefix_str,
        verbose=False
    )
    registered_img_ants = transformation['warpedmovout']
    registered_img_ants.to_file(out_path_str)
    logger.debug("    ANTs регистрация завершена.")
    # Возвращаем пути к файлам трансформаций
    return transformation.get('fwdtransforms', []), transformation.get('invtransforms', [])


def original_skull_stripping(input_img_path_str: str, out_file_base_str: str) -> tuple[str | None, str | None]:
    """
    Выполняет удаление черепа с помощью FSL BET.
    Запрашивает выход в формате NIFTI_GZ.
    """
    global fsl_present
    if not fsl_present: # Проверяем доступность FSL
         logger.error("  Skull Stripping: FSL 'bet' команда не найдена. Шаг пропускается.")
         return None, None

    logger.info(f"  Skull Stripping (FSL BET): {Path(input_img_path_str).name} -> {out_file_base_str}.nii.gz")
    bet = BET()
    bet.inputs.in_file = input_img_path_str
    bet.inputs.out_file = out_file_base_str # BET добавит .nii.gz
    bet.inputs.frac = 0.5      # Параметр из оригинала
    bet.inputs.mask = True     # Запрашиваем создание маски
    bet.inputs.robust = True   # Параметр из оригинала
    bet.inputs.output_type = 'NIFTI_GZ' # Явно указываем формат вывода

    try:
        logger.debug(f"    Команда BET: {bet.cmdline}")
        result = bet.run() # Запускаем BET

        # Получаем пути к выходным файлам из результатов Nipype
        stripped_path_str = result.outputs.out_file
        mask_path_str = result.outputs.mask_file

        # --- Проверка существования файлов после BET ---
        # Nipype иногда может неверно возвращать пути или они могут быть Undefined
        # Поэтому проверяем ожидаемые пути, если вывод Nipype некорректен.

        expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
        if stripped_path_str is Undefined or not Path(stripped_path_str).exists():
            if expected_stripped_path.exists():
                logger.warning(f"    Выходной путь BET (stripped) от Nipype некорректен/отсутствует. Используется ожидаемый путь: {expected_stripped_path}")
                stripped_path_str = str(expected_stripped_path)
            else:
                logger.error(f"    BET не создал файл удаленного черепа по ожидаемому пути: {expected_stripped_path}")
                if hasattr(result, 'runtime') and result.runtime and result.runtime.stderr:
                     logger.error(f"    BET stderr:\n{result.runtime.stderr.strip()}")
                return None, None # Ошибка, если основной файл не найден

        expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
        if mask_path_str is Undefined or not Path(mask_path_str).exists():
            if expected_mask_path.exists():
                logger.warning(f"    Выходной путь BET (mask) от Nipype некорректен/отсутствует. Используется ожидаемый путь: {expected_mask_path}")
                mask_path_str = str(expected_mask_path)
            else:
                logger.error(f"    BET не создал файл маски по ожидаемому пути: {expected_mask_path}")
                # Продолжаем, если основной файл есть, но маски нет (хотя это странно для mask=True)
                mask_path_str = None # Указываем, что маска не найдена

        logger.debug("    Удаление черепа (BET) завершено.")
        return stripped_path_str, mask_path_str

    except Exception as e:
         logger.error(f"    Ошибка при выполнении Nipype BET: {e}", exc_info=True)
         # Дополнительная проверка, если интерфейс упал, но файлы могли создаться
         expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
         expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
         if expected_stripped_path.exists():
              logger.warning("    Интерфейс Nipype BET завершился с ошибкой, но выходной файл (stripped) найден.")
              mask_found_after_error = str(expected_mask_path) if expected_mask_path.exists() else None
              return str(expected_stripped_path), mask_found_after_error
         else:
              logger.error("    Выходные файлы BET не найдены после ошибки интерфейса Nipype.")
              return None, None


def save_parameters(params_dict: dict, output_path: Path):
    """Сохраняет словарь параметров в JSON файл."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Сериализация Path объектов в строки для JSON
        params_serializable = {}
        for key, value in params_dict.items():
             if isinstance(value, Path):
                 params_serializable[key] = str(value.resolve()) # Сохраняем абсолютный путь
             elif isinstance(value, list) and value and isinstance(value[0], (Path, str)):
                 params_serializable[key] = [str(Path(p).resolve()) for p in value]
             else:
                 params_serializable[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params_serializable, f, indent=4, ensure_ascii=False)
        logger.debug(f"Параметры этапа сохранены: {output_path}")
    except Exception as e:
        logger.error(f"Не удалось сохранить параметры в {output_path}: {e}", exc_info=True)


# --- Функции-обертки для каждого шага с сохранением параметров ---

def run_intensity_normalization_and_save(
    current_input_path: Path,
    step_output_path: Path,
    template_path: Path,
    transform_dir: Path
) -> tuple[dict, bool]:
    """Обертка для нормализации интенсивности с сохранением параметров."""
    step_name = "1_intensity_normalization"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    params = {
        "method": "HistogramMatching",
        "template_file": str(template_path.resolve())
    }
    try:
        original_intensity_normalization(str(current_input_path), str(step_output_path), str(template_path))
        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        params["error"] = str(e)
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, False


def run_bias_field_correction_and_save(
    current_input_path: Path,
    step_output_path: Path,
    transform_dir: Path
) -> tuple[dict, bool]:
    """Обертка для N4 коррекции с сохранением параметров и поля смещения."""
    step_name = "2_bias_field_correction"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    params = {
        "method": "N4BiasFieldCorrection",
        "sitk_shrinkFactor": 4, # Из оригинального кода
        "sitk_mask_method": "LiThreshold" # Из оригинального кода
    }
    log_bias_field_path = transform_dir / "log_bias_field.nii.gz"
    bias_field_path = transform_dir / "bias_field.nii.gz"
    try:
        log_bias_field_sitk = original_bias_field_correction(str(current_input_path), str(step_output_path))
        # Сохраняем логарифм поля смещения и само поле смещения
        sitk.WriteImage(log_bias_field_sitk, str(log_bias_field_path))
        params["output_log_bias_field_path"] = str(log_bias_field_path.resolve())
        bias_field_sitk = sitk.Exp(log_bias_field_sitk) # Преобразуем логарифм в поле
        sitk.WriteImage(bias_field_sitk, str(bias_field_path))
        params["output_bias_field_path"] = str(bias_field_path.resolve())
        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        params["error"] = str(e)
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, False


def run_registration_and_save(
    current_input_path: Path,
    step_output_path: Path,
    template_path: Path,
    transform_dir: Path,
    transform_prefix: str
) -> tuple[dict, bool]:
    """Обертка для регистрации ANTsPy с сохранением параметров и трансформаций."""
    step_name = "3_registration"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    # Префикс для файлов трансформаций ANTs должен включать путь к директории трансформаций
    ants_output_prefix_str = str(transform_dir / transform_prefix)
    params = {
        "method": "ANTsPy Registration",
        "type_of_transform": "SyN", # Из оригинального кода
        "template_file": str(template_path.resolve()),
        "output_prefix": ants_output_prefix_str, # Префикс для ANTs файлов
        "forward_transforms_paths": [],
        "inverse_transforms_paths": []
    }
    try:
        fwd_tforms, inv_tforms = original_registration(
            str(current_input_path), str(step_output_path), str(template_path), ants_output_prefix_str
        )
        # ANTs возвращает список путей, сохраняем их
        params["forward_transforms_paths"] = [str(Path(p).resolve()) for p in fwd_tforms]
        params["inverse_transforms_paths"] = [str(Path(p).resolve()) for p in inv_tforms]
        if not params["forward_transforms_paths"]:
            logger.warning(f"  Не удалось найти файлы прямых трансформаций ANTs по префиксу {ants_output_prefix_str}")
        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        params["error"] = str(e)
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, False


def run_skull_stripping_and_save(
    current_input_path: Path,
    final_output_file_path: Path, # Это уже финальный путь для предобработанного файла
    transform_dir: Path,
    transform_prefix: str # Префикс для именования маски
) -> tuple[dict, bool]:
    """Обертка для удаления черепа с сохранением параметров и маски."""
    step_name = "4_skull_stripping"
    logger.info(f"Запуск шага: {step_name} для {current_input_path.name}")
    params = {
        "method": "FSL BET",
        "inputs.frac": 0.5,
        "inputs.robust": True,
        "output_mask_path": None,
        "output_stripped_path": None
    }
    # Базовое имя для временных файлов BET (без расширения .nii.gz)
    # Файлы будут созданы в директории transform_dir
    bet_temp_output_base_str = str(transform_dir / f"{transform_prefix}_bet_temp")
    # Финальный путь для маски
    final_mask_path = transform_dir / f"{transform_prefix}_brain_mask.nii.gz"

    try:
        # original_skull_stripping вернет пути к созданным файлам (stripped и mask)
        stripped_path_from_bet_str, mask_path_from_bet_str = original_skull_stripping(
            str(current_input_path), bet_temp_output_base_str
        )

        if stripped_path_from_bet_str is None: # Если BET не смог создать основной файл
            raise RuntimeError("FSL BET не создал файл с удаленным черепом.")

        generated_stripped_path = Path(stripped_path_from_bet_str)

        # Перемещаем или копируем файл с удаленным черепом в его финальное местоположение
        final_output_file_path.parent.mkdir(parents=True, exist_ok=True) # Убедимся, что папка существует
        logger.debug(f"  Перемещение {generated_stripped_path} -> {final_output_file_path}")
        shutil.move(str(generated_stripped_path), str(final_output_file_path))
        params["output_stripped_path"] = str(final_output_file_path.resolve())

        # Обрабатываем маску, если она была создана
        if mask_path_from_bet_str:
            generated_mask_path = Path(mask_path_from_bet_str)
            if generated_mask_path.exists():
                logger.debug(f"  Перемещение маски {generated_mask_path} -> {final_mask_path}")
                shutil.move(str(generated_mask_path), str(final_mask_path))
                params["output_mask_path"] = str(final_mask_path.resolve())
            else:
                logger.warning(f"  Файл маски {generated_mask_path} не найден после BET, хотя ожидался.")
        else:
            logger.warning("  BET не вернул путь к файлу маски.")

        logger.info(f"  Шаг {step_name} успешно завершен.")
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        return params, True
    except Exception as e:
        logger.error(f"  Шаг {step_name} завершился с ошибкой: {e}", exc_info=True)
        params["error"] = str(e)
        save_parameters(params, transform_dir / f"{step_name}_params.json")
        # Попытка удалить временные файлы BET, если они есть
        bet_temp_stripped = Path(bet_temp_output_base_str + ".nii.gz")
        bet_temp_mask = Path(bet_temp_output_base_str + "_mask.nii.gz")
        if bet_temp_stripped.exists():
            try: os.remove(bet_temp_stripped)
            except OSError: logger.warning(f"Не удалось удалить временный файл: {bet_temp_stripped}")
        if bet_temp_mask.exists():
            try: os.remove(bet_temp_mask)
            except OSError: logger.warning(f"Не удалось удалить временный файл: {bet_temp_mask}")
        return params, False


# --- Основная функция конвейера предобработки ---
def run_preprocessing_pipeline(
    input_root_str: str,
    output_root_prep_str: str,
    output_root_tfm_str: str,
    template_path_str: str,
    keep_intermediate: bool = False
):
    """
    Запускает полный конвейер предобработки для NIfTI файлов в BIDS-подобной структуре.
    """
    input_root = Path(input_root_str)
    output_root_prep = Path(output_root_prep_str)
    output_root_tfm = Path(output_root_tfm_str)
    template_path = Path(template_path_str)

    # --- Проверки ---
    if not input_root.is_dir():
        logger.error(f"Корневая входная директория BIDS не найдена: {input_root}")
        raise FileNotFoundError(f"Корневая входная директория BIDS не найдена: {input_root}")
    if not template_path.is_file():
        logger.error(f"Файл шаблона МРТ не найден: {template_path}")
        raise FileNotFoundError(f"Файл шаблона МРТ не найден: {template_path}")
    check_fsl_availability() # Проверяем FSL один раз в начале

    # Создаем корневые выходные директории
    try:
        output_root_prep.mkdir(parents=True, exist_ok=True)
        output_root_tfm.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Не удалось создать корневые выходные директории: {e}")
        raise

    logger.info(f"Начало предобработки NIfTI файлов из: {input_root.resolve()}")
    logger.info(f"  Предобработанные файлы будут сохранены в: {output_root_prep.resolve()}")
    logger.info(f"  Файлы трансформаций и параметры в: {output_root_tfm.resolve()}")
    logger.info(f"  Используемый шаблон: {template_path.resolve()}")
    logger.info(f"  Сохранять промежуточные файлы: {'Да' if keep_intermediate else 'Нет'}")

    # Поиск NIfTI файлов (оригинальный паттерн)
    # Сначала ищем с ses-, потом без, если первый не дал результатов
    nii_files_pattern1 = str(input_root / 'sub-*/ses-*/anat/*.nii.gz')
    nii_files_pattern2 = str(input_root / 'sub-*/ses-*/anat/*.nii') # Для .nii
    nii_files_pattern3 = str(input_root / 'sub-*/anat/*.nii.gz') # Без ses
    nii_files_pattern4 = str(input_root / 'sub-*/anat/*.nii')    # Без ses и .gz

    # Используем glob.glob для поиска, затем объединяем и убираем дубликаты
    found_files_paths = set()
    found_files_paths.update(glob.glob(nii_files_pattern1, recursive=True))
    found_files_paths.update(glob.glob(nii_files_pattern2, recursive=True))
    # Если с ses- ничего не найдено, ищем без ses-
    if not found_files_paths:
        logger.info("Не найдено файлов по шаблону sub-*/ses-*/anat/, поиск по sub-*/anat/...")
        found_files_paths.update(glob.glob(nii_files_pattern3, recursive=True))
        found_files_paths.update(glob.glob(nii_files_pattern4, recursive=True))

    nii_files_to_process = sorted(list(found_files_paths)) # Сортируем для воспроизводимости

    if not nii_files_to_process:
        logger.warning(f"NIfTI файлы для обработки не найдены во входной директории: {input_root}")
        return True # Считаем успешным, если нечего обрабатывать

    logger.info(f"Найдено {len(nii_files_to_process)} NIfTI файлов для обработки.")
    processed_count = 0
    error_count = 0

    # --- Цикл обработки каждого найденного NIfTI файла ---
    for input_nifti_file_str in nii_files_to_process:
        input_nifti_path = Path(input_nifti_file_str)
        logger.info(f"--- Начало обработки файла: {input_nifti_path.relative_to(input_root)} ---")

        # Словарь для сбора всех параметров этого файла
        overall_processing_params = {
            "input_file": str(input_nifti_path.resolve()),
            "processing_timestamp": datetime.now().isoformat(),
            "steps_parameters": {} # Параметры каждого шага
        }
        file_processing_failed = False # Флаг ошибки для текущего файла
        per_file_log_handler = None    # Обработчик лога для конкретного файла
        temp_processing_dir = None     # Временная директория для промежуточных файлов

        try:
            # Создаем пути и директории для вывода этого файла
            final_preprocessed_file_path, individual_transform_dir = create_output_paths(
                input_nifti_file_str, input_root_str, output_root_prep_str, output_root_tfm_str
            )
            # Префикс для файлов трансформаций и временных файлов, основанный на имени NIfTI
            nifti_file_stem = input_nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
            individual_log_file_path = individual_transform_dir / f"{nifti_file_stem}_processing_log.txt"
            overall_params_json_path = individual_transform_dir / f"{nifti_file_stem}_processing_summary.json"

            # --- Настройка логгера для текущего файла (пишет в его transform_dir) ---
            per_file_log_handler = logging.FileHandler(individual_log_file_path, mode='w', encoding='utf-8') # 'w' - перезапись для каждого файла
            per_file_log_handler.setFormatter(formatter)
            per_file_log_handler.setLevel(logging.DEBUG) # Пишем все детали в файл
            logger.addHandler(per_file_log_handler) # Добавляем к основному логгеру
            logger.info(f"Лог для файла {input_nifti_path.name} будет сохранен в: {individual_log_file_path}")

            # --- Создание временной директории для промежуточных файлов этого NIfTI ---
            # Это более надежно, чем создавать файлы напрямую в transform_dir и потом удалять
            temp_processing_dir = Path(tempfile.mkdtemp(prefix=f"{nifti_file_stem}_temp_", dir=individual_transform_dir))
            logger.debug(f"Создана временная директория для промежуточных файлов: {temp_processing_dir}")

            current_step_input_path = input_nifti_path # Начинаем с исходного файла
            all_steps_succeeded = True

            # --- Шаг 1: Нормализация интенсивности ---
            norm_output_path = temp_processing_dir / f"{nifti_file_stem}_norm.nii.gz"
            params, success = run_intensity_normalization_and_save(current_step_input_path, norm_output_path, template_path, individual_transform_dir)
            overall_processing_params["steps_parameters"]["1_intensity_normalization"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге нормализации интенсивности.")
            current_step_input_path = norm_output_path # Выход этого шага - вход для следующего

            # --- Шаг 2: Коррекция поля смещения ---
            biascorr_output_path = temp_processing_dir / f"{nifti_file_stem}_norm_biascorr.nii.gz"
            params, success = run_bias_field_correction_and_save(current_step_input_path, biascorr_output_path, individual_transform_dir)
            overall_processing_params["steps_parameters"]["2_bias_field_correction"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге коррекции поля смещения.")
            current_step_input_path = biascorr_output_path

            # --- Шаг 3: Регистрация ---
            reg_output_path = temp_processing_dir / f"{nifti_file_stem}_norm_biascorr_reg.nii.gz"
            ants_transform_prefix = f"{nifti_file_stem}_reg" # Префикс для файлов трансформаций ANTs
            params, success = run_registration_and_save(current_step_input_path, reg_output_path, template_path, individual_transform_dir, ants_transform_prefix)
            overall_processing_params["steps_parameters"]["3_registration"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге регистрации.")
            current_step_input_path = reg_output_path

            # --- Шаг 4: Удаление черепа ---
            # Финальный предобработанный файл сохраняется сразу в `final_preprocessed_file_path`
            skullstrip_transform_prefix = f"{nifti_file_stem}_strip" # Префикс для маски и временных файлов BET
            params, success = run_skull_stripping_and_save(current_step_input_path, final_preprocessed_file_path, individual_transform_dir, skullstrip_transform_prefix)
            overall_processing_params["steps_parameters"]["4_skull_stripping"] = params
            if not success: all_steps_succeeded = False; raise RuntimeError("Ошибка на шаге удаления черепа.")

            # Если все шаги успешны
            processed_count += 1
            logger.info(f"--- Успешно завершена обработка файла: {input_nifti_path.relative_to(input_root)} ---")

        except Exception as e: # Ловим ошибки любого шага для текущего файла
            file_processing_failed = True
            error_count += 1
            logger.error(
                f"--- ОШИБКА при обработке файла: {input_nifti_path.relative_to(input_root)} ---"
            )
            logger.error(f"Сообщение об ошибке: {e}", exc_info=False) # Не выводим traceback в основной лог здесь
            # Traceback будет в индивидуальном логе файла
            if per_file_log_handler: # Логируем traceback в файл этого субъекта
                logger.debug("Полный traceback ошибки:", exc_info=True)


        finally:
            # --- Сохранение общего JSON с параметрами для этого файла ---
            # Добавляем статус обработки
            overall_processing_params["processing_status"] = "success" if not file_processing_failed else "failed"
            if file_processing_failed and 'e' in locals() and isinstance(e, Exception):
                overall_processing_params["error_message"] = str(e)

            save_parameters(overall_processing_params, overall_params_json_path)

            # --- Закрытие и удаление файлового обработчика лога для текущего файла ---
            if per_file_log_handler:
                logger.removeHandler(per_file_log_handler)
                per_file_log_handler.close()

            # --- Удаление временной директории, если не указано обратное ---
            if temp_processing_dir and temp_processing_dir.exists():
                if not keep_intermediate and not file_processing_failed: # Удаляем только если успешно и не просили оставить
                    try:
                        shutil.rmtree(temp_processing_dir)
                        logger.debug(f"Временная директория {temp_processing_dir} удалена.")
                    except Exception as e_rm:
                        logger.warning(f"Не удалось удалить временную директорию {temp_processing_dir}: {e_rm}")
                else:
                    logger.info(f"Промежуточные файлы сохранены в: {temp_processing_dir.resolve()}")
            logger.info("-" * 30) # Разделитель между файлами в основном логе


    # --- Итоговая статистика по всему запуску ---
    logger.info("=" * 50)
    logger.info(f"Сводка по предобработке:")
    logger.info(f"  Всего найдено NIfTI файлов для обработки: {len(nii_files_to_process)}")
    logger.info(f"  Успешно обработано файлов: {processed_count}")
    logger.info(f"  Файлов с ошибками обработки: {error_count}")
    logger.info("=" * 50)

    return True # Скрипт завершил свою работу (успех/неуспех отдельных файлов логируется)


# --- Точка входа при запуске скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Выполняет конвейер предобработки NIfTI МРТ изображений: нормализация интенсивности, "
                    "N4 коррекция поля смещения, регистрация и удаление черепа (FSL BET). "
                    "Сохраняет параметры и файлы трансформаций.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Корневая директория входного BIDS датасета с NIfTI файлами (например, 'bids_data_nifti')."
    )
    parser.add_argument(
        "--output_prep_dir", # Изменено для ясности
        required=True,
        type=str,
        help="Корневая директория для сохранения финальных предобработанных NIfTI файлов (например, 'preprocessed_data')."
    )
    parser.add_argument(
        "--output_transform_dir", # Изменено для ясности
        required=True,
        type=str,
        help="Корневая директория для сохранения файлов трансформаций, параметров и индивидуальных логов (например, 'transformations')."
    )
    parser.add_argument(
        "--template_path", # Изменено для ясности
        required=True,
        type=str,
        help="Путь к файлу шаблона МРТ NIfTI (например, T1 MNI шаблон)."
    )
    parser.add_argument(
        "--main_log_file", # Изменено для ясности
        default=None,
        type=str,
        help="Путь к основному лог-файлу скрипта. Если не указан, будет создан 'preprocessing_main.log' "
             "в директории, указанной в --output_transform_dir (или в текущей, если та недоступна)."
    )
    parser.add_argument(
        "--console_log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования для вывода в консоль."
    )
    parser.add_argument(
        "--keep_intermediate_files", # Изменено для ясности
        action='store_true', # Если флаг есть, значение True
        help="Сохранять промежуточные файлы каждого шага предобработки во временных директориях."
    )

    args = parser.parse_args()

    # --- Настройка основного логирования ---
    main_log_path = args.main_log_file
    if main_log_path is None:
        # По умолчанию кладем основной лог в папку с трансформациями
        # Это делается здесь, т.к. output_transform_dir обязателен
        default_log_name = 'preprocessing_main.log'
        try:
            # output_transform_dir должен существовать к этому моменту или быть создан в run_preprocessing_pipeline
            # Но для лога лучше создать заранее, если возможно
            os.makedirs(args.output_transform_dir, exist_ok=True)
            main_log_path = os.path.join(args.output_transform_dir, default_log_name)
        except OSError as e:
            # Если не удалось создать output_transform_dir, пишем лог в текущую папку
            main_log_path = default_log_name
            print(f"Предупреждение: Не удалось использовать {args.output_transform_dir} для основного лога. "
                  f"Лог будет в {main_log_path}. Ошибка: {e}")

    setup_main_logging(main_log_path, args.console_log_level)

    # --- Основной блок выполнения ---
    try:
        logger.info("=" * 50)
        logger.info(f"Запуск скрипта предобработки: preprocessing.py")
        logger.info(f"  Входная директория (NIfTI): {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория (предобработанные): {os.path.abspath(args.output_prep_dir)}")
        logger.info(f"  Выходная директория (трансформации): {os.path.abspath(args.output_transform_dir)}")
        logger.info(f"  Файл шаблона: {os.path.abspath(args.template_path)}")
        logger.info(f"  Сохранять промежуточные файлы: {'Да' if args.keep_intermediate_files else 'Нет'}")
        logger.info(f"  Основной лог-файл: {os.path.abspath(main_log_path)}")
        logger.info(f"  Уровень логирования консоли: {args.console_log_level}")
        logger.info("=" * 50)

        success = run_preprocessing_pipeline(
            args.input_dir,
            args.output_prep_dir,
            args.output_transform_dir,
            args.template_path,
            args.keep_intermediate_files
        )

        if success:
            logger.info("Скрипт предобработки успешно завершил свою работу.")
            sys.exit(0)
        else:
            # Эта ветка не должна достигаться при текущей логике
            logger.error("Скрипт предобработки завершился с непредвиденной ошибкой в основной функции.")
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Необходимый файл или директория не найдены. {e}")
        sys.exit(1)
    except OSError as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"КРИТИЧЕСКАЯ НЕПРЕДВИДЕННАЯ ОШИБКА на верхнем уровне: {e}")
        sys.exit(1)