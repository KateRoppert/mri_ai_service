# import SimpleITK as sitk
# from nipype.interfaces.fsl import BET
# import os
# import glob
# import ants
# import shutil

# # === Функция коррекции поля смещения ===
# def bias_field_correction(input_img, out_path):
#     raw_img_sitk = sitk.ReadImage(input_img, sitk.sitkFloat32)
#     head_mask = sitk.LiThreshold(raw_img_sitk, 0, 1)
#     shrinkFactor = 4
#     inputImage = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
#     maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())
#     bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     corrected = bias_corrector.Execute(inputImage, maskImage)
#     log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
#     corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)
#     sitk.WriteImage(corrected_image_full_resolution, out_path)

# # === Функция нормализации интенсивности ===
# def intensity_normalization(input_img, out_path):
#     template_img_path = "/home/roppert/work/brain_tumour2/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii"
#     template_img_sitk = sitk.ReadImage(template_img_path, sitk.sitkFloat32)
#     template_img_sitk = sitk.DICOMOrient(template_img_sitk, 'RPS')
#     raw_img_sitk = sitk.ReadImage(input_img, sitk.sitkFloat32)
#     transformed = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)
#     sitk.WriteImage(transformed, out_path)

# # === Функция регистрации изображения и сохранения трансформаций ===
# def registration(input_img, out_path, transforms_path):
#     template_img_path = "/home/roppert/work/brain_tumour2/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii"
#     template_img_ants = ants.image_read(template_img_path, reorient='IAL')
#     raw_img_ants = ants.image_read(input_img, reorient='IAL')

#     transformation = ants.registration(
#         fixed=template_img_ants,
#         moving=raw_img_ants,
#         type_of_transform='SyN',
#         verbose=False
#     )
#     registered_img_ants = transformation['warpedmovout']
#     registered_img_ants.to_file(out_path)

#     # Сохраняем трансформации
#     os.makedirs(os.path.dirname(transforms_path), exist_ok=True)
#     for i, tfm in enumerate(transformation['fwdtransforms']):
#         base = transforms_path.replace('.nii.gz', f'_fwd_{i}.mat')
#         shutil.copy(tfm, base)

# # === Функция удаления черепа ===
# def skull_stripping(input_img, out_path):
#     bet = BET()
#     bet.inputs.in_file = input_img
#     bet.inputs.out_file = out_path
#     bet.inputs.frac = 0.5
#     bet.inputs.mask = True
#     bet.inputs.robust = True
#     bet.run()

# # === Основная функция обработки всех файлов ===
# def process_nifti_files(input_root, output_root, transforms_root):
#     os.makedirs(output_root, exist_ok=True)
#     os.makedirs(transforms_root, exist_ok=True)

#     for patient_folder in sorted(os.listdir(input_root)):
#         patient_input_path = os.path.join(input_root, patient_folder)
#         if not os.path.isdir(patient_input_path):
#             continue

#         patient_output_path = os.path.join(output_root, patient_folder)
#         patient_transform_path = os.path.join(transforms_root, patient_folder)
#         os.makedirs(patient_output_path, exist_ok=True)
#         os.makedirs(patient_transform_path, exist_ok=True)

#         nii_files = sorted(glob.glob(os.path.join(patient_input_path, '**/anat/*.nii.gz'), recursive=True))
#         if not nii_files:
#             print(f"[ИНФО] Нет файлов для обработки в {patient_folder}")
#             continue

#         print(f"[ИНФО] Обработка пациента: {patient_folder}")
#         log_path = os.path.join(patient_transform_path, 'processing_log.txt')
#         log = open(log_path, 'w')

#         # 1. Intensity Normalization
#         norm_files = []
#         intensity_norm_path = os.path.join(patient_output_path, "intensity_normalized")
#         os.makedirs(intensity_norm_path, exist_ok=True)
#         log.write("[ШАГ] Нормализация интенсивности\n")
#         for file in nii_files:
#             filename = os.path.basename(file)
#             out_file = os.path.join(intensity_norm_path, filename)
#             try:
#                 intensity_normalization(file, out_file)
#                 norm_files.append(out_file)
#                 log.write(f"Нормализовано: {file} -> {out_file}\n")
#             except Exception as e:
#                 log.write(f"Ошибка нормализации {filename}: {str(e)}\n")

#         # 2. Bias Field Correction
#         corrected_files = []
#         bias_corrected_path = os.path.join(patient_output_path, "bias_field_corrected")
#         os.makedirs(bias_corrected_path, exist_ok=True)
#         log.write("\n[ШАГ] Коррекция поля смещения\n")
#         for file in norm_files:
#             filename = os.path.basename(file)
#             out_file = os.path.join(bias_corrected_path, filename)
#             try:
#                 bias_field_correction(file, out_file)
#                 corrected_files.append(out_file)
#                 log.write(f"Скорректировано: {file} -> {out_file}\n")
#             except Exception as e:
#                 log.write(f"Ошибка коррекции {filename}: {str(e)}\n")

#         # 3. Registration
#         reg_files = []
#         registered_path = os.path.join(patient_output_path, "registered")
#         transforms_save_path = os.path.join(patient_transform_path, "registered")
#         os.makedirs(registered_path, exist_ok=True)
#         os.makedirs(transforms_save_path, exist_ok=True)
#         log.write("\n[ШАГ] Регистрация\n")
#         for file in corrected_files:
#             filename = os.path.basename(file)
#             out_file = os.path.join(registered_path, filename)
#             tfm_file = os.path.join(transforms_save_path, filename)
#             try:
#                 registration(file, out_file, tfm_file)
#                 reg_files.append(out_file)
#                 log.write(f"Зарегистрировано: {file} -> {out_file}\n")
#                 log.write(f"Трансформация сохранена: {tfm_file}_fwd_*.mat\n")
#             except Exception as e:
#                 log.write(f"Ошибка регистрации {filename}: {str(e)}\n")

#         # 4. Skull Stripping
#         skull_stripped_path = os.path.join(patient_output_path, "skull_stripped")
#         os.makedirs(skull_stripped_path, exist_ok=True)
#         log.write("\n[ШАГ] Удаление черепа\n")
#         for file in reg_files:
#             filename = os.path.basename(file)
#             out_file = os.path.join(skull_stripped_path, filename)
#             try:
#                 skull_stripping(file, out_file)
#                 log.write(f"Череп удалён: {file} -> {out_file}\n")
#             except Exception as e:
#                 log.write(f"Ошибка удаления черепа {filename}: {str(e)}\n")

#         log.close()
#         print(f"[ИНФО] Пациент {patient_folder} обработан.\n")

# if __name__ == "__main__":
#     input_directory = "bids_data_nifti"
#     output_directory = "preprocessed_data"
#     transformations_directory = "transformations"
#     process_nifti_files(input_directory, output_directory, transformations_directory)

# -*- coding: utf-8 -*-
import SimpleITK as sitk
from nipype.interfaces.fsl import BET, FSLCommand
from nipype.interfaces.base import Undefined # Импортируем Undefined
import os
import glob
import ants
import shutil
import json
import logging
from pathlib import Path
import tempfile
import subprocess
import argparse 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('PreprocessingPipeline')

# --- Проверка доступности FSL ---
fsl_present = None
try:
    FSLCommand.check_fsl()
    logger.info("FSL installation found by Nipype.")
    fsl_present = True
except Exception as e:
    logger.warning(f"Nipype could not automatically detect FSL installation: {e}")
    try:
        subprocess.run(['which', 'bet'], check=True, capture_output=True)
        logger.info("FSL 'bet' command found in PATH via subprocess check.")
        fsl_present = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FSL 'bet' command NOT found in PATH. Skull stripping will fail.")
        fsl_present = False

# --- (Остальные функции без изменений) ---
def create_output_paths(input_nifti_path, input_root, output_root_prep, output_root_tfm):
    input_path = Path(input_nifti_path); input_root = Path(input_root)
    output_root_prep = Path(output_root_prep); output_root_tfm = Path(output_root_tfm)
    try: relative_path = input_path.relative_to(input_root)
    except ValueError: relative_path = Path(input_path.name)
    final_prep_path = output_root_prep / relative_path
    transform_dir_name = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    transform_dir = output_root_tfm / relative_path.parent / transform_dir_name
    final_prep_path.parent.mkdir(parents=True, exist_ok=True)
    transform_dir.mkdir(parents=True, exist_ok=True)
    return final_prep_path, transform_dir

def original_bias_field_correction(input_img_path_str, out_path_str):
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)
    head_mask = sitk.LiThreshold(raw_img_sitk, 0, 1)
    shrinkFactor = 4; inputImage = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_shrink = bias_corrector.Execute(inputImage, maskImage)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
    corrected_image_full_resolution = raw_img_sitk / (sitk.Exp(log_bias_field) + 1e-6)
    sitk.WriteImage(corrected_image_full_resolution, out_path_str)
    return log_bias_field

def original_intensity_normalization(input_img_path_str, out_path_str, template_img_path_str):
    template_img_sitk = sitk.ReadImage(template_img_path_str, sitk.sitkFloat32)
    raw_img_sitk = sitk.ReadImage(input_img_path_str, sitk.sitkFloat32)
    transformed = sitk.HistogramMatching(raw_img_sitk, template_img_sitk)
    sitk.WriteImage(transformed, out_path_str)

def original_registration(input_img_path_str, out_path_str, template_img_path_str, transforms_prefix_str):
    template_img_ants = ants.image_read(template_img_path_str)
    raw_img_ants = ants.image_read(input_img_path_str)
    transformation = ants.registration(fixed=template_img_ants, moving=raw_img_ants, type_of_transform='SyN', outprefix=transforms_prefix_str, verbose=False)
    registered_img_ants = transformation['warpedmovout']
    registered_img_ants.to_file(out_path_str)
    return transformation['fwdtransforms'], transformation['invtransforms']


# === Оригинальная функция удаления черепа (ИСПРАВЛЕНА ПРОВЕРКА ПУТИ ВЫВОДА) ===
def original_skull_stripping(input_img_path_str, out_file_base_str):
    """
    Runs FSL BET, requesting NIFTI_GZ output.

    Args:
        input_img_path_str (str): Path to input NIfTI file.
        out_file_base_str (str): Base path and name for output files (without extension).

    Returns:
        tuple: (path_to_stripped_image, path_to_mask_image) as strings.
               Returns (None, None) on failure.
    """
    if not fsl_present:
         logger.error("FSL 'bet' command not found, cannot perform skull stripping.")
         return None, None

    bet = BET()
    bet.inputs.in_file = input_img_path_str
    bet.inputs.out_file = out_file_base_str # Базовое имя
    bet.inputs.frac = 0.5
    bet.inputs.mask = True      # Запрашиваем маску
    bet.inputs.robust = True
    bet.inputs.output_type = 'NIFTI_GZ' # Явно просим .nii.gz

    try:
        logger.info(f"Running BET command: {bet.cmdline}")
        result = bet.run()

        stripped_path_str = result.outputs.out_file
        mask_path_str = result.outputs.mask_file

        # --- ИЗМЕНЕНИЕ: Корректная проверка пути для stripped_path_str при mask=True ---
        if stripped_path_str is Undefined or not Path(stripped_path_str).exists():
             # Ожидаемый путь для stripped image, когда mask=True
             expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
             # Проверим также вариант без расширения на всякий случай
             expected_stripped_path_noext = Path(out_file_base_str)

             if expected_stripped_path.exists():
                  stripped_path_str = str(expected_stripped_path)
                  logger.warning(f"Nipype output path incorrect/missing, using manual path: {stripped_path_str}")
             elif expected_stripped_path_noext.exists():
                 stripped_path_str = str(expected_stripped_path_noext)
                 logger.warning(f"Nipype output path incorrect/missing and .nii.gz not found, using manual path without extension: {stripped_path_str}")
             else:
                  logger.error(f"BET stripped output file is Undefined or not found. Expected: {expected_stripped_path} or {expected_stripped_path_noext}")
                  if hasattr(result, 'runtime') and result.runtime and hasattr(result.runtime, 'stderr') and result.runtime.stderr:
                       logger.error(f"BET stderr:\n{result.runtime.stderr}")
                  return None, None

        # Проверка пути маски остается прежней (ожидаем _mask.nii.gz)
        if mask_path_str is Undefined or not Path(mask_path_str).exists():
             expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
             if expected_mask_path.exists():
                  mask_path_str = str(expected_mask_path)
                  logger.warning(f"Nipype mask path incorrect/missing, using manual path: {mask_path_str}")
             else:
                  logger.error(f"BET mask output file is Undefined or not found. Expected: {expected_mask_path}")
                  if hasattr(result, 'runtime') and result.runtime and hasattr(result.runtime, 'stderr') and result.runtime.stderr:
                       logger.error(f"BET stderr:\n{result.runtime.stderr}")
                  return None, None

        return stripped_path_str, mask_path_str

    except Exception as e:
         logger.error(f"Error running Nipype BET interface: {e}", exc_info=True)
         # Проверяем ожидаемые пути вручную после ошибки интерфейса
         expected_stripped_path = Path(f"{out_file_base_str}.nii.gz")
         expected_mask_path = Path(f"{out_file_base_str}_mask.nii.gz")
         if expected_stripped_path.exists() and expected_mask_path.exists():
              logger.warning("Nipype interface failed, but output files seem to exist. Returning paths.")
              return str(expected_stripped_path), str(expected_mask_path)
         else:
              logger.error("Output files not found after Nipype interface error.")
              return None, None

# === Функции-обертки (run_skull_stripping_and_save ИСПРАВЛЕНА для обработки пути) ===
def run_intensity_normalization_and_save(input_file_path, output_file_path, template_file_path, transform_dir):
    logger.info(f"Running Intensity Normalization on {input_file_path.name}...")
    params = {"method": "HistogramMatching", "template_file": str(template_file_path.resolve())}
    try:
        original_intensity_normalization(str(input_file_path), str(output_file_path), str(template_file_path))
        logger.info("Intensity Normalization successful.")
        save_parameters(params, transform_dir / "intensity_normalization_params.json")
        return params, True
    except Exception as e: logger.error(f"Intensity Normalization failed: {e}", exc_info=True); params["error"] = str(e); save_parameters(params, transform_dir / "intensity_normalization_params.json"); return params, False

def run_bias_field_correction_and_save(input_file_path, output_file_path, transform_dir):
    logger.info(f"Running N4 Bias Field Correction on {input_file_path.name}...")
    params = {"method": "N4BiasFieldCorrection", "shrinkFactor": 4, "mask_method": "LiThreshold", "output_log_bias_field_path": None, "output_bias_field_path": None}
    log_bias_field_path = transform_dir / "log_bias_field.nii.gz"; bias_field_path = transform_dir / "bias_field.nii.gz"
    try:
        log_bias_field_sitk = original_bias_field_correction(str(input_file_path), str(output_file_path))
        sitk.WriteImage(log_bias_field_sitk, str(log_bias_field_path)); params["output_log_bias_field_path"] = str(log_bias_field_path.resolve())
        bias_field_sitk = sitk.Exp(log_bias_field_sitk); sitk.WriteImage(bias_field_sitk, str(bias_field_path)); params["output_bias_field_path"] = str(bias_field_path.resolve())
        logger.info("N4 Bias Field Correction successful.")
        save_parameters(params, transform_dir / "bias_field_correction_params.json")
        return params, True
    except Exception as e: logger.error(f"N4 Bias Field Correction failed: {e}", exc_info=True); params["error"] = str(e); save_parameters(params, transform_dir / "bias_field_correction_params.json"); return params, False

def run_registration_and_save(input_file_path, output_file_path, template_file_path, transform_dir, transform_prefix):
    logger.info(f"Running ANTs Registration on {input_file_path.name}...")
    params = {"method": "ANTs Registration", "type_of_transform": "SyN", "template_file": str(template_file_path.resolve()), "output_prefix": str(transform_dir / transform_prefix), "forward_transforms_paths": [], "inverse_transforms_paths": []}
    try:
        fwd_transforms_orig_paths, inv_transforms_orig_paths = original_registration(str(input_file_path), str(output_file_path), str(template_file_path), params["output_prefix"])
        affine_fwd = transform_dir / f"{transform_prefix}0GenericAffine.mat"; warp_fwd = transform_dir / f"{transform_prefix}1Warp.nii.gz"; warp_inv = transform_dir / f"{transform_prefix}1InverseWarp.nii.gz"
        if affine_fwd.exists(): params["forward_transforms_paths"].append(str(affine_fwd.resolve()))
        if warp_fwd.exists(): params["forward_transforms_paths"].append(str(warp_fwd.resolve()))
        if warp_inv.exists(): params["inverse_transforms_paths"].append(str(warp_inv.resolve()))
        if affine_fwd.exists(): params["inverse_transforms_paths"].append(str(affine_fwd.resolve()))
        if not params["forward_transforms_paths"]: logger.warning(f"Не удалось найти файлы трансформаций ANTs по префиксу {params['output_prefix']}")
        logger.info("ANTs Registration successful.")
        save_parameters(params, transform_dir / "registration_params.json")
        return params, True
    except Exception as e: logger.error(f"ANTs Registration failed: {e}", exc_info=True); params["error"] = str(e); save_parameters(params, transform_dir / "registration_params.json"); return params, False

def run_skull_stripping_and_save(input_file_path, final_output_file_path, transform_dir, transform_prefix):
    """Обертка для удаления черепа с сохранением маски."""
    logger.info(f"Running Skull Stripping (BET) on {input_file_path.name}...")
    params = {"method": "FSL BET", "inputs.frac": 0.5, "inputs.robust": True, "output_mask_path": None, "output_stripped_path": None}
    # Базовое имя для временных файлов BET (без расширения)
    bet_output_base_str = str(transform_dir / f"{transform_prefix}_bet_temp")
    # Целевой путь для маски (с расширением)
    mask_output_path = transform_dir / f"{transform_prefix}_brain_mask.nii.gz"

    try:
        stripped_path_bet_str, mask_path_bet_str = original_skull_stripping(str(input_file_path), bet_output_base_str)

        if stripped_path_bet_str is None or mask_path_bet_str is None:
             raise RuntimeError("original_skull_stripping returned None, BET likely failed internally.")

        generated_stripped_path = Path(stripped_path_bet_str)
        generated_mask_path = Path(mask_path_bet_str)

        # Проверка существования файлов (уже сделана в original_skull_stripping)
        # if not generated_stripped_path.exists(): ...
        # if not generated_mask_path.exists(): ...

        # Перемещаем маску
        if generated_mask_path != mask_output_path:
            logger.debug(f"Moving BET mask from {generated_mask_path} to {mask_output_path}")
            shutil.move(str(generated_mask_path), str(mask_output_path))
        else:
             logger.debug(f"BET mask already at correct location: {mask_output_path}")
        params["output_mask_path"] = str(mask_output_path.resolve())

        # Перемещаем основной результат в финальный путь
        final_output_path_obj = Path(final_output_file_path)
        final_output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if generated_stripped_path != final_output_path_obj:
            logger.debug(f"Moving BET stripped image from {generated_stripped_path} to {final_output_path_obj}")
            shutil.move(str(generated_stripped_path), str(final_output_path_obj))
        else:
             logger.debug(f"BET stripped image already at correct location: {final_output_path_obj}")
        params["output_stripped_path"] = str(final_output_path_obj.resolve())

        logger.info("Skull Stripping successful.")
        save_parameters(params, transform_dir / "skull_stripping_params.json")
        return params, True
    except Exception as e:
        logger.error(f"Skull Stripping failed in wrapper: {e}", exc_info=True)
        params["error"] = str(e); save_parameters(params, transform_dir / "skull_stripping_params.json")
        # Удаляем временные файлы, если они остались
        if 'bet_output_base_str' in locals():
             path_strip_gz = Path(bet_output_base_str + ".nii.gz")
             path_strip_noext = Path(bet_output_base_str) # На случай, если output_type не сработал
             path_mask = Path(bet_output_base_str + "_mask.nii.gz")
             if path_strip_gz.exists(): os.remove(path_strip_gz)
             if path_strip_noext.exists(): os.remove(path_strip_noext)
             if path_mask.exists(): os.remove(path_mask)
        return params, False


# === Функция сохранения параметров (Без изменений) ===
def save_parameters(params_dict, output_path):
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        params_serializable = {}
        for key, value in params_dict.items():
             if isinstance(value, Path): params_serializable[key] = str(value)
             elif isinstance(value, list) and value and isinstance(value[0], (Path, str)): params_serializable[key] = [str(p) for p in value]
             else: params_serializable[key] = value
        with open(output_path, 'w') as f: json.dump(params_serializable, f, indent=4)
        logger.debug(f"Parameters saved to: {output_path}")
    except Exception as e: logger.error(f"Failed to save parameters to {output_path}: {e}", exc_info=True)

# === Основная функция обработки (Без изменений) ===
def process_nifti_files(input_root, output_root_prep, output_root_tfm, template_path_str, keep_intermediate=False):
    input_root = Path(input_root); output_root_prep = Path(output_root_prep)
    output_root_tfm = Path(output_root_tfm); template_path = Path(template_path_str)
    if not template_path.is_file(): logger.error(f"Template file not found: {template_path_str}"); return
    processed_files_count = 0; error_files_count = 0
    nii_files = sorted(glob.glob(os.path.join(input_root, 'sub-*/ses-*/**/anat/*.nii.gz'), recursive=True))
    if not nii_files: nii_files = sorted(glob.glob(os.path.join(input_root, 'sub-*/**/anat/*.nii.gz'), recursive=True))
    if not nii_files: logger.warning(f"No NIfTI files found matching pattern in {input_root}"); return
    logger.info(f"Found {len(nii_files)} NIfTI files to process.")

    for input_nifti_file_str in nii_files:
        input_nifti_file_path = Path(input_nifti_file_str)
        logger.info(f"--- Processing File: {input_nifti_file_str} ---")
        overall_params = {"input_file": str(input_nifti_file_path.resolve()), "steps": {}}
        file_had_error = False
        temp_dir = None
        try:
            final_prep_file_path, transform_dir = create_output_paths(input_nifti_file_str, input_root, output_root_prep, output_root_tfm)
            transform_prefix = input_nifti_file_path.name.replace(".nii.gz", "").replace(".nii", "")
            params_json_path = transform_dir / f"{transform_prefix}_processing_parameters.json"
            log_file_path = transform_dir / f"{transform_prefix}_processing_log.txt"
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{transform_prefix}_", dir=transform_dir))
            logger.debug(f"Created temporary directory: {temp_dir}")
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s', datefmt='%H:%M:%S')
            file_handler.setFormatter(file_formatter); file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            current_input_path = input_nifti_file_path
            step_success = True
            # 1. Intensity Normalization
            step_output_path = temp_dir / f"{transform_prefix}_norm.nii.gz"
            norm_params, step_success = run_intensity_normalization_and_save(current_input_path, step_output_path, template_path, transform_dir)
            overall_params["steps"]["1_normalization"] = norm_params
            if not step_success: raise RuntimeError("Intensity Normalization failed")
            current_input_path = step_output_path
            # 2. Bias Field Correction
            step_output_path = temp_dir / f"{transform_prefix}_norm_biascorr.nii.gz"
            bias_params, step_success = run_bias_field_correction_and_save(current_input_path, step_output_path, transform_dir)
            overall_params["steps"]["2_bias_correction"] = bias_params
            if not step_success: raise RuntimeError("Bias Field Correction failed")
            current_input_path = step_output_path
            # 3. Registration
            step_output_path = temp_dir / f"{transform_prefix}_norm_biascorr_reg.nii.gz"
            ants_tfm_prefix = f"{transform_prefix}_reg"
            reg_params, step_success = run_registration_and_save(current_input_path, step_output_path, template_path, transform_dir, ants_tfm_prefix)
            overall_params["steps"]["3_registration"] = reg_params
            if not step_success: raise RuntimeError("Registration failed")
            current_input_path = step_output_path
            # 4. Skull Stripping
            step_output_path = final_prep_file_path
            bet_tfm_prefix = f"{transform_prefix}_strip"
            bet_params, step_success = run_skull_stripping_and_save(current_input_path, step_output_path, transform_dir, bet_tfm_prefix)
            overall_params["steps"]["4_skull_stripping"] = bet_params
            if not step_success: raise RuntimeError("Skull Stripping failed")
            processed_files_count += 1
            logger.info(f"--- Successfully processed file: {input_nifti_file_str} ---")
        except Exception as e:
            file_had_error = True; error_files_count += 1
            logger.error(f"--- FAILED to process file: {input_nifti_file_str} ---")
            logger.error(f"Error during pipeline step: {e}", exc_info=False)
        finally:
            save_parameters(overall_params, params_json_path)
            if 'file_handler' in locals() and file_handler in logger.handlers:
                 logger.removeHandler(file_handler); file_handler.close()
            if temp_dir and temp_dir.exists() and not keep_intermediate:
                try: shutil.rmtree(temp_dir); logger.debug(f"Removed temporary directory: {temp_dir}")
                except Exception as e: logger.warning(f"Could not remove temporary directory {temp_dir}: {e}")
            elif temp_dir: logger.info(f"Keeping intermediate files in: {temp_dir}")

    logger.info("="*40); logger.info(f"Preprocessing Summary:")
    logger.info(f"Total files found: {len(nii_files)}"); logger.info(f"Successfully processed: {processed_files_count}")
    logger.info(f"Failed: {error_files_count}"); logger.info("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NIfTI MRI data (Original Logic + Saving Transforms).")
    parser.add_argument("input_dir", type=str, help="Root directory of the input BIDS dataset (NIfTI files).")
    parser.add_argument("output_dir_prep", type=str, help="Root directory for final preprocessed NIfTI files.")
    parser.add_argument("output_dir_tfm", type=str, help="Root directory for transformation files and parameters.")
    parser.add_argument("--template", type=str, required=True, help="Path to the T1 MNI template NIfTI file.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set console logging level.")
    parser.add_argument("--keep_intermediate", action='store_true', help="Keep intermediate files in the transformation directory.")
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    process_nifti_files(args.input_dir, args.output_dir_prep, args.output_dir_tfm, args.template, args.keep_intermediate)