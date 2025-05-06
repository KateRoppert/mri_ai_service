import os
import subprocess
import shutil

def bids_modality_from_folder(folder_name):
    name = folder_name.lower()
    if 't1c' in name:
        return 'ce-gd_T1w'
    elif 't2fl' in name:
        return 'FLAIR'
    elif 't1' in name and 'c' not in name:
        return 'T1w'
    elif 't2' in name:
        return 'T2w'
    else:
        return None

def convert_bids_dicom_to_nifti(bids_dicom_dir, bids_nifti_dir, dcm2niix_path='dcm2niix'):
    for root, dirs, files in os.walk(bids_dicom_dir):
        dicom_files = [f for f in files if f.endswith('.dcm')]
        if not dicom_files:
            continue

        rel_path = os.path.relpath(root, bids_dicom_dir)
        path_parts = rel_path.split(os.sep)

        # Ждём структуру: sub-XXX/ses-XXX/anat/<modality>
        if len(path_parts) < 4 or path_parts[-2] != 'anat':
            print(f"[SKIP] Неизвестная структура: {rel_path}")
            continue

        sub_id = path_parts[0]
        ses_id = path_parts[1]
        modality_folder = path_parts[-1]
        bids_modality = bids_modality_from_folder(modality_folder)

        if bids_modality is None:
            print(f"[SKIP] Не удалось определить модальность из {modality_folder}")
            continue

        output_path = os.path.join(bids_nifti_dir, sub_id, ses_id, 'anat')
        os.makedirs(output_path, exist_ok=True)

        # Временное имя файла, чтобы потом переименовать
        temp_prefix = "tmp_output"

        cmd = [
            dcm2niix_path,
            '-z', 'y',
            '-o', output_path,
            '-f', temp_prefix,
            root
        ]

        print(f"[INFO] Конвертация {root} → {output_path}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ошибка при конвертации {root}: {e}")
            continue

        # Формируем новое имя
        new_base = f"{sub_id}_{ses_id}_{bids_modality}"
        temp_nii = os.path.join(output_path, f"{temp_prefix}.nii.gz")
        temp_json = os.path.join(output_path, f"{temp_prefix}.json")
        final_nii = os.path.join(output_path, f"{new_base}.nii.gz")
        final_json = os.path.join(output_path, f"{new_base}.json")

        if os.path.exists(temp_nii):
            shutil.move(temp_nii, final_nii)
            print(f"[RENAME] {temp_nii} → {final_nii}")
        else:
            print(f"[WARN] Не найден .nii.gz файл после конвертации в {root}")

        if os.path.exists(temp_json):
            shutil.move(temp_json, final_json)
            print(f"[RENAME] {temp_json} → {final_json}")

    print("[DONE] Конвертация и переименование завершены.")


convert_bids_dicom_to_nifti(
    bids_dicom_dir='bids_data_dicom',
    bids_nifti_dir='bids_data_nifti'
)