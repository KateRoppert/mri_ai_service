import os
import shutil
import pydicom
from collections import defaultdict

def is_dicom_file(file_path):
    """Проверяет, является ли файл валидным DICOM-файлом"""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False

def determine_modality(ds, file_path):
    """Автоматически определяет модальность используя комбинацию метаданных DICOM и анализа пути"""
    # Анализ DICOM тегов
    series_desc = ds.get('SeriesDescription', '').lower()
    protocol = ds.get('ProtocolName', '').lower()
    
    # Ключевые слова для модальностей
    modality_keywords = {
        't1c': ['t1c', 't1+c', 't1-ce', 't1contrast', 't1gd', 'contrast'],
        't1': ['t1', 't1w', 't1-weighted', 't1weighted', 'spgr', 'mprage'],
        't2fl': ['t2fl', 't2-flair', 'flair', 't2flair'],
        't2': ['t2', 't2w', 't2-weighted', 't2weighted', 'tse']
    }

    # Проверка тегов DICOM
    for modality, keys in modality_keywords.items():
        if any(key in series_desc or key in protocol for key in keys):
            return modality

    # Анализ пути к файлу
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        part_lower = part.lower()
        for modality, keys in modality_keywords.items():
            if any(key in part_lower for key in keys):
                return modality

    # Анализ параметров сканирования
    try:
        tr = float(ds.get('RepetitionTime', 0))
        te = float(ds.get('EchoTime', 0))
        
        if 300 < tr < 800 and te < 30:
            return 't1'
        elif 2000 < tr < 5000 and te > 80:
            return 't2'
    except:
        pass

    return 'unknown'

def organize_dicom_to_bids(input_dir, output_dir='bids_data_dicom'):
    # Создаем корневую выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Обрабатываем пациентов (первый уровень вложенности)
    for patient_idx, patient_folder in enumerate(sorted(os.listdir(input_dir)), 1):
        patient_path = os.path.join(input_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue
        
        sub_id = f"sub-{patient_idx:03d}"
        sub_path = os.path.join(output_dir, sub_id)
        
        # Обрабатываем сессии (второй уровень вложенности)
        for session_idx, session_folder in enumerate(sorted(os.listdir(patient_path)), 1):
            session_path = os.path.join(patient_path, session_folder)
            if not os.path.isdir(session_path):
                continue
            
            ses_id = f"ses-{session_idx:03d}"
            ses_bids_path = os.path.join(sub_path, ses_id, 'anat')
            
            # Собираем все DICOM-файлы в сессии
            dcm_files = []
            for root, _, files in os.walk(session_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_dicom_file(file_path):
                        dcm_files.append(file_path)
            
            # Группируем файлы по модальности
            modality_groups = defaultdict(list)
            
            for file_path in dcm_files:
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    modality = determine_modality(ds, file_path)
                    
                    if modality == 'unknown':
                        print(f"Не удалось определить модальность для файла: {file_path}")
                        continue
                        
                    modality_groups[modality].append(file_path)
                except Exception as e:
                    print(f"Ошибка обработки файла {file_path}: {str(e)}")
                    continue
            
            # Копируем файлы в BIDS-структуру
            for modality, files in modality_groups.items():
                modality_path = os.path.join(ses_bids_path, modality)
                os.makedirs(modality_path, exist_ok=True)
                
                for idx, src_file in enumerate(files, 1):
                    dst_file = os.path.join(
                        modality_path,
                        f"{sub_id}_{ses_id}_{modality}_{idx:03d}.dcm"
                    )
                    shutil.copy(src_file, dst_file)
    
    print("Конвертация в BIDS формат завершена!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BIDS организация DICOM данных')
    parser.add_argument('input_dir', help='Входная директория с сырыми данными')
    parser.add_argument('--output_dir', default='bids_data_dicom', help='Выходная BIDS директория')
    args = parser.parse_args()
    
    organize_dicom_to_bids(args.input_dir, args.output_dir)
