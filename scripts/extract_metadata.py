import os
import pydicom
import json

def clean_element(element):
    """Рекурсивная очистка элементов DICOM от бинарных данных"""
    if isinstance(element, dict):
        # Обрабатываем DICOM элементы с VR и Value
        if 'vr' in element and 'Value' in element:
            vr = element['vr']
            value = element['Value']
            
            if vr == 'SQ':  # Обработка последовательностей
                return {
                    'vr': vr,
                    'Value': [clean_element(item) for item in value]
                }
            else:  # Обработка обычных элементов
                return {
                    'vr': vr,
                    'Value': [
                        "BINARY_DATA" if isinstance(v, bytes) else v
                        for v in value
                    ]
                }
        return {k: clean_element(v) for k, v in element.items()}
    elif isinstance(element, list):
        return [clean_element(item) for item in element]
    return element

def extract_metadata(dicom_dir, output_dir):
    """Извлечение метаданных с сохранением BIDS-структуры"""
    for root, _, files in os.walk(dicom_dir):
        for file in files:
            if file.startswith('.'):
                continue
            
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, dicom_dir)
            json_path = os.path.join(
                output_dir, 
                os.path.splitext(rel_path)[0] + '_meta.json'
            )
            
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            try:
                ds = pydicom.dcmread(src_path, stop_before_pixels=True)
                meta_dict = ds.to_json_dict()
                cleaned_meta = {tag: clean_element(el) for tag, el in meta_dict.items()}
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_meta, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Ошибка обработки {src_path}: {str(e)}")

input_dir = "bids_data_dicom"
output_dir = "dicom_metadata"
extract_metadata(input_dir, output_dir)
