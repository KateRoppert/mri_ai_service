import os
import json
import subprocess
import argparse

def create_dataset_description(bids_dir):
    """Создание обязательного файла dataset_description.json для BIDS"""
    description = {
        "Name": "Brain Tumour Dataset",
        "BIDSVersion": "1.8.0",
        "License": "CC0",
        "Authors": ["LAPDIMO"],
        "DatasetDOI": "10.1234/example.doi"
    }
    
    os.makedirs(bids_dir, exist_ok=True)
    output_path = os.path.join(bids_dir, "dataset_description.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(description, f, indent=2, ensure_ascii=False)
    
    print(f"Файл {output_path} успешно создан")

def validate_bids(bids_dir):
    """Проверка BIDS-валидатором"""
    try:
        result = subprocess.run(
            ["bids-validator", bids_dir],
            capture_output=True,
            text=True,
            check=True
        )
        print("\nРезультат проверки:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("\nНайдены ошибки в структуре:")
        print(e.stderr)
        
    except FileNotFoundError:
        print("Ошибка: bids-validator не установлен")
        print("Установите через: npm install -g bids-validator")

bids_dir = "bids_data_nifti"

create_dataset_description(bids_dir)
validate_bids(bids_dir)
