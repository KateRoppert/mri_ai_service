"""
Тестовый скрипт для проверки kappa_client.
Запуск: python test_kappa_client.py
"""
import asyncio
import getpass
import httpx
from pathlib import Path

from kappa_auth import kappa_login
from kappa_client import create_dataset, upload_entity


async def main():
    # --- 1. Авторизация ---
    print("=== Авторизация в Kappa ===")
    login_id = input("Логин: ").strip()
    passwd = getpass.getpass("Пароль: ").strip()

    result = await kappa_login(login_id, passwd)
    if not result:
        print("Ошибка авторизации!")
        return

    print(f"Авторизация успешна: {result['first_name']} {result['last_name']}")

    # Достаём данные сессии для API-запросов
    from kappa_auth import get_session
    session = get_session(result["session_id"])
    token = session["kappa_token"]
    user_id = session["user_id"]
    user_type_id = session["user_type_id"]

    print(f"user_id={user_id}, user_type_id={user_type_id}")

    # --- 2. Создание датасета ---
    print("\n=== Создание тестового датасета (BIDS DICOM) ===")
    dataset_id = await create_dataset(
        token=token,
        user_id=user_id,
        user_type_id=user_type_id,
        dataset_name="test_bids_dicom_session_1918",
        dataset_short_info="Тест загрузки DICOM сессии из BIDS",
        dataset_type=1,
        dataset_tags="test,mri,dicom,bids",
    )

    if dataset_id is None:
        print("Ошибка создания датасета!")
        return

    print(f"Датасет создан, ID: {dataset_id}")

    # --- 3. Загрузка DICOM сессии как zip-архива ---
    print("\n=== Загрузка DICOM сессии (zip) ===")

    session_dir = Path("/home/ubuntu/mri_ai_service/demo_workspace/input/13_03_1850/bids_organized/sub-001/ses-001")
    dicom_files = sorted(session_dir.rglob("*.dcm"))
    modalities = sorted(set(f.parent.name for f in dicom_files))

    print(f"Файлов: {len(dicom_files)}, модальности: {modalities}")

    resp = await upload_entity(
        token=token, user_id=user_id, user_type_id=user_type_id,
        dataset_id=dataset_id,
        entity_name="sub-001_ses-001",
        file_paths=dicom_files,
        zip_as_archive=True,
        entity_info={
            "patient_id": "sub-001",
            "session_id": "ses-001",
            "pipeline_stage": "01_bids_organized",
            "modalities": modalities,
            "file_count": len(dicom_files),
            "archive_format": "zip",
            "files_per_modality": {
                mod: len([f for f in dicom_files if f.parent.name == mod])
                for mod in modalities
            },
        },
    )

    if resp:
        print(f"DICOM zip загружен: {resp}")
    else:
        print("Ошибка загрузки DICOM zip!")

    # --- 4. NIfTI (без архива, напрямую) ---
    print("\n=== Загрузка NIfTI ===")
    nifti_dir = Path("/home/ubuntu/mri_ai_service/demo_workspace/input/13_03_1850/nifti/sub-001/ses-001")
    nifti_files = sorted(nifti_dir.rglob("*.nii.gz")) if nifti_dir.exists() else []

    if nifti_files:
        nifti_dataset_id = await create_dataset(
            token=token, user_id=user_id, user_type_id=user_type_id,
            dataset_name="test_nifti_session",
            dataset_short_info="NIfTI файлы после конвертации",
            dataset_type=1,
            dataset_tags="test,mri,nifti",
        )
        print(f"NIfTI датасет ID: {nifti_dataset_id}")

        resp = await upload_entity(
            token=token, user_id=user_id, user_type_id=user_type_id,
            dataset_id=nifti_dataset_id,
            entity_name="sub-001_ses-001",
            file_paths=nifti_files,
            entity_info={
                "patient_id": "sub-001",
                "session_id": "ses-001",
                "pipeline_stage": "03_nifti_conversion",
                "modalities": [f.stem.split("_")[-1] for f in nifti_files],
                "file_count": len(nifti_files),
            },
        )
        print(f"NIfTI: {'OK — ' + resp if resp else 'FAIL'}")

    print("\n=== Тесты завершены ===")


if __name__ == "__main__":
    asyncio.run(main())