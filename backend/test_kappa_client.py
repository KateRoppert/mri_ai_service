"""
Тестовый скрипт для проверки kappa_client.
Запуск: python test_kappa_client.py
"""
import asyncio
import getpass
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
        dataset_name="test_bids_dicom_session",
        dataset_short_info="Тест загрузки DICOM сессии из BIDS",
        dataset_type=1,
        dataset_tags="test,mri,dicom,bids",
    )

    if dataset_id is None:
        print("Ошибка создания датасета!")
        return

    print(f"Датасет создан, ID: {dataset_id}")

    # --- 3. Загрузка сущности: DICOM сессия (все модальности) ---
    print("\n=== Загрузка DICOM сессии как сущности ===")

    session_dir = Path("/home/ubuntu/mri_ai_service/demo_workspace/input/13_03_1850/bids_organized/sub-001/ses-001")

    if not session_dir.exists():
        print(f"Директория не найдена: {session_dir}")
        return

    # Собираем все .dcm рекурсивно из всех модальностей
    dicom_files = sorted(session_dir.rglob("*.dcm"))

    # Определяем модальности (имена подпапок в anat/)
    modalities = sorted(set(f.parent.name for f in dicom_files))

    print(f"Найдено файлов: {len(dicom_files)}")
    print(f"Модальности: {modalities}")
    for mod in modalities:
        mod_files = [f for f in dicom_files if f.parent.name == mod]
        print(f"  {mod}: {len(mod_files)} файлов")

    entity_response = await upload_entity(
        token=token,
        user_id=user_id,
        user_type_id=user_type_id,
        dataset_id=dataset_id,
        entity_name="sub-001_ses-001",
        file_paths=dicom_files,
        entity_info={
            "patient_id": "sub-001",
            "session_id": "ses-001",
            "pipeline_stage": "01_bids_organized",
            "modalities": modalities,
            "source_dir": str(session_dir),
            "file_count": len(dicom_files),
            "files_per_modality": {
                mod: len([f for f in dicom_files if f.parent.name == mod])
                for mod in modalities
            },
        },
    )

    if entity_response is None:
        print("Ошибка загрузки DICOM сущности!")
        return

    print(f"DICOM сущность загружена: {entity_response}")

    # --- 4. Тест загрузки NIfTI (все модальности одной сессии) ---
    print("\n=== Загрузка NIfTI файлов ===")

    nifti_session_dir = Path("/home/ubuntu/mri_ai_service/demo_workspace/input/13_03_1850/nifti/sub-001/ses-001")

    if not nifti_session_dir.exists():
        print(f"NIfTI директория не найдена: {nifti_session_dir}, пропускаем")
    else:
        nifti_files = sorted(nifti_session_dir.rglob("*.nii.gz"))
        print(f"NIfTI файлов: {len(nifti_files)}")
        for f in nifti_files:
            print(f"  {f.name} ({f.stat().st_size} bytes)")

        nifti_dataset_id = await create_dataset(
            token=token,
            user_id=user_id,
            user_type_id=user_type_id,
            dataset_name="test_nifti_converted",
            dataset_short_info="Тест загрузки NIfTI после конвертации",
            dataset_type=1,
            dataset_tags="test,mri,nifti",
        )

        if nifti_dataset_id is None:
            print("Ошибка создания NIfTI датасета!")
        else:
            print(f"NIfTI датасет создан, ID: {nifti_dataset_id}")

            nifti_response = await upload_entity(
                token=token,
                user_id=user_id,
                user_type_id=user_type_id,
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

            if nifti_response:
                print(f"NIfTI сущность загружена: {nifti_response}")
            else:
                print("Ошибка загрузки NIfTI сущности!")

    print("\n=== Тест завершён ===")


if __name__ == "__main__":
    asyncio.run(main())