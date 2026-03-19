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
    print("\n=== Создание тестового датасета ===")
    dataset_id = await create_dataset(
        token=token,
        user_id=user_id,
        user_type_id=user_type_id,
        dataset_name="test_mri_pipeline_bids",
        dataset_short_info="Тестовый датасет для проверки интеграции",
        dataset_tags="test,mri,pipeline",
    )

    if dataset_id is None:
        print("Ошибка создания датасета!")
        return

    print(f"Датасет создан, ID: {dataset_id}")

    # --- 3. Загрузка сущности ---
    print("\n=== Загрузка тестовой сущности ===")

    # Ищем любой файл в demo_workspace для теста
    demo_dir = Path(__file__).parent.parent / "demo_workspace" / "input"
    test_files = list(demo_dir.rglob("*.json"))[:1]  # берём первый JSON

    if not test_files:
        # Если JSON нет, создадим тестовый файл
        test_file = Path(__file__).parent / "test_entity.json"
        test_file.write_text('{"test": true, "source": "mri_ai_service"}')
        test_files = [test_file]
        print(f"Тестовый файл создан: {test_file}")
    else:
        print(f"Найден файл для теста: {test_files[0]}")

    entity_response = await upload_entity(
        token=token,
        user_id=user_id,
        user_type_id=user_type_id,
        dataset_id=dataset_id,
        entity_name="test_sub-001_ses-001",
        file_paths=test_files,
        entity_info={
            "patient_id": "sub-001",
            "session_id": "ses-001",
            "pipeline_stage": "bids_organized",
            "test": True,
        },
    )

    if entity_response is None:
        print("Ошибка загрузки сущности!")
        return

    print(f"Сущность загружена, ответ: {entity_response}")

    print("\n=== Тест завершён успешно ===")


if __name__ == "__main__":
    asyncio.run(main())