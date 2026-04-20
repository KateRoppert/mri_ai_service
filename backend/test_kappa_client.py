"""
Тест загрузки результатов пайплайна в Kappa через KappaUploader.
Запуск: python test_kappa_client.py
"""
import asyncio
import getpass

from kappa_auth import kappa_login, get_session
from kappa_uploader import KappaUploader


async def main():
    # --- Авторизация ---
    print("=== Авторизация в Kappa ===")
    login_id = input("Логин: ").strip()
    passwd = getpass.getpass("Пароль: ").strip()

    result = await kappa_login(login_id, passwd)
    if not result:
        print("Ошибка авторизации!")
        return

    session = get_session(result["session_id"])
    print(f"OK: user_id={session['user_id']}")

    # --- Создаём uploader для демо-запуска ---
    output_path = "/home/ubuntu/mri_ai_service/demo_workspace/input/19_03_1812"
    run_id = "test_upload_19_03_v6"

    uploader = KappaUploader(
        run_id=run_id,
        output_path=output_path,
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
    )

    # --- Загружаем все папки ---
    print("\n=== Загрузка всех папок ===")
    results = await uploader.upload_all_new()

    print("\n=== Результаты ===")
    for folder, count in results.items():
        print(f"  {folder}: {count} сущностей загружено")

    if not results:
        print("  Ничего не загружено (возможно, всё уже было загружено ранее)")

    # --- Повторный вызов (должен вернуть 0 — всё уже загружено) ---
    print("\n=== Повторная проверка (должно быть 0) ===")
    results2 = await uploader.upload_all_new()
    print(f"  Повторно загружено: {sum(results2.values())} сущностей")

    print("\n=== Тест завершён ===")

# async def main():
#     print("=== Авторизация в Kappa ===")
#     login_id = input("Логин: ").strip()
#     passwd = getpass.getpass("Пароль: ").strip()

#     result = await kappa_login(login_id, passwd)
#     if not result:
#         print("Ошибка авторизации!")
#         return

#     session = get_session(result["session_id"])
#     print(f"OK: user_id={session['user_id']}")

#     output_path = "/home/ubuntu/mri_ai_service/demo_workspace/input/19_03_1812"
#     run_id = "test_debug_01"

#     uploader = KappaUploader(
#         run_id=run_id,
#         output_path=output_path,
#         token=session["kappa_token"],
#         user_id=session["user_id"],
#         user_type_id=session["user_type_id"],
#     )

#     # Тестируем только одну папку — metadata (маленькая, JSON)
#     print("\n=== Тест одной папки: metadata ===")

#     folder_path = uploader.output_path / "metadata"
#     print(f"  Папка существует: {folder_path.exists()}")

#     sessions = uploader._discover_sessions(folder_path)
#     print(f"  Найдено сессий: {len(sessions)}")
#     for key, files in sessions.items():
#         print(f"    {key}: {len(files)} файлов")

#     dataset_id = await uploader._ensure_dataset("metadata")
#     print(f"  dataset_id: {dataset_id}")

#     if dataset_id:
#         count = await uploader.upload_folder("metadata")
#         print(f"  Загружено сущностей: {count}")
#     else:
#         print("  ОШИБКА: dataset_id = None, загрузка пропущена")

#     print("\n=== Тест завершён ===")

if __name__ == "__main__":
    asyncio.run(main())