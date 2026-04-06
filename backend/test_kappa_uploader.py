"""
Тест загрузки итоговых результатов в Kappa через рефакторенный KappaUploader.
Запуск: python test_kappa_uploader.py
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

    # --- Параметры ---
    output_path = "/home/ubuntu/mri_ai_service/demo_workspace/input/19_03_1812"
    preprocessing_config = "/home/ubuntu/mri_ai_service/configs/preprocessing_config.yaml"

    # --- Создаём uploader ---
    uploader = KappaUploader(
        run_id="test_refactored_v1",
        output_path=output_path,
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        lesion_type="glioblastoma",
        preprocessing_config_path=preprocessing_config,
    )

    print(f"\nPreprocessing ID: {uploader.preprocessing_id}")

    # --- Тест discover_sessions ---
    print("\n=== Обнаружение сессий ===")
    sessions = uploader._discover_sessions()
    for sk, sd in sessions.items():
        print(f"\n  Session: {sk}")
        print(f"    Preprocessed: {[f.name for f in sd['preprocessed']]}")
        print(f"    Masks: {[f.name for f in sd['masks']]}")
        print(f"    Quality: {[f.name for f in sd['quality_reports']]}")
        print(f"    Volume: {sd['volume_report'].name if sd['volume_report'] else None}")
        print(f"    Lobar: {sd['lobar_report'].name if sd['lobar_report'] else None}")

    # --- Тест build_entity_info ---
    print("\n=== Entity Info ===")
    for sk, sd in sessions.items():
        info = uploader._build_entity_info(sk, sd)
        print(f"\n  {sk}:")
        print(f"    modalities: {info.get('modalities')}")
        print(f"    file_count: {info.get('file_count')}")
        print(f"    study_hash: {info.get('study_hash')}")
        if info.get("quality_reports"):
            for qr in info["quality_reports"]:
                print(f"    quality {qr['modality']}: {qr['quality_score']} ({qr['quality_category']})")
        if info.get("volume_report"):
            vr = info["volume_report"]
            print(f"    tumor total: {vr.get('total_tumor_cm3')} cm3")
            for cls, data in vr.get("classes", {}).items():
                print(f"      {cls}: {data.get('cm3')} cm3")
        if info.get("lobar_report"):
            lr = info["lobar_report"]
            print(f"    lobar total: {lr.get('total_lesion_cm3')} cm3")
            for lobe, data in lr.get("lobes", {}).items():
                print(f"      {lobe}: {data.get('cm3')} cm3 ({data.get('percent')}%)")

    # --- Тест upload ---
    print("\n=== Загрузка в Kappa (dataset_id=133) ===")
    confirm = input("Загрузить? (y/n): ").strip().lower()
    if confirm != "y":
        print("Отменено")
        return

    report = await uploader.upload_results()
    print(f"\n=== Результат ===")
    print(f"  Dataset ID: {report.get('dataset_id')}")
    print(f"  Uploaded: {report.get('uploaded')}/{report.get('total')}")
    for sr in report.get("sessions", []):
        status = "OK" if sr.get("success") else f"FAIL: {sr.get('error')}"
        print(f"  {sr['session']}: {status} ({sr.get('files', 0)} files)")

    print("\n=== Тест завершён ===")


if __name__ == "__main__":
    asyncio.run(main())