import os
import subprocess
import shutil
import traceback
from datetime import datetime

def run_mriqc(report_type):
    bids_dir = "bids_data_nifti"
    output_dir = "mriqc_output"
    base_work_dir = "mriqc_work"
    error_log = "mriqc_failed_subjects.log"

    os.makedirs(output_dir, exist_ok=True)

    # очищаем лог перед запуском
    with open(error_log, "w") as f:
        f.write("Список участников с ошибками при обработке MRIQC:\n\n")

    if report_type in ["individual", "both"]:
        subjects = [d for d in os.listdir(bids_dir) if d.startswith("sub-")]
        for subject in subjects:
            print(f"\n▶ Обработка {subject}")
            work_dir = os.path.join(base_work_dir, subject)
            os.makedirs(work_dir, exist_ok=True)

            try:
                subprocess.run([
                    "mriqc", bids_dir, output_dir, "participant",
                    "--participant_label", subject.replace("sub-", ""),
                    "--no-sub", "--nprocs", "1", "--mem", "16G",
                    "--work-dir", work_dir
                ], check=True)
                print(f"✅ Успешно завершена: {subject}")
            except subprocess.CalledProcessError as e:
                error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"❌ MRIQC завершился с ошибкой на {subject}, продолжаю...")
                with open(error_log, "a") as f:
                    f.write(f"[{error_time}] {subject} — Ошибка выполнения MRIQC\n")
                    f.write(f"  Код возврата: {e.returncode}\n\n")
            except Exception as e:
                error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"❌ Неожиданная ошибка при обработке {subject}: {str(e)}")
                with open(error_log, "a") as f:
                    f.write(f"[{error_time}] {subject} — Неизвестная ошибка:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")

            # Удаляем временную рабочую директорию для участника
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)

    if report_type in ["group", "both"]:
        print("\n▶ Генерация группового отчёта")
        try:
            subprocess.run([
                "mriqc", bids_dir, output_dir, "group", "--no-sub",
                "--work-dir", base_work_dir
            ], check=True)
            print("✅ Групповой отчёт успешно создан")
        except subprocess.CalledProcessError as e:
            error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("❌ Ошибка при генерации группового отчёта")
            with open(error_log, "a") as f:
                f.write(f"[{error_time}] Ошибка при генерации группового отчёта\n")
                f.write(f"  Код возврата: {e.returncode}\n\n")

if __name__ == "__main__":
    report_type = input("Выберите тип отчёта (individual/group/both): ").strip().lower()
    run_mriqc(report_type)

    print("Процесс MRIQC завершён! Отчёты и данные сохранены.")