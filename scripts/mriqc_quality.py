# import os
# import subprocess
# import shutil
# import traceback
# from datetime import datetime

# def run_mriqc(report_type):
#     bids_dir = "bids_data_nifti"
#     output_dir = "mriqc_output"
#     base_work_dir = "mriqc_work"
#     error_log = "mriqc_failed_subjects.log"

#     os.makedirs(output_dir, exist_ok=True)

#     # очищаем лог перед запуском
#     with open(error_log, "w") as f:
#         f.write("Список участников с ошибками при обработке MRIQC:\n\n")

#     if report_type in ["individual", "both"]:
#         subjects = [d for d in os.listdir(bids_dir) if d.startswith("sub-")]
#         for subject in subjects:
#             print(f"\n▶ Обработка {subject}")
#             work_dir = os.path.join(base_work_dir, subject)
#             os.makedirs(work_dir, exist_ok=True)

#             try:
#                 subprocess.run([
#                     "mriqc", bids_dir, output_dir, "participant",
#                     "--participant_label", subject.replace("sub-", ""),
#                     "--no-sub", "--nprocs", "2", "--mem", "16G",
#                     "--work-dir", work_dir
#                 ], check=True)
#                 print(f"✅ Успешно завершена: {subject}")
#             except subprocess.CalledProcessError as e:
#                 error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 print(f"❌ MRIQC завершился с ошибкой на {subject}, продолжаю...")
#                 with open(error_log, "a") as f:
#                     f.write(f"[{error_time}] {subject} — Ошибка выполнения MRIQC\n")
#                     f.write(f"  Код возврата: {e.returncode}\n\n")
#             except Exception as e:
#                 error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 print(f"❌ Неожиданная ошибка при обработке {subject}: {str(e)}")
#                 with open(error_log, "a") as f:
#                     f.write(f"[{error_time}] {subject} — Неизвестная ошибка:\n")
#                     f.write(traceback.format_exc())
#                     f.write("\n")

#             # Удаляем временную рабочую директорию для участника
#             if os.path.exists(work_dir):
#                 shutil.rmtree(work_dir)

#     if report_type in ["group", "both"]:
#         print("\n▶ Генерация группового отчёта")
#         try:
#             subprocess.run([
#                 "mriqc", bids_dir, output_dir, "group", "--no-sub",
#                 "--work-dir", base_work_dir
#             ], check=True)
#             print("✅ Групповой отчёт успешно создан")
#         except subprocess.CalledProcessError as e:
#             error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print("❌ Ошибка при генерации группового отчёта")
#             with open(error_log, "a") as f:
#                 f.write(f"[{error_time}] Ошибка при генерации группового отчёта\n")
#                 f.write(f"  Код возврата: {e.returncode}\n\n")

# if __name__ == "__main__":
#     report_type = input("Выберите тип отчёта (individual/group/both): ").strip().lower()
#     run_mriqc(report_type)

#     print("Процесс MRIQC завершён! Отчёты и данные сохранены.")

import os
import subprocess
import shutil
import traceback
from datetime import datetime
import argparse
import logging
import sys
import shutil
from pathlib import Path

# --- Настройка логгера (без изменений) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path: str):
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(formatter); logger.addHandler(ch)
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG); fh.setFormatter(formatter); logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)

# === ИЗМЕНЕНИЯ В ЭТОЙ ФУНКЦИИ ===
def run_mriqc_analysis(
    bids_dir_str: str,
    output_dir_str: str,
    report_type: str,
    mriqc_path: str = 'mriqc',
    n_procs: int = 1,
    mem_gb: int = 4,
    error_log_path: str = "mriqc_failed_subjects.log",
    subject_labels: list[str] | None = None # <<< ИЗМЕНЕНО: теперь список строк
    ):
    """
    Запускает MRIQC для анализа данных BIDS.

    Args:
        # ... (остальные аргументы без изменений) ...
        subject_labels (list[str] | None): Список ID участников для обработки (например, ['001', 'sub-005']).
                                           Если None, обрабатываются все участники.
    """
    logger.info(f"Запуск анализа MRIQC.")
    logger.info(f"  BIDS директория: {bids_dir_str}")
    # ... (остальные логи параметров) ...
    if subject_labels: # Если передан список
        logger.info(f"  Обработка указанных участников: {', '.join(subject_labels)}")
    else:
        logger.info(f"  Обработка всех участников в BIDS директории.")


    bids_dir = Path(bids_dir_str)
    output_dir = Path(output_dir_str)
    base_work_dir = output_dir / "mriqc_work"
    error_log = Path(error_log_path)

    # --- Проверки (без изменений) ---
    if not bids_dir.is_dir():
        logger.error(f"BIDS директория не найдена: {bids_dir}")
        raise FileNotFoundError(f"BIDS директория не найдена: {bids_dir}")
    mriqc_found_path = shutil.which(mriqc_path)
    if mriqc_found_path is None:
        logger.error(f"Исполняемый файл mriqc не найден: '{mriqc_path}'.")
        raise FileNotFoundError(f"mriqc не найден: {mriqc_path}")
    else:
        logger.info(f"Исполняемый файл mriqc найден: {mriqc_found_path}")
        mriqc_path = mriqc_found_path
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log, "w", encoding='utf-8') as f:
            f.write(f"Лог ошибок MRIQC ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
    except OSError as e: logger.error(f"Не удалось создать директории/лог: {e}"); raise

    subjects_processed_successfully = 0
    subjects_failed = 0

    # --- Запуск для индивидуальных участников ---
    if report_type in ["participant", "both"]:
        target_subjects_full_ids = []
        if subject_labels: # Если передан список участников
            for subj_label in subject_labels:
                # Нормализуем имя (добавляем 'sub-' если нет)
                if not subj_label.startswith("sub-"):
                    full_id = f"sub-{subj_label}"
                else:
                    full_id = subj_label

                if not (bids_dir / full_id).is_dir():
                    logger.warning(f"Участник '{full_id}' (из списка --subjects) не найден в {bids_dir}. Пропуск.")
                else:
                    target_subjects_full_ids.append(full_id)
            if not target_subjects_full_ids:
                 logger.warning("Ни один из указанных участников не найден. Индивидуальные отчеты не будут созданы.")
            else:
                 logger.info(f"Будут обработаны участники: {', '.join(target_subjects_full_ids)}")
        else:
            # Если список не передан, берем всех
            target_subjects_full_ids = sorted([d.name for d in bids_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
            if not target_subjects_full_ids:
                logger.warning("Не найдено участников (папок sub-*) для индивидуального анализа.")
            else:
                logger.info(f"Найдено {len(target_subjects_full_ids)} участников для индивидуального анализа.")

        # Обрабатываем выбранных участников
        for subject_id_full in target_subjects_full_ids:
            subject_label_for_mriqc = subject_id_full.replace("sub-", "")
            logger.info(f"--- Обработка участника: {subject_id_full} ---")
            subject_work_dir = base_work_dir / subject_id_full
            try: subject_work_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"  Не удалось создать work_dir для {subject_id_full}: {e}. Пропуск."); subjects_failed += 1
                with open(error_log, "a", encoding='utf-8') as f: f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — ОШИБКА СОЗДАНИЯ WORK DIR\n  Сообщение: {e}\n\n")
                continue

            cmd_participant = [ mriqc_path, str(bids_dir), str(output_dir), "participant",
                                "--participant_label", subject_label_for_mriqc, "--no-sub",
                                "--nprocs", str(n_procs), "--mem_gb", str(mem_gb),
                                "--work-dir", str(subject_work_dir) ]
            logger.debug(f"  Команда (participant): {' '.join(cmd_participant)}")
            try:
                result = subprocess.run(cmd_participant, capture_output=True, text=True, check=False, timeout=3600)
                if result.stdout: logger.debug(f"  MRIQC stdout ({subject_id_full}):\n{result.stdout.strip()}")
                if result.stderr: logger.debug(f"  MRIQC stderr ({subject_id_full}):\n{result.stderr.strip()}")
                if result.returncode == 0:
                    logger.info(f"  Успешно: {subject_id_full}"); subjects_processed_successfully += 1
                else:
                    logger.error(f"  MRIQC ошибка ({subject_id_full}, код: {result.returncode}). См. {error_log.name}."); subjects_failed += 1
                    with open(error_log, "a", encoding='utf-8') as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — ОШИБКА MRIQC\n  Код: {result.returncode}\n")
                        if result.stderr: f.write(f"  stderr (часть):\n  ...\n  {' '.join(result.stderr.strip().splitlines()[-5:])}\n")
                        f.write("\n")
            except subprocess.TimeoutExpired:
                logger.error(f"  TIMEOUT для {subject_id_full}."); subjects_failed += 1
                with open(error_log, "a", encoding='utf-8') as f: f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — ОШИБКА: TIMEOUT\n\n")
            except Exception as e:
                logger.exception(f"  Непредвиденная ошибка ({subject_id_full}): {e}"); subjects_failed += 1
                with open(error_log, "a", encoding='utf-8') as f: f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — НЕИЗВЕСТНАЯ ОШИБКА:\n{traceback.format_exc()}\n\n")
            finally:
                if subject_work_dir.exists():
                    logger.debug(f"  Удаление work_dir: {subject_work_dir}")
                    try: shutil.rmtree(subject_work_dir)
                    except Exception as e_rm: logger.warning(f"  Не удалось удалить {subject_work_dir}: {e_rm}")

    # --- Запуск для группового отчета ---
    group_report_successful = False
    can_run_group = True
    # Не запускаем групповой отчет, если обрабатывали конкретный список участников, и все они провалились
    # или если список был пуст (никого не нашли)
    if subject_labels and (subjects_failed == len(subject_labels) or not target_subjects_full_ids):
        can_run_group = False
        logger.info("Групповой отчет не будет создан: не было успешно обработанных участников из указанного списка или список был пуст.")

    if report_type in ["group", "both"] and can_run_group:
        # Также не запускаем, если вообще не было успешных индивидуальных запусков
        # (это актуально, если subject_labels был None, но все участники упали)
        if not subject_labels and subjects_processed_successfully == 0 and subjects_failed > 0:
             logger.info("Групповой отчет не будет создан: не было успешно обработанных индивидуальных отчетов.")
        else:
            logger.info("--- Генерация группового отчета MRIQC ---")
            work_dir_group = base_work_dir / "group_work"
            try:
                work_dir_group.mkdir(parents=True, exist_ok=True)
                cmd_group = [ mriqc_path, str(bids_dir), str(output_dir), "group",
                              "--no-sub", "--work-dir", str(work_dir_group) ]
                logger.debug(f"  Команда (group): {' '.join(cmd_group)}")
                result = subprocess.run(cmd_group, capture_output=True, text=True, check=False, timeout=1800)
                if result.stdout: logger.debug(f"  MRIQC stdout (group):\n{result.stdout.strip()}")
                if result.stderr: logger.debug(f"  MRIQC stderr (group):\n{result.stderr.strip()}")
                if result.returncode == 0:
                    logger.info("  Групповой отчет MRIQC успешно создан."); group_report_successful = True
                else:
                    logger.error(f"  Ошибка генерации группового отчета (код: {result.returncode}). См. {error_log.name}.")
                    with open(error_log, "a", encoding='utf-8') as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ГРУППОВОЙ ОТЧЕТ — ОШИБКА\n  Код: {result.returncode}\n")
                        if result.stderr: f.write(f"  stderr (часть):\n  ...\n  {' '.join(result.stderr.strip().splitlines()[-5:])}\n")
                        f.write("\n")
            except subprocess.TimeoutExpired:
                logger.error("  TIMEOUT для группового отчета.")
                with open(error_log, "a", encoding='utf-8') as f: f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ГРУППОВОЙ ОТЧЕТ — ОШИБКА: TIMEOUT\n\n")
            except Exception as e:
                logger.exception(f"  Непредвиденная ошибка (групповой отчет): {e}")
                with open(error_log, "a", encoding='utf-8') as f: f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ГРУППОВОЙ ОТЧЕТ — НЕИЗВЕСТНАЯ ОШИБКА:\n{traceback.format_exc()}\n\n")
            finally:
                if work_dir_group.exists():
                    logger.debug(f"  Удаление work_dir (group): {work_dir_group}")
                    try: shutil.rmtree(work_dir_group)
                    except Exception as e_rm: logger.warning(f"  Не удалось удалить {work_dir_group}: {e_rm}")

    # --- Итоговая информация (без изменений) ---
    logger.info("=" * 50); logger.info("Анализ MRIQC завершен.")
    if report_type in ["participant", "both"]:
        logger.info(f"  Обработано участников успешно: {subjects_processed_successfully}")
        logger.info(f"  Ошибок при обработке участников: {subjects_failed}")
    if report_type in ["group", "both"]:
        if group_report_successful: logger.info("  Групповой отчет: Успешно создан.")
        elif not can_run_group: logger.info("  Групповой отчет: Пропущен.")
        else: logger.info("  Групповой отчет: Ошибка или не запускался.")
    logger.info(f"  Подробности об ошибках MRIQC см. в: {error_log.resolve()}"); logger.info("=" * 50)
    return True


# --- Точка входа при запуске скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запускает MRIQC для анализа качества данных МРТ в формате BIDS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bids_dir", required=True, help="Путь к BIDS датасету.")
    parser.add_argument("--output_dir", required=True, help="Директория для отчетов MRIQC.")
    parser.add_argument(
        "--report_type", required=True, choices=['participant', 'group', 'both'],
        help="Тип отчета: 'participant', 'group', или 'both'."
    )
    # === ИЗМЕНЕНИЯ В ЭТОМ АРГУМЕНТЕ ===
    parser.add_argument(
        "--subjects", default=None, type=str, nargs='+', # Принимает ОДИН ИЛИ БОЛЕЕ аргументов
        help="Список ID участников для обработки (например, 001 005 или sub-001 sub-005). Если не указан, обрабатываются все."
    )
    parser.add_argument("--mriqc_path", default='mriqc', help="Путь к mriqc.")
    parser.add_argument("--n_procs", type=int, default=1, help="Кол-во процессоров.")
    parser.add_argument("--mem_gb", type=int, default=4, help="Память (GB).")
    parser.add_argument("--error_log", default="mriqc_processing_errors.log", help="Файл для ошибок MRIQC.")
    parser.add_argument("--log_file", default=None, help="Основной лог-файл скрипта.")
    args = parser.parse_args()

    # --- Настройка логирования (без изменений) ---
    log_file_path_main = args.log_file; output_dir_path_main = args.output_dir
    default_main_log_filename = 'mriqc_quality.log'
    if log_file_path_main is None:
        try:
            if output_dir_path_main and not os.path.exists(output_dir_path_main): os.makedirs(output_dir_path_main, exist_ok=True)
            log_file_path_final = os.path.join(output_dir_path_main or '.', default_main_log_filename)
        except OSError: log_file_path_final = default_main_log_filename; print(f"Предупреждение: Не удалось исп. {output_dir_path_main} для лога...")
    else: log_file_path_final = log_file_path_main
    setup_logging(log_file_path_final)

    # --- Основной блок выполнения ---
    try:
        logger.info("=" * 50); logger.info(f"Запуск mriqc_quality.py")
        logger.info(f"  BIDS директория: {os.path.abspath(args.bids_dir)}")
        logger.info(f"  Выходная директория MRIQC: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Тип отчета: {args.report_type}")
        if args.subjects: # Логируем, если обрабатываем список
            logger.info(f"  Обработка указанных участников: {', '.join(args.subjects)}")
        logger.info(f"  Путь mriqc: {args.mriqc_path}"); logger.info(f"  Процессоры: {args.n_procs}, Память: {args.mem_gb}GB")
        logger.info(f"  Лог ошибок MRIQC: {os.path.abspath(args.error_log)}"); logger.info(f"  Основной лог скрипта: {os.path.abspath(log_file_path_final)}"); logger.info("=" * 50)

        error_log_abs_path = Path(args.error_log)
        if not error_log_abs_path.is_absolute():
            error_log_abs_path = (Path(args.output_dir) if Path(args.output_dir).exists() else Path('.')) / args.error_log

        success = run_mriqc_analysis(
            bids_dir_str=args.bids_dir,
            output_dir_str=args.output_dir,
            report_type=args.report_type,
            mriqc_path=args.mriqc_path,
            n_procs=args.n_procs,
            mem_gb=args.mem_gb,
            error_log_path=str(error_log_abs_path.resolve()),
            subject_labels=args.subjects # <<< ПЕРЕДАЕМ СПИСОК
        )

        if success: logger.info("Скрипт mriqc_quality.py успешно завершил работу."); sys.exit(0)
        else: logger.error("Скрипт mriqc_quality.py завершился с ошибкой."); sys.exit(1)
    except FileNotFoundError as e: logger.error(f"Критическая ошибка: Файл/директория не найдены. {e}"); sys.exit(1)
    except OSError as e: logger.error(f"Критическая ошибка ФС: {e}", exc_info=True); sys.exit(1)
    except Exception as e: logger.exception(f"Непредвиденная критическая ошибка: {e}"); sys.exit(1)