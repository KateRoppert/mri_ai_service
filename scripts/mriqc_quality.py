import os
import subprocess
import shutil
import traceback
from datetime import datetime
import argparse
import logging
import sys
import shutil # Для shutil.which
from pathlib import Path

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s') # Добавим имя функции

def setup_logging(log_file_path: str, console_level: str = "INFO"):
    """
    Настраивает основной логгер скрипта: вывод в консоль и в указанный файл.
    Очищает предыдущие обработчики основного логгера.

    Args:
        log_file_path (str): Полный путь к основному лог-файлу скрипта.
        console_level (str): Уровень логирования для консоли (DEBUG, INFO, etc.).
    """
    if logger.hasHandlers():
        logger.handlers.clear()

    # Консольный обработчик
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        ch.setLevel(logging.INFO)
        print(f"Предупреждение: Неверный уровень логирования для консоли '{console_level}'. Используется INFO.")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файловый обработчик
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # mode='a' для дозаписи
        fh.setLevel(logging.DEBUG) # В файл пишем все, начиная с DEBUG
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Основное логирование в файл настроено: {log_file_path}")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить основной файловый лог {log_file_path}: {e}")
        # Можно рассмотреть sys.exit(1) здесь, если основной лог критичен


def run_mriqc_analysis(
    bids_dir_str: str,
    output_dir_str: str,
    report_type: str,
    mriqc_path: str = 'mriqc',
    n_procs: int = 1,
    n_threads: int = 1,
    mem_gb: int = 4,
    error_log_path: str = "mriqc_failed_subjects.log",
    subject_labels: list[str] | None = None
    ) -> bool:
    """
    Основная функция для запуска анализа MRIQC.

    Выполняет шаги:
    1. Проверка входных данных и доступности MRIQC.
    2. Определение списка участников для обработки (всех или указанных).
    3. Запуск MRIQC в режиме 'participant' для каждого выбранного участника.
    4. (Опционально) Запуск MRIQC в режиме 'group'.
    5. Обработка ошибок, логирование, очистка временных директорий.

    Args:
        bids_dir_str (str): Путь к корневой директории BIDS датасета.
        output_dir_str (str): Путь к директории для сохранения отчетов MRIQC.
        report_type (str): Тип отчета ('participant', 'group', 'both').
        mriqc_path (str): Путь к исполняемому файлу mriqc.
        n_procs (int): Количество параллельных процессов для Nipype (флаг --nprocs MRIQC).
        n_threads (int): Количество потоков для OpenMP (устанавливается через OMP_NUM_THREADS).
        mem_gb (int): Объем памяти (в ГБ) для Nipype (флаг --mem_gb MRIQC).
        error_log_path (str): Путь к файлу для записи специфических ошибок MRIQC.
        subject_labels (list[str] | None): Список ID участников (например, '001', 'sub-005')
                                           для обработки. Если None, обрабатываются все.

    Returns:
        bool: True, если скрипт дошел до конца (даже если были ошибки MRIQC), False при критической ошибке скрипта.
    """
    logger.info(f"Запуск анализа MRIQC.")
    logger.info(f"  BIDS директория: {bids_dir_str}")
    logger.info(f"  Выходная директория: {output_dir_str}")
    logger.info(f"  Тип отчета: {report_type}")
    logger.info(f"  Исполняемый файл mriqc: {mriqc_path}")
    logger.info(f"  Количество процессов (nprocs): {n_procs}")
    logger.info(f"  Количество потоков (omp_threads): {n_threads}")
    logger.info(f"  Память (GB): {mem_gb}")
    logger.info(f"  Лог ошибок MRIQC: {error_log_path}")
    if subject_labels:
        logger.info(f"  Обработка указанных участников: {', '.join(subject_labels)}")
    else:
        logger.info(f"  Обработка всех участников в BIDS директории.")

    bids_dir = Path(bids_dir_str)
    output_dir = Path(output_dir_str)
    # Временная рабочая директория внутри output_dir
    base_work_dir = output_dir / "mriqc_work"
    error_log = Path(error_log_path)

    # --- Предварительные проверки ---
    if not bids_dir.is_dir():
        logger.error(f"BIDS директория не найдена: {bids_dir}")
        raise FileNotFoundError(f"BIDS директория не найдена: {bids_dir}")

    mriqc_found_path = shutil.which(mriqc_path)
    if mriqc_found_path is None:
        logger.error(
            f"Исполняемый файл mriqc не найден: '{mriqc_path}'. "
            f"Убедитесь, что он установлен и доступен в PATH."
        )
        raise FileNotFoundError(f"mriqc не найден: {mriqc_path}")
    else:
        logger.info(f"Исполняемый файл mriqc найден: {mriqc_found_path}")
        mriqc_path = mriqc_found_path # Используем полный найденный путь

    # Создание необходимых директорий и файла лога ошибок
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        error_log.parent.mkdir(parents=True, exist_ok=True) # Создаем папку для лога ошибок
        # Очищаем/создаем лог ошибок MRIQC
        with open(error_log, "w", encoding='utf-8') as f:
            f.write(f"Лог ошибок при выполнении MRIQC ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
        logger.debug(f"Файл лога ошибок MRIQC очищен/создан: {error_log}")
    except OSError as e:
        logger.error(f"Не удалось создать выходные директории или лог ошибок MRIQC: {e}")
        raise # Это критическая ошибка

    subjects_processed_successfully = 0
    subjects_failed = 0

    # --- Подготовка окружения для запуска subprocess ---
    # Устанавливаем OMP_NUM_THREADS для контроля многопоточности внутри MRIQC
    run_env = os.environ.copy()
    run_env["OMP_NUM_THREADS"] = str(n_threads)
    logger.debug(f"Установлена переменная окружения OMP_NUM_THREADS={run_env['OMP_NUM_THREADS']}")

    # --- Запуск MRIQC для индивидуальных участников ---
    if report_type in ["participant", "both"]:
        target_subjects_full_ids = [] # Список полных ID (sub-XXX) для обработки

        # Определяем список участников
        if subject_labels: # Если пользователь указал конкретных участников
            logger.info(f"Обработка участников по списку: {subject_labels}")
            for subj_label in subject_labels:
                # Нормализуем ID, добавляя 'sub-' при необходимости
                full_id = f"sub-{subj_label}" if not subj_label.startswith("sub-") else subj_label
                # Проверяем существование папки участника
                if not (bids_dir / full_id).is_dir():
                    logger.warning(f"Участник '{full_id}' (из списка --subjects) не найден в {bids_dir}. Пропуск.")
                else:
                    target_subjects_full_ids.append(full_id)
            if not target_subjects_full_ids:
                 logger.warning("Ни один из указанных участников не найден. Индивидуальные отчеты не будут созданы.")
            else:
                 logger.info(f"Список участников для обработки: {', '.join(target_subjects_full_ids)}")
        else: # Если участники не указаны, берем всех из папки BIDS
            logger.info("Определение списка всех участников в BIDS директории...")
            target_subjects_full_ids = sorted([d.name for d in bids_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
            if not target_subjects_full_ids:
                logger.warning("Не найдено участников (папок sub-*) для индивидуального анализа.")
            else:
                logger.info(f"Найдено {len(target_subjects_full_ids)} участников для анализа.")

        # Итерация по списку выбранных участников
        for subject_id_full in target_subjects_full_ids:
            subject_label_for_mriqc = subject_id_full.replace("sub-", "") # MRIQC требует ID без 'sub-'
            logger.info(f"--- Обработка участника: {subject_id_full} ---")

            # Создаем уникальную временную рабочую директорию
            subject_work_dir = base_work_dir / subject_id_full
            try:
                subject_work_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"  Создана временная рабочая директория: {subject_work_dir}")
            except OSError as e:
                logger.error(f"  Не удалось создать рабочую директорию для {subject_id_full}: {e}. Пропуск участника.")
                subjects_failed += 1
                # Запись в лог ошибок MRIQC
                with open(error_log, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — ОШИБКА СОЗДАНИЯ WORK DIR\n")
                    f.write(f"  Сообщение: {e}\n\n")
                continue # Переход к следующему участнику

            # Формируем команду для запуска MRIQC participant-level
            cmd_participant = [
                mriqc_path,
                str(bids_dir),                 # Путь к BIDS данным
                str(output_dir),               # Путь для выходных отчетов
                "participant",                 # Уровень анализа
                "--participant_label", subject_label_for_mriqc, # ID участника
                "--no-sub",                    # Не отправлять анонимную статистику
                "--nprocs", str(n_procs),      # Количество процессов Nipype
                "--mem_gb", str(mem_gb),       # Выделенная память для Nipype
                "--work-dir", str(subject_work_dir) # Временная директория
                # Дополнительные полезные флаги MRIQC можно добавить здесь
                # Например: '--fft-spikes-detector', '--ica', '-vv' для более подробного вывода
            ]
            logger.debug(f"  Команда MRIQC (participant): {' '.join(cmd_participant)}")

            try:
                # Запускаем MRIQC с настроенным окружением (OMP_NUM_THREADS)
                result = subprocess.run(
                    cmd_participant,
                    env=run_env,           # Передаем настроенное окружение
                    capture_output=True,   # Захватываем stdout и stderr
                    text=True,             # Декодируем вывод как текст
                    check=False,           # Не выбрасывать исключение при ошибке MRIQC
                    timeout=3600           # Таймаут 1 час на участника (можно настроить)
                )

                # Логируем вывод MRIQC (полезно для отладки)
                if result.stdout:
                    logger.debug(f"  MRIQC stdout для {subject_id_full}:\n{result.stdout.strip()}")
                if result.stderr:
                    logger.debug(f"  MRIQC stderr для {subject_id_full}:\n{result.stderr.strip()}")

                # Проверяем код возврата MRIQC
                if result.returncode == 0:
                    logger.info(f"  Успешно завершена обработка участника: {subject_id_full}")
                    subjects_processed_successfully += 1
                else:
                    # Если MRIQC вернул ошибку
                    logger.error(
                        f"  MRIQC завершился с ошибкой для {subject_id_full} "
                        f"(код возврата: {result.returncode}). "
                        f"Подробности см. в логе ошибок: {error_log.name}."
                    )
                    subjects_failed += 1
                    # Запись в лог ошибок MRIQC
                    with open(error_log, "a", encoding='utf-8') as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — ОШИБКА ВЫПОЛНЕНИЯ MRIQC\n")
                        f.write(f"  Код возврата: {result.returncode}\n")
                        if result.stderr:
                            # Записываем последние N строк stderr для диагностики
                            stderr_lines = result.stderr.strip().splitlines()
                            f.write(f"  stderr (последние {min(10, len(stderr_lines))} строк):\n  ...\n")
                            f.write("  " + "\n  ".join(stderr_lines[-10:])) # Последние 10 строк
                            f.write("\n")
                        f.write("\n")

            except subprocess.TimeoutExpired:
                logger.error(f"  Время ожидания MRIQC (1 час) истекло для участника {subject_id_full}.")
                subjects_failed += 1
                with open(error_log, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — ОШИБКА: TIMEOUT\n\n")
            except Exception as e: # Ловим другие возможные ошибки при запуске subprocess
                logger.exception(f"  Непредвиденная ошибка Python при обработке {subject_id_full}: {e}")
                subjects_failed += 1
                with open(error_log, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject_id_full} — НЕИЗВЕСТНАЯ ОШИБКА PYTHON:\n")
                    f.write(traceback.format_exc()) # Записываем полный traceback Python
                    f.write("\n\n")
            finally:
                # Очистка временной директории участника в любом случае
                if subject_work_dir.exists():
                    logger.debug(f"  Удаление временной рабочей директории: {subject_work_dir}")
                    try:
                        shutil.rmtree(subject_work_dir)
                    except Exception as e_rm:
                        logger.warning(f"  Не удалось удалить временную рабочую директорию {subject_work_dir}: {e_rm}")

    # --- Запуск MRIQC для группового отчета ---
    group_report_successful = False
    can_run_group = True # Флаг, можно ли запускать групповой отчет

    # Проверяем условия, при которых групповой отчет запускать не стоит
    if subject_labels and (subjects_failed == len(subject_labels) or not target_subjects_full_ids):
        can_run_group = False
        logger.info(
            "Групповой отчет не будет создан: не было успешно обработанных участников "
            "из указанного списка или список был пуст/не найден."
        )
    if not subject_labels and subjects_processed_successfully == 0 and subjects_failed > 0:
         # Если обрабатывали всех, но все упали
         can_run_group = False
         logger.info("Групповой отчет не будет создан: не было успешно обработанных индивидуальных отчетов.")

    if report_type in ["group", "both"] and can_run_group:
        logger.info("--- Генерация группового отчета MRIQC ---")
        work_dir_group = base_work_dir / "group_work" # Отдельная workdir для группы
        try:
            work_dir_group.mkdir(parents=True, exist_ok=True)

            # Формируем команду для group-level
            cmd_group = [
                mriqc_path,
                str(bids_dir),
                str(output_dir),
                "group",
                "--no-sub", # Не отправлять статистику
                "--work-dir", str(work_dir_group)
                # Параметры --nprocs и --mem_gb обычно не так критичны для group,
                # но можно добавить, если нужно и поддерживается вашей версией MRIQC
            ]
            logger.debug(f"  Команда MRIQC (group): {' '.join(cmd_group)}")

            # Запускаем MRIQC group с настроенным окружением
            result = subprocess.run(
                cmd_group,
                env=run_env,
                capture_output=True, text=True, check=False,
                timeout=1800 # Таймаут 30 минут на групповой отчет
            )

            # Логируем вывод
            if result.stdout:
                logger.debug(f"  MRIQC stdout (group):\n{result.stdout.strip()}")
            if result.stderr:
                logger.debug(f"  MRIQC stderr (group):\n{result.stderr.strip()}")

            if result.returncode == 0:
                logger.info("  Групповой отчет MRIQC успешно создан.")
                group_report_successful = True
            else:
                logger.error(
                    f"  Ошибка при генерации группового отчета MRIQC "
                    f"(код возврата: {result.returncode}). Подробности см. в {error_log.name}."
                )
                with open(error_log, "a", encoding='utf-8') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ГРУППОВОЙ ОТЧЕТ — ОШИБКА ВЫПОЛНЕНИЯ\n")
                    f.write(f"  Код возврата: {result.returncode}\n")
                    if result.stderr:
                        stderr_lines = result.stderr.strip().splitlines()
                        f.write(f"  stderr (последние {min(10, len(stderr_lines))} строк):\n  ...\n")
                        f.write("  " + "\n  ".join(stderr_lines[-10:]))
                        f.write("\n")
                    f.write("\n")

        except subprocess.TimeoutExpired:
            logger.error("  Время ожидания MRIQC истекло при генерации группового отчета.")
            with open(error_log, "a", encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ГРУППОВОЙ ОТЧЕТ — ОШИБКА: TIMEOUT\n\n")
        except Exception as e:
            logger.exception(f"  Непредвиденная ошибка Python при генерации группового отчета: {e}")
            with open(error_log, "a", encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ГРУППОВОЙ ОТЧЕТ — НЕИЗВЕСТНАЯ ОШИБКА PYTHON:\n")
                f.write(traceback.format_exc())
                f.write("\n\n")
        finally:
            # Очистка временной директории для группы
            if work_dir_group.exists():
                logger.debug(f"  Удаление временной рабочей директории (group): {work_dir_group}")
                try:
                    shutil.rmtree(work_dir_group)
                except Exception as e_rm:
                     logger.warning(f"  Не удалось удалить временную рабочую директорию (group) {work_dir_group}: {e_rm}")

    # --- Итоговая информация ---
    logger.info("=" * 50)
    logger.info("Анализ MRIQC завершен.")
    if report_type in ["participant", "both"]:
        logger.info(f"  Обработано участников успешно: {subjects_processed_successfully}")
        logger.info(f"  Ошибок при обработке участников: {subjects_failed}")
    if report_type in ["group", "both"]:
        if group_report_successful:
            logger.info("  Групповой отчет: Успешно создан.")
        elif not can_run_group:
            logger.info("  Групповой отчет: Пропущен (из-за ошибок или выбора участников).")
        else:
            logger.info("  Групповой отчет: Ошибка при создании или не запускался.")
    logger.info(f"  Подробности об ошибках MRIQC (если были) см. в файле: {error_log.resolve()}")
    logger.info("=" * 50)

    # Считаем выполнение скрипта успешным, если он дошел до конца
    return True


# --- Точка входа при запуске скрипта ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запускает MRIQC для анализа качества данных МРТ в формате BIDS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--bids_dir",
        required=True,
        help="Путь к корневой директории BIDS датасета (например, 'bids_data_nifti')."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Путь к директории для сохранения отчетов MRIQC (например, 'mriqc_output')."
    )
    parser.add_argument(
        "--report_type",
        required=True,
        choices=['participant', 'group', 'both'],
        help="Тип отчета для генерации: 'participant' (индивидуальные), "
             "'group' (групповой), 'both' (оба)."
    )
    parser.add_argument(
        "--subjects",
        default=None,
        type=str,
        nargs='+', # Ожидает один или более ID участников
        help="Список ID участников для обработки (например, '001' '005' 'sub-007'). "
             "Если не указан, обрабатываются все участники."
    )
    parser.add_argument(
        "--mriqc_path",
        default='mriqc',
        help="Путь к исполняемому файлу mriqc (или команда, если в PATH)."
    )
    parser.add_argument(
        "--n_procs",
        type=int,
        default=1,
        help="Количество процессов для использования Nipype (--nprocs)."
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=1,
        help="Количество потоков для использования OpenMP (устанавливает OMP_NUM_THREADS)."
    )
    parser.add_argument(
        "--mem_gb",
        type=int,
        default=4,
        help="Объем памяти (в ГБ) для использования Nipype (--mem_gb)."
    )
    parser.add_argument(
        "--error_log",
        default="mriqc_processing_errors.log",
        help="Файл для записи ошибок, специфичных для MRIQC (например, проваленные участники)."
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Путь к основному лог-файлу скрипта. Если не указан, "
             "будет создан 'mriqc_quality.log' внутри --output_dir."
    )
    parser.add_argument(
        "--console_log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования для вывода в консоль."
    )

    args = parser.parse_args()

    # --- Настройка основного логирования ---
    main_log_path_arg = args.log_file
    output_dir_path_arg = args.output_dir
    default_main_log_filename = 'mriqc_quality.log'

    if main_log_path_arg is None:
        try:
            # Кладем основной лог в output_dir MRIQC
            if output_dir_path_arg and not os.path.exists(output_dir_path_arg):
                 os.makedirs(output_dir_path_arg, exist_ok=True)
            main_log_path_final = os.path.join(output_dir_path_arg or '.', default_main_log_filename)
        except OSError as e:
             main_log_path_final = default_main_log_filename
             print(f"Предупреждение: Не удалось использовать {output_dir_path_arg} для основного лог-файла. "
                   f"Лог будет записан в {main_log_path_final}. Ошибка: {e}")
    else:
        main_log_path_final = main_log_path_arg

    # Настраиваем логгер перед вызовом основной функции
    setup_logging(main_log_path_final, args.console_log_level)

    # --- Основной блок выполнения ---
    try:
        logger.info("=" * 50)
        logger.info(f"Запуск скрипта: mriqc_quality.py")
        logger.info(f"  BIDS директория: {os.path.abspath(args.bids_dir)}")
        logger.info(f"  Выходная директория MRIQC: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Тип отчета: {args.report_type}")
        if args.subjects:
            logger.info(f"  Обработка указанных участников: {', '.join(args.subjects)}")
        logger.info(f"  Путь mriqc: {args.mriqc_path}")
        logger.info(f"  Ресурсы: Процессы={args.n_procs}, Потоки={args.n_threads}, Память={args.mem_gb}GB")
        logger.info(f"  Лог ошибок MRIQC: {os.path.abspath(args.error_log)}")
        logger.info(f"  Основной лог скрипта: {os.path.abspath(main_log_path_final)}")
        logger.info(f"  Уровень лога консоли: {args.console_log_level}")
        logger.info("=" * 50)

        # Определяем абсолютный путь к логу ошибок MRIQC
        error_log_abs_path = Path(args.error_log)
        if not error_log_abs_path.is_absolute():
            error_log_abs_path = (Path(args.output_dir) if Path(args.output_dir).exists() else Path('.')) / args.error_log

        # Вызываем основную функцию анализа
        success_pipeline = run_mriqc_analysis(
            bids_dir_str=args.bids_dir,
            output_dir_str=args.output_dir,
            report_type=args.report_type,
            mriqc_path=args.mriqc_path,
            n_procs=args.n_procs,
            n_threads=args.n_threads, # Передаем количество потоков
            mem_gb=args.mem_gb,
            error_log_path=str(error_log_abs_path.resolve()),
            subject_labels=args.subjects # Передаем список участников
        )

        # Завершаем скрипт с кодом 0, т.к. основная функция обрабатывает ошибки внутри
        # Наличие ошибок MRIQC не означает провал самого скрипта
        if success_pipeline:
            logger.info(
                "Скрипт mriqc_quality.py успешно завершил свою работу. "
                "Проверьте основной лог и лог ошибок MRIQC на предмет проблем."
            )
            sys.exit(0)
        else:
            # Эта ветка не должна достигаться при текущей логике run_mriqc_analysis
            logger.error("Скрипт mriqc_quality.py завершился с непредвиденной ошибкой в основной функции.")
            sys.exit(1)

    except FileNotFoundError as e:
        # Ошибка, если BIDS директория или mriqc не найдены на старте
        logger.error(f"Критическая ошибка: Файл или директория не найдены. {e}")
        sys.exit(1)
    except OSError as e:
        # Ошибка создания директорий
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Ловим все остальные непредвиденные ошибки на самом верхнем уровне
        logger.exception(f"Непредвиденная критическая ошибка на верхнем уровне выполнения: {e}")
        sys.exit(1)