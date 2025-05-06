import os
import subprocess
import argparse
import logging
import sys
from pathlib import Path
import shutil 

# --- Глобальная настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file_path):
    """Настраивает вывод логов в консоль (INFO) и файл (DEBUG)."""
    if logger.hasHandlers():
        logger.handlers.clear()

    # Обработчик для консоли
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Обработчик для файла
    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.debug(f"Логирование в файл настроено: {log_file_path}")
    except Exception as e:
        logger.error(f"Не удалось настроить логирование в файл {log_file_path}: {e}", exc_info=False)


def run_dciodvfy_check(input_dir, report_dir, dciodvfy_executable='dciodvfy'):
    """
    Запускает dciodvfy для DICOM файлов в input_dir и сохраняет отчеты в report_dir.

    Args:
        input_dir (str): Путь к директории с DICOM файлами (ожидается структура BIDS).
        report_dir (str): Путь к директории для сохранения отчетов dciodvfy.
        dciodvfy_executable (str): Путь к исполняемому файлу dciodvfy.
    """
    logger.info(f"Начало проверки DICOM стандарта в '{input_dir}'. Отчеты будут сохранены в '{report_dir}'.")
    logger.info(f"Исполняемый файл dciodvfy: '{dciodvfy_executable}'")

    # --- Проверки входных данных ---
    input_path = Path(input_dir)
    report_path = Path(report_dir)

    if not input_path.is_dir():
        logger.error(f"Входная директория не найдена: {input_dir}")
        raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")

    # Проверка наличия dciodvfy (простая проверка, полная будет при первом вызове)
    dciodvfy_found_path = shutil.which(dciodvfy_executable)
    if dciodvfy_found_path is None:
        logger.error(f"Исполняемый файл dciodvfy не найден по пути или в системном PATH: '{dciodvfy_executable}'. Убедитесь, что он установлен и путь указан верно.")
        raise FileNotFoundError(f"dciodvfy не найден: {dciodvfy_executable}")
    else:
        # Логируем полный путь, по которому он был найден
        logger.debug(f"Утилита dciodvfy найдена: {dciodvfy_found_path}")
        # Можно обновить переменную, если она была неполной (например, просто 'dciodvfy')
        dciodvfy_executable = dciodvfy_found_path

    # Создание корневой папки отчетов
    try:
        report_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Директория для отчетов создана или уже существует: {report_path}")
    except OSError as e:
        logger.error(f"Не удалось создать директорию для отчетов {report_path}: {e}")
        raise

    overall_summary = {} # Для сбора общей статистики (хотя функция ее не возвращает)

    # --- Обход входной директории ---
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        # Относительный путь для сохранения структуры в отчетах
        rel_path = root_path.relative_to(input_path)
        output_root = report_path / rel_path
        logger.debug(f"Обработка директории: {root_path}")

        # Создаем соответствующую поддиректорию в папке отчетов
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"  Не удалось создать поддиректорию для отчетов {output_root}: {e}. Пропуск файлов в {root_path}.")
            continue # Пропускаем файлы в этой директории

        dir_summary = [] # Сводка для текущей директории
        found_dicom_in_dir = False

        # --- Обработка файлов в директории ---
        for file in files:
            # Проверяем расширение файла (регистронезависимо)
            if not file.lower().endswith(".dcm"):
                continue

            found_dicom_in_dir = True
            input_file = root_path / file
            # Имя файла отчета
            report_filename = input_file.stem + '_report.txt'
            output_file = output_root / report_filename
            logger.debug(f"  Проверка файла: {input_file}")

            errors = 0
            warnings = 0
            error_running = False
            report_content = ""

            # --- Запуск dciodvfy для файла ---
            try:
                cmd = [dciodvfy_executable, str(input_file)]
                logger.debug(f"    Запуск команды: {' '.join(cmd)}")
                # Увеличим таймаут, т.к. проверка может быть долгой
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False) # check=False, т.к. dciodvfy может возвращать ненулевой код при наличии ошибок/предупреждений
                report_content = f"--- dciodvfy Report for {input_file} ---\n"
                report_content += f"Command: {' '.join(cmd)}\n"
                report_content += f"Return Code: {result.returncode}\n\n"
                report_content += "--- STDOUT ---\n"
                report_content += result.stdout or "[No stdout output]\n"
                report_content += "\n--- STDERR ---\n"
                report_content += result.stderr or "[No stderr output]\n"
                report_content += "--- End Report ---"

                # Подсчёт ошибок и предупреждений в отчете
                report_lines = (result.stdout + result.stderr).splitlines()
                errors = sum(line.strip().startswith("Error") for line in report_lines)
                warnings = sum(line.strip().startswith("Warning") for line in report_lines)
                logger.debug(f"    Проверка завершена. Ошибок: {errors}, Предупреждений: {warnings}. Код возврата: {result.returncode}")

            except FileNotFoundError:
                # Эта ошибка должна была быть поймана раньше, но на всякий случай
                error_msg = f"Критическая ошибка: Исполняемый файл dciodvfy не найден: {dciodvfy_executable}"
                logger.error(error_msg)
                report_content = error_msg
                error_running = True
                raise # Прерываем всю работу, если dciodvfy пропал
            except subprocess.TimeoutExpired:
                error_msg = f"Ошибка: Время ожидания dciodvfy истекло для файла {input_file}."
                logger.error(f"    {error_msg}")
                report_content = error_msg
                error_running = True
            except Exception as e:
                error_msg = f"Ошибка при запуске dciodvfy для файла {input_file}: {e}"
                logger.error(f"    {error_msg}", exc_info=True)
                report_content = error_msg
                error_running = True

            # --- Запись файла отчета ---
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.debug(f"    Отчет сохранен в: {output_file}")
            except OSError as e:
                logger.error(f"    Не удалось записать файл отчета {output_file}: {e}")
                error_running = True # Считаем это ошибкой выполнения проверки

            # Добавляем результат в сводку для директории
            if error_running:
                dir_summary.append((file, -1, -1)) # -1 означает ошибку запуска/записи
            else:
                dir_summary.append((file, errors, warnings))

        # --- Запись файла сводки для директории ---
        if found_dicom_in_dir and dir_summary: # Записываем сводку, только если были DICOM файлы и есть результаты
            summary_file = output_root / "summary.txt"
            logger.debug(f"  Запись сводки для директории: {summary_file}")
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Результаты проверки DICOM стандарта в директории: {rel_path}\n")
                    f.write("=" * (len(str(rel_path)) + 40) + "\n\n")
                    total_files = len(dir_summary)
                    total_errors = sum(e for _, e, _ in dir_summary if e != -1)
                    total_warnings = sum(w for _, _, w in dir_summary if w != -1)
                    files_with_errors = sum(1 for _, e, _ in dir_summary if e > 0)
                    files_with_warnings = sum(1 for _, _, w in dir_summary if w > 0)
                    files_failed = sum(1 for _, e, _ in dir_summary if e == -1)

                    f.write(f"Всего проверено файлов: {total_files}\n")
                    f.write(f"  С ошибками: {files_with_errors} (всего {total_errors} ошибок)\n")
                    f.write(f"  С предупреждениями: {files_with_warnings} (всего {total_warnings} предупреждений)\n")
                    if files_failed > 0:
                        f.write(f"  Не удалось проверить: {files_failed}\n")
                    f.write("\n--- Детали по файлам ---\n")

                    for filename, errors, warnings in sorted(dir_summary): # Сортируем для порядка
                        if errors == -1:
                            f.write(f"  {filename}: ❌ ОШИБКА ПРОВЕРКИ (см. {filename.replace('.dcm', '_report.txt')})\n")
                        elif errors > 0:
                            f.write(f"  {filename}: ❗ {errors} ошибок, {warnings} предупреждений\n")
                        elif warnings > 0:
                            f.write(f"  {filename}: ⚠️ {warnings} предупреждений\n")
                        else:
                            f.write(f"  {filename}: ✅ OK\n")

                overall_summary[str(rel_path)] = dir_summary # Добавляем в общую сводку (хотя она не используется)
            except OSError as e:
                logger.error(f"  Не удалось записать файл сводки {summary_file}: {e}")

    logger.info("Проверка DICOM стандарта завершена.")

# --- Точка входа при запуске скрипта ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Запускает утилиту dciodvfy для проверки DICOM файлов на соответствие стандарту.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        help="Входная директория с данными в формате BIDS DICOM (например, bids_data_dicom)."
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help="Выходная директория для сохранения отчетов dciodvfy (например, dciodvfy_reports)."
    )
    parser.add_argument(
        '--dciodvfy_path',
        default='dciodvfy', # По умолчанию ищем в PATH
        help="Путь к исполняемому файлу dciodvfy."
    )
    parser.add_argument(
        '--log_file',
        default=None,
        help="Путь к файлу для записи логов. Если не указан, будет создан файл 'dicom_standard_check.log' внутри --output_dir."
    )

    args = parser.parse_args()

    # --- Настройка логирования ---
    log_file_path = args.log_file
    if log_file_path is None:
        output_dir_path = args.output_dir
        log_filename = 'dicom_standard_check.log'
        try:
            # Создаем output_dir заранее, если надо
            if output_dir_path and not os.path.exists(output_dir_path):
                 os.makedirs(output_dir_path, exist_ok=True)
            log_file_path = os.path.join(output_dir_path or '.', log_filename)
        except OSError as e:
             log_file_path = log_filename # Пишем в текущую папку при ошибке
             print(f"Предупреждение: Не удалось использовать {output_dir_path} для лог-файла. Лог будет записан в {log_file_path}. Ошибка: {e}")

    setup_logging(log_file_path)

    # --- Основной блок выполнения ---
    try:
        logger.info("="*50)
        logger.info(f"Запуск dicom_standard_check.py")
        logger.info(f"  Входная директория: {os.path.abspath(args.input_dir)}")
        logger.info(f"  Выходная директория: {os.path.abspath(args.output_dir)}")
        logger.info(f"  Путь dciodvfy: {args.dciodvfy_path}")
        logger.info(f"  Лог-файл: {os.path.abspath(log_file_path)}")
        logger.info("="*50)

        run_dciodvfy_check(args.input_dir, args.output_dir, args.dciodvfy_path)

        logger.info("Скрипт успешно завершил работу.")
        sys.exit(0) # Успешный выход

    except FileNotFoundError as e:
        # Ошибка отсутствия входной директории или dciodvfy
        logger.error(f"Критическая ошибка: Файл или директория не найдены. {e}")
        sys.exit(1) # Выход с кодом ошибки
    except OSError as e:
        logger.error(f"Критическая ошибка файловой системы: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Ловим все остальные непредвиденные ошибки
        logger.exception(f"Непредвиденная критическая ошибка: {e}")
        sys.exit(1)