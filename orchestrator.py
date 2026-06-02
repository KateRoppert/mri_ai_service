"""
Главный управляющий скрипт для запуска всего pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
import subprocess
import logging
import time
import sys
import argparse
from datetime import datetime, timedelta

# Импорт из utils
from utils.config_loader import (
    load_config,
    resolve_paths,
    get_stage_input_dir,
    get_stage_output_dir,
    get_enabled_stages,
    ConfigValidationError
)


# ============================================
# ПОСТРОЕНИЕ КОМАНД
# ============================================

def build_command(
    stage_name: str,
    config: Dict[str, Any],
    project_root: Path
) -> List[str]:
    """
    Строит командную строку для запуска этапа.
    """
    stage_config = config['stages'][stage_name]
    
    # Базовая команда
    script_path = project_root / stage_config['script']
    cmd = ['python', str(script_path)]
    
    # Позиционные аргументы
    input_dir = get_stage_input_dir(stage_name, config)
    output_dir = get_stage_output_dir(stage_name, config)
    cmd.extend([str(input_dir), str(output_dir)])
    
    # Лог-файл
    logs_dir = config['general']['root_output_dir'] / config['output_structure']['logs'] / 'stages'
    log_file = logs_dir / f"{stage_name}.log"
    cmd.extend(['--log_file', str(log_file)])
    
    # Global max-subjects — only inject if the stage does not declare its own.
    # A stage with max_subjects: null in its args section explicitly means "no limit"
    # and should override the global setting.
    stage_args = stage_config.get('args', {})
    if 'max_subjects' not in stage_args:
        max_subjects = config['general'].get('max_subjects')
        if max_subjects is not None:
            cmd.extend(['--max-subjects', str(max_subjects)])

    # Lesion type — global per-run, only injected into segmentation-related stages
    if stage_name in (
        'stage_01_reorganize',
        'stage_04_quality',
        'stage_05_preprocessing',
        'stage_06_segmentation',
        'stage_07_inverse_transform',
        'stage_08_lobar_localization',
    ):
        lesion_type = config['general'].get('lesion_type', 'glioblastoma')
        cmd.extend(['--lesion-type', lesion_type])
    
    # Опциональные аргументы - БЕЗ replace()
    for arg_name, arg_value in stage_config['args'].items():
        if arg_value is None or arg_value is False:
            continue
        
        if arg_value is True:
            # Булевы флаги - имя как есть
            cmd.append(f'--{arg_name}')
        else:
            # Специальная обработка путей
            if arg_name == 'config':
                config_path = project_root / arg_value
                cmd.extend([f'--{arg_name}', str(config_path)])

            elif arg_name == 'preprocessing-config':
                config_path = project_root / arg_value
                cmd.extend([f'--{arg_name}', str(config_path)])

            elif arg_name == 'results_dir':
                if arg_value:
                    results_path = config['general']['root_output_dir'] / arg_value
                else:
                    global_results = config['general'].get('benchmark_results_dir')
                    if global_results:
                        results_path = config['general']['root_output_dir'] / global_results
                    else:
                        continue
                cmd.extend([f'--{arg_name}', str(results_path)])
            elif arg_name == 'metadata-dir':
                # metadata-dir относительно root_output_dir
                if arg_value:
                    metadata_path = config['general']['root_output_dir'] / arg_value
                    cmd.extend([f'--{arg_name}', str(metadata_path)])
            else:
                # Обычные аргументы - имя как есть, БЕЗ replace
                cmd.extend([f'--{arg_name}', str(arg_value)])
    
    return cmd


# ============================================
# ЗАПУСК ЭТАПА
# ============================================

def run_stage(
    stage_name: str,
    config: Dict[str, Any],
    project_root: Path,
    logger: logging.Logger,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Запускает один этап pipeline.
    
    Args:
        stage_name: Имя этапа
        config: Конфигурация pipeline
        project_root: Корневая директория проекта
        logger: Logger для записи в master log
        dry_run: Если True, только показывает команду без выполнения
        
    Returns:
        Словарь с результатами
    """
    stage_config = config['stages'][stage_name]
    
    # Проверка, включен ли этап
    if not stage_config['enabled']:
        logger.info(f"[{stage_name}] SKIPPED (disabled in config)")
        return {
            'status': 'SKIPPED',
            'time': 0.0,
            'command': '',
            'critical': False
        }
    
    # Построение команды
    cmd = build_command(stage_name, config, project_root)
    cmd_str = ' '.join(cmd)
    
    # Dry-run режим
    if dry_run:
        logger.info(f"[{stage_name}] DRY-RUN")
        logger.info(f"  Command: {cmd_str}")
        return {
            'status': 'DRY_RUN',
            'time': 0.0,
            'command': cmd_str,
            'critical': False
        }
    
    # Запуск этапа
    logger.info(f"[{stage_name}] STARTED")
    logger.debug(f"  Command: {cmd_str}")
    
    start_time = time.time()
    
    try:
        # Запускаем процесс
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        elapsed_str = format_elapsed_time(elapsed)
        
        logger.info(f"[{stage_name}] SUCCESS ({elapsed_str})")
        
        return {
            'status': 'SUCCESS',
            'time': elapsed,
            'command': cmd_str,
            'critical': False
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        elapsed_str = format_elapsed_time(elapsed)
        
        logger.error(f"[{stage_name}] FAILED ({elapsed_str})")
        logger.error(f"  Return code: {e.returncode}")
        
        # Логируем stderr если есть
        if e.stderr:
            logger.error(f"  Error output (last 500 chars):")
            logger.error(f"  {e.stderr[-500:]}")
        
        # Определяем критичность ошибки
        is_critical = check_if_critical_error(e, stage_name, logger)
        
        return {
            'status': 'FAILED',
            'time': elapsed,
            'command': cmd_str,
            'error': e.stderr if e.stderr else str(e),
            'critical': is_critical
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        elapsed_str = format_elapsed_time(elapsed)
        
        logger.error(f"[{stage_name}] FAILED ({elapsed_str})")
        logger.error(f"  Unexpected error: {type(e).__name__}: {e}")
        
        return {
            'status': 'FAILED',
            'time': elapsed,
            'command': cmd_str,
            'error': str(e),
            'critical': True
        }


def check_if_critical_error(
    error: subprocess.CalledProcessError,
    stage_name: str,
    logger: logging.Logger
) -> bool:
    """
    Определяет, является ли ошибка критичной (требует остановки pipeline).
    """
    stderr = error.stderr if error.stderr else ""
    returncode = error.returncode
    
    # Критичные признаки
    critical_patterns = [
        "FileNotFoundError",
        "ModuleNotFoundError",
        "SyntaxError",
        "ImportError",
        "MemoryError",
        "No such file or directory",
        "Input directory does not exist",
        "Input directory is empty"
    ]
    
    for pattern in critical_patterns:
        if pattern in stderr:
            logger.warning(f"  Critical error detected: {pattern}")
            return True
    
    # Return code >= 2 часто означает серьезную ошибку
    if returncode >= 2:
        logger.warning(f"  High return code: {returncode} (likely critical)")
        return True
    
    # Если есть признаки частичного успеха
    if "processed" in stderr.lower() or "completed" in stderr.lower():
        logger.info(f"  Partial success detected")
        return False
    
    # По умолчанию критична
    logger.warning(f"  Treating as critical")
    return True


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def format_elapsed_time(seconds: float) -> str:
    """Форматирует время в читаемый вид."""
    td = timedelta(seconds=int(seconds))
    
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_output_directories(config: Dict[str, Any]) -> None:
    """Создает структуру выходных директорий."""
    root_output = config['general']['root_output_dir']
    output_struct = config['output_structure']
    
    directories = [
        root_output / output_struct['stage_01'],
        root_output / output_struct['stage_02'],
        root_output / output_struct['stage_03'],
        root_output / output_struct['stage_04'],
        root_output / output_struct['stage_05_data'],
        root_output / output_struct['stage_06'],
        root_output / output_struct['stage_07'],
        root_output / output_struct['stage_08'],
        root_output / output_struct['logs'],
        root_output / output_struct['logs'] / 'stages',
        root_output / output_struct['reports']
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: Path) -> logging.Logger:
    """Настраивает master logger."""
    logger = logging.getLogger('pipeline_master')
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================
# ОСНОВНАЯ ЛОГИКА PIPELINE
# ============================================

def run_pipeline(
    config: Dict[str, Any],
    project_root: Path,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Запускает весь pipeline.
    
    Args:
        config: Конфигурация pipeline
        project_root: Корневая директория проекта
        dry_run: Режим dry-run
        
    Returns:
        Статистика выполнения
    """
    # Создаем директории
    create_output_directories(config)
    
    # Настраиваем logger
    log_file = config['general']['root_output_dir'] / config['output_structure']['logs'] / 'pipeline_master.log'
    logger = setup_logger(log_file)
    
    # Начало pipeline
    logger.info("="*70)
    logger.info("PIPELINE STARTED")
    logger.info("="*70)
    logger.info(f"Config: {config.get('_config_path', 'N/A')}")
    logger.info(f"Root input: {config['general']['root_input_dir']}")
    logger.info(f"Root output: {config['general']['root_output_dir']}")
    logger.info(f"Max subjects: {config['general'].get('max_subjects', 'all')}")
    logger.info(f"Skip existing: {config['general'].get('skip_existing', False)}")
    logger.info(f"Dry-run: {dry_run}")
    logger.info("")
    
    # Получаем список включенных этапов
    enabled_stages = get_enabled_stages(config)
    logger.info(f"Enabled stages: {len(enabled_stages)}/8")
    for stage in enabled_stages:
        logger.info(f"  ✓ {stage}")
    logger.info("")
    
    # Выполнение этапов
    pipeline_stats = {}
    pipeline_start_time = time.time()
    
    for i, stage_name in enumerate(enabled_stages, 1):
        logger.info(f"[STAGE {i}/{len(enabled_stages)}] {stage_name}")
        
        result = run_stage(stage_name, config, project_root, logger, dry_run)
        pipeline_stats[stage_name] = result
        
        # Проверка на критическую ошибку
        if result['status'] == 'FAILED' and result['critical']:
            logger.error("")
            logger.error("="*70)
            logger.error(f"PIPELINE STOPPED due to critical error in {stage_name}")
            logger.error("="*70)
            break
        
        logger.info("")
    
    # Итоги
    pipeline_elapsed = time.time() - pipeline_start_time
    logger.info("="*70)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*70)
    logger.info(f"Total time: {format_elapsed_time(pipeline_elapsed)}")
    
    # Подсчет статистики
    total_stages = len(enabled_stages)
    completed = sum(1 for r in pipeline_stats.values() if r['status'] == 'SUCCESS')
    failed = sum(1 for r in pipeline_stats.values() if r['status'] == 'FAILED')
    skipped = sum(1 for r in pipeline_stats.values() if r['status'] == 'SKIPPED')
    
    logger.info(f"Completed stages: {completed}/{total_stages}")
    if failed > 0:
        logger.info(f"Failed stages: {failed}")
    if skipped > 0:
        logger.info(f"Skipped stages: {skipped}")
    
    # Детализация по этапам
    logger.info("")
    logger.info("Stage details:")
    for stage_name, result in pipeline_stats.items():
        status_symbol = {
            'SUCCESS': '✓',
            'FAILED': '✗',
            'SKIPPED': '-',
            'DRY_RUN': '?'
        }.get(result['status'], '?')
        
        time_str = format_elapsed_time(result['time']) if result['time'] > 0 else '-'
        logger.info(f"  {status_symbol} {stage_name}: {result['status']} ({time_str})")
    
    logger.info("="*70)
    
    return pipeline_stats


# ============================================
# MAIN
# ============================================

def main():
    """Точка входа."""
    parser = argparse.ArgumentParser(
        description='Pipeline orchestrator for brain MRI lesion segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('pipeline_config.yaml'),
        help='Path to pipeline configuration file (default: pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show commands without executing them'
    )
    
    args = parser.parse_args()
    
    # Проверка существования конфига
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Определяем project_root (директория где лежит orchestrator.py)
    project_root = Path(__file__).parent.resolve()
    
    try:
        # Загрузка конфига
        print(f"Loading config: {args.config}")
        config = load_config(args.config)
        config['_config_path'] = str(args.config)  # Сохраняем для логирования
        
        # Разрешение путей
        config = resolve_paths(config)
        
        # Запуск pipeline
        pipeline_stats = run_pipeline(config, project_root, args.dry_run)
        
        # Определяем exit code
        failed_stages = [
            name for name, result in pipeline_stats.items()
            if result['status'] == 'FAILED'
        ]
        
        if failed_stages:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except ConfigValidationError as e:
        print(f"ERROR: Configuration validation failed:")
        print(f"  {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"ERROR: Unexpected error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()