"""
Загрузка и валидация конфигурации pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
import yaml
import sys


class ConfigValidationError(Exception):
    """Исключение при ошибке валидации конфига."""
    pass


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Загружает конфигурацию из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Словарь с конфигурацией
        
    Raises:
        ConfigValidationError: Если конфиг некорректен
    """
    # Проверка существования файла
    if not config_path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")
    
    # Загрузка YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Failed to parse YAML: {e}")
    
    # Валидация структуры
    validate_config(config, config_path.parent)
    
    return config


def validate_config(config: Dict[str, Any], config_dir: Path) -> None:
    """
    Валидирует структуру и содержимое конфигурации.
    
    Args:
        config: Словарь конфигурации
        config_dir: Директория где лежит конфиг (для относительных путей)
        
    Raises:
        ConfigValidationError: Если конфиг некорректен
    """
    # Проверка обязательных секций
    required_sections = ['general', 'output_structure', 'stages']
    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: {section}")
    
    # Валидация general секции
    _validate_general_section(config['general'])
    
    # Валидация output_structure
    _validate_output_structure(config['output_structure'])
    
    # Валидация stages
    _validate_stages_section(config['stages'], config_dir)
    
    # Валидация parallel_stages (опционально)
    if 'parallel_stages' in config:
        _validate_parallel_stages(config['parallel_stages'], config['stages'])


def _validate_general_section(general: Dict[str, Any]) -> None:
    """Валидация секции general."""
    required_fields = ['root_input_dir', 'root_output_dir']
    
    for field in required_fields:
        if field not in general:
            raise ConfigValidationError(f"Missing required field in 'general': {field}")
    
    # Проверка типов
    if not isinstance(general.get('max_subjects'), (int, type(None))):
        raise ConfigValidationError("'max_subjects' must be an integer or null")
    
    if not isinstance(general.get('skip_existing'), bool):
        raise ConfigValidationError("'skip_existing' must be a boolean")


def _validate_output_structure(output_structure: Dict[str, str]) -> None:
    """Валидация секции output_structure."""
    required_keys = ['stage_01', 'stage_02', 'stage_03', 'stage_04', 
                     'stage_05_data', 'stage_06', 'stage_07', 'stage_08',
                     'logs', 'reports']
    
    for key in required_keys:
        if key not in output_structure:
            raise ConfigValidationError(f"Missing key in 'output_structure': {key}")
        
        if not isinstance(output_structure[key], str):
            raise ConfigValidationError(f"'{key}' in output_structure must be a string")


def _validate_stages_section(stages: Dict[str, Any], config_dir: Path) -> None:
    """Валидация секции stages."""
    required_stages = ['stage_01_reorganize', 'stage_02_metadata',
                       'stage_03_convert', 'stage_04_quality',
                       'stage_05_preprocessing', 'stage_06_segmentation',
                       'stage_07_inverse_transform', 'stage_08_anatomical_analysis']
    
    for stage_name in required_stages:
        if stage_name not in stages:
            raise ConfigValidationError(f"Missing required stage: {stage_name}")
        
        stage = stages[stage_name]
        
        # Проверка обязательных полей
        if 'enabled' not in stage:
            raise ConfigValidationError(f"Missing 'enabled' in {stage_name}")
        
        if not isinstance(stage['enabled'], bool):
            raise ConfigValidationError(f"'enabled' must be boolean in {stage_name}")
        
        if 'script' not in stage:
            raise ConfigValidationError(f"Missing 'script' in {stage_name}")
        
        # Проверка существования скрипта (только для включённых этапов)
        if stage['enabled']:
            script_path = config_dir / stage['script']
            if not script_path.exists():
                raise ConfigValidationError(
                    f"Script not found for {stage_name}: {script_path}"
                )
        
        if 'args' not in stage:
            raise ConfigValidationError(f"Missing 'args' in {stage_name}")
        
        if not isinstance(stage['args'], dict):
            raise ConfigValidationError(f"'args' must be a dict in {stage_name}")


def _validate_parallel_stages(parallel_stages: Dict[str, Any], 
                               stages: Dict[str, Any]) -> None:
    """Валидация секции parallel_stages."""
    if 'enabled' not in parallel_stages:
        raise ConfigValidationError("Missing 'enabled' in parallel_stages")
    
    if not isinstance(parallel_stages['enabled'], bool):
        raise ConfigValidationError("'enabled' must be boolean in parallel_stages")
    
    if parallel_stages['enabled']:
        if 'groups' not in parallel_stages:
            raise ConfigValidationError("Missing 'groups' in parallel_stages")
        
        if not isinstance(parallel_stages['groups'], list):
            raise ConfigValidationError("'groups' must be a list in parallel_stages")
        
        # Проверка корректности номеров этапов в группах
        for group in parallel_stages['groups']:
            if not isinstance(group, list):
                raise ConfigValidationError("Each group must be a list of stage numbers")
            
            for stage_num in group:
                if not isinstance(stage_num, int) or stage_num < 1 or stage_num > 6:
                    raise ConfigValidationError(
                        f"Invalid stage number in parallel group: {stage_num}"
                    )


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует строковые пути в Path объекты и разрешает относительные пути.
    
    Args:
        config: Словарь конфигурации
        
    Returns:
        Конфиг с разрешенными путями
    """
    # Преобразуем root пути
    config['general']['root_input_dir'] = Path(config['general']['root_input_dir']).resolve()
    config['general']['root_output_dir'] = Path(config['general']['root_output_dir']).resolve()
    
    return config


def get_stage_input_dir(stage_name: str, config: Dict[str, Any]) -> Path:
    """
    Определяет входную директорию для этапа.
    
    Args:
        stage_name: Имя этапа (например, 'stage_02_metadata')
        config: Конфигурация
        
    Returns:
        Path к входной директории
    """
    root_input = config['general']['root_input_dir']
    root_output = config['general']['root_output_dir']
    output_struct = config['output_structure']
    
    # Маппинг: какой этап использует чей выход
    input_mapping = {
        'stage_01_reorganize': root_input,
        'stage_02_metadata': root_output / output_struct['stage_01'],
        'stage_03_convert': root_output / output_struct['stage_01'],
        'stage_04_quality': root_output / output_struct['stage_03'],
        'stage_05_preprocessing': root_output / output_struct['stage_03'],
        'stage_06_segmentation': root_output / output_struct['stage_05_data'],
        'stage_07_inverse_transform': root_output / output_struct['stage_06'],
        'stage_08_anatomical_analysis': root_output / output_struct['stage_06'],
    }
    
    return input_mapping[stage_name]


def get_stage_output_dir(stage_name: str, config: Dict[str, Any]) -> Path:
    """
    Определяет выходную директорию для этапа.
    
    Args:
        stage_name: Имя этапа
        config: Конфигурация
        
    Returns:
        Path к выходной директории
    """
    root_output = config['general']['root_output_dir']
    output_struct = config['output_structure']
    
    # Маппинг этапов на выходные директории
    output_mapping = {
        'stage_01_reorganize': root_output / output_struct['stage_01'],
        'stage_02_metadata': root_output / output_struct['stage_02'],
        'stage_03_convert': root_output / output_struct['stage_03'],
        'stage_04_quality': root_output / output_struct['stage_04'],
        'stage_05_preprocessing': root_output / output_struct['stage_05_data'],
        'stage_06_segmentation': root_output / output_struct['stage_06'],
        'stage_07_inverse_transform': root_output / output_struct['stage_07'],
        'stage_08_anatomical_analysis': root_output / output_struct['stage_08'],
    }
    
    return output_mapping[stage_name]


def get_enabled_stages(config: Dict[str, Any]) -> List[str]:
    """
    Возвращает список включенных этапов в правильном порядке.

    Args:
        config: Конфигурация

    Returns:
        Список имен включенных этапов
    """
    all_stages = [
        'stage_01_reorganize',
        'stage_02_metadata',
        'stage_03_convert',
        'stage_04_quality',
        'stage_05_preprocessing',
        'stage_06_segmentation',
        'stage_07_inverse_transform',
        'stage_08_anatomical_analysis'
    ]

    return [stage for stage in all_stages if config['stages'][stage]['enabled']]


def load_lesion_type_config(lesion_type: str) -> Dict[str, Any]:
    """
    Load per-lesion-type configuration from configs/lesion_types.yaml.

    Args:
        lesion_type: Name of the lesion type (e.g., 'glioblastoma', 'multiple_sclerosis')

    Returns:
        Dict with keys: required_modalities, reference_modality, reports.

    Raises:
        ConfigValidationError: If the file is missing or YAML is invalid.
        KeyError: If lesion_type not found in the YAML.
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'lesion_types.yaml'

    # Check file existence
    if not config_path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    # Load and parse YAML with error handling
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Failed to parse YAML: {e}")

    if lesion_type not in all_configs:
        raise KeyError(
            f"Unknown lesion_type '{lesion_type}'. "
            f"Available: {list(all_configs.keys())}"
        )
    return all_configs[lesion_type]


def load_series_scoring_config() -> Dict[str, Any]:
    """
    Load series scoring configuration from configs/series_scoring.yaml.

    Used by Stage 01 (scripts/01_reorganize_folders.py) to select the best
    series among multiple candidates for the same modality, and to exclude
    non-brain anatomy series. See KI-027, KI-001 in KNOWN_ISSUES.md.

    Returns:
        Dict with keys: anatomy_exclude, failure_markers, text_weights,
        resolution_scoring, flair_ti_bonus.

    Raises:
        ConfigValidationError: If the file is missing or YAML is invalid.
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'series_scoring.yaml'

    if not config_path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Failed to parse YAML: {e}")

    return config