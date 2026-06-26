"""
Тесты для config_loader.py
"""

from pathlib import Path
import tempfile
import pytest
import yaml
import sys

# Добавляем путь к utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import (
    load_config,
    validate_config,
    resolve_paths,
    get_stage_input_dir,
    get_stage_output_dir,
    get_enabled_stages,
    load_series_scoring_config,
    ConfigValidationError
)


def create_test_config(tmp_path: Path, config_dict: dict) -> Path:
    """Создает временный конфиг файл для тестирования."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)
    return config_file


def create_minimal_valid_config() -> dict:
    """Создает минимальный валидный конфиг."""
    return {
        'general': {
            'root_input_dir': '/tmp/input',
            'root_output_dir': '/tmp/output',
            'max_subjects': None,
            'skip_existing': True
        },
        'output_structure': {
            'stage_01': 'bids',
            'stage_02': 'metadata',
            'stage_03': 'nifti',
            'stage_04': 'quality',
            'stage_05_data': 'preprocessed',
            'stage_05_transforms': 'transforms',
            'stage_06': 'segmentation',
            'stage_07': 'segmentation',
            'stage_08': 'segmentation',
            'logs': 'logs',
            'reports': 'reports'
        },
        'stages': {
            'stage_01_reorganize': {
                'enabled': True,
                'script': 'scripts/01_reorganize_folders.py',
                'args': {}
            },
            'stage_02_metadata': {
                'enabled': True,
                'script': 'scripts/02_extract_metadata.py',
                'args': {}
            },
            'stage_03_convert': {
                'enabled': True,
                'script': 'scripts/03_convert_to_nifti.py',
                'args': {}
            },
            'stage_04_quality': {
                'enabled': True,
                'script': 'scripts/04_assess_quality.py',
                'args': {}
            },
            'stage_05_preprocessing': {
                'enabled': True,
                'script': 'scripts/05_preprocessing.py',
                'args': {}
            },
            'stage_06_segmentation': {
                'enabled': True,
                'script': 'scripts/06_segmentation.py',
                'args': {}
            },
            'stage_07_inverse_transform': {
                'enabled': True,
                'script': 'scripts/07_inverse_transform.py',
                'args': {}
            },
            'stage_08_anatomical_analysis': {
                'enabled': True,
                'script': 'scripts/08_anatomical_analysis.py',
                'args': {}
            }
        }
    }


def create_script_files(tmp_path: Path) -> Path:
    """Создает структуру с фейковыми скриптами."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    
    # Создаем файлы с правильными именами
    script_names = [
        '01_reorganize_folders.py',
        '02_extract_metadata.py',
        '03_convert_to_nifti.py',
        '04_assess_quality.py',
        '05_preprocessing.py',
        '06_segmentation.py',
        '07_inverse_transform.py',
        '08_anatomical_analysis.py'
    ]
    
    for script_name in script_names:
        (scripts_dir / script_name).touch()
    
    return scripts_dir


def test_load_valid_config(tmp_path):
    """Тест загрузки корректного конфига."""
    # Создаем структуру скриптов
    scripts_dir = create_script_files(tmp_path)
    
    config_dict = create_minimal_valid_config()
    config_file = create_test_config(tmp_path, config_dict)
    
    # Загружаем конфиг
    config = load_config(config_file)
    
    assert config is not None
    assert 'general' in config
    assert 'stages' in config


def test_missing_required_section(tmp_path):
    """Тест отсутствия обязательной секции."""
    create_script_files(tmp_path)
    
    config_dict = create_minimal_valid_config()
    del config_dict['general']  # Удаляем обязательную секцию
    
    config_file = create_test_config(tmp_path, config_dict)
    
    with pytest.raises(ConfigValidationError, match="Missing required section: general"):
        load_config(config_file)


def test_missing_required_field_in_general(tmp_path):
    """Тест отсутствия обязательного поля в general."""
    create_script_files(tmp_path)
    
    config_dict = create_minimal_valid_config()
    del config_dict['general']['root_input_dir']
    
    config_file = create_test_config(tmp_path, config_dict)
    
    with pytest.raises(ConfigValidationError, match="Missing required field"):
        load_config(config_file)


def test_invalid_max_subjects_type(tmp_path):
    """Тест некорректного типа max_subjects."""
    create_script_files(tmp_path)
    
    config_dict = create_minimal_valid_config()
    config_dict['general']['max_subjects'] = "invalid"  # Должно быть int или None
    
    config_file = create_test_config(tmp_path, config_dict)
    
    with pytest.raises(ConfigValidationError, match="max_subjects"):
        load_config(config_file)


def test_missing_stage(tmp_path):
    """Тест отсутствия обязательного этапа."""
    create_script_files(tmp_path)
    
    config_dict = create_minimal_valid_config()
    del config_dict['stages']['stage_01_reorganize']
    
    config_file = create_test_config(tmp_path, config_dict)
    
    with pytest.raises(ConfigValidationError, match="Missing required stage"):
        load_config(config_file)


def test_script_not_found(tmp_path):
    """Тест отсутствия файла скрипта."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    # НЕ создаем файлы скриптов
    
    config_dict = create_minimal_valid_config()
    config_file = create_test_config(tmp_path, config_dict)
    
    with pytest.raises(ConfigValidationError, match="Script not found"):
        load_config(config_file)


def test_resolve_paths():
    """Тест преобразования путей."""
    config = {
        'general': {
            'root_input_dir': '/tmp/input',
            'root_output_dir': '/tmp/output'
        }
    }
    
    resolved = resolve_paths(config)
    
    assert isinstance(resolved['general']['root_input_dir'], Path)
    assert isinstance(resolved['general']['root_output_dir'], Path)


def test_get_stage_input_dir():
    """Тест получения входной директории для этапа."""
    config = {
        'general': {
            'root_input_dir': Path('/tmp/input'),
            'root_output_dir': Path('/tmp/output')
        },
        'output_structure': {
            'stage_01': 'bids',
            'stage_03': 'nifti',
            'stage_05_data': 'preprocessed',
            'stage_06': 'segmentation'
        }
    }
    
    # Этап 1 использует root_input
    assert get_stage_input_dir('stage_01_reorganize', config) == Path('/tmp/input')
    
    # Этап 2 использует выход этапа 1
    assert get_stage_input_dir('stage_02_metadata', config) == Path('/tmp/output/bids')
    
    # Этап 4 использует выход этапа 3
    assert get_stage_input_dir('stage_04_quality', config) == Path('/tmp/output/nifti')
    
    # Этап 6 использует выход этапа 5
    assert get_stage_input_dir('stage_06_segmentation', config) == Path('/tmp/output/preprocessed')


def test_get_stage_output_dir():
    """Тест получения выходной директории для этапа."""
    config = {
        'general': {
            'root_output_dir': Path('/tmp/output')
        },
        'output_structure': {
            'stage_01': 'bids',
            'stage_02': 'metadata',
            'stage_03': 'nifti',
            'stage_04': 'quality',
            'stage_05_data': 'preprocessed',
            'stage_06': 'segmentation',
            'stage_07': 'segmentation',
            'stage_08': 'segmentation'
        }
    }

    assert get_stage_output_dir('stage_01_reorganize', config) == Path('/tmp/output/bids')
    assert get_stage_output_dir('stage_02_metadata', config) == Path('/tmp/output/metadata')
    assert get_stage_output_dir('stage_06_segmentation', config) == Path('/tmp/output/segmentation')


def test_get_enabled_stages():
    """Тест получения списка включенных этапов."""
    config = {
        'stages': {
            'stage_01_reorganize': {'enabled': True},
            'stage_02_metadata': {'enabled': False},
            'stage_03_convert': {'enabled': True},
            'stage_04_quality': {'enabled': False},
            'stage_05_preprocessing': {'enabled': True},
            'stage_06_segmentation': {'enabled': True},
            'stage_07_inverse_transform': {'enabled': False},
            'stage_08_anatomical_analysis': {'enabled': False}
        }
    }
    
    enabled = get_enabled_stages(config)
    
    assert len(enabled) == 4
    assert 'stage_01_reorganize' in enabled
    assert 'stage_03_convert' in enabled
    assert 'stage_05_preprocessing' in enabled
    assert 'stage_06_segmentation' in enabled
    assert 'stage_02_metadata' not in enabled
    assert 'stage_04_quality' not in enabled


def test_nonexistent_config_file():
    """Тест загрузки несуществующего файла."""
    with pytest.raises(ConfigValidationError, match="Config file not found"):
        load_config(Path("/nonexistent/config.yaml"))


@pytest.fixture
def _series_scoring_yaml_path():
    """Temporarily overwrites the real configs/series_scoring.yaml and restores it afterward.

    load_series_scoring_config() takes no arguments and always reads from the
    hardcoded path Path(__file__).parent.parent / 'configs' / 'series_scoring.yaml'.
    There is no way to inject a test path, so to genuinely exercise the production
    function we must temporarily mutate the real file. The original content is
    backed up and restored in a finally block so the file is never left corrupted,
    even if the test fails or raises.
    """
    real_path = Path(__file__).parent / 'configs' / 'series_scoring.yaml'
    original_content = real_path.read_text(encoding='utf-8')
    try:
        yield real_path
    finally:
        real_path.write_text(original_content, encoding='utf-8')


def test_load_series_scoring_config_empty_file(_series_scoring_yaml_path):
    """Regression test: empty series_scoring.yaml should raise ConfigValidationError, not return None.

    This tests the behavior when yaml.safe_load returns None for an empty file.
    The function should raise ConfigValidationError instead of returning None,
    which would silently disable scoring and anatomy exclusion downstream.
    """
    _series_scoring_yaml_path.write_text("", encoding='utf-8')

    with pytest.raises(ConfigValidationError, match="must be a non-empty YAML mapping"):
        load_series_scoring_config()


def test_load_series_scoring_config_whitespace_only_file(_series_scoring_yaml_path):
    """Regression test: whitespace-only series_scoring.yaml should raise ConfigValidationError.

    Whitespace-only files also parse to None and should fail with the same error.
    """
    _series_scoring_yaml_path.write_text("   \n  \n   ", encoding='utf-8')

    with pytest.raises(ConfigValidationError, match="must be a non-empty YAML mapping"):
        load_series_scoring_config()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])