"""
Тесты для orchestrator.py
"""

from pathlib import Path
import tempfile
import pytest
import yaml
import sys
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Добавляем пути
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from orchestrator import (
    build_command,
    run_stage,
    check_if_critical_error,
    format_elapsed_time,
    create_output_directories,
    setup_logger
)


def create_test_config() -> dict:
    """Создает тестовый конфиг."""
    return {
        'general': {
            'root_input_dir': Path('/tmp/input'),
            'root_output_dir': Path('/tmp/output'),
            'max_subjects': 10,
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
                'args': {
                    'mode': 'parallel',
                    'workers': 4,
                    'force': False,
                    'benchmark': True
                }
            },
            'stage_02_metadata': {
                'enabled': True,
                'script': 'scripts/02_extract_metadata.py',
                'args': {
                    'config': 'config/dicom_tags.yaml',
                    'mode': 'sequential',
                    'workers': 1
                }
            }
        }
    }


def test_build_command_basic():
    """Тест построения базовой команды."""
    config = create_test_config()
    project_root = Path('/project')
    
    cmd = build_command('stage_01_reorganize', config, project_root)
    
    # Проверяем структуру команды
    assert cmd[0] == 'python'
    assert 'scripts/01_reorganize_folders.py' in cmd[1]
    assert '/tmp/input' in cmd  # input_dir
    assert '/tmp/output/bids' in cmd  # output_dir
    assert '--mode' in cmd
    assert 'parallel' in cmd
    assert '--workers' in cmd
    assert '4' in cmd
    assert '--benchmark' in cmd
    assert '--max-subjects' in cmd
    assert '10' in cmd


def test_build_command_with_config_arg():
    """Тест построения команды с config аргументом."""
    config = create_test_config()
    project_root = Path('/project')
    
    cmd = build_command('stage_02_metadata', config, project_root)
    
    # Проверяем что config путь преобразован
    assert '--config' in cmd
    config_idx = cmd.index('--config')
    config_path = cmd[config_idx + 1]
    assert 'config/dicom_tags.yaml' in config_path
    assert config_path.startswith('/project')


def test_build_command_skips_false_and_none():
    """Тест что False и None аргументы не добавляются."""
    config = create_test_config()
    config['stages']['stage_01_reorganize']['args']['force'] = False
    config['stages']['stage_01_reorganize']['args']['dry_run'] = None
    
    project_root = Path('/project')
    cmd = build_command('stage_01_reorganize', config, project_root)
    
    # force=False не должно быть в команде
    assert '--force' not in cmd
    # dry_run=None не должно быть в команде
    assert '--dry-run' not in cmd


def test_format_elapsed_time():
    """Тест форматирования времени."""
    assert format_elapsed_time(45) == "45s"
    assert format_elapsed_time(90) == "1m 30s"
    assert format_elapsed_time(3665) == "1h 1m 5s"
    assert format_elapsed_time(7200) == "2h 0m 0s"


def test_check_if_critical_error():
    """Тест определения критичности ошибки."""
    logger = Mock()
    
    # Критичная ошибка - FileNotFoundError
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=['python'],
        stderr="FileNotFoundError: No such file"
    )
    assert check_if_critical_error(error, 'stage_01', logger) == True
    
    # Критичная ошибка - высокий return code
    error = subprocess.CalledProcessError(
        returncode=3,
        cmd=['python'],
        stderr="Some error"
    )
    assert check_if_critical_error(error, 'stage_01', logger) == True
    
    # Некритичная - есть признаки успеха
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=['python'],
        stderr="Warning: some issues, but 10 subjects processed successfully"
    )
    assert check_if_critical_error(error, 'stage_01', logger) == False


def test_create_output_directories(tmp_path):
    """Тест создания структуры директорий."""
    config = create_test_config()
    config['general']['root_output_dir'] = tmp_path
    
    create_output_directories(config)
    
    # Проверяем что директории созданы
    assert (tmp_path / 'bids').exists()
    assert (tmp_path / 'metadata').exists()
    assert (tmp_path / 'nifti').exists()
    assert (tmp_path / 'logs').exists()
    assert (tmp_path / 'logs' / 'stages').exists()
    assert (tmp_path / 'reports').exists()


def test_setup_logger(tmp_path):
    """Тест настройки logger."""
    log_file = tmp_path / 'test.log'
    
    logger = setup_logger(log_file)
    
    assert logger is not None
    assert logger.name == 'pipeline_master'
    assert log_file.exists()
    
    # Тестовое сообщение
    logger.info("Test message")
    
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Test message" in content


@patch('orchestrator.subprocess.run')
def test_run_stage_success(mock_run):
    """Тест успешного выполнения этапа."""
    # Мокируем успешный subprocess
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
    
    config = create_test_config()
    project_root = Path('/project')
    logger = Mock()
    
    result = run_stage('stage_01_reorganize', config, project_root, logger, dry_run=False)
    
    assert result['status'] == 'SUCCESS'
    assert result['time'] > 0
    assert result['critical'] == False
    assert 'command' in result


@patch('orchestrator.subprocess.run')
def test_run_stage_failure(mock_run):
    """Тест провального выполнения этапа."""
    # Мокируем ошибку subprocess
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=['python'],
        stderr="FileNotFoundError: Input not found"
    )
    
    config = create_test_config()
    project_root = Path('/project')
    logger = Mock()
    
    result = run_stage('stage_01_reorganize', config, project_root, logger, dry_run=False)
    
    assert result['status'] == 'FAILED'
    assert result['critical'] == True
    assert 'error' in result


def test_run_stage_disabled():
    """Тест пропуска отключенного этапа."""
    config = create_test_config()
    config['stages']['stage_01_reorganize']['enabled'] = False
    
    project_root = Path('/project')
    logger = Mock()
    
    result = run_stage('stage_01_reorganize', config, project_root, logger, dry_run=False)
    
    assert result['status'] == 'SKIPPED'
    assert result['time'] == 0.0


def test_run_stage_dry_run():
    """Тест dry-run режима."""
    config = create_test_config()
    project_root = Path('/project')
    logger = Mock()
    
    result = run_stage('stage_01_reorganize', config, project_root, logger, dry_run=True)
    
    assert result['status'] == 'DRY_RUN'
    assert result['time'] == 0.0
    assert 'command' in result
    # subprocess.run не должен вызываться
    assert result['command'] != ''


@patch('orchestrator.subprocess.run')
def test_run_stage_with_log_file(mock_run, tmp_path):
    """Тест что лог-файл создается."""
    mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
    
    config = create_test_config()
    config['general']['root_output_dir'] = tmp_path
    
    # Создаем директорию для логов
    (tmp_path / 'logs' / 'stages').mkdir(parents=True)
    
    project_root = Path('/project')
    logger = Mock()
    
    result = run_stage('stage_01_reorganize', config, project_root, logger, dry_run=False)
    
    # Проверяем что в команде есть --log_file
    assert '--log_file' in result['command']
    assert 'stage_01_reorganize.log' in result['command']


def test_build_command_boolean_args():
    """Тест обработки булевых аргументов."""
    config = create_test_config()
    config['stages']['stage_01_reorganize']['args']['force'] = True
    config['stages']['stage_01_reorganize']['args']['validate'] = True
    
    project_root = Path('/project')
    cmd = build_command('stage_01_reorganize', config, project_root)
    
    # True аргументы должны быть флагами без значений
    assert '--force' in cmd
    assert '--validate' in cmd
    
    # Проверяем что после флагов нет "True"
    force_idx = cmd.index('--force')
    # Следующий элемент не должен быть "True"
    if force_idx + 1 < len(cmd):
        assert cmd[force_idx + 1] != 'True'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])