"""
Тест загрузки реального конфига.
"""

from pathlib import Path
import sys

# Добавляем путь к utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import (
    load_config,
    resolve_paths,
    get_stage_input_dir,
    get_stage_output_dir,
    get_enabled_stages,
    ConfigValidationError
)


def test_real_config(config_path: str):
    """
    Тестирует загрузку реального конфига.
    
    Args:
        config_path: Путь к реальному pipeline_config.yaml
    """
    print(f"{'='*70}")
    print(f"Testing config: {config_path}")
    print(f"{'='*70}\n")
    
    try:
        # Загрузка конфига
        config = load_config(Path(config_path))
        print("✓ Config loaded successfully\n")
        
        # Разрешение путей
        config = resolve_paths(config)
        print("✓ Paths resolved\n")
        
        # Вывод общей информации
        print("GENERAL SETTINGS:")
        print(f"  Root input:  {config['general']['root_input_dir']}")
        print(f"  Root output: {config['general']['root_output_dir']}")
        print(f"  Max subjects: {config['general']['max_subjects']}")
        print(f"  Skip existing: {config['general']['skip_existing']}\n")
        
        # Вывод включенных этапов
        enabled_stages = get_enabled_stages(config)
        print(f"ENABLED STAGES ({len(enabled_stages)}/6):")
        for stage_name in enabled_stages:
            print(f"  ✓ {stage_name}")
        print()
        
        # Вывод отключенных этапов
        all_stages = [
            'stage_01_reorganize',
            'stage_02_metadata',
            'stage_03_convert',
            'stage_04_quality',
            'stage_05_preprocessing',
            'stage_06_segmentation'
        ]
        disabled_stages = [s for s in all_stages if s not in enabled_stages]
        if disabled_stages:
            print(f"DISABLED STAGES ({len(disabled_stages)}/6):")
            for stage_name in disabled_stages:
                print(f"  ✗ {stage_name}")
            print()
        
        # Вывод путей для каждого этапа
        print("STAGE PATHS:")
        for stage_name in all_stages:
            enabled = config['stages'][stage_name]['enabled']
            status = "✓" if enabled else "✗"
            input_dir = get_stage_input_dir(stage_name, config)
            output_dir = get_stage_output_dir(stage_name, config)
            print(f"  {status} {stage_name}:")
            print(f"      Input:  {input_dir}")
            print(f"      Output: {output_dir}")
        print()
        
        # Вывод аргументов для включенных этапов
        print("STAGE ARGUMENTS:")
        for stage_name in enabled_stages:
            stage_config = config['stages'][stage_name]
            print(f"  {stage_name}:")
            print(f"    Script: {stage_config['script']}")
            print(f"    Args:")
            for arg_name, arg_value in stage_config['args'].items():
                print(f"      --{arg_name}: {arg_value}")
        print()
        
        # Проверка parallel_stages
        if 'parallel_stages' in config:
            parallel = config['parallel_stages']
            print("PARALLEL STAGES:")
            print(f"  Enabled: {parallel['enabled']}")
            if parallel['enabled'] and 'groups' in parallel:
                print(f"  Groups: {parallel['groups']}")
            print()
        
        # Проверка reporting
        if 'reporting' in config:
            reporting = config['reporting']
            print("REPORTING:")
            print(f"  Enabled: {reporting['enabled']}")
            if reporting['enabled']:
                print(f"  Formats: {reporting.get('formats', [])}")
                print(f"  Include: {reporting.get('include', [])}")
            print()
        
        print(f"{'='*70}")
        print("✓ ALL CHECKS PASSED")
        print(f"{'='*70}")
        
    except ConfigValidationError as e:
        print(f"\n✗ CONFIG VALIDATION ERROR:")
        print(f"  {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR:")
        print(f"  {type(e).__name__}: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test real pipeline config")
    parser.add_argument(
        "config",
        type=str,
        help="Path to pipeline_config.yaml"
    )
    
    args = parser.parse_args()
    test_real_config(args.config)