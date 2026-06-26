import sys
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from utils.config_loader import load_series_scoring_config, ConfigValidationError


def test_loads_real_config_with_expected_top_level_keys():
    config = load_series_scoring_config()
    assert {'anatomy_exclude', 'failure_markers', 'text_weights',
            'resolution_scoring', 'flair_ti_bonus'} <= config.keys()


def test_text_weights_has_all_four_modalities():
    config = load_series_scoring_config()
    assert {'t1', 't1c', 't2', 't2fl'} <= config['text_weights'].keys()


def test_anatomy_exclude_keywords_present():
    config = load_series_scoring_config()
    keywords = config['anatomy_exclude']['keywords']
    assert 'spine' in keywords
    assert 'pituitary' in keywords


def test_missing_file_raises_config_validation_error(monkeypatch, tmp_path):
    import utils.config_loader as config_loader_mod
    fake_module_file = tmp_path / "subdir" / "config_loader.py"
    fake_module_file.parent.mkdir(parents=True)
    monkeypatch.setattr(config_loader_mod, "__file__", str(fake_module_file))
    with pytest.raises(ConfigValidationError):
        config_loader_mod.load_series_scoring_config()
