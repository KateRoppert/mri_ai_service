import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config


def test_glioblastoma_modalities():
    cfg = load_lesion_type_config('glioblastoma')
    assert set(cfg['required_modalities']) == {'t1', 't1c', 't2', 't2fl'}
    assert cfg['reference_modality'] == 't1'
    assert 'volume' in cfg['reports']


def test_multiple_sclerosis_modalities():
    cfg = load_lesion_type_config('multiple_sclerosis')
    assert set(cfg['required_modalities']) == {'t1', 't2', 't2fl'}
    assert 't1c' not in cfg['required_modalities']
    assert cfg['reference_modality'] == 't1'
    assert 'lesion_count' in cfg['reports']


def test_unknown_lesion_type_raises():
    with pytest.raises(KeyError):
        load_lesion_type_config('brain_metastasis')
