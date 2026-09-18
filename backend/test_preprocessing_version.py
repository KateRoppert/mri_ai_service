"""
Тесты для версионирования препроцессинга и маппинга датасетов.
Запуск: python -m pytest test_preprocessing_version.py -v
или:    python test_preprocessing_version.py
"""
import json
import tempfile
from pathlib import Path

import yaml


def test_compute_id_stability():
    """Один и тот же конфиг даёт одинаковый hash."""
    from preprocessing_version import compute_preprocessing_id

    config_path = Path(__file__).parent.parent / "configs" / "preprocessing_config.yaml"
    if not config_path.exists():
        print(f"SKIP: config not found at {config_path}")
        return

    id1 = compute_preprocessing_id(str(config_path))
    id2 = compute_preprocessing_id(str(config_path))

    assert id1 == id2, f"Hash should be stable: {id1} != {id2}"
    assert len(id1) == 8, f"Hash should be 8 chars: {id1}"
    print(f"OK: stable hash = {id1}")


def test_compute_id_ignores_paths():
    """Изменение fsl_dir не влияет на hash."""
    from preprocessing_version import compute_preprocessing_id

    config_base = {
        "fsl": {"fsl_dir": "/path/A"},
        "atlas": {"name": "SRI24", "filename": "sri24.nii.gz", "url": "http://a", "cache_dir": "/tmp"},
        "steps": [{"name": "reorient", "enabled": True, "params": {"target_orientation": "LAS"}}],
        "modalities": ["t1", "t2"],
        "logging": {"level": "INFO"},
    }

    config_changed = dict(config_base)
    config_changed["fsl"] = {"fsl_dir": "/path/B"}
    config_changed["atlas"] = dict(config_base["atlas"])
    config_changed["atlas"]["url"] = "http://b"
    config_changed["atlas"]["cache_dir"] = "/other"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
        yaml.dump(config_base, f1)
        path1 = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
        yaml.dump(config_changed, f2)
        path2 = f2.name

    id1 = compute_preprocessing_id(path1)
    id2 = compute_preprocessing_id(path2)

    Path(path1).unlink()
    Path(path2).unlink()

    assert id1 == id2, f"Hash should ignore paths: {id1} != {id2}"
    print(f"OK: paths ignored, hash = {id1}")


def test_compute_id_changes_on_param_change():
    """Изменение параметра шага меняет hash."""
    from preprocessing_version import compute_preprocessing_id

    config1 = {
        "atlas": {"name": "SRI24", "filename": "sri24.nii.gz"},
        "steps": [{"name": "reorient", "enabled": True, "params": {"target_orientation": "LAS"}}],
        "modalities": ["t1", "t2"],
    }

    config2 = {
        "atlas": {"name": "SRI24", "filename": "sri24.nii.gz"},
        "steps": [{"name": "reorient", "enabled": True, "params": {"target_orientation": "RAS"}}],
        "modalities": ["t1", "t2"],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f1:
        yaml.dump(config1, f1)
        path1 = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f2:
        yaml.dump(config2, f2)
        path2 = f2.name

    id1 = compute_preprocessing_id(path1)
    id2 = compute_preprocessing_id(path2)

    Path(path1).unlink()
    Path(path2).unlink()

    assert id1 != id2, f"Hash should differ on param change: {id1} == {id2}"
    print(f"OK: different params → different hash ({id1} vs {id2})")


# The versions registry and the Kappa mapping live in configs/, which is
# bind-mounted into the running web container. Tests used to overwrite the real
# files and restore them afterwards — while the stack was up, the service could
# read a test's temporary mapping in that window, and test_real_config never
# restored at all, leaving a stray version in a tracked file. Every test below
# points the module at a temp file instead.


def test_register_and_retrieve_version(tmp_path, monkeypatch):
    """Регистрация версии и получение конфига по ID."""
    import preprocessing_version
    from preprocessing_version import register_version, get_version_config

    monkeypatch.setattr(preprocessing_version, "VERSIONS_FILE",
                        tmp_path / "preprocessing_versions.json")

    config = {
        "atlas": {"name": "TestAtlas", "filename": "test.nii.gz"},
        "steps": [{"name": "test_step", "enabled": True, "params": {"value": 42}}],
        "modalities": ["t1"],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    prep_id = register_version(str(config_path))
    assert len(prep_id) == 8

    retrieved = get_version_config(prep_id)
    assert retrieved is not None
    assert retrieved["steps"][0]["params"]["value"] == 42

    # Повторная регистрация не должна ломать
    assert register_version(str(config_path)) == prep_id


def test_dataset_mapping(tmp_path, monkeypatch):
    """Маппинг (user_id + lesion_type + preprocessing_id) → dataset_id."""
    import kappa_dataset_mapping
    from kappa_dataset_mapping import get_dataset_id, set_dataset_id, get_lesion_types

    monkeypatch.setattr(kappa_dataset_mapping, "MAPPING_FILE",
                        tmp_path / "kappa_datasets.yaml")

    # smoke: не падает для пользователя без датасетов
    get_lesion_types(user_id=1)

    set_dataset_id(1, "glioblastoma", "abc12345", 999)
    assert get_dataset_id(1, "glioblastoma", "abc12345") == 999
    assert get_dataset_id(1, "glioblastoma", "other") == 999   # :current fallback
    assert get_dataset_id(2, "glioblastoma", "abc12345") is None  # user isolation


def test_real_config(tmp_path, monkeypatch):
    """The project's real config gets an id, and the real mapping resolves it.

    The mapping half is the guard that matters: on 2026-09-18 an image built
    before the per-user key format met a kappa_datasets.yaml written after it,
    every lookup returned None, and the validation tab went silently empty for
    every account. Reading the real file with the current code catches a format
    drift like that at test time.
    """
    import preprocessing_version
    from preprocessing_version import compute_preprocessing_id, register_version
    from kappa_dataset_mapping import get_dataset_id

    config_path = Path(__file__).parent.parent / "configs" / "preprocessing_config.yaml"
    assert config_path.exists(), f"config not found at {config_path}"

    monkeypatch.setattr(preprocessing_version, "VERSIONS_FILE",
                        tmp_path / "preprocessing_versions.json")

    prep_id = compute_preprocessing_id(str(config_path))
    assert register_version(str(config_path)) == prep_id

    # Read-only against the real mapping: the owner account must resolve.
    assert get_dataset_id(26, "glioblastoma", prep_id) is not None
    assert get_dataset_id(26, "multiple_sclerosis", prep_id) is not None


if __name__ == "__main__":
    # The tests take pytest fixtures (tmp_path, monkeypatch), so run them
    # through pytest rather than calling them by hand.
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
