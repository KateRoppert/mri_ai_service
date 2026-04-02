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


def test_register_and_retrieve_version():
    """Регистрация версии и получение конфига по ID."""
    from preprocessing_version import (
        register_version,
        get_version_config,
        VERSIONS_FILE,
    )

    config = {
        "atlas": {"name": "TestAtlas", "filename": "test.nii.gz"},
        "steps": [{"name": "test_step", "enabled": True, "params": {"value": 42}}],
        "modalities": ["t1"],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    # Сохраняем текущий файл версий
    original_versions = None
    if VERSIONS_FILE.exists():
        original_versions = VERSIONS_FILE.read_text()

    try:
        prep_id = register_version(config_path)
        assert len(prep_id) == 8

        retrieved = get_version_config(prep_id)
        assert retrieved is not None
        assert retrieved["steps"][0]["params"]["value"] == 42

        # Повторная регистрация не должна ломать
        prep_id2 = register_version(config_path)
        assert prep_id == prep_id2

        print(f"OK: registered and retrieved version {prep_id}")

    finally:
        Path(config_path).unlink()
        # Восстанавливаем
        if original_versions is not None:
            VERSIONS_FILE.write_text(original_versions)
        elif VERSIONS_FILE.exists():
            VERSIONS_FILE.unlink()


def test_dataset_mapping():
    """Маппинг lesion_type + preprocessing_id → dataset_id."""
    from kappa_dataset_mapping import (
        get_dataset_id,
        set_dataset_id,
        get_lesion_types,
        MAPPING_FILE,
    )

    # Тест чтения существующего маппинга
    lesion_types = get_lesion_types()
    print(f"Lesion types: {lesion_types}")

    # Тест получения dataset_id для glioblastoma:current
    ds_id = get_dataset_id("glioblastoma", "any_hash")
    print(f"glioblastoma:any_hash → {ds_id} (should be 133 via current fallback)")
    assert ds_id == 133, f"Expected 133, got {ds_id}"

    # Тест установки конкретного маппинга
    original = MAPPING_FILE.read_text() if MAPPING_FILE.exists() else None

    try:
        set_dataset_id("glioblastoma", "abc12345", 999)
        ds_id_exact = get_dataset_id("glioblastoma", "abc12345")
        assert ds_id_exact == 999, f"Expected 999, got {ds_id_exact}"

        print(f"OK: glioblastoma:abc12345 → {ds_id_exact}")

    finally:
        # Восстанавливаем
        if original is not None:
            MAPPING_FILE.write_text(original)


def test_real_config():
    """Тест с реальным конфигом проекта."""
    from preprocessing_version import compute_preprocessing_id, register_version

    config_path = Path(__file__).parent.parent / "configs" / "preprocessing_config.yaml"
    if not config_path.exists():
        print(f"SKIP: config not found at {config_path}")
        return

    prep_id = compute_preprocessing_id(str(config_path))
    print(f"Current preprocessing_id: {prep_id}")

    # Регистрируем
    registered_id = register_version(str(config_path))
    assert prep_id == registered_id

    # Проверяем маппинг
    from kappa_dataset_mapping import get_dataset_id
    ds_id = get_dataset_id("glioblastoma", prep_id)
    print(f"Dataset ID for glioblastoma:{prep_id} → {ds_id}")

    print("OK: real config test passed")


if __name__ == "__main__":
    print("=== test_compute_id_stability ===")
    test_compute_id_stability()

    print("\n=== test_compute_id_ignores_paths ===")
    test_compute_id_ignores_paths()

    print("\n=== test_compute_id_changes_on_param_change ===")
    test_compute_id_changes_on_param_change()

    print("\n=== test_register_and_retrieve_version ===")
    test_register_and_retrieve_version()

    print("\n=== test_dataset_mapping ===")
    test_dataset_mapping()

    print("\n=== test_real_config ===")
    test_real_config()

    print("\n=== Все тесты пройдены ===")