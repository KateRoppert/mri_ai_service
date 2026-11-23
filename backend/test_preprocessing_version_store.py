"""По preprocessing_id должны находиться настройки, которыми считали.

KI-085, вторая часть: register_version существовал, но в продакшене не
вызывался нигде, и у трёх из четырёх использованных идентификаторов за ними
не стояло ничего.
"""
import json

import yaml

import preprocessing_version as pv


def _config(path, sigma):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "atlas": {"name": "MNI152", "filename": "mni.nii.gz"},
        "modalities": ["t1", "t1c"],
        "steps": [{"name": "smooth", "enabled": True,
                   "params": {"sigma": sigma}}],
    }), encoding="utf-8")
    return path


def test_register_version_makes_the_id_resolvable(tmp_path):
    path = _config(tmp_path / "c.yaml", sigma=1.0)
    prep_id = pv.register_version(str(path))

    stored = pv.get_version_config(prep_id)
    assert stored is not None
    assert stored["steps"][0]["params"]["sigma"] == 1.0


def test_register_version_is_idempotent(tmp_path):
    path = _config(tmp_path / "c.yaml", sigma=1.0)
    assert pv.register_version(str(path)) == pv.register_version(str(path))


def test_register_version_does_not_touch_the_json_file(tmp_path, monkeypatch):
    """Приложение не имеет права делать рабочую копию грязной."""
    legacy = tmp_path / "preprocessing_versions.json"
    legacy.write_text(json.dumps({"old1234": {"config": {"steps": []}}}),
                      encoding="utf-8")
    before = legacy.read_text(encoding="utf-8")
    monkeypatch.setattr(pv, "VERSIONS_FILE", legacy)

    pv.register_version(str(_config(tmp_path / "c.yaml", sigma=2.0)))

    assert legacy.read_text(encoding="utf-8") == before


def test_legacy_json_is_still_readable(tmp_path, monkeypatch):
    """Единственная унаследованная запись 1e2b93ad должна остаться доступной."""
    legacy = tmp_path / "preprocessing_versions.json"
    legacy.write_text(json.dumps({
        "1e2b93ad": {"created_at": "2026-04-02T08:49:56",
                     "config": {"steps": [{"name": "legacy"}]}},
    }), encoding="utf-8")
    monkeypatch.setattr(pv, "VERSIONS_FILE", legacy)

    assert pv.get_version_config("1e2b93ad")["steps"][0]["name"] == "legacy"


def test_unknown_id_resolves_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "VERSIONS_FILE", tmp_path / "нет.json")
    assert pv.get_version_config("ffffffff") is None


def test_a_broken_config_does_not_raise(tmp_path):
    """Старт прогона важнее метки: битый конфиг не должен валить запуск."""
    broken = tmp_path / "broken.yaml"
    broken.write_text("steps: [unclosed", encoding="utf-8")
    assert pv.register_version(str(broken)) is None
