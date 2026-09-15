import pytest


@pytest.fixture
def mapping(tmp_path, monkeypatch):
    import kappa_dataset_mapping as m
    monkeypatch.setattr(m, "MAPPING_FILE", tmp_path / "kappa_datasets.yaml")
    return m


def test_set_then_get_exact_and_current(mapping):
    mapping.set_dataset_id(26, "glioblastoma", "3a183dc7", 249)
    assert mapping.get_dataset_id(26, "glioblastoma", "3a183dc7") == 249
    # any other prep for the same user falls back to :current
    assert mapping.get_dataset_id(26, "glioblastoma", "whatever") == 249


def test_users_are_isolated(mapping):
    mapping.set_dataset_id(26, "glioblastoma", "3a183dc7", 249)
    # a different user has no mapping yet -> miss (triggers create upstream)
    assert mapping.get_dataset_id(52, "glioblastoma", "3a183dc7") is None
    assert mapping.get_dataset_id(52, "glioblastoma", "current") is None


def test_missing_user_id_returns_none(mapping):
    mapping.set_dataset_id(26, "glioblastoma", "3a183dc7", 249)
    assert mapping.get_dataset_id(None, "glioblastoma", "3a183dc7") is None


def test_get_lesion_types_attaches_per_user_dataset_id(mapping):
    mapping.set_dataset_id(26, "glioblastoma", "3a183dc7", 249)
    # seed the lesion_types list the file normally carries
    data = mapping._load_mapping()
    data["lesion_types"] = [{"id": "glioblastoma", "name": "Глиобластома"},
                            {"id": "multiple_sclerosis", "name": "РС"}]
    mapping._save_mapping(data)

    for lt in mapping.get_lesion_types(26):
        if lt["id"] == "glioblastoma":
            assert lt["dataset_id"] == 249
        else:
            assert lt["dataset_id"] is None
    # a fresh user sees no datasets yet
    assert all(lt["dataset_id"] is None for lt in mapping.get_lesion_types(52))
