"""Восстановление строк реестра из дампа, снятого перед догрузкой.

Скрипт лежит в scripts/, а тест здесь: ему нужна настроенная БД из
backend/conftest.py. Отдельный scripts/conftest.py потребовал бы
sys.path.insert на backend/, и тогда `import config` в тестах scripts/ начал
бы разрешаться в backend/config.py.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from database import SessionLocal
from patient_registry import ensure_tables, find_by_study_hash, register_patient
from registry_models import PatientRegistry

_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "restore_registry_from_dump.py")


def _load_script():
    """Import the one-off script by path, without touching sys.path."""
    spec = importlib.util.spec_from_file_location(
        "restore_registry_from_dump", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


restore = _load_script().restore


def _clean():
    db = SessionLocal()
    try:
        db.query(PatientRegistry).filter(
            PatientRegistry.study_hash.like("restore_%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


@pytest.fixture
def registry_with_row_on_375():
    """One study already delivered to dataset 375, as the backfill left it."""
    ensure_tables()
    _clean()
    register_patient(
        bids_id="sub-009_ses-001",
        study_hash="restore_one",
        original_patient_id="P-9",
        kappa_entity_id="entity-375",
        kappa_dataset_id=375,
        pipeline_run_id="run-backfill",
        lesion_type="glioblastoma",
    )
    yield
    _clean()


def _dump(tmp_path, **overrides):
    row = {
        "study_hash": "restore_one",
        "bids_id": "sub-009_ses-001",
        "original_patient_id": "P-9",
        "kappa_entity_id": "entity-347",
        "kappa_dataset_id": 347,
    }
    row.update(overrides)
    path = tmp_path / "dump.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    return path


def test_dry_run_changes_nothing(tmp_path, registry_with_row_on_375):
    result = restore(_dump(tmp_path), apply=False)

    assert result["would_restore"] == 1
    assert result["restored"] == 0
    assert len(find_by_study_hash("restore_one")) == 1


def test_apply_recreates_the_old_row_and_keeps_the_new_one(
    tmp_path, registry_with_row_on_375
):
    restore(_dump(tmp_path), apply=True)

    rows = {r["kappa_dataset_id"]: r["kappa_entity_id"]
            for r in find_by_study_hash("restore_one")}
    assert rows == {347: "entity-347", 375: "entity-375"}


def test_running_twice_is_idempotent(tmp_path, registry_with_row_on_375):
    dump = _dump(tmp_path)
    restore(dump, apply=True)
    second = restore(dump, apply=True)

    assert len(find_by_study_hash("restore_one")) == 2
    assert second["already_there"] == 1
    assert second["restored"] == 0


def test_rows_without_an_address_are_skipped(tmp_path, registry_with_row_on_375):
    """Строку без датасета восстанавливать не из чего — это была бы догадка."""
    result = restore(_dump(tmp_path, kappa_dataset_id=None), apply=True)

    assert result["skipped"] == 1
    assert result["restored"] == 0
    assert len(find_by_study_hash("restore_one")) == 1
