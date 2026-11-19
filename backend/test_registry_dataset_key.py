"""Одно исследование может лежать в нескольких датасетах Kappa.

KI-084: реестр ключевался одним study_hash, и выгрузка в другой датасет
перезаписывала kappa_dataset_id/kappa_entity_id, стирая адрес сущности в
прежнем датасете. Наблюдалось на догрузке test500 в датасет 375: строки
перетекли 347 205->167, 348 114->79, 375 0->74 при неизменном общем числе.
"""
from database import SessionLocal
from patient_registry import (
    ensure_tables,
    find_by_bids_id,
    find_by_study_hash,
    register_patient,
)
from registry_models import PatientRegistry


def setup_function():
    ensure_tables()
    db = SessionLocal()
    try:
        db.query(PatientRegistry).filter(
            PatientRegistry.study_hash.like("dskey_%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _register(dataset_id, entity_id, study_hash="dskey_one", bids="sub-007"):
    return register_patient(
        bids_id=f"{bids}_ses-001",
        study_hash=study_hash,
        original_patient_id="P-1",
        kappa_entity_id=entity_id,
        kappa_dataset_id=dataset_id,
        pipeline_run_id="run-x",
        lesion_type="glioblastoma",
    )


def test_same_study_in_two_datasets_keeps_both_addresses():
    _register(347, "entity-347")
    _register(375, "entity-375")

    rows = find_by_study_hash("dskey_one")
    assert len(rows) == 2
    assert {r["kappa_dataset_id"]: r["kappa_entity_id"] for r in rows} == {
        347: "entity-347", 375: "entity-375",
    }


def test_reupload_into_the_same_dataset_updates_one_row():
    _register(347, "entity-a")
    _register(347, "entity-b")

    rows = find_by_study_hash("dskey_one")
    assert len(rows) == 1
    assert rows[0]["kappa_entity_id"] == "entity-b"


def test_undelivered_row_is_claimed_by_the_upload():
    """Передача session_registry -> KappaUploader обязана уцелеть."""
    _register(None, None)            # как пишет session_registry
    _register(375, "entity-375")     # как дополняет выгрузка

    rows = find_by_study_hash("dskey_one")
    assert len(rows) == 1
    assert rows[0]["kappa_dataset_id"] == 375
    assert rows[0]["kappa_entity_id"] == "entity-375"


def test_two_finishes_without_kappa_do_not_duplicate():
    """UNIQUE в SQLite считает NULL различными — отсюда COALESCE в индексе."""
    _register(None, None)
    _register(None, None)

    assert len(find_by_study_hash("dskey_one")) == 1


def test_rows_carry_their_own_dataset():
    _register(347, "entity-347")
    _register(375, "entity-375")

    for row in find_by_study_hash("dskey_one"):
        assert row["kappa_entity_id"] == f"entity-{row['kappa_dataset_id']}"


def test_mask_replacement_lookup_sees_the_right_dataset():
    """Именно этот путь ломался: app.py ищет сущность по bids_id + датасету."""
    _register(347, "entity-347")
    _register(375, "entity-375")

    records = find_by_bids_id("sub-007_ses-001", {347})
    entity = next((r["kappa_entity_id"] for r in records
                   if r.get("kappa_entity_id")), None)
    assert entity == "entity-347"


# --- миграция индекса -------------------------------------------------------
#
# Отдельный движок на временном файле: трогать общую тестовую БД опасно, а
# _migrate_study_hash_index импортирует engine внутри себя, поэтому подмена
# database.engine работает.

_OLD_SCHEMA = """
CREATE TABLE patient_registry (
    id INTEGER NOT NULL PRIMARY KEY,
    study_hash VARCHAR NOT NULL,
    bids_id VARCHAR NOT NULL,
    original_patient_id VARCHAR NOT NULL,
    kappa_entity_id VARCHAR,
    kappa_dataset_id INTEGER
);
CREATE UNIQUE INDEX ix_patient_registry_study_hash
    ON patient_registry (study_hash);
"""


def _old_db(tmp_path, rows=()):
    """A DB in the pre-migration shape, optionally pre-filled."""
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_OLD_SCHEMA)
        for study_hash, dataset_id in rows:
            conn.execute(
                "INSERT INTO patient_registry "
                "(study_hash, bids_id, original_patient_id, kappa_dataset_id) "
                "VALUES (?, 'sub-001_ses-001', 'P', ?)",
                (study_hash, dataset_id),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _indexes(path):
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='patient_registry'"
        )}
    finally:
        conn.close()


def _migrate_against(path, monkeypatch):
    import sqlalchemy

    import database
    import patient_registry as pr

    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(database, "engine", engine)
    try:
        pr._migrate_study_hash_index()
    finally:
        engine.dispose()


def test_migration_swaps_the_index_and_is_idempotent(tmp_path, monkeypatch):
    path = _old_db(tmp_path, rows=[("a", 347), ("b", 348)])

    _migrate_against(path, monkeypatch)
    after = _indexes(path)
    assert "ix_patient_registry_study_dataset" in after
    assert "ix_patient_registry_study_hash" not in after

    _migrate_against(path, monkeypatch)        # второй раз — ничего не делает
    assert _indexes(path) == after


def test_migration_refuses_on_duplicates_and_changes_nothing(
    tmp_path, monkeypatch
):
    """Клинические данные: какая из двух строк верна — решение не автоматики."""
    import sqlite3

    path = _old_db(tmp_path)
    conn = sqlite3.connect(path)
    try:
        # Дубль по (исследование, датасет) — старый индекс такого не держал
        # только потому, что он был по одному study_hash.
        conn.execute("DROP INDEX ix_patient_registry_study_hash")
        for _ in range(2):
            conn.execute(
                "INSERT INTO patient_registry "
                "(study_hash, bids_id, original_patient_id, kappa_dataset_id) "
                "VALUES ('dup', 'sub-001_ses-001', 'P', 347)")
        conn.commit()
    finally:
        conn.close()

    _migrate_against(path, monkeypatch)

    assert "ix_patient_registry_study_dataset" not in _indexes(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM patient_registry").fetchone()[0] == 2
    finally:
        conn.close()
