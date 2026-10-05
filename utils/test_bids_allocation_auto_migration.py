"""
KI-060: an old per-lesion-type allocation table is migrated on first use.

feat/bids-per-dataset moved numbering to per-dataset scopes, but converting
an existing table was a manual script nobody ran. On the demob laptop the
first run after `git pull` died in stage 01 with "no such column: scope"
(2026-09-30). Every user of the table opens it through
bids_allocator._connect, so that is where the migration now happens.
"""
import logging
import sqlite3
import threading

import pytest

from utils import bids_allocator
from utils.bids_allocator import get_allocations, get_bids_id


def _old_db(path, registry_rows, allocation_rows, with_registry=True):
    """A DB as it was before feat/bids-per-dataset."""
    con = sqlite3.connect(path)
    if with_registry:
        con.execute("""CREATE TABLE patient_registry (
            bids_id TEXT, original_patient_id TEXT, lesion_type TEXT,
            kappa_dataset_id INTEGER)""")
        con.executemany("INSERT INTO patient_registry VALUES (?,?,?,?)", registry_rows)
    con.execute("""CREATE TABLE bids_patient_allocation (
        lesion_type TEXT NOT NULL, original_patient_id TEXT NOT NULL,
        bids_id TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (lesion_type, original_patient_id),
        UNIQUE (lesion_type, bids_id))""")
    con.executemany("INSERT INTO bids_patient_allocation VALUES (?,?,?,'2026-01-01')",
                    allocation_rows)
    con.commit()
    con.close()
    return path


def _columns(path, table="bids_patient_allocation"):
    con = sqlite3.connect(path)
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def _tables(path):
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def _backups(path):
    return sorted(path.parent.glob(path.name + ".bak-*-before-scope-migration"))


@pytest.fixture
def old_db(tmp_path):
    return _old_db(
        tmp_path / "brain_lesion.db",
        registry_rows=[("sub-005_ses-001", "P100", "glioblastoma", 249),
                       ("sub-002_ses-001", "P200", "multiple_sclerosis", 353)],
        allocation_rows=[("glioblastoma", "P100", "sub-005"),
                         ("glioblastoma", "P900", "sub-300"),
                         ("multiple_sclerosis", "P200", "sub-002")],
    )


def test_first_use_of_an_old_db_migrates_it(old_db):
    # Exactly the call that crashed stage 01 on 2026-09-30.
    assert get_allocations("ds:249", old_db) == {"P100": "sub-005"}

    assert "scope" in _columns(old_db)
    assert get_bids_id("ds:353", "P200", old_db) == "sub-002"
    assert get_bids_id("local:glioblastoma", "P900", old_db) == "sub-300"
    assert "bids_patient_allocation_legacy" in _tables(old_db)


def test_dataset_floors_are_set(old_db):
    get_allocations("ds:249", old_db)
    con = sqlite3.connect(old_db)
    floors = dict(con.execute("SELECT scope, floor FROM bids_scope_floor"))
    con.close()
    assert floors == {"ds:249": 5, "ds:353": 2}


def test_a_backup_of_the_old_db_is_kept(old_db):
    get_allocations("ds:249", old_db)

    backups = _backups(old_db)
    assert len(backups) == 1
    assert "lesion_type" in _columns(backups[0])
    assert "scope" not in _columns(backups[0])


def test_second_use_does_not_migrate_again(old_db, caplog):
    get_allocations("ds:249", old_db)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        get_allocations("ds:249", old_db)

    assert len(_backups(old_db)) == 1
    assert not [r for r in caplog.records if "migrat" in r.getMessage().lower()]


def test_migration_is_logged(old_db, caplog):
    with caplog.at_level(logging.WARNING):
        get_allocations("ds:249", old_db)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "2" in text and "1" in text          # 2 to datasets, 1 to local
    assert str(_backups(old_db)[0].name) in text


def test_a_fresh_db_is_created_without_migration(tmp_path, caplog):
    db = tmp_path / "brain_lesion.db"
    with caplog.at_level(logging.WARNING):
        assert get_allocations("ds:1", db) == {}
    assert "scope" in _columns(db)
    assert _backups(db) == []
    assert not caplog.records


def test_old_db_without_registry_goes_to_local(tmp_path):
    """A CLI-only machine: nothing was ever uploaded, so nothing has a dataset."""
    db = _old_db(tmp_path / "brain_lesion.db", [], [("multiple_sclerosis", "P1", "sub-001")],
                 with_registry=False)

    assert get_allocations("local:multiple_sclerosis", db) == {"P1": "sub-001"}


def test_conflict_stops_with_a_readable_error_and_changes_nothing(tmp_path):
    """Two people under one number in one dataset cannot be split
    automatically — refuse, never guess."""
    db = _old_db(
        tmp_path / "brain_lesion.db",
        registry_rows=[("sub-003_ses-001", "P1", "glioblastoma", 249),
                       ("sub-003_ses-001", "P2", "glioblastoma", 249)],
        allocation_rows=[("glioblastoma", "P1", "sub-003"), ("glioblastoma", "P2", "sub-004")],
    )

    with pytest.raises(RuntimeError) as err:
        get_allocations("ds:249", db)

    message = str(err.value)
    assert "sub-003" in message and "249" in message
    assert "migrate_bids_allocation_scopes" in message
    assert "lesion_type" in _columns(db) and "scope" not in _columns(db)
    assert "bids_patient_allocation_legacy" not in _tables(db)


def test_two_processes_racing_migrate_once(old_db):
    """The backend and stage 01 can open the DB at the same moment."""
    results, errors = [], []
    start = threading.Barrier(4)

    def use():
        try:
            start.wait()
            results.append(get_allocations("ds:249", old_db))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=use) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results == [{"P100": "sub-005"}] * 4
    assert len(_backups(old_db)) == 1


def test_db_path_resolution_is_unchanged(monkeypatch, tmp_path):
    """The migration must hit the same file the allocator resolves."""
    db = tmp_path / "env.db"
    monkeypatch.setenv("BRAIN_LESION_DB", str(db))
    assert bids_allocator._resolve_db_path(None) == db
