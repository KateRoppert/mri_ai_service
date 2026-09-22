import sqlite3
import pytest
from scripts.migrate_bids_allocation_scopes import find_conflicts, migrate


def _db(tmp_path, registry_rows, allocation_rows):
    path = tmp_path / "brain_lesion.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE patient_registry (
        bids_id TEXT, original_patient_id TEXT, lesion_type TEXT, kappa_dataset_id INTEGER)""")
    con.executemany("INSERT INTO patient_registry VALUES (?,?,?,?)", registry_rows)
    con.execute("""CREATE TABLE bids_patient_allocation (
        lesion_type TEXT, original_patient_id TEXT, bids_id TEXT, created_at TEXT,
        PRIMARY KEY (lesion_type, original_patient_id))""")
    con.executemany("INSERT INTO bids_patient_allocation VALUES (?,?,?,'2026-01-01')",
                    allocation_rows)
    con.commit(); con.close()
    return path


def test_uploaded_patients_move_to_their_dataset(tmp_path):
    db = _db(tmp_path,
             [("sub-005_ses-001", "P100", "glioblastoma", 249)],
             [("glioblastoma", "P100", "sub-005")])

    result = migrate(db, dry_run=False)

    con = sqlite3.connect(db)
    row = con.execute("SELECT scope, bids_id FROM bids_patient_allocation "
                      "WHERE original_patient_id='P100'").fetchone()
    assert row == ("ds:249", "sub-005")
    assert result["moved_to_datasets"] == 1


def test_never_uploaded_patients_go_to_the_local_scope(tmp_path):
    db = _db(tmp_path, [], [("glioblastoma", "P900", "sub-300")])

    migrate(db, dry_run=False)

    con = sqlite3.connect(db)
    row = con.execute("SELECT scope, bids_id FROM bids_patient_allocation "
                      "WHERE original_patient_id='P900'").fetchone()
    assert row == ("local:glioblastoma", "sub-300")


def test_no_number_changes(tmp_path):
    db = _db(tmp_path,
             [("sub-005_ses-001", "P100", "glioblastoma", 249),
              ("sub-007_ses-001", "P200", "glioblastoma", 266)],
             [("glioblastoma", "P100", "sub-005"), ("glioblastoma", "P200", "sub-007")])

    migrate(db, dry_run=False)

    con = sqlite3.connect(db)
    got = dict(con.execute("SELECT original_patient_id, bids_id FROM bids_patient_allocation"))
    assert got == {"P100": "sub-005", "P200": "sub-007"}


def test_conflict_inside_one_dataset_aborts(tmp_path):
    """Two people under one number in the same dataset cannot be split
    automatically — stop and report instead of guessing."""
    db = _db(tmp_path,
             [("sub-001_ses-001", "P100", "glioblastoma", 249),
              ("sub-001_ses-001", "P200", "glioblastoma", 249)],
             [("glioblastoma", "P100", "sub-001")])

    conflicts = find_conflicts(db)
    assert conflicts

    with pytest.raises(SystemExit):
        migrate(db, dry_run=False)


def test_collision_across_datasets_is_not_a_conflict(tmp_path):
    """This is the real sub-003 case: same number, different datasets, two
    people — exactly what dataset scoping is meant to separate."""
    db = _db(tmp_path,
             [("sub-003_ses-001", "P100", "glioblastoma", 133),
              ("sub-003_ses-001", "P200", "glioblastoma", 249)],
             [])

    assert find_conflicts(db) == []


def test_dry_run_changes_nothing(tmp_path):
    db = _db(tmp_path,
             [("sub-005_ses-001", "P100", "glioblastoma", 249)],
             [("glioblastoma", "P100", "sub-005")])

    migrate(db, dry_run=True)

    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(bids_patient_allocation)")]
    assert "scope" not in cols
