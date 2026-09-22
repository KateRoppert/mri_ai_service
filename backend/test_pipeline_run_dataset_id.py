"""
Tests for kappa_dataset_id: the Kappa dataset a run numbers its subjects in
and uploads to, fixed once at run start (backend/numbering.py).
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent))

import database
from database import Base, create_pipeline_run, get_pipeline_run


@pytest.fixture
def session():
    """Isolated in-memory SQLite session with the schema created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        yield db
    finally:
        db.close()


def test_run_stores_its_dataset(session):
    run = create_pipeline_run(
        session, input_path="/in", output_path="/out",
        lesion_type="glioblastoma", kappa_dataset_id=337)

    assert get_pipeline_run(session, run.run_id).kappa_dataset_id == 337


def test_dataset_is_optional(session):
    """CLI runs and runs started without a Kappa session have none."""
    run = create_pipeline_run(session, input_path="/in", output_path="/out")

    assert run.kappa_dataset_id is None


def test_migrate_add_kappa_dataset_id_adds_column_to_old_table():
    # Simulate a pre-migration database: no kappa_dataset_id column at all.
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE pipeline_runs "
            "(run_id VARCHAR PRIMARY KEY, input_path VARCHAR, output_path VARCHAR, status VARCHAR)"
        ))
        conn.commit()

    original_engine = database.engine
    database.engine = engine
    try:
        database._migrate_add_kappa_dataset_id()
        with engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(pipeline_runs)"))]
        assert "kappa_dataset_id" in cols
    finally:
        database.engine = original_engine


def test_migrate_add_kappa_dataset_id_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE pipeline_runs "
            "(run_id VARCHAR PRIMARY KEY, input_path VARCHAR, output_path VARCHAR, status VARCHAR)"
        ))
        conn.commit()

    original_engine = database.engine
    database.engine = engine
    try:
        database._migrate_add_kappa_dataset_id()
        database._migrate_add_kappa_dataset_id()  # must not raise
    finally:
        database.engine = original_engine
