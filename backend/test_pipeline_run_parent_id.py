"""
Tests for parent_run_id: links a requeued pipeline run back to the run it
continues from, so the frontend can show a "this continues run X" banner.
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent))

import database
from database import Base, create_pipeline_run


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


def test_create_pipeline_run_sets_parent_run_id(session):
    run = create_pipeline_run(
        session, input_path="/in", output_path="/out", parent_run_id="orig-run-id",
    )
    assert run.parent_run_id == "orig-run-id"


def test_create_pipeline_run_parent_run_id_defaults_to_none(session):
    run = create_pipeline_run(session, input_path="/in", output_path="/out")
    assert run.parent_run_id is None


def test_migrate_add_parent_run_id_adds_column_to_old_table():
    # Simulate a pre-migration database: a pipeline_runs table with no
    # parent_run_id column at all (raw CREATE TABLE, bypassing the ORM model
    # which already declares the column).
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
        database._migrate_add_parent_run_id()
        with engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(pipeline_runs)"))]
        assert "parent_run_id" in cols
    finally:
        database.engine = original_engine


def test_migrate_add_parent_run_id_is_idempotent():
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
        database._migrate_add_parent_run_id()
        database._migrate_add_parent_run_id()  # must not raise (e.g. "duplicate column")
    finally:
        database.engine = original_engine
