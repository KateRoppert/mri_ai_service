"""The stopped status and its columns must exist and survive migration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import PipelineStatus


def test_stopped_status_exists():
    assert PipelineStatus.STOPPED.value == "stopped"


def test_stopped_is_distinct_from_failed():
    # A stopped run is a deliberate act, not a defect. Conflating them would
    # corrupt any reading of how often the pipeline actually fails.
    assert PipelineStatus.STOPPED != PipelineStatus.FAILED


def test_migration_adds_columns(tmp_path, monkeypatch):
    import sqlalchemy

    db_file = tmp_path / "test.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db_file}")

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "CREATE TABLE pipeline_runs (run_id VARCHAR PRIMARY KEY, status VARCHAR)"
        ))
        conn.commit()

    import database
    monkeypatch.setattr(database, "engine", engine)
    database._migrate_add_stop_columns()

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]

    assert "stopped_at_stage" in cols
    assert "stopped_by" in cols
