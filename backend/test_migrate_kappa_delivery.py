"""The delivery migration: idempotent, and bounded in what it revives."""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlalchemy

import database as db_mod


def _legacy_db(tmp_path: Path) -> Path:
    """A pipeline_runs table as it looked BEFORE this feature — no delivery
    columns — with four runs that exercise each arm of the backfill's WHERE."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE pipeline_runs ("
        " run_id VARCHAR PRIMARY KEY, input_path VARCHAR, output_path VARCHAR,"
        " status VARCHAR, completed_at DATETIME, kappa_dataset_id INTEGER)"
    )
    recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(" ")
    ancient = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat(" ")
    conn.executemany(
        "INSERT INTO pipeline_runs VALUES (?, '/in', '/out', ?, ?, ?)",
        [
            ("recent_with_dataset", "completed", recent, 350),
            ("recent_no_dataset", "completed", recent, None),
            ("ancient_with_dataset", "completed", ancient, 350),
            ("failed_with_dataset", "failed", recent, 350),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _statuses(path: Path) -> dict:
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT run_id, kappa_upload_status FROM pipeline_runs"
    ).fetchall()
    conn.close()
    return dict(rows)


def test_migration_adds_columns_and_backfills_only_recent_dataset_runs(
    tmp_path, monkeypatch
):
    path = _legacy_db(tmp_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(db_mod, "engine", engine)

    db_mod._migrate_add_kappa_delivery()

    with engine.connect() as conn:
        cols = [r[1] for r in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]
    for name in ("kappa_upload_status", "kappa_upload_next_attempt",
                 "kappa_upload_detail", "kappa_user_id"):
        assert name in cols

    statuses = _statuses(path)
    assert statuses["recent_with_dataset"] == "pending"
    assert statuses["recent_no_dataset"] is None      # no dataset to upload to
    assert statuses["ancient_with_dataset"] is None   # outside the 30-day window
    assert statuses["failed_with_dataset"] is None    # never completed


def test_migration_is_idempotent_and_does_not_re_backfill(tmp_path, monkeypatch):
    path = _legacy_db(tmp_path)
    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    monkeypatch.setattr(db_mod, "engine", engine)

    db_mod._migrate_add_kappa_delivery()

    # Operator resolves the run; a second run of the migration must not
    # resurrect it, or every restart would re-queue finished work.
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE pipeline_runs SET kappa_upload_status = 'done' "
        " WHERE run_id = 'recent_with_dataset'"
    )
    conn.commit()
    conn.close()

    db_mod._migrate_add_kappa_delivery()

    assert _statuses(path)["recent_with_dataset"] == "done"
