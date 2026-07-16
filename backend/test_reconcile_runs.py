"""
Tests for startup reconciliation of orphaned pipeline runs.

A pipeline run is a subprocess of the backend and cannot survive a restart, so
any run left 'running'/'pending' at boot is orphaned. Leaving it 'running' makes
the frontend history auto-refresh poll forever; reconciliation fixes it to
'failed'. These tests run against an isolated in-memory DB (never the real one).
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent))

from database import Base, PipelineRun, reconcile_orphaned_runs


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


def _add(db, run_id, status):
    db.add(PipelineRun(
        run_id=run_id, input_path="/in", output_path="/out", status=status,
    ))


def test_reconciles_running_and_pending_to_failed(session):
    _add(session, "r-running", "running")
    _add(session, "r-pending", "pending")
    _add(session, "r-done", "completed")
    _add(session, "r-failed", "failed")
    session.commit()

    n = reconcile_orphaned_runs(session)

    assert n == 2
    statuses = {r.run_id: r.status for r in session.query(PipelineRun).all()}
    assert statuses["r-running"] == "failed"
    assert statuses["r-pending"] == "failed"
    assert statuses["r-done"] == "completed"   # terminal states untouched
    assert statuses["r-failed"] == "failed"


def test_sets_completed_at_and_error_message(session):
    _add(session, "r1", "running")
    session.commit()

    reconcile_orphaned_runs(session)

    run = session.query(PipelineRun).filter_by(run_id="r1").one()
    assert run.completed_at is not None
    assert isinstance(run.completed_at, datetime)
    assert run.error_message  # non-empty explanation set


def test_preserves_existing_error_message(session):
    session.add(PipelineRun(
        run_id="r1", input_path="/in", output_path="/out",
        status="running", error_message="OOM during stage 05",
    ))
    session.commit()

    reconcile_orphaned_runs(session)

    run = session.query(PipelineRun).filter_by(run_id="r1").one()
    assert run.status == "failed"
    assert run.error_message == "OOM during stage 05"  # original cause kept


def test_noop_when_nothing_orphaned(session):
    _add(session, "r-done", "completed")
    session.commit()

    assert reconcile_orphaned_runs(session) == 0
