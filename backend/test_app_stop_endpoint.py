"""Stop endpoint: kill the run, record why, never lie about the outcome."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _run(run_id="run-1", status="running", current_stage=5, config_path=None):
    return SimpleNamespace(
        run_id=run_id,
        status=status,
        current_stage=current_stage,
        output_path="/out",
        input_path="/in",
        lesion_type="glioblastoma",
        config_path=config_path,
    )


def test_404_when_run_unknown():
    with patch("app.get_pipeline_run", return_value=None):
        response = client.post("/api/pipeline-runs/nope/stop")

    assert response.status_code == 404


def test_409_when_run_already_finished():
    with patch("app.get_pipeline_run", return_value=_run(status="completed")):
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 409


def test_kills_process_group_and_records_state():
    proc = MagicMock()
    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=proc), \
         patch("app.pipeline_manager.runtime_config_path",
               return_value=Path("/runtime_configs/config_run-1.yaml")), \
         patch("app._kill_process_tree") as kill, \
         patch("app.update_pipeline_run") as update:
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    kill.assert_called_once_with(proc)
    kwargs = update.call_args.kwargs
    assert kwargs["status"] == "stopped"
    assert kwargs["stopped_at_stage"] == 5
    assert kwargs["config_path"] == "/runtime_configs/config_run-1.yaml"


def test_records_stopped_before_kill():
    """Stop commits status+config_path before killing the process tree."""
    proc = MagicMock()
    order = []

    def _record_update(*args, **kwargs):
        order.append("update")

    def _record_kill(*args, **kwargs):
        order.append("kill")

    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=proc), \
         patch("app.pipeline_manager.runtime_config_path",
               return_value=Path("/runtime_configs/config_run-1.yaml")), \
         patch("app._kill_process_tree", side_effect=_record_kill), \
         patch("app.update_pipeline_run", side_effect=_record_update):
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    assert order == ["update", "kill"]


def test_marks_stopped_even_when_process_already_gone():
    # After a backend restart the DB can still say "running" while no process
    # exists. The process is provably dead, so stop corrects the record
    # instead of failing.
    with patch("app.get_pipeline_run", return_value=_run()), \
         patch("app.pipeline_manager.get_process", return_value=None), \
         patch("app.pipeline_manager.runtime_config_path",
               return_value=Path("/runtime_configs/config_run-1.yaml")), \
         patch("app._kill_process_tree") as kill, \
         patch("app.update_pipeline_run") as update:
        response = client.post("/api/pipeline-runs/run-1/stop")

    assert response.status_code == 200
    kill.assert_not_called()
    assert update.call_args.kwargs["status"] == "stopped"
    assert update.call_args.kwargs["config_path"].endswith("config_run-1.yaml")


def test_failed_update_does_not_overwrite_stopped(tmp_path):
    """A late failed write must not clobber a stop that already committed."""
    import sqlalchemy
    from sqlalchemy.orm import sessionmaker
    from database import (
        PipelineRun,
        update_pipeline_run_if_active,
        get_pipeline_run,
    )

    db_file = tmp_path / "stop_race.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db_file}")
    PipelineRun.__table__.create(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    db = Session()

    db.add(PipelineRun(
        run_id="run-1",
        input_path="/in",
        output_path="/out",
        status="stopped",
        stopped_at_stage=5,
        config_path="/runtime_configs/config_run-1.yaml",
    ))
    db.commit()

    result = update_pipeline_run_if_active(
        db,
        "run-1",
        status="failed",
        error_message="killed exit code",
    )

    assert result is None
    run = get_pipeline_run(db, "run-1")
    assert run.status == "stopped"
    assert run.config_path == "/runtime_configs/config_run-1.yaml"
    db.close()


def _session_with_run(tmp_path, run_id="run-1", status="pending"):
    import sqlalchemy
    from sqlalchemy.orm import sessionmaker
    from database import PipelineRun

    db_file = tmp_path / f"{run_id}.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db_file}")
    PipelineRun.__table__.create(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    db = Session()
    db.add(PipelineRun(
        run_id=run_id,
        input_path="/in",
        output_path="/out",
        status=status,
    ))
    db.commit()
    return db


def test_already_stopped_does_not_start_pipeline(tmp_path):
    """Stop committed before the running write — do not Popen an orchestrator."""
    from app import run_pipeline_background
    from database import get_pipeline_run

    db = _session_with_run(tmp_path, status="stopped")
    try:
        with patch("app.pipeline_manager.start_pipeline") as start, \
             patch("app.pipeline_manager.cleanup_runtime_config"):
            run_pipeline_background("run-1", "/in", "/out", db)

        start.assert_not_called()
        assert get_pipeline_run(db, "run-1").status == "stopped"
    finally:
        db.close()


def test_stop_during_start_kills_process_without_waiting(tmp_path):
    """Stop between the running write and register_process must kill, not wait."""
    from app import run_pipeline_background, pipeline_manager
    from database import get_pipeline_run

    db = _session_with_run(tmp_path, status="pending")
    proc = MagicMock()
    proc.pid = 4242
    proc.communicate.side_effect = AssertionError(
        "must not wait on a process started after stop"
    )

    def fake_start(run_id, *args, **kwargs):
        # Concurrent Stop: status is already terminal, nothing to kill yet
        # because register_process has not run. Then we Popen and register —
        # the background task must notice and kill.
        run = get_pipeline_run(db, run_id)
        run.status = "stopped"
        db.commit()
        pipeline_manager.register_process(run_id, proc)
        return proc

    try:
        with patch("app.pipeline_manager.start_pipeline", side_effect=fake_start), \
             patch("app._kill_process_tree") as kill, \
             patch("app.pipeline_manager.cleanup_runtime_config"):
            run_pipeline_background("run-1", "/in", "/out", db)

        kill.assert_called_once_with(proc)
        proc.communicate.assert_not_called()
        assert pipeline_manager.get_process("run-1") is None
        assert get_pipeline_run(db, "run-1").status == "stopped"
    finally:
        pipeline_manager.unregister_process("run-1")
        db.close()


def test_failed_start_does_not_overwrite_stopped(tmp_path):
    """start_pipeline returning None must not clobber a stop that already committed."""
    from app import run_pipeline_background
    from database import get_pipeline_run

    db = _session_with_run(tmp_path, status="pending")

    def fake_start(run_id, *args, **kwargs):
        run = get_pipeline_run(db, run_id)
        run.status = "stopped"
        db.commit()
        return None

    try:
        with patch("app.pipeline_manager.start_pipeline", side_effect=fake_start), \
             patch("app.pipeline_manager.cleanup_runtime_config"):
            run_pipeline_background("run-1", "/in", "/out", db)

        assert get_pipeline_run(db, "run-1").status == "stopped"
    finally:
        db.close()
