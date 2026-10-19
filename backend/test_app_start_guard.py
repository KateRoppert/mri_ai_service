"""Starting a run must refuse a path another run is already writing to.

KI-068, found 2026-10-09: two clicks on «Запуск» 40 s apart started two runs
of P000067 into the same folder. They shared `bids_organized/`,
`preprocessed/` and `temp/`; HD-BET was OOM-killed (return code -9), some
sessions failed, and together with a third job the laptop ran out of memory
and the global OOM killer took down VS Code.

`database.get_active_run_by_output_path` already existed and was called from
three other endpoints — just not from the one that starts a run.

The frontend's `loading` button state is not the fix: start_pipeline hands
the work to BackgroundTasks and returns PENDING within about a second, so
the button re-enables while the run keeps going for minutes.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

_BODY = {
    "input_path": "/in",
    "output_path": "/out",
    "use_default_output": False,
    "lesion_type": "multiple_sclerosis",
}


def _occupying_run(run_id="run-busy"):
    return SimpleNamespace(
        run_id=run_id,
        status="running",
        input_path="/in",
        output_path="/out",
        lesion_type="multiple_sclerosis",
    )


def _new_run():
    return SimpleNamespace(
        run_id="run-new",
        input_path="/in",
        output_path="/out",
        lesion_type="multiple_sclerosis",
        created_at=datetime.now(timezone.utc),
    )


def _scope():
    """(scope, dataset_id, warning) — what numbering.scope_for_run returns."""
    return ("ds:158", 158, None)


def test_start_is_refused_when_another_run_holds_the_path():
    with patch("app.pipeline_manager.validate_input_path", return_value=True), \
         patch("app.get_active_run_by_output_path", return_value=_occupying_run()), \
         patch("app.create_pipeline_run") as mock_create:
        response = client.post("/api/pipeline/start", json=_BODY)

    assert response.status_code == 409
    # The occupying run's id has to be in the message: the operator needs to
    # know WHICH run to wait for, and it is routinely not the one they see.
    assert "run-busy" in response.json()["detail"]
    mock_create.assert_not_called()


def test_refusal_does_not_touch_kappa():
    """The guard runs before numbering, which may CREATE a Kappa dataset.

    numbering.scope_for_run -> kappa_dataset_resolver.resolve_or_create is a
    side effect on an external system, and it also raises the numbering floor.
    A request about to be refused must not get that far.
    """
    with patch("app.pipeline_manager.validate_input_path", return_value=True), \
         patch("app.get_active_run_by_output_path", return_value=_occupying_run()), \
         patch("app.numbering.scope_for_run", new=AsyncMock()) as mock_scope:
        response = client.post("/api/pipeline/start", json=_BODY)

    assert response.status_code == 409
    mock_scope.assert_not_awaited()


def test_start_proceeds_when_the_path_is_free():
    with patch("app.pipeline_manager.validate_input_path", return_value=True), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.numbering.scope_for_run", new=AsyncMock(return_value=_scope())), \
         patch("app.create_pipeline_run", return_value=_new_run()), \
         patch("kappa_run_log.log_start"), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()):
        response = client.post("/api/pipeline/start", json=_BODY)

    assert response.status_code == 200, response.text
    assert response.json()["run_id"] == "run-new"


def test_the_path_is_rechecked_after_the_numbering_await():
    """Two simultaneous starts must not both get through.

    `await numbering.scope_for_run` yields the event loop between the first
    check and the row write, so in a single-threaded loop a second request can
    pass the same check while the first is suspended and neither has written
    its row yet. The late re-check is what closes that window; the button's
    loading state only covers callers that are a browser.

    Simulated here by a guard that reports the path free on the first call
    (the early check) and busy on the second (the late one).
    """
    calls = []

    def _guard(_db, _path):
        calls.append(_path)
        return None if len(calls) == 1 else _occupying_run("run-raced")

    with patch("app.pipeline_manager.validate_input_path", return_value=True), \
         patch("app.get_active_run_by_output_path", side_effect=_guard), \
         patch("app.numbering.scope_for_run", new=AsyncMock(return_value=_scope())), \
         patch("app.create_pipeline_run") as mock_create:
        response = client.post("/api/pipeline/start", json=_BODY)

    assert response.status_code == 409
    assert "run-raced" in response.json()["detail"]
    assert len(calls) == 2, "путь должен проверяться дважды: до и после await"
    mock_create.assert_not_called()
