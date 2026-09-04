"""The registry is how a stop request reaches a running process."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def test_registered_process_is_retrievable():
    mgr = PipelineManager()
    proc = MagicMock()

    mgr.register_process("run-1", proc)

    assert mgr.get_process("run-1") is proc


def test_unknown_run_returns_none():
    assert PipelineManager().get_process("never-registered") is None


def test_unregister_removes_it():
    mgr = PipelineManager()
    mgr.register_process("run-1", MagicMock())

    mgr.unregister_process("run-1")

    assert mgr.get_process("run-1") is None


def test_unregister_is_idempotent():
    # Called from a finally block that also runs when the process never
    # started; it must not raise there.
    PipelineManager().unregister_process("never-registered")


def test_registry_is_shared_across_instances():
    # app.py constructs its own PipelineManager per request path; the stop
    # endpoint has to see the process the start path registered.
    PipelineManager().register_process("run-shared", MagicMock())

    assert PipelineManager().get_process("run-shared") is not None
