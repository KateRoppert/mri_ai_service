"""
KI-052: on timeout, the backend called process.kill() on the orchestrator
process only. orchestrator.py runs each pipeline stage as ITS OWN child
subprocess — killing only the orchestrator leaves that grandchild running,
orphaned, invisible to the DB/UI, for however long it takes to finish on
its own. Confirmed on real data: a run the DB marked "failed" at a 2h
timeout kept producing real segmentation output for almost 2 more hours.

This test spawns a real two-level process tree (a shell that backgrounds
a sleep and waits on it — same shape as orchestrator.py spawning a stage
subprocess) to prove the fix actually reaches the grandchild, not just
mocks the call.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import _kill_process_tree


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_kill_process_tree_kills_grandchild_not_just_direct_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    process = subprocess.Popen(
        ["bash", "-c", f"sleep 60 & echo $! > {pid_file}; wait"],
        start_new_session=True,
    )
    # give the shell time to background the sleep and write its pid
    for _ in range(20):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    grandchild_pid = int(pid_file.read_text().strip())
    assert _process_alive(grandchild_pid), "test setup failed: grandchild never started"

    _kill_process_tree(process)
    process.wait(timeout=5)

    time.sleep(0.2)
    assert not _process_alive(grandchild_pid), (
        "grandchild (stage subprocess) survived — only the direct child was killed"
    )


def test_kill_process_tree_does_not_raise_if_already_dead(tmp_path):
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait(timeout=5)
    # process has already exited on its own — killing its group must not raise
    _kill_process_tree(process)
