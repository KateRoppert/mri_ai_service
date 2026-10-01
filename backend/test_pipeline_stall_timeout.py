"""
The pipeline is killed for being stuck, not for being long (KI-052).

A total-duration timeout sized by input folders killed a healthy 11-session
MS run mid-stage 05 on 2026-09-28: the formula counted 3 patient folders,
not the sessions or the per-session cost. The backend now waits for the
orchestrator and kills it only when nothing under the run's output_path has
changed for the stall threshold.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import newest_mtime, wait_for_pipeline, append_master_log_note


class FakeProcess:
    """communicate() times out `timeouts` times, then returns the output.

    `on_timeout` runs on every timeout — tests use it to simulate the
    pipeline writing files (or not) while the backend waits.
    """

    def __init__(self, timeouts, returncode=0, on_timeout=None):
        self.timeouts = timeouts
        self.returncode = None
        self._final_rc = returncode
        self.on_timeout = on_timeout
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.calls <= self.timeouts:
            if self.on_timeout:
                self.on_timeout(self.calls)
            raise subprocess.TimeoutExpired(cmd="orchestrator", timeout=timeout)
        self.returncode = self._final_rc
        return "out", "err"


def _age(path: Path, seconds_ago: float) -> None:
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


def test_newest_mtime_missing_dir_is_none(tmp_path):
    assert newest_mtime(tmp_path / "nope") is None


def test_newest_mtime_sees_nested_files(tmp_path):
    old = tmp_path / "logs" / "old.log"
    new = tmp_path / "preprocessed" / "sub-001" / "ses-002" / "anat" / "t1.nii.gz"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text("x")
    new.write_text("x")
    for p in (old, old.parent, new.parent, new.parent.parent, new.parent.parent.parent,
              new.parent.parent.parent.parent, tmp_path):
        _age(p, 1000)
    _age(new, 10)

    assert abs(newest_mtime(tmp_path) - (time.time() - 10)) < 2


def test_normal_exit_returns_output(tmp_path):
    result = wait_for_pipeline(FakeProcess(timeouts=2, returncode=0),
                               tmp_path, stall_seconds=3600, poll_seconds=0)
    assert result.stalled is False
    assert result.returncode == 0
    assert (result.stdout, result.stderr) == ("out", "err")


def test_no_activity_past_threshold_is_a_stall(tmp_path):
    f = tmp_path / "logs" / "pipeline_master.log"
    f.parent.mkdir()
    f.write_text("x")
    for p in (f, f.parent, tmp_path):
        _age(p, 2 * 3600)

    # First clock read is the run start (now); every later read is 2 h on,
    # with nothing written in between.
    start = time.time()
    reads = iter([start] + [start + 2 * 3600] * 1000)
    process = FakeProcess(timeouts=100)
    result = wait_for_pipeline(process, tmp_path, stall_seconds=3600, poll_seconds=0,
                               clock=lambda: next(reads))

    assert result.stalled is True
    assert result.idle_seconds >= 3600
    assert process.calls == 1  # decided on the first check, no extra waiting


def test_recent_activity_is_not_a_stall(tmp_path):
    """A long run that keeps writing files must outlive any fixed duration."""
    out = tmp_path / "preprocessed"
    out.mkdir()

    def pipeline_writes_a_file(call):
        time.sleep(0.05)
        (out / f"sub-{call:03d}.nii.gz").write_text("x")

    # 30 checks x 50 ms = 1.5 s of waiting against a 0.5 s threshold: a fixed
    # duration limit would have fired, but every check sees a fresh file.
    process = FakeProcess(timeouts=30, returncode=0, on_timeout=pipeline_writes_a_file)
    result = wait_for_pipeline(process, tmp_path, stall_seconds=0.5, poll_seconds=0)

    assert result.stalled is False
    assert result.returncode == 0


def test_run_start_counts_as_activity(tmp_path):
    """An old, reused output folder must not look stalled the moment we start."""
    (tmp_path / "old.txt").write_text("x")
    _age(tmp_path / "old.txt", 10 * 3600)
    _age(tmp_path, 10 * 3600)

    process = FakeProcess(timeouts=3, returncode=0)
    result = wait_for_pipeline(process, tmp_path, stall_seconds=3600, poll_seconds=0)

    assert result.stalled is False


def test_stall_note_is_appended_to_master_log(tmp_path):
    log = tmp_path / "logs" / "pipeline_master.log"
    log.parent.mkdir()
    log.write_text("2026-09-30 09:09:37 | INFO    | PIPELINE STARTED\n")

    append_master_log_note(tmp_path, "PIPELINE KILLED: no activity for 60 min")

    lines = log.read_text().splitlines()
    assert lines[0].endswith("PIPELINE STARTED")
    assert "| ERROR   | PIPELINE KILLED: no activity for 60 min" in lines[-1]


def test_stall_note_without_logs_dir_does_not_raise(tmp_path):
    append_master_log_note(tmp_path / "missing", "whatever")
