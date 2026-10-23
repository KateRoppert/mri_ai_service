"""KI-032: what a pool worker logs lands in the stage log file.

Stages 01 and 03 attached their file handler to a named logger while workers
logged through other loggers, so per-patient and per-series detail went to
stderr only. Everything relied on fork copying handlers into workers, which
Python 3.14 stops doing by default on Linux (forkserver).
"""
import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import stage_logging  # noqa: E402


def _work(tag):
    # An arbitrary logger, as helper modules use (getLogger(__name__)).
    logging.getLogger(f"some.helper.module").warning("worker says %s", tag)
    return tag


@pytest.fixture(autouse=True)
def _clean_root():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    yield
    for h in list(root.handlers):
        if h not in saved[0]:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved[1])


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_worker_warnings_reach_the_file(tmp_path, method):
    log = tmp_path / "stage.log"
    stage_logging.configure_stage_logging(log)

    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context(method),
                             initializer=stage_logging.worker_initializer,
                             initargs=stage_logging.worker_initargs()) as ex:
        assert sorted(ex.map(_work, ["a", "b", "c"])) == ["a", "b", "c"]

    for h in logging.getLogger().handlers:
        h.flush()
    text = log.read_text()
    for tag in "abc":
        assert text.count(f"worker says {tag}") == 1, text
    assert "some.helper.module" in text          # logger name is in the line


def test_stage_logger_has_no_own_handlers_and_reaches_the_file(tmp_path):
    log = tmp_path / "stage.log"
    stage = stage_logging.configure_stage_logging(log, name="reorganize_folders")
    stage.getChild("patient_KA03").warning("no date in series name")

    for h in logging.getLogger().handlers:
        h.flush()
    assert stage.handlers == []
    assert log.read_text().count("no date in series name") == 1


def test_reconfiguring_does_not_stack_handlers(tmp_path):
    log = tmp_path / "stage.log"
    stage_logging.configure_stage_logging(log)
    stage_logging.configure_stage_logging(log)
    logging.getLogger("x").warning("once")

    for h in logging.getLogger().handlers:
        h.flush()
    assert log.read_text().count("once") == 1


def test_initargs_without_a_file_are_harmless(tmp_path):
    stage_logging.configure_stage_logging(None)
    stage_logging.worker_initializer(*stage_logging.worker_initargs())


def test_stage_debug_is_kept_third_party_debug_is_not(tmp_path):
    log = tmp_path / "stage.log"
    stage = stage_logging.configure_stage_logging(log, name="nifti_converter")
    stage.getChild("worker").debug("series detail")
    logging.getLogger("urllib3.connectionpool").debug("noise")

    for h in logging.getLogger().handlers:
        h.flush()
    text = log.read_text()
    assert "series detail" in text
    assert "noise" not in text


def test_a_stage_that_configures_root_itself_keeps_its_level_in_workers(tmp_path):
    """Stages 05/07/08 attach their own root FileHandler (not ours). Under fork
    the initializer must leave the inherited setup — including DEBUG — alone,
    and initargs must still find the file for spawn."""
    log = tmp_path / "stage05.log"
    root = logging.getLogger()
    others = list(root.handlers)          # e.g. left by other test modules
    for h in others:
        root.removeHandler(h)
    fh = logging.FileHandler(log)
    root.addHandler(fh)
    root.setLevel(logging.DEBUG)
    try:
        assert stage_logging.worker_initargs()[0] == str(log)
        stage_logging.worker_initializer(*stage_logging.worker_initargs())
        assert root.level == logging.DEBUG
        assert root.handlers.count(fh) == 1 and len([h for h in root.handlers
                                                    if isinstance(h, logging.FileHandler)]) == 1
    finally:
        root.removeHandler(fh)
        fh.close()
        for h in others:
            root.addHandler(h)
