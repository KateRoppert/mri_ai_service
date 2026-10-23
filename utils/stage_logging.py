"""
Logging for pipeline stage scripts: one file per stage that also receives
what pool workers log (KI-032).

Handlers go on the root logger, so every logger in the process — the stage's
own, per-patient children, helper modules' getLogger(__name__) — reaches the
stage log file. Stages 05/07/08 already worked this way; 01 and 03 attached
the file to a named logger while workers logged elsewhere, and their detail
went to stderr only.

Workers started by fork inherit these handlers. Workers started by spawn or
forkserver (the Linux default from Python 3.14) start with none; pass
`initializer=worker_initializer, initargs=worker_initargs()` to every pool so
they re-attach the same file.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

CONSOLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Marks handlers this module added, so reconfiguring replaces them instead of
# stacking a second copy (every line twice).
_OURS = "_stage_logging_handler"


def _remove_ours(root: logging.Logger) -> None:
    for h in list(root.handlers):
        if getattr(h, _OURS, False):
            root.removeHandler(h)
            h.close()


def _file_handler(log_file: Path, level: int) -> logging.Handler:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    setattr(fh, _OURS, True)
    return fh


def configure_stage_logging(
    log_file: Optional[Path],
    name: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    console_stream=None,
) -> logging.Logger:
    """Console + file handlers on the root logger; returns the stage logger
    `name` (or root), which keeps no handlers of its own and propagates."""
    import sys

    root = logging.getLogger()
    _remove_ours(root)
    # Root at the console level: third-party DEBUG (nibabel, urllib3, ...)
    # stays out of the file; the stage's own loggers get file_level below.
    root.setLevel(console_level)

    console = logging.StreamHandler(console_stream or sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    setattr(console, _OURS, True)
    root.addHandler(console)

    if log_file:
        root.addHandler(_file_handler(Path(log_file), file_level))

    stage = logging.getLogger(name) if name else root
    if name:
        # A handler left on the stage logger would print every line twice.
        for h in list(stage.handlers):
            stage.removeHandler(h)
        stage.setLevel(file_level)        # stage and its children: DEBUG to the file
        stage.propagate = True
    return stage


def worker_initargs() -> Tuple[Optional[str], int]:
    """(log file, level) of the root logger's file handler — ours, or the one
    stages 05/07/08 attach themselves."""
    handlers = logging.getLogger().handlers
    for ours_first in (True, False):
        for h in handlers:
            if isinstance(h, logging.FileHandler) and \
                    (getattr(h, _OURS, False) or not ours_first):
                return h.baseFilename, h.level or logging.DEBUG
    return None, logging.DEBUG


def worker_initializer(log_file: Optional[str], level: int) -> None:
    """Pool initializer. Under fork the worker inherited the parent's
    handlers (and level) — leave them be. Under spawn/forkserver it has none:
    attach the stage file."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    if log_file:
        root.addHandler(_file_handler(Path(log_file), level))
