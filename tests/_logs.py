"""Log assertions that also work on Python 3.9."""

from __future__ import annotations

import contextlib
import logging


@contextlib.contextmanager
def assert_no_error_logs(*logger_names):
    """Like unittest's assertNoLogs(name, level="ERROR"), which Python 3.9 lacks."""
    records = []
    handler = logging.Handler(level=logging.ERROR)
    handler.emit = records.append
    loggers = [logging.getLogger(name) for name in logger_names]
    old_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in zip(loggers, old_levels):
            logger.removeHandler(handler)
            logger.setLevel(level)
    assert not records, "unexpected error logs: " + "; ".join(
        f"{r.name}: {r.getMessage()}" for r in records
    )
