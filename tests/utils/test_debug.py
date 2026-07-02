from __future__ import annotations

import logging

from zcu_tools.utils.debug import log_current_exception


def test_log_current_exception_no_active_exception_is_noop(caplog):
    logger = logging.getLogger("tests.utils.debug.no_exception")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_current_exception(logger, "should not appear")

    assert caplog.records == []


def test_log_current_exception_logs_active_traceback(caplog):
    logger = logging.getLogger("tests.utils.debug.active")

    try:
        raise ValueError("bad value")
    except ValueError:
        with caplog.at_level(logging.ERROR, logger=logger.name):
            log_current_exception(logger, "operation failed")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.message == "operation failed"
    assert record.exc_info is not None
    assert "ValueError: bad value" in caplog.text


def test_log_current_exception_logs_pyro_traceback(caplog):
    logger = logging.getLogger("tests.utils.debug.pyro")

    class PyroError(RuntimeError):
        _pyroTraceback = ["Remote traceback:\n", "line 1\n"]

    try:
        raise PyroError("remote failed")
    except PyroError:
        with caplog.at_level(logging.ERROR, logger=logger.name):
            log_current_exception(logger, "remote operation failed")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "remote operation failed" in record.message
    assert "Remote traceback:" in record.message
    assert "line 1" in record.message
    assert record.exc_info is not None
