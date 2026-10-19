"""caplog tests for the key lifecycle log lines added in Phase 157.

Pins that the operation handle settle, the persistence caretaker load failure,
and the background worker exception each emit a log record at the expected level
(the worker-exception one carrying a real traceback) — so a future refactor that
drops a log call is caught.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.services.caretaker import (
    create_persistence_caretaker as PersistenceCaretaker,
)
from zcu_tools.gui.app.measure.services.persistence_types import AppPersistedState
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.operation_handles import (
    OperationHandles,
    OperationOutcome,
)

from tests.gui._qt_waits import qt_until

# Reuse the same quiesce discipline as test_background: a queued cross-thread
# delivery onto a freed C++ object segfaults, so drain every runner before GC.
_LIVE_BG: list[BackgroundRunner] = []


@pytest.fixture(autouse=True, name="_quiesce_bg")
def fixture_quiesce_bg():
    """Quiesce tracked background runners after each test."""
    yield
    for bg in _LIVE_BG:
        # Submitted Qt work has no remaining blocker; drain its worker and delivery.
        assert bg.quiesce(timeout_ms=30_000)
    _LIVE_BG.clear()


def _pump_until(qapp, predicate) -> None:
    qt_until(predicate)


def test_operation_settle_failure_logs_warning(caplog) -> None:
    handles = OperationHandles()
    token = handles.create(origin=EventOrigin(kind="user"))
    with caplog.at_level(
        logging.DEBUG, logger="zcu_tools.gui.session.operation_handles"
    ):
        handles.settle(token, OperationOutcome("failed", "boom"))
    records = [r for r in caplog.records if "operation settle" in r.message]
    assert records
    assert records[0].levelno == logging.WARNING
    assert "boom" in records[0].getMessage()


def test_operation_settle_success_logs_info(caplog) -> None:
    handles = OperationHandles()
    token = handles.create(origin=EventOrigin(kind="user"))
    with caplog.at_level(
        logging.DEBUG, logger="zcu_tools.gui.session.operation_handles"
    ):
        handles.settle(token, OperationOutcome("finished"))
    records = [r for r in caplog.records if "operation settle" in r.message]
    assert records and records[0].levelno == logging.INFO


def test_caretaker_load_failure_logs_warning(tmp_path: Path, caplog) -> None:
    class _Originator:
        def restore_persisted_state(self, state: AppPersistedState) -> None:
            pass

        def capture_persisted_state(self) -> AppPersistedState:  # pragma: no cover
            raise AssertionError

    caretaker = PersistenceCaretaker(_Originator(), cache_dir=tmp_path)
    caretaker.state_path.write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="zcu_tools.gui.session.persistence"):
        outcome = caretaker.restore_all(load=True)
    assert outcome.load_error is not None
    warnings = [
        record
        for record in caplog.records
        if record.name == "zcu_tools.gui.session.persistence"
        and record.message == str(outcome.load_error)
    ]
    assert warnings and warnings[0].levelno == logging.WARNING
    assert "Failed to read GUI state" in warnings[0].message
    assert "using defaults" in warnings[0].message


def test_background_worker_exception_logs_traceback(qapp, caplog) -> None:
    bg = BackgroundRunner()
    _LIVE_BG.append(bg)
    errors: list[Exception] = []

    def boom() -> None:
        raise RuntimeError("worker-kaboom")

    with caplog.at_level(
        logging.ERROR, logger="zcu_tools.gui.session.adapters.qt_background"
    ):
        bg.submit(
            boom,
            run_in_pool=False,
            on_done=lambda _: None,
            on_error=errors.append,
        )
        _pump_until(qapp, lambda: bool(errors))

    records = [r for r in caplog.records if "worker failed" in r.message]
    assert records, "worker exception was not logged"
    rec = records[0]
    assert rec.levelno == logging.ERROR
    # exc_info must be captured so the real traceback survives the marshal.
    assert rec.exc_info is not None
    assert "worker-kaboom" in caplog.text
