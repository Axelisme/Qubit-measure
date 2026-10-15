"""Completion-driven owner pumping and joined worker error propagation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import TypeVar

from qtpy.QtCore import QEventLoop, QTimer
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler

_T = TypeVar("_T")


def qt_until(completed: Callable[[], object]) -> bool:
    """Pump real Qt deliveries until an arranged producer satisfies completed.

    Callers must arrange the producer before entering. The deadline diagnoses a
    stuck producer, not a promptness contract. Predicate errors return to caller.
    """
    loop = QEventLoop()
    probe = QTimer()
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    errors: list[BaseException] = []

    def check() -> None:
        try:
            if completed():
                loop.quit()
        except BaseException as exc:
            errors.append(exc)
            loop.quit()

    probe.timeout.connect(check)
    watchdog.timeout.connect(loop.quit)
    # The caller's scheduled worker/callback produces completion; only guard hangs.
    watchdog.start(30_000)
    probe.start(0)
    try:
        if not completed():
            loop.exec()
        if errors:
            raise errors[0]
        assert completed(), "scheduled Qt delivery did not complete"
        return True
    finally:
        probe.stop()
        watchdog.stop()


def manual_until(owner: ManualOwnerScheduler, completed: Callable[[], object]) -> None:
    """Pump the arranged producer's owner callbacks, with a hang-only deadline."""
    # Submitted work posts progress/terminal callbacks; queue arrival wakes pump_once.
    deadline = time.monotonic() + 30
    while not completed():
        remaining = deadline - time.monotonic()
        assert remaining > 0, "scheduled owner delivery did not complete"
        assert owner.pump_once(block=True, timeout=remaining)


@contextmanager
def running(work: Callable[[], _T]) -> Iterator[Future[_T]]:
    """Run work off-thread; join and propagate its error even if caller fails.

    The caller must release any test-controlled blockers in its inner finally.
    """
    result: Future[_T] = Future()

    def invoke() -> None:
        try:
            result.set_result(work())
        except BaseException as exc:
            result.set_exception(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    try:
        yield result
    finally:
        # Arranged work terminates after caller releases blockers / pumps its queue.
        thread.join(timeout=30)
        assert not thread.is_alive(), "test worker did not finish"
        result.result()
