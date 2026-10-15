"""Completion-driven owner pumping and joined worker error propagation."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import TypeVar

import pytest
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def manual_until(owner: ManualOwnerScheduler, completed: Callable[[], object]) -> None:
    """Pump the arranged producer's owner callbacks, with a hang-only deadline."""
    # Submitted work posts progress/terminal callbacks; queue arrival wakes pump_once.
    deadline = time.monotonic() + 30
    while not completed():
        remaining = deadline - time.monotonic()
        assert remaining > 0, "scheduled owner delivery did not complete"
        assert owner.pump_once(block=True, timeout=remaining)


@contextmanager
def running(work: Callable[[], _T]) -> Generator[Future[_T]]:
    """Run work off-thread; join and propagate its error even if caller fails.

    The caller must release any test-controlled blockers in its inner finally.
    """
    result: Future[_T] = Future()

    def invoke() -> None:
        try:
            result.set_result(work())
        except BaseException as exc:
            logger.exception("test worker failed")
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


@contextmanager
def waiting_call(
    monkeypatch: pytest.MonkeyPatch, work: Callable[[], _T]
) -> Generator[tuple[threading.Event, Future[_T]]]:
    """Observe this worker entering stdlib Condition.wait, then delegate unchanged.

    The arrival is emitted while the condition remains held, before its atomic
    release-and-wait. No owner state or lock is inspected. The caller must finish
    the public operation (even on failure) before leaving this context.
    """
    arrived = threading.Event()
    worker_ids: list[int] = []
    original_wait = threading.Condition.wait

    def observe(condition: threading.Condition, timeout: float | None = None) -> bool:
        if worker_ids == [threading.get_ident()]:
            arrived.set()
        return original_wait(condition, timeout)

    def invoke() -> _T:
        worker_ids.append(threading.get_ident())
        return work()

    with monkeypatch.context() as patch:
        patch.setattr(threading.Condition, "wait", observe)
        with running(invoke) as result:
            yield arrived, result
