"""Completion-driven waits for explicitly pumped, non-Qt test owners."""

from collections.abc import Callable
from concurrent.futures import Future
from threading import Event, Timer
from typing import TypeVar

from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler

_T = TypeVar("_T")


def owner_until(
    owner: ManualOwnerScheduler,
    completed: Callable[[], object],
    *,
    label: str,
) -> None:
    """Deliver queued callbacks until an arranged owner completion is observable.

    The caller must arrange a producer that posts the completing owner callback.
    No polling interval or callback count proves completion. A separate 30-second
    watchdog wakes the queue only to diagnose a stuck producer. Predicate and
    owner callback errors propagate; the watchdog is cancelled and joined on exit.
    """
    expired = Event()

    def wake() -> None:
        expired.set()
        owner.post(lambda: None)

    watchdog = Timer(30, wake)
    # The arranged producer posts completion; the timer only guards a broken handoff.
    watchdog.start()
    try:
        while not completed():
            assert not expired.is_set(), f"{label}: arranged owner completion missing"
            owner.pump_once(block=True)
    finally:
        watchdog.cancel()
        watchdog.join()


def owner_result(owner: ManualOwnerScheduler, future: Future[_T], *, label: str) -> _T:
    """Pump a submitted Future's owner calls and return its result or exact error.

    A done callback posts a wake-up after the worker returns, including when no
    owner call was needed. This closes the return-after-owner-delivery race.
    """
    future.add_done_callback(lambda _completed: owner.post(lambda: None))
    owner_until(owner, future.done, label=label)
    return future.result()
