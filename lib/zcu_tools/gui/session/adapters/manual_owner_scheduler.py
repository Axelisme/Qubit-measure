"""Deterministic owner-loop scheduler for tests and headless composition."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


class ManualOwnerScheduler:
    """Thread-safe callback queue pumped explicitly by its constructing thread.

    This adapter is intentionally not an inline scheduler: worker-thread posts stay
    queued until the owner calls :meth:`pump_once` or :meth:`pump_all`, so terminal
    State writes can never leak onto a worker thread.
    """

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._callbacks: queue.Queue[Callable[[], None]] = queue.Queue()

    def is_owner_thread(self) -> bool:
        return threading.get_ident() == self._owner_thread_id

    def post(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("owner callback must be callable")
        self._callbacks.put(callback)

    def call(self, callback: Callable[[], _T]) -> _T:
        if self.is_owner_thread():
            raise RuntimeError("OwnerScheduler.call cannot run on the owner thread")

        done = threading.Event()
        results: list[_T] = []
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                results.append(callback())
            # tolerated: Return every callback failure, including interrupts, to the waiting caller.
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                done.set()

        self.post(invoke)
        done.wait()
        if errors:
            raise errors[0]
        return results[0]

    def pump_once(self, *, block: bool = False, timeout: float | None = None) -> bool:
        """Execute one queued callback on the owner thread; return whether one ran."""
        self._assert_owner()
        try:
            if block:
                callback = self._callbacks.get(timeout=timeout)
            else:
                callback = self._callbacks.get_nowait()
        except queue.Empty:
            return False
        callback()
        return True

    def pump_all(self) -> int:
        """Drain callbacks already queued and return the number executed."""
        self._assert_owner()
        count = 0
        while self.pump_once():
            count += 1
        return count

    def _assert_owner(self) -> None:
        if not self.is_owner_thread():
            raise RuntimeError("ManualOwnerScheduler may only be pumped by its owner")


__all__ = ["ManualOwnerScheduler"]
