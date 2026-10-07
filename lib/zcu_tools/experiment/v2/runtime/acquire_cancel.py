"""Acquire-local cancellation projected from a borrowed run stop signal."""

from __future__ import annotations

import threading

from zcu_tools.experiment.stop_signal import StopSignal


class AcquireCancelFlag:
    """Combine run cancellation with an independent stop flag for one acquire.

    external is the borrowed Schedule stop signal. Create a fresh instance for
    each program acquire. set() stops only that acquire; it never mutates
    external or records a Schedule outcome.
    """

    def __init__(self, external: StopSignal) -> None:
        self._external = external
        self._local = threading.Event()

    def is_set(self) -> bool:
        """Return whether external cancellation or the acquire-local flag is set."""
        return self._external.is_set() or self._local.is_set()

    def set(self) -> None:
        """Set only the acquire-local stop flag, leaving the run signal unchanged."""
        self._local.set()
