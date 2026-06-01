"""Test doubles for the progress subsystem.

``DirectProgressTransport`` is a synchronous ``ProgressTransport``: ``emit``
calls the receiver inline (no thread, no Qt). Valid for single-threaded tests
where everything already runs on one thread, so the port's "receiver runs on the
consumer thread" contract is trivially met.
"""

from __future__ import annotations

from typing import Callable, Optional

from zcu_tools.gui.services.ports import ProgressEvent


class DirectProgressTransport:
    def __init__(self) -> None:
        self._receiver: Optional[Callable[[ProgressEvent], None]] = None

    def emit(self, event: ProgressEvent) -> None:
        if self._receiver is not None:
            self._receiver(event)

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
        self._receiver = receiver
