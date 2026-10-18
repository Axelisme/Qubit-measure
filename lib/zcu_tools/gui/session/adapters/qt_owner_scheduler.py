"""Qt implementation of the owner-loop scheduler port."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

# tolerated: qtpy exposes Signal at runtime but its QtCore stub does not export it.
from qtpy.QtCore import QObject, Qt, Signal  # pyright: ignore[reportPrivateImportUsage]

_T = TypeVar("_T")


class QtOwnerScheduler(QObject):
    """Marshal callbacks to the Qt thread that constructs this adapter."""

    _invoke = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._owner_thread_id = threading.get_ident()
        # tolerated: PyQt6 connect stub omits the connection type needed for owner-thread delivery.
        self._invoke.connect(self._run, type=Qt.ConnectionType.QueuedConnection)  # pyright: ignore[reportCallIssue]

    def is_owner_thread(self) -> bool:
        return threading.get_ident() == self._owner_thread_id

    def post(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("owner callback must be callable")
        self._invoke.emit(callback)

    def call(self, callback: Callable[[], _T]) -> _T:
        if self.is_owner_thread():
            raise RuntimeError("OwnerScheduler.call cannot run on the owner thread")

        done = threading.Event()
        results: list[_T] = []
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                results.append(callback())
            except BaseException as exc:  # propagate the exact callback failure
                errors.append(exc)
            finally:
                done.set()

        self.post(invoke)
        done.wait()
        if errors:
            raise errors[0]
        return results[0]

    def _run(self, callback: Callable[[], None]) -> None:
        callback()


__all__ = ["QtOwnerScheduler"]
