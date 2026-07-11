"""Qt implementation of the owner-loop scheduler port."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar, cast

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]

_T = TypeVar("_T")


class QtOwnerScheduler(QObject):
    """Marshal callbacks to the Qt thread that constructs this adapter."""

    _invoke = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._owner_thread_id = threading.get_ident()
        self._invoke.connect(self._run, type=Qt.ConnectionType.QueuedConnection)  # type: ignore[call-arg]

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
        holder: dict[str, object] = {}

        def invoke() -> None:
            try:
                holder["result"] = callback()
            except BaseException as exc:  # propagate the exact callback failure
                holder["error"] = exc
            finally:
                done.set()

        self.post(invoke)
        done.wait()
        error = holder.get("error")
        if isinstance(error, BaseException):
            raise error
        return cast(_T, holder["result"])

    def _run(self, callback: Callable[[], None]) -> None:
        callback()


__all__ = ["QtOwnerScheduler"]
