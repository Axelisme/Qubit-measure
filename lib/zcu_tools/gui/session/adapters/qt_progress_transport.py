"""Qt driven adapter for the ``ProgressTransport`` port.

Realises the port's cross-thread contract with a queued Qt signal: ``emit`` is
called from any worker thread, and the connected slot — hence the registered
receiver — always runs on the thread this QObject lives on. Construct it on the
**main thread** so the receiver (ProgressService) runs there, keeping that
service main-thread-only and lock-free.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.session.ports import ProgressEvent


class QtProgressTransport(QObject):
    # object payload = ProgressEvent; queued so it hops to this QObject's thread.
    _event: Signal = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._receiver: Callable[[ProgressEvent], None] | None = None
        # Queued so the slot runs on this QObject's owning thread (main thread),
        # marshalling worker-thread emits across the boundary.
        self._event.connect(self._on_event, type=Qt.ConnectionType.QueuedConnection)  # type: ignore[call-arg]

    def emit(self, event: ProgressEvent) -> None:  # type: ignore[reportIncompatibleMethodOverride]
        # Thread-safe: Qt marshals the queued signal to the owning thread.
        self._event.emit(event)

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
        self._receiver = receiver

    def _on_event(self, event: ProgressEvent) -> None:
        # Runs on this QObject's thread (the main thread).
        if self._receiver is not None:
            self._receiver(event)
