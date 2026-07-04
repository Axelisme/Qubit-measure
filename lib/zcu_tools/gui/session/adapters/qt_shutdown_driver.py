from __future__ import annotations

import logging
import time
from collections.abc import Callable

from qtpy.QtCore import QObject, QTimer  # type: ignore[attr-defined]

from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.services.shutdown import (
    DEFAULT_SHUTDOWN_TIMEOUT,
    ShutdownCoordinator,
    ShutdownState,
)

logger = logging.getLogger(__name__)

# How often the driver polls the coordinator while waiting for operations to
# settle.
_POLL_INTERVAL_MS = 50


class QtShutdownDriver(QObject):
    """Qt driving adapter for the Qt-free ShutdownCoordinator."""

    def __init__(
        self,
        handles: OperationHandles,
        *,
        timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._coordinator = ShutdownCoordinator(
            handles, timeout=timeout, now=time.monotonic
        )
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)
        self._on_closed: Callable[[], None] | None = None

    def begin(self, on_closed: Callable[[], None]) -> None:
        """Cancel every live operation and begin polling for them to settle."""
        if self._coordinator.is_active:
            return
        self._on_closed = on_closed
        self._coordinator.begin()
        # Poll once immediately: with no active operation the coordinator settles
        # on the first tick and we close without an event-loop round-trip.
        self._on_tick()

    def _on_tick(self) -> None:
        state = self._coordinator.tick()
        if state is ShutdownState.WAITING:
            if not self._timer.isActive():
                self._timer.start()
            return
        self._timer.stop()
        on_closed = self._on_closed
        self._on_closed = None
        if on_closed is not None:
            on_closed()
