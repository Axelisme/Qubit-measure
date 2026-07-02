from __future__ import annotations

import logging
import time
from collections.abc import Callable

from qtpy.QtCore import QObject, QTimer  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.services.shutdown import (
    DEFAULT_SHUTDOWN_TIMEOUT,
    ShutdownCoordinator,
    ShutdownState,
)
from zcu_tools.gui.session.operation_handles import OperationHandles

logger = logging.getLogger(__name__)

# How often the driver polls the coordinator while waiting for operations to
# settle. The project's first *periodic* timer (all others are singleShot).
_POLL_INTERVAL_MS = 50


class QtShutdownDriver(QObject):
    """Qt driving adapter for the Qt-free ShutdownCoordinator (ADR-0005).

    Owns a periodic QTimer that pumps ``coordinator.tick`` until it reports a
    terminal state, then stops the timer and invokes the supplied ``on_closed``
    callback (the window's actual teardown). The coordinator decides *when* to
    close; this adapter only supplies the clock and the timer. A blocked connect
    that never settles dies with the process once the window closes, so
    TIMED_OUT and SETTLED both just call ``on_closed``.
    """

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
        """Cancel every live operation and begin polling for them to settle.

        ``on_closed`` runs once on the main thread when the wait ends (settled
        or timed out). Re-entrant begins are absorbed by the coordinator; the
        first ``on_closed`` wins.
        """
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
