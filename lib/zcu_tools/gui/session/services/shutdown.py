from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum

from zcu_tools.gui.session.operation_handles import OperationHandles

logger = logging.getLogger(__name__)

# How long to wait for every cancelled operation to settle before forcing the
# window shut. A connect has no cancellation point, so a remote connect that is
# mid-handshake when the user closes will only stop on this timeout.
DEFAULT_SHUTDOWN_TIMEOUT = 10.0


class ShutdownState(Enum):
    WAITING = "waiting"  # some cancelled operation has not settled yet
    SETTLED = "settled"  # every cancelled operation reached a terminal outcome
    TIMED_OUT = "timed_out"  # deadline passed with operations still pending


class ShutdownCoordinator:
    """Qt-free coordinator for "cancel everything, wait for it to stop, then close"."""

    def __init__(
        self,
        handles: OperationHandles,
        *,
        timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        now: Callable[[], float],
    ) -> None:
        self._handles = handles
        self._timeout = timeout
        self._now = now
        self._tokens: list[int] = []
        self._deadline: float | None = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def begin(self) -> None:
        """Cancel every live operation and start the wait."""
        if self._active:
            return
        self._active = True
        self._tokens = self._handles.cancel_all()
        self._deadline = self._now() + self._timeout
        logger.info(
            "ShutdownCoordinator.begin: cancelled %d operation(s)", len(self._tokens)
        )

    def tick(self) -> ShutdownState:
        """Poll the cancelled operations once and report the wait state."""
        if not self._active or self._deadline is None:
            raise RuntimeError("ShutdownCoordinator.tick called before begin")
        pending = [t for t in self._tokens if self._handles.poll(t) is None]
        if not pending:
            logger.info("ShutdownCoordinator: all operations settled")
            self._active = False
            return ShutdownState.SETTLED
        if self._now() >= self._deadline:
            logger.warning(
                "ShutdownCoordinator: timed out with %d operation(s) still pending",
                len(pending),
            )
            self._active = False
            return ShutdownState.TIMED_OUT
        return ShutdownState.WAITING
