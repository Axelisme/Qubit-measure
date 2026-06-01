from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional

from zcu_tools.gui.services.operation_gate import OperationGate

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
    """Qt-free coordinator for "cancel everything, wait for it to stop, then
    close (forcing past a deadline)".

    Pure logic so it is unit-testable without a Qt event loop: ``begin`` cancels
    every live operation through the gate and records their tokens + a deadline;
    ``tick`` polls those tokens (non-blocking) and reports whether they have all
    settled, the deadline passed, or we are still waiting. The periodic driving
    (a QTimer) and the actual window teardown live in the Qt layer; this object
    only decides *when* it is time to close.

    Cancellation is an async request — ``cancel`` sets each operation's
    stop_event but does not wait. A run / device setup self-judges 'cancelled'
    and settles; a connect (no stop point) never settles, so TIMED_OUT is its
    only exit. Timing is injected (``now``) so tests need no real clock.
    """

    def __init__(
        self,
        gate: OperationGate,
        *,
        timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        now: Callable[[], float],
    ) -> None:
        self._gate = gate
        self._timeout = timeout
        self._now = now
        self._tokens: list[int] = []
        self._deadline: Optional[float] = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def begin(self) -> None:
        """Cancel every live operation and start the wait. Idempotent: a second
        begin while already active is a no-op (a re-entrant close request)."""
        if self._active:
            return
        self._active = True
        self._tokens = self._gate.cancel_all()
        self._deadline = self._now() + self._timeout
        logger.info(
            "ShutdownCoordinator.begin: cancelled %d operation(s)", len(self._tokens)
        )

    def tick(self) -> ShutdownState:
        """Poll the cancelled operations once and report the wait state.

        Must be called only while active (after ``begin``). Returns SETTLED once
        every token reached a terminal outcome, TIMED_OUT once the deadline
        passed with any still pending, else WAITING.
        """
        if not self._active or self._deadline is None:
            raise RuntimeError("ShutdownCoordinator.tick called before begin")
        pending = [t for t in self._tokens if self._gate.poll(t) is None]
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
