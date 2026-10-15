"""Completion-driven pumping of real Qt owner deliveries."""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def qt_until(completed: Callable[[], object], *, label: str = "completion") -> None:
    """Pump real Qt deliveries until an arranged producer satisfies completed.

    Callers must arrange the producer before entering. A zero-interval probe
    checks the predicate on owner-loop turns; neither its ticks nor elapsed time
    prove arrival or completion. Only the predicate ends a successful wait. The
    separate single-shot 30-second watchdog diagnoses a stuck producer, not a
    promptness contract; label identifies that failure. Predicate errors return
    to caller. Both timers stop on every exit.
    """
    from qtpy.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    probe = QTimer()
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    errors: list[BaseException] = []

    def check() -> None:
        try:
            if completed():
                loop.quit()
        except BaseException as exc:
            logger.exception("Qt completion predicate failed")
            errors.append(exc)
            loop.quit()

    probe.timeout.connect(check)
    watchdog.timeout.connect(loop.quit)
    # The caller's scheduled worker/callback produces completion; only guard hangs.
    watchdog.start(30_000)
    probe.start(0)
    try:
        if not completed():
            loop.exec()
        if errors:
            raise errors[0]
        assert completed(), f"{label}: scheduled Qt delivery did not complete"
    finally:
        probe.stop()
        watchdog.stop()
