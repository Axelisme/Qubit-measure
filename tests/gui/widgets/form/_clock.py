"""Manual external Qt timer shared by FormEditor debounce scenarios."""

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QApplication


class ManualTimer(QObject):
    """Qt timer replacement driven only by explicit virtual clock advances."""

    timeout = pyqtSignal()

    def __init__(self, parent: QObject | None, clock: "Clock") -> None:
        super().__init__(parent)
        self.clock = clock
        self.interval_ms = 0
        self.deadline: int | None = None
        clock.timers.append(self)
        self.destroyed.connect(lambda: clock.timers.remove(self))

    def setSingleShot(self, single: bool) -> None:
        """Require the one-shot timer used by the debounce policy."""
        assert single

    def setInterval(self, msec: int) -> None:
        """Record the externally requested interval."""
        self.interval_ms = msec

    def start(self, msec: int | None = None) -> None:
        """Replace the pending deadline, just like QTimer.start."""
        if msec is not None:
            self.interval_ms = msec
        self.deadline = self.clock.now + self.interval_ms

    def stop(self) -> None:
        """Cancel pending expiry."""
        self.deadline = None

    def isActive(self) -> bool:
        """Report whether expiry remains scheduled."""
        return self.deadline is not None


class Clock:
    """Deliver only timer expiries reached by an explicit advance."""

    def __init__(self) -> None:
        self.now = 0
        self.timers: list[ManualTimer] = []

    def timer(self, parent: QObject | None = None) -> ManualTimer:
        """Construct a parent-owned Qt timer replacement."""
        return ManualTimer(parent, self)

    def advance(self, msec: int) -> None:
        """Advance virtual milliseconds and synchronously deliver due signals."""
        self.now += msec
        for timer in tuple(self.timers):
            if timer.deadline is not None and timer.deadline <= self.now:
                timer.stop()
                timer.timeout.emit()
        QApplication.processEvents()
