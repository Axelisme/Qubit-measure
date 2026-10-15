"""Virtual time for the waveform dialog's external, single-shot Qt timer seam."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class PreviewClock:
    """Deliver real Qt timeout signals at controlled deadlines on the owner thread.

    Restart replaces a deadline, stop cancels it, and draining exhausts all
    pending single-shot work. No dialog callback or render logic is replaced.
    ``now`` is virtual milliseconds from zero; ``deliveries`` counts emitted
    timeout signals. ``timers`` retains the timers constructed by this clock
    until fixture cleanup cancels them.
    """

    def __init__(self) -> None:
        self.now = 0
        self.deliveries = 0
        self.timers: list[_PreviewTimer] = []

    def timer(self, parent: QObject) -> _PreviewTimer:
        """Construct a parent-owned single-shot timer with a real Qt signal."""
        timer = _PreviewTimer(self, parent)
        self.timers.append(timer)
        return timer

    def advance(self, milliseconds: int) -> None:
        """Advance virtual time and synchronously deliver due single-shot signals."""
        self.now += milliseconds
        for timer in self.timers:
            if timer.deadline is not None and timer.deadline <= self.now:
                timer.stop()  # A single-shot is inactive before its callback runs.
                self.deliveries += 1
                timer.timeout.emit()

    def drain(self) -> None:
        """Deliver pending work, then require the single-shot queue to be empty."""
        deadlines = [t.deadline for t in self.timers if t.deadline is not None]
        if deadlines:
            self.advance(max(deadlines) - self.now)
        assert all(t.deadline is None for t in self.timers), (
            "timer rearmed during drain"
        )

    def close(self) -> None:
        """Cancel pending signals before the dialog's Qt children are disposed."""
        for timer in self.timers:
            timer.stop()


class _PreviewTimer(QObject):
    timeout = pyqtSignal()

    def __init__(self, clock: PreviewClock, parent: QObject) -> None:
        super().__init__(parent)
        self.clock = clock
        self.deadline: int | None = None

    def setSingleShot(self, enabled: bool) -> None:
        assert enabled, "waveform clock only supports single-shot timers"

    def start(self, milliseconds: int) -> None:
        self.deadline = self.clock.now + milliseconds

    def stop(self) -> None:
        self.deadline = None
