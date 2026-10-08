"""UTC wall-clock adapter shared by GUI and offline workflow hosts."""

from datetime import UTC, datetime

from zcu_tools.experiment.stop_signal import StopSignal


class UtcClock:
    """Provide real UTC time and cooperative waits without owning a worker.

    GUI and offline hosts may share this adapter. No hardware, Qt, progress,
    thread creation or control exception belongs to the clock.
    """

    def now(self) -> datetime:
        """Return the current timezone-aware UTC wall time without side effects."""
        return datetime.now(UTC)

    def wait_until(self, target: datetime, cancel_signal: StopSignal) -> None:
        """Block until an aware target or the run's cancellation event is set.

        target may use any timezone; a past target returns immediately.
        cancel_signal is the current run's native StopSignal. Cancellation
        returns normally, leaving the engine to decide the control outcome.
        A naive target raises TypeError; datetime arithmetic errors propagate.
        Do not change the signal or read hardware. This call blocks its caller.
        """
        remaining = (target - self.now()).total_seconds()
        if remaining > 0:
            cancel_signal.event.wait(remaining)
