"""Notebook-owned SIGINT handling for a synchronous core invocation."""

import signal
import threading
from types import FrameType, TracebackType
from typing import Self

from zcu_tools.experiment.stop_signal import StopSignal


class _RunInterrupts:
    """Own the temporary main-thread handler and the frontend stop-request fact."""

    def __init__(self, stop: StopSignal) -> None:
        self.stop_requested = False
        self._stop = stop
        self._previous = signal.getsignal(signal.SIGINT)
        self._main_thread = threading.current_thread() is threading.main_thread()

    def __enter__(self) -> Self:
        if self._main_thread:
            signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.stop_requested = True
        self._stop.set(reason="Notebook SIGINT")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._main_thread:
            signal.signal(signal.SIGINT, self._previous)
