"""Notebook-owned SIGINT handling for a synchronous core invocation."""

import _thread
import signal
import threading
from types import FrameType, TracebackType
from typing import Self

from zcu_tools.experiment.stop_signal import StopSignal


class _RunInterrupts:
    """Own the temporary main-thread handler and the frontend stop-request fact."""

    def __init__(self, stop: StopSignal, timeout: float) -> None:
        self.stop_requested = False
        self._stop = stop
        self._timeout = timeout
        self._previous = signal.getsignal(signal.SIGINT)
        self._main_thread = threading.current_thread() is threading.main_thread()
        self._thread_id = threading.get_ident()
        self._active = False
        self._timer: threading.Timer | None = None

    def __enter__(self) -> Self:
        if self._main_thread:
            self._active = True
            signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if not self._active:
            return
        if self.stop_requested:
            raise KeyboardInterrupt
        self.stop_requested = True
        self._stop.set(reason="Notebook SIGINT")
        self._timer = threading.Timer(self._timeout, self._interrupt_main)
        self._timer.daemon = True
        self._timer.start()

    def _interrupt_main(self) -> None:
        if self._active:
            if hasattr(signal, "pthread_kill"):
                # Target the main thread to interrupt blocking POSIX C calls.
                signal.pthread_kill(self._thread_id, signal.SIGINT)
            else:
                _thread.interrupt_main()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._active = False
        try:
            if self._timer is not None:
                self._timer.cancel()
                # Drain an already-running callback before returning the handler.
                self._timer.join()
        finally:
            if self._main_thread:
                signal.signal(signal.SIGINT, self._previous)
