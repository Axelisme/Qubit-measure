"""Single-flight, observable startup preparation without acquisition side effects."""

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from logging import getLogger
from threading import Lock, Thread
from typing import Literal

_logger = getLogger(__name__)


@dataclass(frozen=True)
class WarmupStatus:
    """Immutable progress snapshot, safe to poll from any thread.

    state is pending before start, running during preparation, ready on success,
    or failed after an item raises. completed contains successful item names in
    order; current names the active item, otherwise None. total counts all items.
    error contains the exception type/message on failure, otherwise None.
    Failed preparation permits on-demand acquisition; ready is not a promise
    that every possible numerical signature has been compiled.
    """

    state: Literal["pending", "running", "ready", "failed"]
    completed: tuple[str, ...]
    current: str | None
    total: int
    error: str | None = None


class EnvironmentWarmup:
    """Run named zero-argument numeric preparation items on one background thread.

    items must have unique nonempty names. Callables must own disposable resources
    and release them even on failure. They must not sample or change live devices.
    start is thread-safe and always returns the same non-cancellable Future.
    Its result is the terminal WarmupStatus, including failures, not an exception.
    Future callbacks execute on the completing thread (or immediately if done);
    UI callers must marshal delivery to their owner thread. Poll status for
    intermediate progress. No worker pool remains after completion. Process
    exit waits for the worker: an active Numba compiler cannot be safely cancelled.
    """

    def __init__(self, items: tuple[tuple[str, Callable[[], None]], ...]) -> None:
        names = tuple(name for name, _ in items)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("warmup item names must be nonempty and unique")
        self._items = items
        self._lock = Lock()
        self._future: Future[WarmupStatus] | None = None
        self._status = WarmupStatus("pending", (), None, len(items))

    @property
    def status(self) -> WarmupStatus:
        """Return an immutable progress snapshot without waiting for the worker."""
        with self._lock:
            return self._status

    def start(self) -> Future[WarmupStatus]:
        """Start once and return the shared future; concurrent calls never duplicate work."""
        with self._lock:
            if self._future is None:
                self._future = Future()
                self._future.set_running_or_notify_cancel()
                self._status = WarmupStatus("running", (), None, len(self._items))
                Thread(
                    target=self._run,
                    args=(self._future,),
                    name="simulation-warmup",
                    daemon=False,
                ).start()
            return self._future

    def wait_ready(self, timeout: float | None = None) -> WarmupStatus:
        """Start if needed and wait for success/failure; timeout raises TimeoutError.

        timeout is seconds, or None for no deadline. Timeout does not cancel work.
        A failed status lets callers continue with ordinary on-demand compilation.
        """
        return self.start().result(timeout=timeout)

    def _run(self, future: Future[WarmupStatus]) -> None:
        completed: tuple[str, ...] = ()
        try:
            for name, prepare in self._items:
                with self._lock:
                    self._status = WarmupStatus(
                        "running", completed, name, len(self._items)
                    )
                prepare()
                completed += (name,)
            status = WarmupStatus("ready", completed, None, len(self._items))
        except BaseException as error:
            _logger.exception("Simulation startup preparation failed")
            # Startup compilation is optional. Retain the diagnostic and let
            # acquisition try its ordinary on-demand numeric path.
            status = WarmupStatus(
                "failed",
                completed,
                None,
                len(self._items),
                f"{type(error).__name__}: {error}",
            )
        with self._lock:
            self._status = status
        future.set_result(status)
