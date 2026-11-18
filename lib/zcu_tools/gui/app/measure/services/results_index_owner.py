"""Owner-loop state for the one retained Measure Results Browser scan."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Literal

from zcu_tools.gui.session.ports import (
    BackgroundAdmissionClosedError,
    BackgroundError,
    BackgroundExecutor,
    OwnerScheduler,
)

from .results_index import (
    ResultEntryBatch,
    ResultsIndex,
    ResultsScanCancelled,
    scan_results,
)


@dataclass(frozen=True)
class ResultsScanState:
    """One owner-published view of the browser index.

    index includes only delivered runs and issues. phase starts as indexing and
    becomes ready for a complete index, cancelled for a cooperative stop, or
    failed on an unexpected error. The latter two retain the delivered data.
    completed_entries counts delivered entries. total_entries is unknown until
    the first batch or successful completion. error is a user-facing message
    on failure, otherwise None.
    """

    index: ResultsIndex
    phase: Literal["indexing", "ready", "cancelled", "failed"] = "indexing"
    completed_entries: int = 0
    total_entries: int | None = None
    error: str | None = None


class ResultsIndexOwner:
    """Start one project scan on first watch; publish detached owner-loop states.

    Construct and call on the application owner thread. project_root anchors
    Database and result; this owner never writes them. background and scheduler
    are borrowed from the app. Only the app stops admission and drains them.
    """

    def __init__(
        self,
        project_root: Path,
        background: BackgroundExecutor,
        scheduler: OwnerScheduler,
    ) -> None:
        self._project_root = project_root.absolute()
        self._background = background
        self._scheduler = scheduler
        self._state = ResultsScanState(ResultsIndex(self._project_root, ()))
        self._stop = Event()
        self._started = False
        self._listeners: list[Callable[[ResultsScanState], None]] = []

    def watch(self, listener: Callable[[ResultsScanState], None]) -> Callable[[], None]:
        """Deliver the current state then updates on the owner; return idempotent unsubscribe.

        The first watch starts the index. Unsubscribing never cancels or restarts
        it. New watches after cancel raise BackgroundAdmissionClosedError.
        """
        if self._stop.is_set():
            raise BackgroundAdmissionClosedError("Results indexing has shut down")
        if not self._started:
            # The app owns the executor and drains it on close (ADR-0066).
            self._background.submit(
                self._scan,
                run_in_pool=True,
                on_done=lambda _value: None,
                on_error=self._scan_failed,
            )
            self._started = True

        def deliver(state: ResultsScanState) -> None:
            listener(state)

        deliver(self._state)
        self._listeners.append(deliver)

        def unsubscribe() -> None:
            if deliver in self._listeners:
                self._listeners.remove(deliver)

        return unsubscribe

    def cancel(self) -> None:
        """Close admission and request cooperative stop, without waiting."""
        self._stop.set()

    def _scan(self) -> None:
        def deliver(batch: ResultEntryBatch) -> None:
            self._scheduler.post(lambda: self._commit_batch(batch))

        index: ResultsIndex | None
        try:
            index = scan_results(
                self._project_root, on_entry=deliver, cancelled=self._stop.is_set
            )
        except ResultsScanCancelled:
            index = None
        # Batches and normal completion share one ordered owner queue.
        self._scheduler.post(lambda: self._complete(index))

    def _commit_batch(self, batch: ResultEntryBatch) -> None:
        if self._stop.is_set() or self._state.phase != "indexing":
            return
        previous = self._state.index
        index = ResultsIndex(
            batch.project_root,
            previous.runs + batch.runs,
            previous.issues + batch.issues,
        )
        self._publish(
            ResultsScanState(
                index, "indexing", batch.completed_entries, batch.total_entries
            )
        )

    def _complete(self, index: ResultsIndex | None) -> None:
        if index is None or self._stop.is_set():
            self._publish(replace(self._state, phase="cancelled"))
            return
        count = self._state.completed_entries
        self._publish(ResultsScanState(index, "ready", count, count))

    def _scan_failed(self, error: BackgroundError) -> None:
        # BackgroundRunner logs the original worker exception.
        self._publish(replace(self._state, phase="failed", error=str(error)))

    def _publish(self, state: ResultsScanState) -> None:
        self._state = state
        for listener in tuple(self._listeners):
            listener(state)
