"""BackgroundRunner — the app-agnostic OffMain execution mechanism (ADR-0019).

One runner owns "run a unit of work off the main thread, optionally inside a
caller-supplied context manager, and deliver the outcome back on the main
thread". It is policy-agnostic AND app-agnostic: it knows nothing about leases,
exclusion, operation_id, the agent, figure routing, liveplot, progress bars or
cancellation. Any ambient scope a unit of work needs is the *caller's* job,
passed as the optional ``enter`` context manager, which the runner enters on the
worker thread, inside the thunk (never at build time).

Two execution substrates behind one ``submit``:
- ``run_in_pool=False`` → a dedicated ``QThread`` (a long operation; never shares
  a pool slot, so it can't starve short helpers).
- ``run_in_pool=True``  → a shared ``QThreadPool`` (a short fire-and-callback
  helper).

``on_done`` / ``on_error`` always run on the Qt main thread (queued from the
worker), so callers settle State / handles there without touching worker-owned
state — the State main-thread invariant.

This module imports ONLY ``qtpy`` + stdlib; the app-specific scopes (routing,
liveplot, stop scopes, pbar) live with the callers that need them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

logger = logging.getLogger(__name__)

from qtpy.QtCore import (  # type: ignore[attr-defined]
    QCoreApplication,
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    Signal,  # type: ignore[reportPrivateImportUsage]
)

# Sentinel distinguishing "the thunk never produced a value" from "the thunk
# returned None" (a legitimate result, e.g. save). Defined in the Qt-free
# session-core (``operation_handles``) and re-exported here so the executor and
# the operation layer share one sentinel identity; importing it does not make
# ``operation_handles`` (or ``operation_runner``) pull Qt.
from zcu_tools.gui.session.operation_handles import NO_RESULT


class _OpSignals(QObject):
    """Main-thread-affinity signal carrier for a pooled runnable (a QRunnable is
    not a QObject, so it cannot own signals)."""

    done: Signal = Signal(object)
    failed: Signal = Signal(object)


class _PoolRunnable(QRunnable):
    """Run ``thunk`` on a pool thread; emit the outcome via main-thread signals."""

    def __init__(self, thunk: Callable[[], Any], signals: _OpSignals) -> None:
        super().__init__()
        self._thunk = thunk
        self._signals = signals

    def run(self) -> None:  # pragma: no cover - exercised via QThreadPool
        try:
            result = self._thunk()
        except Exception as exc:  # noqa: BLE001 - forwarded to on_error on main
            # Log here (worker thread) where the real traceback is live: the
            # exception is otherwise only carried as a value to on_error, so the
            # stack would evaporate. ERROR with exc_info captures it.
            logger.error("background pool worker failed", exc_info=exc)
            self._signals.failed.emit(exc)
        else:
            self._signals.done.emit(result)


class _OpWorker(QThread):
    """Dedicated-thread worker: run ``thunk`` in ``run()`` (worker thread), then
    emit ``done``/``failed`` from ``_emit`` (which runs on the main thread, the
    worker QObject's affinity, via the ``finished`` signal)."""

    done: Signal = Signal(object)
    failed: Signal = Signal(object)

    def __init__(self, thunk: Callable[[], Any], parent: QObject) -> None:
        super().__init__(parent)
        self._thunk = thunk
        self._result: Any = NO_RESULT
        self._error: Exception | None = None
        self.finished.connect(self._emit)

    def run(self) -> None:
        try:
            self._result = self._thunk()
        except Exception as exc:  # noqa: BLE001 - forwarded to on_error on main
            # Log here (worker thread) where the real traceback is live: _emit
            # only re-emits the stored exception as a value, losing the stack.
            logger.error("background dedicated worker failed", exc_info=exc)
            self._error = exc

    def _emit(self) -> None:
        if self._error is not None:
            self.failed.emit(self._error)
        elif self._result is not NO_RESULT:
            self.done.emit(self._result)
        else:
            raise RuntimeError("BackgroundRunner worker settled without an outcome")


class BackgroundRunner(QObject):
    """Run thunks off the main thread, optionally inside a caller ``enter`` scope."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # Hold dedicated workers until they settle so the Python wrapper is not
        # GC'd before ``finished`` fires (parent= keeps the C++ object; the set
        # keeps the Python ref symmetric and explicit).
        self._workers: set[_OpWorker] = set()
        # Hold pool signal carriers until their queued done/failed delivery fires,
        # for the same reason: the carrier's C++ half must outlive the cross-thread
        # ``QMetaCallEvent`` that delivers the outcome on the main thread.
        self._pool_signals: set[_OpSignals] = set()

    def submit(
        self,
        work: Callable[[], Any],
        *,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        run_in_pool: bool = True,
        enter: AbstractContextManager[Any] | None = None,
    ) -> None:
        """Run ``work`` off-main inside ``enter``; deliver its result to
        ``on_done`` or its exception to ``on_error``, both on the main thread.

        ``run_in_pool=True`` uses the shared pool (short helper); ``False`` spawns
        a dedicated thread (long operation). ``enter`` (if given) is entered on the
        worker thread when the thunk is invoked, never here at build time.
        """

        logger.debug("background submit: run_in_pool=%s", run_in_pool)

        def thunk() -> Any:
            if enter is None:
                return work()
            with enter:
                return work()

        if run_in_pool:
            # Parent the signal carrier to the runner so it outlives the pool
            # runnable and its C++ lifetime is tied to the runner's. The carrier is
            # tracked in ``_pool_signals`` and freed only once its delivery has
            # fired (see below), so a queued cross-thread emit never lands on a
            # destroyed carrier — that is what ``quiesce`` guarantees on teardown.
            signals = _OpSignals(self)
            signals.done.connect(on_done)
            signals.failed.connect(on_error)
            self._pool_signals.add(signals)
            signals.done.connect(lambda _=None, s=signals: self._free_signals(s))
            signals.failed.connect(lambda _=None, s=signals: self._free_signals(s))
            self._pool.start(_PoolRunnable(thunk, signals))
        else:
            worker = _OpWorker(thunk, parent=self)
            worker.done.connect(on_done)
            worker.failed.connect(on_error)
            self._workers.add(worker)
            worker.finished.connect(lambda w=worker: self._workers.discard(w))
            worker.finished.connect(worker.deleteLater)
            worker.start()

    def _free_signals(self, signals: _OpSignals) -> None:
        """Release a pool signal carrier once its (already-delivered) outcome has
        fired: drop the Python ref and schedule the C++ deletion. Runs on the main
        thread (queued from the carrier's own done/failed signal), so by the time it
        runs the delivery to ``on_done``/``on_error`` is already complete."""
        self._pool_signals.discard(signals)
        signals.deleteLater()

    def quiesce(self, timeout_ms: int = 5000) -> bool:
        """Wait for all in-flight work AND flush its queued main-thread deliveries.

        ``QThreadPool.waitForDone`` only joins the worker *threads*; the outcome of a
        pooled task is delivered to ``on_done``/``on_error`` via a *queued* cross-thread
        signal that is still sitting in the main-thread event queue when ``run()``
        returns. If the runner (and its signal carriers) are destroyed before that
        queued ``QMetaCallEvent`` is dispatched — e.g. a short-lived runner GC'd at the
        end of a test — ``processEvents`` later crashes delivering to a dead carrier.

        ``quiesce`` closes that gap: join the pool and the dedicated workers, then
        ``processEvents`` so every pending delivery (and the carrier's own deferred
        delete) lands while its target is still alive. Call it from widget close and
        test teardown before the runner goes out of scope. Returns ``True`` if every
        substrate drained within ``timeout_ms``.
        """
        drained = self._pool.waitForDone(timeout_ms)
        for worker in list(self._workers):
            drained = worker.wait(timeout_ms) and drained
        # Dispatch the queued done/failed deliveries (and the deferred-delete events
        # they schedule) now, while the carriers/workers are still alive.
        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()
            app.processEvents()
        return drained
