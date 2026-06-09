"""BackgroundService — the OffMain *execution strategy* as a clean mechanism (ADR-0019).

One service owns "run a unit of work off the main thread, with a chosen set of
standard ambient scopes, and deliver the outcome back on the main thread". It is
policy-agnostic: it knows nothing about leases, exclusion, operation_id or the
agent — those are the OperationGate's job (a sibling leaf the domain service
composes alongside this). ``run`` / ``analyze`` / ``save`` and the interactive
auto-align step are all instances of this single mechanism; they differ only in
the thunk they submit and which scopes they opt into.

Two execution substrates behind one ``submit``:
- ``run_in_pool=False`` → a dedicated ``QThread`` (a long operation; never shares
  a pool slot, so it can't starve short helpers).
- ``run_in_pool=True``  → a shared ``QThreadPool`` (a short fire-and-callback
  helper, e.g. interactive auto-align).

``on_done`` / ``on_error`` always run on the Qt main thread (queued from the
worker), so callers settle State / handles there without touching worker-owned
state — the State main-thread invariant.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from qtpy.QtCore import (  # type: ignore[attr-defined]
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    Signal,  # type: ignore[reportPrivateImportUsage]
)

from zcu_tools.experiment.v2.runner.base import ActiveTask
from zcu_tools.gui.plotting import routing_scope
from zcu_tools.liveplot.backend import set_liveplot_backend
from zcu_tools.progress_bar.interface import use_pbar_factory

if TYPE_CHECKING:
    from zcu_tools.gui.plotting import FigureContainer

# Sentinel distinguishing "the thunk never produced a value" from "the thunk
# returned None" (a legitimate result, e.g. save). The dedicated worker also
# uses it to assert it never settles without an outcome.
NO_RESULT: Any = object()


@dataclass(frozen=True)
class OffMainScopes:
    """The opt-in ambient scopes a thunk runs inside (ADR-0019). Each is
    independently ``None``-able; only non-``None`` ones are entered, on the
    worker thread.

    - ``figure_container``: GUI matplotlib routing — sets the routing ContextVar
      *and* installs ``QtLivePlotBackend`` together (one facet: both direct
      ``plt.subplots`` and liveplot calls land in this container on the main
      thread; the liveplot backend requires the routing container, so they are
      co-dependent and driven by this single field).
    - ``pbar_factory``: progress — the per-operation pbar factory (the Progress
      facet's injection point; the owner mints it bound to the operation token).
    - ``stop_event``: cancel — installs ``ActiveTask`` so the work can self-
      interrupt cooperatively (the Cancel facet's off-main realisation).
    """

    figure_container: Optional["FigureContainer"] = None
    pbar_factory: Optional[Callable[..., Any]] = None
    stop_event: Optional[threading.Event] = None


@contextmanager
def _entered(scopes: OffMainScopes) -> Iterator[None]:
    """Enter the opted-in scopes on the *current* (worker) thread, in the same
    canonical nesting order the per-op workers used: routing → liveplot →
    ActiveTask → pbar."""
    with ExitStack() as stack:
        if scopes.figure_container is not None:
            # Lazy import: QtLivePlotBackend pulls in pyplot; keeping it out of
            # module load preserves the gui import-clean invariant, and it is
            # only needed when actually routing figures.
            from zcu_tools.gui.app.main.adapters.qt_liveplot_backend import (
                QtLivePlotBackend,
            )

            stack.enter_context(routing_scope(scopes.figure_container))
            stack.enter_context(set_liveplot_backend(QtLivePlotBackend()))
        if scopes.stop_event is not None:
            stack.enter_context(ActiveTask(scopes.stop_event))
        if scopes.pbar_factory is not None:
            stack.enter_context(use_pbar_factory(scopes.pbar_factory))
        yield


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
        self._error: Optional[Exception] = None
        self.finished.connect(self._emit)

    def run(self) -> None:
        try:
            self._result = self._thunk()
        except Exception as exc:  # noqa: BLE001 - forwarded to on_error on main
            self._error = exc

    def _emit(self) -> None:
        if self._error is not None:
            self.failed.emit(self._error)
        elif self._result is not NO_RESULT:
            self.done.emit(self._result)
        else:
            raise RuntimeError("BackgroundService worker settled without an outcome")


class BackgroundService(QObject):
    """Run thunks off the main thread with opt-in ambient scopes (ADR-0019)."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # Hold dedicated workers until they settle so the Python wrapper is not
        # GC'd before ``finished`` fires (parent= keeps the C++ object; the set
        # keeps the Python ref symmetric and explicit).
        self._workers: set[_OpWorker] = set()

    def submit(
        self,
        work: Callable[[], Any],
        scopes: Optional[OffMainScopes] = None,
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Run ``work`` off-main inside ``scopes``; deliver its result to
        ``on_done`` or its exception to ``on_error``, both on the main thread.

        ``run_in_pool=True`` uses the shared pool (short helper); ``False`` spawns
        a dedicated thread (long operation). The scopes are entered on the worker
        thread when the thunk is invoked, never here at build time.
        """
        effective_scopes = scopes if scopes is not None else OffMainScopes()

        def thunk() -> Any:
            with _entered(effective_scopes):
                return work()

        if run_in_pool:
            # Parent the signal carrier to bg so it outlives the pool runnable;
            # deleteLater (queued after on_done/on_error) frees it on the main
            # thread once the outcome has been delivered.
            signals = _OpSignals(self)
            signals.done.connect(on_done)
            signals.failed.connect(on_error)
            signals.done.connect(signals.deleteLater)
            signals.failed.connect(signals.deleteLater)
            self._pool.start(_PoolRunnable(thunk, signals))
        else:
            worker = _OpWorker(thunk, parent=self)
            worker.done.connect(on_done)
            worker.failed.connect(on_error)
            self._workers.add(worker)
            worker.finished.connect(lambda w=worker: self._workers.discard(w))
            worker.finished.connect(worker.deleteLater)
            worker.start()
