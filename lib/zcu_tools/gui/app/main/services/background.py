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

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator, Optional

from qtpy.QtCore import QObject  # type: ignore[attr-defined]

from zcu_tools.experiment.v2.runner.base import ActiveTask
from zcu_tools.gui.background import NO_RESULT, BackgroundRunner
from zcu_tools.gui.plotting import routing_scope
from zcu_tools.gui.session.ports import OffMainScopes
from zcu_tools.liveplot.backend import set_liveplot_backend
from zcu_tools.progress_bar.interface import use_pbar_factory

# ``OffMainScopes`` now lives in the session seam (``gui/session/ports``); it is
# re-exported here so main's call sites keep importing it from ``.background``.
# ``NO_RESULT``'s identity is what matters; it is defined once in the shared runner.
__all__ = ["NO_RESULT", "OffMainScopes", "BackgroundService"]


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


class BackgroundService(QObject):
    """Run thunks off the main thread with opt-in ambient scopes (ADR-0019).

    The off-main execution itself is the app-agnostic ``BackgroundRunner``; this
    service composes one and owns only the main-local *scope* policy — which
    standard ambient scopes (routing+liveplot / pbar / cancel) a thunk opts into,
    realised by the ``_entered`` context manager passed as the runner's ``enter``.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._runner = BackgroundRunner(self)

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
        self._runner.submit(
            work,
            on_done=on_done,
            on_error=on_error,
            run_in_pool=run_in_pool,
            enter=_entered(effective_scopes),
        )
