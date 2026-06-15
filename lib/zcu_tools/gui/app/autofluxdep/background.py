"""BackgroundService — autofluxdep's pure OffMain executor (ADR-0019, ADR-0026 §2).

One service owns "run a unit of work off the main thread and deliver the
outcome back on the main thread". It is policy-agnostic AND scope-agnostic:
it knows nothing about leases, exclusion, operation_id, figure routing,
liveplot, progress bars or cancellation. Every ambient scope a thunk needs is
the *caller's* responsibility — built into the work thunk as a closure before
``submit`` is called.

autofluxdep's workers never draw matplotlib figures (they acquire numeric rows
the main thread plots, ADR-0018), so ``figure_ambient`` is never used here.
Device-setup work thunks use ``progress_ambient`` (session layer, no Qt).

``on_done`` / ``on_error`` always run on the Qt main thread (queued from the
worker), so callers settle State / handles there without touching worker-owned
state — the State main-thread invariant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QObject  # type: ignore[attr-defined]

from zcu_tools.gui.background import NO_RESULT, BackgroundRunner

# ``NO_RESULT``'s identity is what matters; it is defined once in the shared runner.
__all__ = ["NO_RESULT", "BackgroundService"]


class BackgroundService(QObject):
    """Pure OffMain executor: submit work, deliver results on main thread.

    Scope entering (pbar, ActiveTask) is the *caller's* responsibility — build
    it into the work thunk closure before calling ``submit``.  See
    ``gui.session.scopes.progress_ambient`` (session layer) for device-setup
    pbar ambient.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._runner = BackgroundRunner(self)

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Run ``work`` off-main; deliver its result to ``on_done`` or its
        exception to ``on_error``, both on the main thread.

        ``run_in_pool=True`` uses the shared pool (short helper); ``False``
        spawns a dedicated thread (long operation).  All ambient scopes must be
        built into ``work`` by the caller before this call.
        """
        self._runner.submit(
            work,
            on_done=on_done,
            on_error=on_error,
            run_in_pool=run_in_pool,
        )

    def quiesce(self, timeout_ms: int = 5000) -> bool:
        """Wait for in-flight work and flush its queued main-thread deliveries
        (delegates to the underlying ``BackgroundRunner``; call from widget close)."""
        return self._runner.quiesce(timeout_ms)
