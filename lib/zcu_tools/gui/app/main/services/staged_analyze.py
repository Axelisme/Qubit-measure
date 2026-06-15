"""_StagedAnalyzeService — the shared spine of the two off-main analyze stages.

``AnalyzeService`` (primary FIT/INTERACTIVE) and ``PostAnalyzeService`` (second
layer) run the identical operation lifecycle: a per-tab async handle (no exclusion,
ADR-0019), ``set_tab_analyzing`` for the duration, a single failure terminal path,
and a finished path that records into State before settling the handle. Only three
things differ between them — the gate + request assembly that starts the work, what
``record`` does with the worker's result, and which Qt signals fire on
finish/fail. This base owns the lifecycle skeleton; each subclass fills the three
seams.

The base deliberately holds NO operation-kind knowledge beyond "off-main work that
settles a per-tab handle": the INTERACTIVE branch (no worker, user-paced) and the
distinct gates (run_result vs analyze_result) live in the subclasses, not here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.session.operation_handles import (
    CancelHook,
    OperationHandles,
    OperationOutcome,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.ports import BackgroundExecutor


class _StagedAnalyzeService(QObject):
    """Lifecycle skeleton shared by the primary- and post-analyze services.

    Subclasses declare their own ``finished``/``failed`` Qt signals (a signal must
    be bound on the concrete class) and expose them through ``_finished_signal`` /
    ``_failed_signal`` so the base can emit the right one. The per-tab token map,
    ``_release``, the failure terminal path, and the start/finish templates are
    owned here.
    """

    def __init__(
        self,
        state: State,
        bg: BackgroundExecutor,
        bus: EventBus,
        handles: OperationHandles,
    ) -> None:
        super().__init__()
        self._state = state
        self._bg = bg
        self._bus = bus
        self._handles = handles
        # Per-tab token map: analyze has NO exclusion gate (ADR-0019), so two
        # different tabs can run concurrently. A single token would let the second
        # start clobber the first, leaking the first's handle. Keyed by tab_id,
        # every terminal path settles exactly the token its own start created.
        self._active_tokens: dict[str, int] = {}

    # -- subclass seam: which concrete Qt signal to emit on finish / fail --
    #
    # Typed Any: at runtime each accessor returns a *bound* signal (it has .emit),
    # but a class-level ``Signal`` annotation is the unbound descriptor type, which
    # the type checker would (wrongly) reject ``.emit`` on.

    @property
    def _finished_signal(self) -> Any:
        raise NotImplementedError

    @property
    def _failed_signal(self) -> Any:
        raise NotImplementedError

    # -- shared handle bookkeeping -----------------------------------------

    def _release(self, tab_id: str, outcome: OperationOutcome) -> None:
        token = self._active_tokens.pop(tab_id, None)
        if token is not None:
            self._handles.settle(token, outcome)

    def _open_token(self, tab_id: str, cancel_hook: CancelHook | None = None) -> int:
        """Mint and register the per-tab operation handle (pending).

        ``cancel_hook`` is forwarded to OperationChannel.create; pass None for
        FIT-analyze (not cancellable) or a teardown callable for interactive.
        """
        token = self._handles.create(cancel_hook=cancel_hook)
        self._active_tokens[tab_id] = token
        return token

    def _begin(self, tab_id: str) -> None:
        """Mark the tab analyzing and announce the interaction change.

        Shared tail of every start path (FIT worker, INTERACTIVE no-worker, post):
        the tab is busy for the duration so concurrent run/analyze is gated out.
        """
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))

    def _submit(
        self,
        tab_id: str,
        work: Callable[[], Any],
        on_finished: Callable[[str, Any], None],
        start_fail_message: str,
    ) -> None:
        """Submit ``work`` off-main, releasing the already-open token if it fails
        to start, then mark the tab analyzing. The token must already be open
        (``_open_token``) so a synchronous submit failure settles the right one.

        All ambient scopes must be built into ``work`` by the caller (ADR-0026
        §2): the work thunk owns its own ``figure_ambient`` context manager.
        """
        try:
            self._bg.submit(
                work,
                run_in_pool=False,
                on_done=lambda result: on_finished(tab_id, result),
                on_error=lambda exc: self._on_failed(tab_id, exc),
            )
        except Exception:
            self._release(tab_id, OperationOutcome("failed", start_fail_message))
            raise
        self._begin(tab_id)

    def _finish(
        self,
        tab_id: str,
        result: Any,
        record: Callable[[str, Any], None],
    ) -> None:
        """The shared finished terminal path: record into State, then settle.

        ``record`` performs the subclass-specific State write (writeback compute +
        ``update_tab_analyze`` for primary; ``update_tab_post_analyze`` for post).
        On a recording failure the handle would otherwise never settle and the tab
        would stay analyzing forever — so route it through the single ``_fail``
        path. On success: clear analyzing, settle the handle only after State is
        observable (an awaiter that wakes sees a ready figure), then emit.
        """
        try:
            record(tab_id, result)
        except Exception as exc:
            logger.exception("%s finished post-processing failed: %r", tab_id, exc)
            self._fail(tab_id, exc)
            return
        self._state.set_tab_analyzing(tab_id, False)
        self._release(tab_id, OperationOutcome("finished"))
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        self._finished_signal.emit(tab_id, result)

    def _on_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("staged-analyze failed: tab_id=%r error=%r", tab_id, error)
        self._fail(tab_id, error)

    def _fail(self, tab_id: str, error: Exception) -> None:
        """The single failure terminal path: clear analyzing, settle the tab's
        handle failed, emit the interaction + failed signals. Shared by the
        worker-side failure and a slot-internal failure during finish, so both
        leave identical observable state."""
        self._state.set_tab_analyzing(tab_id, False)
        self._release(tab_id, OperationOutcome("failed", str(error)))
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        self._failed_signal.emit(tab_id, error)
