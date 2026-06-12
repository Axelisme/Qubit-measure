from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import PostAnalyzeRequest
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.ports import BackgroundExecutor

from .background import OffMainScopes

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus


class PostAnalyzeService(QObject):
    """Second-layer analysis service — mirrors :class:`AnalyzeService`.

    Runs a tab's ``adapter.post_analyze`` off the main thread on top of the
    primary analyze result, then records the result + figure into ``State`` on
    the main thread (the State main-thread invariant). Like FIT analyze, it takes
    a handle only (no exclusion, ADR-0019): post-analysis is a pure CPU recompute
    that never conflicts with hardware.

    Gate: the primary analyze result must exist; ``start_post_analyze`` fast-fails
    otherwise (the post-analysis builds on the primary fit it carries).
    """

    post_analyze_finished: Signal = Signal(str, object)
    post_analyze_failed: Signal = Signal(str, object)

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
        # Per-tab token map: post-analyze has NO exclusion gate (ADR-0019), so two
        # different tabs can run concurrently. A single token would let the second
        # start clobber the first, leaking the first's handle. Keyed by tab_id,
        # every terminal path settles exactly the token its own start created.
        self._active_tokens: dict[str, int] = {}

    def _release(self, tab_id: str, outcome: OperationOutcome) -> None:
        token = self._active_tokens.pop(tab_id, None)
        if token is not None:
            self._handles.settle(token, outcome)

    def start_post_analyze(
        self,
        tab_id: str,
        post_analyze_params_instance: object,
        figure_container: FigureContainer | None = None,
    ) -> int:
        """Begin a post-analysis for ``tab_id``. Returns the operation token.

        Gates on: the tab is not busy, and a primary analyze result exists (post-
        analysis depends on it). The worker reads run_result + analyze_result +
        params off the tab and calls ``adapter.post_analyze``.
        """
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        analyze_result = tab.analyze_result
        if analyze_result is None:
            raise RuntimeError(
                f"Tab {tab_id!r} has no primary analyze result to post-analyze"
            )

        ctx = self._state.exp_context
        req = PostAnalyzeRequest(
            run_result=tab.run_result,
            analyze_result=analyze_result,
            post_analyze_params=post_analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        logger.info(
            "start_post_analyze: tab_id=%r post_params_type=%s",
            tab_id,
            type(post_analyze_params_instance).__name__,
        )
        token = self._handles.create()
        self._active_tokens[tab_id] = token
        adapter = tab.adapter
        scopes = OffMainScopes(figure_container=figure_container)
        try:
            self._bg.submit(
                lambda: adapter.post_analyze(req),
                scopes,
                run_in_pool=False,
                on_done=lambda result: self._on_post_analyze_finished(tab_id, result),
                on_error=lambda exc: self._on_post_analyze_failed(tab_id, exc),
            )
        except Exception:
            self._release(
                tab_id, OperationOutcome("failed", "post-analyze failed to start")
            )
            raise
        # The tab is marked analyzing for the duration so concurrent run/analyze
        # is gated out (is_tab_busy covers analyzing).
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        return token

    def _on_post_analyze_finished(self, tab_id: str, post_result: Any) -> None:
        logger.info(
            "_on_post_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(post_result).__name__,
        )
        try:
            # Record result + figure through the single State mutator (bumps the
            # post_analyze version); it fast-fails if the primary result vanished
            # mid-flight (invalidated by a concurrent re-run/re-analyze).
            self._state.update_tab_post_analyze(tab_id, post_result, post_result.figure)
        except Exception as exc:
            # Service-side recording failed after a successful worker. Without this
            # the tab would stay analyzing forever and the handle would never
            # settle, since the worker's _failed path is not on this code path.
            logger.exception(
                "_on_post_analyze_finished post-processing failed: %r", exc
            )
            self._fail(tab_id, exc)
            return
        self._state.set_tab_analyzing(tab_id, False)
        # Settle the handle only after State is observable, then emit.
        self._release(tab_id, OperationOutcome("finished"))
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        self.post_analyze_finished.emit(tab_id, post_result)

    def _on_post_analyze_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_post_analyze_failed: tab_id=%r error=%r", tab_id, error)
        self._fail(tab_id, error)

    def _fail(self, tab_id: str, error: Exception) -> None:
        """The single failure terminal path: clear analyzing, settle the tab's
        handle failed, emit the interaction + post_analyze_failed signals. Shared
        by the worker-side failure (_on_post_analyze_failed) and a slot-internal
        failure during _on_post_analyze_finished so both leave identical state."""
        self._state.set_tab_analyzing(tab_id, False)
        self._release(tab_id, OperationOutcome("failed", str(error)))
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        self.post_analyze_failed.emit(tab_id, error)
