from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.ports import BackgroundExecutor

from .background import OffMainScopes
from .guard import AnalyzePermit
from .staged_analyze import _StagedAnalyzeService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import InteractiveSession
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import WritebackLifecyclePort


class AnalyzeService(_StagedAnalyzeService):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: State,
        bg: BackgroundExecutor,
        bus: EventBus,
        writeback: WritebackLifecyclePort,
        handles: OperationHandles,
    ) -> None:
        # FIT analyze is the OffMain-thread strategy with only the figure-routing
        # scope (no progress, no cancel). It takes **only a Handle, no exclusion**
        # (ADR-0019): analyze never conflicts with hardware, so it no longer fakes
        # an exclusion lease just to obtain the async handle (operation_id + await).
        # The handle is settled exactly once on the terminal slot (_finish /
        # _fail), the per-tab token map of which lives in the base.
        super().__init__(state, bg, bus, handles)
        self._writeback = writeback

    @property
    def _finished_signal(self) -> Any:
        return self.analyze_finished

    @property
    def _failed_signal(self) -> Any:
        return self.analyze_failed

    def start_analyze(
        self,
        permit: AnalyzePermit,
        analyze_params_instance: object,
        figure_container: FigureContainer | None = None,
    ) -> int:
        # Context + run-result preconditions are proven by the AnalyzePermit;
        # tab-busy is the dynamic check that stays at the operation boundary.
        # Returns the operation token (handle) so the caller can await it.
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        req = AnalyzeRequest(
            run_result=tab.run_result,
            analyze_params=analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        logger.info(
            "start_analyze: tab_id=%r analyze_params_type=%s",
            tab_id,
            type(analyze_params_instance).__name__,
        )
        # Handle only — no exclusion, no stop_event (analyze never conflicts and
        # is not cancellable in this minimal integration). Settled exactly once on
        # the terminal slot, or here (in _submit) if the worker fails to start.
        token = self._open_token(tab_id)
        adapter = tab.adapter
        scopes = OffMainScopes(figure_container=figure_container)
        self._submit(
            tab_id,
            lambda: adapter.analyze(req),
            scopes,
            self._on_analyze_finished,
            "analyze failed to start",
        )
        return token

    def start_interactive(self, permit: AnalyzePermit) -> int:
        """Begin an INTERACTIVE analysis: open the async handle and mark the tab
        analyzing. There is NO worker — the View mounts the interactive canvas on
        the main thread and the user paces the work (Main-thread-user-paced
        strategy, ADR-0019); the handle is held until ``finish_interactive``
        (Done). Returns the operation token (handle)."""
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")
        token = self._open_token(tab_id)
        self._begin(tab_id)
        return token

    def finish_interactive(self, tab_id: str, session: InteractiveSession) -> None:
        """The user finished the interactive pick (Done): build the result and run
        the SAME terminal path as a FIT analyze (writeback compute + State update +
        lease release + events), so the agent's analyze-result poll resolves."""
        self._on_analyze_finished(tab_id, session.finish())

    def _on_analyze_finished(self, tab_id: str, analyze_result: Any) -> None:
        logger.info(
            "_on_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(analyze_result).__name__,
        )
        self._finish(tab_id, analyze_result, self._record)

    def _on_analyze_failed(self, tab_id: str, error: Exception) -> None:
        # Named worker-failure slot kept as the public entry point (the on_error
        # callback target); delegates to the shared failure terminal path.
        self._on_failed(tab_id, error)

    def _record(self, tab_id: str, analyze_result: Any) -> None:
        # Tear down the previous analyze's writeback editor models before the new
        # draft replaces them (ADR-0008: per-item gc=False models are tied to a
        # specific analyze result). Compute the fresh persistent draft from the new
        # result (passed in, not written to State early), then commit result +
        # figure + items through the single mutator (which bumps the version).
        self._writeback.teardown_tab_items(tab_id)
        items = self._writeback.compute_items_for_tab(tab_id, analyze_result)
        self._state.update_tab_analyze(
            tab_id, analyze_result, analyze_result.figure, writeback_items=items
        )
