from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import OperationRunner

from .guard import AnalyzePermit
from .scopes import figure_ambient
from .staged_analyze import _StagedAnalyzeService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import InteractiveSession
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import AnalyzeStatePort, WritebackLifecyclePort


class AnalyzeService(_StagedAnalyzeService):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: AnalyzeStatePort,
        runner: OperationRunner,
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
        super().__init__(state, runner, bus, handles)
        self._writeback = writeback
        # Tabs whose active token is an INTERACTIVE picker (no worker; settles only
        # on Done / cancel). FIT analyze shares the base's per-tab token map but is
        # NOT tracked here, so cancel_interactive cannot reach into a worker-backed
        # analyze (which would settle the handle while the worker callback is still
        # in flight). Entries are removed by every interactive terminal path.
        self._interactive_tabs: set[str] = set()

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
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

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
        adapter = tab.adapter

        def work(factory: Any) -> Any:  # factory is None (wants_progress=False)
            # Analyze uses only figure_ambient (no pbar or cancellation scope — ADR-0026 §2).
            with figure_ambient(figure_container):
                return adapter.analyze(req)

        return self._submit_with_runner(
            tab_id,
            work,
            self._record,
            "analyze failed to start",
        )

    def start_interactive(self, permit: AnalyzePermit) -> int:
        """Begin an INTERACTIVE analysis: open the async handle and mark the tab
        analyzing. There is NO worker — the View mounts the interactive canvas on
        the main thread and the user paces the work (Main-thread-user-paced
        strategy, ADR-0019); the handle is held until ``finish_interactive``
        (Done). Returns the operation token (handle).

        ADR-0025: cancel_hook triggers cancel_interactive so handles.stop(token)
        causes the channel to directly settle-cancelled, allowing an awaiter's
        Stop event to fold reason correctly before Settled arrives.

        INTERACTIVE does NOT go through OperationRunner (stage2c_spec.md §interactive).
        """
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        # Open the token with a cancel_hook that executes the interactive teardown.
        # The hook runs *after* Stop is enqueued, so Settled(cancelled) from the
        # hook's _release lands after Stop — the consumer folds reason correctly.
        # Wrap cancel_interactive (returns bool) so the hook matches CancelHook
        # signature (returns None). The bool return is irrelevant here — stop()
        # already knows this is an interactive op.
        def _hook() -> None:
            self.cancel_interactive(tab_id)

        token = self._open_token(tab_id, cancel_hook=_hook)
        self._interactive_tabs.add(tab_id)
        self._begin(tab_id)
        return token

    def finish_interactive(self, tab_id: str, session: InteractiveSession) -> None:
        """The user finished the interactive pick (Done): build the result and run
        the SAME terminal path as a FIT analyze (writeback compute + State update +
        lease release + events), so the agent's analyze-result poll resolves."""
        self._interactive_tabs.discard(tab_id)
        self._on_analyze_finished(tab_id, session.finish())

    def is_interactive_active(self, tab_id: str) -> bool:
        """Whether ``tab_id`` currently holds an in-flight INTERACTIVE picker
        (opened by ``start_interactive``, not yet settled by Done / cancel)."""
        return tab_id in self._interactive_tabs

    def active_interactive_tab(self) -> str | None:
        """A tab with an in-flight interactive analyze, or None — the foreground
        op for ``Controller.cancel_active_operation`` to settle. Arbitrary if more
        than one (measure-gui drives one interactive picker at a time)."""
        return next(iter(self._interactive_tabs), None)

    def active_interactive_token(self) -> int | None:
        """The handle token of the active interactive analyze, or None."""
        tab = self.active_interactive_tab()
        if tab is None:
            return None
        return self._active_tokens.get(tab)

    def cancel_interactive(self, tab_id: str) -> bool:
        """Cancel an in-flight INTERACTIVE analyze: settle its handle as cancelled
        and clear ``is_analyzing`` so the tab can close.

        Mirrors the ``_fail`` terminal (set_tab_analyzing(False) + _release +
        interaction event) but with a ``cancelled`` outcome and WITHOUT emitting
        ``analyze_failed`` — a user/agent cancel is not an error, so it must not
        pop the "Analyze failed" diagnostic. Returns False (no-op) when the tab has
        no in-flight interactive picker, so the caller can report a graceful
        message instead of raising.
        """
        if tab_id not in self._interactive_tabs:
            return False
        self._interactive_tabs.discard(tab_id)
        logger.info("cancel_interactive: tab_id=%r", tab_id)
        self._state.set_tab_analyzing(tab_id, False)
        self._release(tab_id, OperationOutcome("cancelled"))
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        return True

    def _on_analyze_finished(self, tab_id: str, analyze_result: Any) -> None:
        """Terminal path used by finish_interactive (interactive → same FIT terminal).

        FIT analyze uses _submit_with_runner's internal _finish directly.
        Interactive calls here, which runs record + clears analyzing + settles
        via _release.
        """
        logger.info(
            "_on_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(analyze_result).__name__,
        )
        # Interactive uses _release (not runner settle) — the token is in
        # _active_tokens from _open_token.
        try:
            self._record(tab_id, analyze_result)
        except Exception as exc:
            logger.exception("%s finished post-processing failed: %r", tab_id, exc)
            self._state.set_tab_analyzing(tab_id, False)
            self._release(tab_id, OperationOutcome("failed", str(exc)))
            self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
            self._failed_signal.emit(tab_id, exc)
            return
        self._state.set_tab_analyzing(tab_id, False)
        self._release(tab_id, OperationOutcome("finished"))
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        self._finished_signal.emit(tab_id, analyze_result)

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
