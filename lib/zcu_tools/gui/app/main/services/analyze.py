from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from zcu_tools.gui.app.main.adapter import AnalyzeRequest, WritebackRequest
from zcu_tools.gui.app.main.events.completion import AnalyzeFailedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.interactive import PluginDefinition, Session
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import OperationRunner

from .guard import AnalyzePermit
from .scopes import figure_ambient
from .staged_analyze import _StagedAnalyzeService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpAdapterProtocol
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.ports import OwnerScheduler
    from zcu_tools.gui.session.types import ExpContext

    from ..state import RetiredPaneResources
    from .ports import AnalyzeStatePort, WritebackLifecyclePort


@dataclass(frozen=True, slots=True)
class _AnalyzeCapture:
    run_result: object | None
    context: ExpContext
    adapter: ExpAdapterProtocol
    params: object | None


@dataclass(frozen=True, slots=True)
class ActiveInteractive:
    """The service-owned plugin and committed session for one active tab."""

    plugin: PluginDefinition[Any, Any]
    session: Session[Any]


class AnalyzeService(_StagedAnalyzeService):
    FAILURE_STAGE = "primary"
    STARTED_FACT = TabInteractionFact.PRIMARY_ANALYZE_STARTED
    SUCCEEDED_FACT = TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED
    FAILED_FACT = TabInteractionFact.PRIMARY_ANALYZE_FAILED
    START_REJECTED_FACT = TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED

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
        self._interactive_instances: dict[str, ActiveInteractive] = {}
        # Captured at operation start so proposal generation cannot accidentally
        # consume a context or run result replaced while the worker was running.
        self._captured_inputs: dict[str, _AnalyzeCapture] = {}

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
            run_result=tab.run.result,
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
        captured_inputs = _AnalyzeCapture(
            run_result=req.run_result,
            context=ctx,
            adapter=adapter,
            params=analyze_params_instance,
        )
        self._captured_inputs[tab_id] = captured_inputs

        def work(factory: Any) -> Any:  # factory is None (wants_progress=False)
            # Analyze uses only figure_ambient (no pbar or cancellation scope — ADR-0026 §2).
            with figure_ambient(figure_container):
                return adapter.analyze(req)

        try:
            return self._submit_with_runner(
                tab_id,
                work,
                lambda record_tab_id, result: self._record(
                    record_tab_id, result, captured_inputs=captured_inputs
                ),
                "analyze failed to start",
            )
        except Exception:
            self._captured_inputs.pop(tab_id, None)
            raise

    def start_plugin(
        self,
        permit: AnalyzePermit,
        plugin: PluginDefinition[Any, Any],
        owner: OwnerScheduler,
    ) -> int:
        """Capture operation inputs and register one service-owned session."""
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        captured = _AnalyzeCapture(
            run_result=tab.run.result,
            context=ctx,
            adapter=tab.adapter,
            params=tab.analysis.params,
        )
        session = plugin.open(owner)

        def cancel() -> None:
            self.cancel_interactive(tab_id)

        try:
            token = self._open_token(tab_id, cancel_hook=cancel)
        except Exception:
            session.dispose()
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED,
                )
            )
            raise
        self._captured_inputs[tab_id] = captured
        self._interactive_instances[tab_id] = ActiveInteractive(plugin, session)
        self._interactive_tabs.add(tab_id)
        self._begin(tab_id)
        return token

    def get_interactive(self, tab_id: str) -> ActiveInteractive | None:
        """Read the active plugin and session; never return a retired binding."""
        return self._interactive_instances.get(tab_id)

    def finish_plugin(self, tab_id: str, figure: Figure | None = None) -> bool:
        """Finish from the committed snapshot; validation errors leave it editable."""
        active = self._interactive_instances.get(tab_id)
        if active is None:
            return False
        try:
            result = active.plugin.finish(active.session, figure)
        except Exception as exc:
            try:
                active.session.ensure_input_open()
            except FailedPreconditionError:
                token = self._active_tokens[tab_id]
                with self._bus.origin(self._handles.event_origin(token)):
                    self._interactive_instances.pop(tab_id)
                    active.session.dispose()
                    self._interactive_tabs.discard(tab_id)
                    self._captured_inputs.pop(tab_id, None)
                    logger.exception(
                        "interactive result construction failed: tab_id=%r", tab_id
                    )
                    self._state.set_tab_analyzing(tab_id, False)
                    self._release(tab_id, OperationOutcome("failed", str(exc)))
                    self._bus.emit(
                        TabInteractionChangedPayload(
                            tab_id=tab_id,
                            fact=TabInteractionFact.PRIMARY_ANALYZE_FAILED,
                        )
                    )
                    self._bus.emit(
                        AnalyzeFailedPayload(
                            tab_id=tab_id, stage="primary", error_message=str(exc)
                        )
                    )
                return True
            raise
        token = self._active_tokens[tab_id]
        captured = self._captured_inputs[tab_id]
        with self._bus.origin(self._handles.event_origin(token)):
            self._interactive_instances.pop(tab_id)
            active.session.dispose()
            self._interactive_tabs.discard(tab_id)
            self._on_analyze_finished(tab_id, result, captured_inputs=captured)
        return True

    def is_interactive_active(self, tab_id: str) -> bool:
        """Whether this tab owns an in-flight plugin session."""
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
        ``AnalyzeFailedPayload`` — a user/agent cancel is not an error, so it must not
        pop the "Analyze failed" diagnostic. Returns False (no-op) when the tab has
        no in-flight interactive picker, so the caller can report a graceful
        message instead of raising.
        """
        if tab_id not in self._interactive_tabs:
            return False
        token = self._active_tokens[tab_id]
        with self._bus.origin(self._handles.event_origin(token)):
            self._interactive_tabs.discard(tab_id)
            active = self._interactive_instances.pop(tab_id, None)
            if active is not None:
                active.session.dispose()
            self._captured_inputs.pop(tab_id, None)
            logger.info("cancel_interactive: tab_id=%r", tab_id)
            self._state.set_tab_analyzing(tab_id, False)
            self._release(tab_id, OperationOutcome("cancelled"))
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=TabInteractionFact.PRIMARY_ANALYZE_CANCELLED,
                )
            )
        return True

    def _on_analyze_finished(
        self,
        tab_id: str,
        analyze_result: Any,
        *,
        captured_inputs: _AnalyzeCapture,
    ) -> None:
        """Terminal path used by finish_plugin (interactive → same FIT terminal).

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
            self._record(tab_id, analyze_result, captured_inputs=captured_inputs)
        except Exception as exc:
            logger.exception("%s finished post-processing failed: %r", tab_id, exc)
            self._state.set_tab_analyzing(tab_id, False)
            self._release(tab_id, OperationOutcome("failed", str(exc)))
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=TabInteractionFact.PRIMARY_ANALYZE_FAILED,
                )
            )
            self._bus.emit(
                AnalyzeFailedPayload(
                    tab_id=tab_id,
                    stage="primary",
                    error_message=str(exc),
                )
            )
            return
        self._state.set_tab_analyzing(tab_id, False)
        self._release(tab_id, OperationOutcome("finished"))
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id,
                fact=TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
            )
        )

    def _teardown_retired(self, retired: RetiredPaneResources | None) -> None:
        if retired is None:
            return
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired analyze draft teardown failed")

    def _record(
        self,
        tab_id: str,
        analyze_result: Any,
        *,
        captured_inputs: _AnalyzeCapture,
    ) -> None:
        # Every terminal receives the operation-start snapshot explicitly; the
        # mutable active context and pane inputs are never terminal fallbacks.
        self._captured_inputs.pop(tab_id, None)
        run_result = captured_inputs.run_result
        ctx = captured_inputs.context
        adapter = captured_inputs.adapter
        analyze_params = captured_inputs.params

        proposal_items: list[Any] = []
        if run_result is not None:
            proposal_items = list(
                adapter.get_writeback_items(
                    WritebackRequest(
                        run_result=run_result,
                        analyze_result=analyze_result,
                        ctx=ctx,
                    )
                )
            )

        draft: Any | None = None
        try:
            draft = self._writeback.create_draft(proposal_items)
            retired = self._state.update_tab_analyze(
                tab_id,
                analyze_result,
                getattr(analyze_result, "figure", None),
                writeback_draft=draft,
                analyze_params_instance=analyze_params,
            )
        except BaseException:
            if draft is not None:
                try:
                    self._writeback.teardown_draft(draft)
                except Exception:
                    logger.exception("new analyze draft teardown failed")
            raise
        self._teardown_retired(retired)
