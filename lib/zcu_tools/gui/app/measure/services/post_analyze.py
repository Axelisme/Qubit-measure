from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from zcu_tools.gui.app.measure.adapter import PostAnalyzeRequest, PostWritebackRequest
from zcu_tools.gui.app.measure.events.tab import TabInteractionFact
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner

from .plot_lifecycle import discard_unpublished_plots, release_retired_plots
from .staged_analyze import _StagedAnalyzeService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import ExpAdapterProtocol
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.plotting.plots import Plots

    from ..state import RetiredPaneResources
    from .ports import AnalyzeStatePort, WritebackLifecyclePort


@dataclass(frozen=True, slots=True)
class _PostAnalyzeCapture:
    run_result: object | None
    analyze_result: object
    context: SessionEnv
    adapter: ExpAdapterProtocol
    params: object
    plots: Plots


class PostAnalyzeService(_StagedAnalyzeService):
    """Second-layer analysis service — mirrors :class:`AnalyzeService`.

    Runs a tab's ``adapter.post_analyze`` off the main thread on top of the
    primary analyze result, then records numeric results + named plots in ``State`` on
    the main thread (the State main-thread invariant). Like FIT analyze, it takes
    a handle only (no exclusion, ADR-0066): post-analysis is a pure CPU recompute
    that never conflicts with hardware. The handle lifecycle + failure path live in
    the shared :class:`_StagedAnalyzeService` base.

    Gate: the primary analyze result must exist; ``start_post_analyze`` fast-fails
    otherwise (the post-analysis builds on the primary fit it carries).
    """

    STARTED_FACT = TabInteractionFact.POST_ANALYZE_STARTED
    SUCCEEDED_FACT = TabInteractionFact.POST_ANALYZE_SUCCEEDED
    FAILED_FACT = TabInteractionFact.POST_ANALYZE_FAILED
    START_REJECTED_FACT = TabInteractionFact.POST_ANALYZE_START_REJECTED
    FAILURE_STAGE = "post"

    def __init__(
        self,
        state: AnalyzeStatePort,
        runner: OperationRunner,
        bus: EventBus,
        handles: OperationHandles,
        writeback: WritebackLifecyclePort,
    ) -> None:
        super().__init__(state, runner, bus, handles)
        self._writeback = writeback

    def start_post_analyze(
        self,
        tab_id: str,
        post_analyze_params_instance: object,
        *,
        plots: Plots,
    ) -> int:
        """Begin a post-analysis for ``tab_id``. Returns the operation token.

        Gates on: the tab is not busy, and a primary analyze result exists (post-
        analysis depends on it). The worker reads run_result + analyze_result +
        params off the tab and calls ``adapter.post_analyze``.
        """
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        analyze_result = tab.analysis.result
        if analyze_result is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no primary analyze result to post-analyze"
            )

        ctx = self._state.session_env
        req = PostAnalyzeRequest(
            run_result=tab.run.result,
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
        adapter = tab.adapter
        captured_inputs = _PostAnalyzeCapture(
            run_result=req.run_result,
            analyze_result=req.analyze_result,
            context=ctx,
            adapter=adapter,
            params=post_analyze_params_instance,
            plots=plots,
        )

        def work(factory: Any) -> Any:  # factory is None (wants_progress=False)
            return adapter.post_analyze(req, plots=plots)

        # The tab is marked analyzing for the duration so concurrent run/analyze is
        # gated out (is_tab_busy covers analyzing) — done by _submit_with_runner's
        # _begin tail (post-begin invariant from stage2c_spec.md).
        try:
            return self._submit_with_runner(
                tab_id,
                work,
                lambda record_tab_id, result: self._record(
                    record_tab_id, result, captured_inputs=captured_inputs
                ),
                lambda: discard_unpublished_plots(plots),
                "post-analyze failed to start",
            )
        except Exception:
            try:
                discard_unpublished_plots(plots)
            except Exception:
                logger.exception("Unpublished post-analysis plot cleanup failed")
            raise

    def _teardown_retired(self, retired: RetiredPaneResources | None) -> None:
        if retired is None:
            return
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired post-analysis draft teardown failed")
        release_retired_plots(retired)

    def _record(
        self,
        tab_id: str,
        post_result: Any,
        *,
        captured_inputs: _PostAnalyzeCapture,
    ) -> None:
        run_result = captured_inputs.run_result
        analyze_result = captured_inputs.analyze_result
        ctx = captured_inputs.context
        adapter = captured_inputs.adapter
        params = captured_inputs.params
        plots = captured_inputs.plots
        writeback = self._writeback
        draft: Any | None = None
        try:
            proposal_items = list(
                adapter.get_post_writeback_items(
                    PostWritebackRequest(
                        run_result=run_result,
                        analyze_result=cast(Any, analyze_result),
                        post_analyze_result=post_result,
                        ctx=ctx,
                    )
                )
            )
            draft = writeback.create_draft(proposal_items)
            plots.finish()
            retired = self._state.update_tab_post_analyze(
                tab_id,
                post_result,
                plots,
                post_analyze_params_instance=params,
                writeback_draft=draft,
            )
        except BaseException:
            if draft is not None:
                try:
                    writeback.teardown_draft(draft)
                except Exception:
                    logger.exception("new post-analysis draft teardown failed")
            raise
        self._teardown_retired(retired)
