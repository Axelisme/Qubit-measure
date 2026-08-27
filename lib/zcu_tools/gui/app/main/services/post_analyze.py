from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

from zcu_tools.gui.app.main.adapter import PostAnalyzeRequest, PostWritebackRequest
from zcu_tools.gui.app.main.events.tab import TabInteractionFact
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner

from .scopes import figure_ambient
from .staged_analyze import _StagedAnalyzeService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .ports import AnalyzeStatePort, WritebackLifecyclePort


class PostAnalyzeService(_StagedAnalyzeService):
    """Second-layer analysis service — mirrors :class:`AnalyzeService`.

    Runs a tab's ``adapter.post_analyze`` off the main thread on top of the
    primary analyze result, then records the result + figure into ``State`` on
    the main thread (the State main-thread invariant). Like FIT analyze, it takes
    a handle only (no exclusion, ADR-0019): post-analysis is a pure CPU recompute
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
        writeback: WritebackLifecyclePort | None = None,
    ) -> None:
        super().__init__(state, runner, bus, handles)
        self._writeback = writeback

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
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        analyze_result = tab.analysis.result
        if analyze_result is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no primary analyze result to post-analyze"
            )

        ctx = self._state.exp_context
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
        captured_inputs = (
            req.run_result,
            req.analyze_result,
            ctx,
            adapter,
            post_analyze_params_instance,
        )

        def work(factory: Any) -> Any:  # factory is None (wants_progress=False)
            # Post-analyze uses only figure_ambient (no pbar or cancellation scope — ADR-0026 §2).
            with figure_ambient(figure_container):
                return adapter.post_analyze(req)

        # The tab is marked analyzing for the duration so concurrent run/analyze is
        # gated out (is_tab_busy covers analyzing) — done by _submit_with_runner's
        # _begin tail (post-begin invariant from stage2c_spec.md).
        return self._submit_with_runner(
            tab_id,
            work,
            lambda record_tab_id, result: self._record(
                record_tab_id, result, captured_inputs=captured_inputs
            ),
            "post-analyze failed to start",
        )

    def _teardown_retired(self, retired: Any) -> None:
        if retired is None or self._writeback is None:
            return
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired post-analysis draft teardown failed")

    def _record(
        self,
        tab_id: str,
        post_result: Any,
        *,
        captured_inputs: tuple[Any, Any, Any, Any, Any] | None = None,
    ) -> None:
        # The worker terminal uses operation-start values rather than rereading
        # the active context or the current primary result. The compatibility
        # direct-call branch below is not used by production terminals:
        # ``_submit_with_runner`` always closes over this tuple.
        if captured_inputs is None:
            tab = self._state.get_tab(tab_id)
            run_result, analyze_result, ctx, adapter, params = (
                tab.run.result,
                tab.analysis.result,
                self._state.exp_context,
                tab.adapter,
                tab.post_analysis.params,
            )
        else:
            run_result, analyze_result, ctx, adapter, params = captured_inputs

        writeback = self._writeback
        assert writeback is not None
        draft: Any | None = None
        try:
            proposal_factory = getattr(adapter, "get_post_writeback_items", None)
            proposal_items: list[Any] = []
            if callable(proposal_factory):
                factory = cast(Callable[..., Iterable[Any]], proposal_factory)
                proposal_items = list(
                    factory(
                        PostWritebackRequest(
                            run_result=run_result,
                            analyze_result=cast(Any, analyze_result),
                            post_analyze_result=post_result,
                            ctx=ctx,
                        )
                    )
                )
            draft = writeback.create_draft(proposal_items)
            retired = self._state.update_tab_post_analyze(
                tab_id,
                post_result,
                getattr(post_result, "figure", None),
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
