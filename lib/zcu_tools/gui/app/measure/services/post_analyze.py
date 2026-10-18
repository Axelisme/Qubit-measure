from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.measure.adapter import PostAnalyzeRequest, PostWritebackRequest
from zcu_tools.gui.app.measure.events.tab import TabInteractionFact
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.resources.entry.ledger_models import JsonObject

from ..writeback import WritebackDraft
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
    run_id: str | None
    experiment_name: str
    device_snapshot: Mapping[str, JsonObject] | None
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
        """Begin Post analysis of the selected Primary result; return its token.

        tab_id selects an idle tab with a Primary result. Adapter-specific
        post_analyze_params_instance and the owned plots supply operation inputs.
        Capture Run/Load parameters and the Primary WritebackDraft.run_id at
        start. The new Post draft retains that identity, including None when the
        Primary draft has none or no draft exists; never infer it from live Run
        metadata or Save state. Later tab/session edits do not replace inputs.
        Busy/missing Primary raises FailedPreconditionError. A present Primary
        draft must be WritebackDraft, otherwise raise TypeError before submission.
        Submission failures propagate after unpublished plot cleanup.
        """
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        analyze_result = tab.analysis.result
        if analyze_result is None:
            raise FailedPreconditionError(
                f"Tab {tab_id!r} has no primary analyze result to post-analyze"
            )

        primary_draft = tab.analysis.writeback_draft
        if primary_draft is not None and not isinstance(primary_draft, WritebackDraft):
            raise TypeError("Primary writeback draft must be WritebackDraft")
        run_id = primary_draft.run_id if primary_draft is not None else None

        ctx = replace(self._state.session_env, parameters=tab.run.parameters)
        req = PostAnalyzeRequest(
            run_result=tab.run.result,
            analyze_result=analyze_result,
            post_analyze_params=post_analyze_params_instance,
            parameters=ctx.parameters,
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
            run_id=run_id,
            experiment_name=primary_draft.experiment_name
            if primary_draft is not None
            else tab.adapter_name,
            device_snapshot=primary_draft.device_snapshot
            if primary_draft is not None
            else None,
            context=ctx,
            adapter=adapter,
            params=post_analyze_params_instance,
            plots=plots,
        )

        def work(_factory: Any) -> Any:  # factory is None (wants_progress=False)
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
                        analyze_result=analyze_result,
                        post_analyze_result=post_result,
                        ctx=ctx,
                    )
                )
            )
            draft = writeback.create_draft(
                proposal_items,
                parameters=ctx.parameters,
                run_id=captured_inputs.run_id,
                experiment_name=captured_inputs.experiment_name,
                device_snapshot=captured_inputs.device_snapshot,
            )
            plots.finish()
            retired = self._state.update_tab_post_analyze(
                tab_id,
                post_result,
                plots,
                post_analyze_params_instance=params,
                source_operation_id=self._active_tokens[tab_id],
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
