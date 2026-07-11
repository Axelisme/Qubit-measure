from __future__ import annotations

import logging
from typing import Any

from qtpy.QtCore import Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import PostAnalyzeRequest
from zcu_tools.gui.app.main.events.tab import TabInteractionFact
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting import FigureContainer

from .scopes import figure_ambient
from .staged_analyze import _StagedAnalyzeService

logger = logging.getLogger(__name__)


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

    post_analyze_finished: Signal = Signal(str, object)
    post_analyze_failed: Signal = Signal(str, object)

    @property
    def _finished_signal(self) -> Any:
        return self.post_analyze_finished

    @property
    def _failed_signal(self) -> Any:
        return self.post_analyze_failed

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
        analyze_result = tab.analyze_result
        if analyze_result is None:
            raise FailedPreconditionError(
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
        adapter = tab.adapter

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
            self._record,
            "post-analyze failed to start",
        )

    def _record(self, tab_id: str, post_result: Any) -> None:
        # Record result + figure through the single State mutator (bumps the
        # post_analyze version); it fast-fails if the primary result vanished
        # mid-flight (invalidated by a concurrent re-run/re-analyze).
        self._state.update_tab_post_analyze(tab_id, post_result, post_result.figure)
