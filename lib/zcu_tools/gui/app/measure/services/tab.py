from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from zcu_tools.gui.app.measure.adapter.analyze_params import describe_analyze_params
from zcu_tools.gui.app.measure.state import Session, TabInteractionState
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.resource import (
    CfgObservation,
    CfgRef,
    CfgResolution,
    CfgStaleError,
)
from zcu_tools.gui.session.types import ParameterContext

from ..adapter import AnalysisMode, WritebackItem
from .plot_lifecycle import release_retired_plots
from .ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
    TabSnapshot,
)
from .writeback import WritebackDraft

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State

    from .ports import WritebackLifecyclePort
    from .tab_cfg import TabCfgResources


# Characters allowed verbatim in a tab-id slug; everything else (notably the
# adapter '/') collapses to '-'. The slug is cosmetic — the 8-hex suffix carries
# uniqueness — so a human/agent reads 'twotone-freq-1a2b3c4d' instead of a bare
# UUID while the id stays an opaque string key.
_SLUG_OK = set("abcdefghijklmnopqrstuvwxyz0123456789")


def _slug(name: str) -> str:
    out = "".join(c if c in _SLUG_OK else "-" for c in name.lower())
    # Collapse runs of '-' and trim, so 'twotone/rabi/amp_rabi' -> 'twotone-rabi-amp-rabi'.
    parts = [p for p in out.split("-") if p]
    return "-".join(parts) or "tab"


@dataclass(frozen=True)
class AnalysisPreparation:
    """Analysis readiness after a committed result, independent of its success."""

    has_params: bool
    error: str | None = None


class TabService:
    """Tab aggregate read model, cfg state, and tab lifecycle primitives."""

    def __init__(
        self,
        state: State,
        registry: Registry,
        writeback: WritebackLifecyclePort,
        cfg_resources: TabCfgResources,
        cfg_resolution: Callable[[ParameterContext | None], CfgResolution],
        captured_parameters: Callable[
            [Mapping[str, str] | None], ParameterContext | None
        ],
    ) -> None:
        """Inject owner-sequence captures and fixed cfg resolution construction.

        captured_parameters freezes current global sources with optional named
        aliases overrides. None and {} mean no override. EMPTY returns None
        without overrides; nonempty aliases reject no_context. Source/alias errors
        propagate before registration. Every cfg refresh/reset captures current
        sources anew; cfg_resolution retains only its supplied fixed capture.
        """
        self._state = state
        self._registry = registry
        # The tab lifecycle previews pane drafts and tears them down on close via
        # one narrow port, without depending on the concrete sibling service.
        self._writeback = writeback
        self._cfg_resources = cfg_resources
        self._cfg_resolution = cfg_resolution
        self._captured_parameters = captured_parameters

    def capture_parameters(
        self, tab_id: str, *, aliases: Mapping[str, str] | None = None
    ) -> ParameterContext | None:
        """Freeze current point/library/aliases for an existing tab's consumer.

        aliases maps alias names to concrete components for this capture only.
        None and {} use global rows. EMPTY yields None without overrides.
        Unknown tab and source or alias failures propagate; no observation guard is refreshed.
        """
        self._state.get_tab(tab_id)
        return self._captured_parameters(aliases)

    def preview_cfg(
        self, tab_id: str, expected: CfgRef, *, aliases: Mapping[str, str] | None = None
    ) -> CfgObservation:
        """Preview cfg with named aliases without publication or Run acceptance.

        Require an existing tab and its exact observed identity/revision.
        Resolve current inputs from the same capture owner as Run admission.
        Stale refs, source/alias failures and cfg notification guards propagate.
        None and {} return the cached observation. Nonempty aliases resolves
        current inputs once. Return Valid or Invalid without changing global rows,
        inputs or revision.
        """
        tab = self._state.get_tab(tab_id)
        observed = tab.cfg.observe()
        if expected != observed.ref:
            raise CfgStaleError(expected, observed.ref)
        if not aliases:
            return observed
        parameters = self.capture_parameters(tab_id, aliases=aliases)
        return tab.cfg.preview(
            expected.revision, resolution=self._cfg_resolution(parameters)
        )

    def get_snapshot(self, tab_id: str) -> TabSnapshot:
        """Build the immutable full render model for one tab (all live fields
        populated). The persist/restore form of ``TabSnapshot`` is produced
        elsewhere (codec / restore) with the live fields left empty."""
        tab = self._state.get_tab(tab_id)
        ctx = self._state.session_env
        is_running = self._state.is_tab_running(tab_id)
        interaction = TabInteractionState(
            global_run_active=self._state.is_run_active() and not is_running,
            has_context=ctx.has_context(),
            has_active_context=ctx.is_active(),
            has_soc=ctx.has_soc(),
            is_running=is_running,
            is_analyzing=tab.is_analyzing,
            has_run_result=tab.has_run_result(),
            has_analyze_result=tab.has_analyze_result(),
            has_figure=tab.has_figure(),
            has_post_analyze_result=tab.has_post_analyze_result(),
        )

        # Pane-owned writeback items via opaque drafts.
        def _items_for_pane(pane) -> tuple[WritebackItem, ...]:
            draft = pane.writeback_draft
            if draft is None:
                return ()
            return tuple(self._writeback.preview_draft(draft))

        analysis_items = _items_for_pane(tab.analysis)
        post_items = _items_for_pane(tab.post_analysis)
        return TabSnapshot(
            adapter_name=tab.adapter_name,
            cfg_schema=tab.cfg.snapshot_inputs(),
            tab_id=tab_id,
            interaction=interaction,
            capabilities=tab.adapter.capabilities,
            run=RunPaneSnapshot(
                result=tab.run.result,
                source_path=tab.run.source_path,
                source_operation_id=tab.run.source_operation_id,
                run_id=tab.run.confirmed_run_id,
            ),
            analysis=AnalysisPaneSnapshot(
                params=tab.analysis.params,
                result=tab.analysis.result,
                figures=tab.analysis.plots,
                writeback_items=analysis_items,
                unavailable_reasons=(
                    tab.analysis.writeback_draft.unavailable_reasons
                    if isinstance(tab.analysis.writeback_draft, WritebackDraft)
                    else ()
                ),
                has_writeback_draft=tab.analysis.writeback_draft is not None,
                source_operation_id=tab.analysis.source_operation_id,
                result_params=tab.analysis.result_params,
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=tab.post_analysis.params,
                result=tab.post_analysis.result,
                figures=tab.post_analysis.plots,
                writeback_items=post_items,
                unavailable_reasons=(
                    tab.post_analysis.writeback_draft.unavailable_reasons
                    if isinstance(tab.post_analysis.writeback_draft, WritebackDraft)
                    else ()
                ),
                has_writeback_draft=tab.post_analysis.writeback_draft is not None,
                source_operation_id=tab.post_analysis.source_operation_id,
                result_params=tab.post_analysis.result_params,
            ),
            save=SavePaneSnapshot(comment=tab.save.comment),
            artifacts=self._state.get_artifact_snapshots(tab_id),
        )

    def new_tab(
        self,
        adapter_name: str,
        from_dict: TabSnapshot | None = None,
    ) -> str:
        """Create a tab cfg whose refresh/reset reads current global sources.

        Defaults use current global aliases, not focus or via inference. Unbound
        aliases still permit tab creation; cfgs that consume them are Invalid.
        The tab retains no source capture; Run admission captures its own inputs.

        ``from_dict is None`` → a fresh tab with the adapter's default cfg.
        ``from_dict`` given (restore) → rebuild the tab from the snapshot's
        detached ``cfg_schema`` input memento. Path overrides are process-local and not
        persisted, so restore creates fresh panes.
        """
        adapter = self._registry.create(adapter_name)
        tab_id = f"{_slug(adapter_name)}-{uuid.uuid4().hex[:8]}"
        logger.info(
            "new_tab: adapter=%r tab_id=%r restore=%s",
            adapter_name,
            tab_id,
            from_dict is not None,
        )

        def current_parameters() -> ParameterContext | None:
            return self._captured_parameters(None)

        cfg = self._cfg_resources.create(
            tab_id,
            lambda: adapter.make_default_cfg(
                replace(self._state.session_env, parameters=current_parameters())
            ),
            resolution=lambda: self._cfg_resolution(current_parameters()),
            initial=from_dict.cfg_schema if from_dict is not None else None,
        )
        self._state.add_tab(
            tab_id,
            Session(
                adapter_name=adapter_name,
                adapter=adapter,
                cfg=cfg,
            ),
        )

        def cfg_changed(_observation: CfgObservation) -> None:
            self._state.version.bump(f"tab:{tab_id}:cfg")

        cfg.watch(cfg_changed)
        return tab_id

    def make_default_cfg(self, adapter_name: str) -> CfgSchema:
        """The adapter's default cfg under the current context — the base schema
        WorkspaceService needs to decode a persisted raw cfg into a live one."""
        adapter = self._registry.create(adapter_name)
        parameters = self._captured_parameters(None)
        return adapter.make_default_cfg(
            replace(self._state.session_env, parameters=parameters)
        )

    def list_adapter_names(self) -> list[str]:
        return self._registry.list_names()

    def adapter_guide(self, adapter_name: str) -> dict[str, str]:
        """Static human-facing orientation guide of an adapter (five fields)."""
        import dataclasses

        return dataclasses.asdict(self._registry.create(adapter_name).guide())

    def analyze_param_definitions(
        self, adapter_name: str, *, stage: Literal["primary", "post"]
    ) -> list[dict[str, Any]]:
        """Describe canonical adapter parameters before or after a result exists."""
        adapter = self._registry.create(adapter_name)
        params_cls = (
            adapter.analyze_params_cls()
            if stage == "primary"
            else adapter.post_analyze_params_cls()
        )
        return describe_analyze_params(params_cls)

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Cannot close busy tab {tab_id!r}")
        self._cfg_resources.retire(tab_id)
        retired = self._state.remove_tab(tab_id)
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("closed-tab draft teardown failed")
        release_retired_plots(retired)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).analysis.result

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._state.get_tab(tab_id).adapter_name

    def prepare_result_analysis(self, tab_id: str) -> AnalysisPreparation:
        """Seed params from this Run/Load result and its saved source.

        tab_id selects a tab with a result. Use its publication parameters,
        including context-free None, never current tab/session bindings.
        Publish returned params; adapter failure becomes AnalysisPreparation.error
        without changing the result. Missing result raises RuntimeError.
        """
        tab = self._state.get_tab(tab_id)
        if tab.run.result is None:
            raise RuntimeError("No run result available to build analyze params")
        if tab.adapter.capabilities.analysis is AnalysisMode.NONE:
            return AnalysisPreparation(has_params=False)
        try:
            instance = tab.adapter.get_analyze_params(
                tab.run.result,
                replace(self._state.session_env, parameters=tab.run.parameters),
            )
        except Exception as exc:
            logger.exception("Analysis preparation failed for tab %s", tab_id)
            return AnalysisPreparation(has_params=False, error=str(exc))
        self._state.update_tab_analyze_param_instance(tab_id, instance)
        return AnalysisPreparation(has_params=True)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._state.update_tab_analyze_param_instance(tab_id, instance)

    def initialize_tab_post_analyze_params(self, tab_id: str) -> object:
        """Seed and store Post params from the primary result and result source.

        tab_id selects a tab with a primary analysis result. Its Run/Load saved
        parameters supply the source, including None without fallback. Return
        the stored params. Missing primary result or adapter failure raises.
        """
        tab = self._state.get_tab(tab_id)
        if tab.analysis.result is None:
            raise RuntimeError(
                "No primary analyze result available to build post-analysis params"
            )
        instance = tab.adapter.get_post_analyze_params(
            tab.analysis.result,
            replace(self._state.session_env, parameters=tab.run.parameters),
        )
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)
        return instance

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)

    def get_tab_post_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).post_analysis.result
