from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from zcu_tools.gui.app.measure.adapter.analyze_params import describe_analyze_params
from zcu_tools.gui.app.measure.state import Session, TabInteractionState
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.resource import CfgObservation, CfgResolution
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import RoleSpec

from ..adapter import AnalysisMode, WritebackItem
from .plot_lifecycle import release_retired_plots
from .ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
    TabSnapshot,
)

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
            [Mapping[str, RoleSpec], str | None], ParameterContext | None
        ],
    ) -> None:
        """Inject fixed cfg sources and the current role capture operation.

        captured_parameters receives ordered role declarations and an optional
        explicit target for the first role. It resolves through the context owner;
        empty declarations return None only when target is omitted. Missing context/roles propagate before
        tab registration. cfg_resolution retains the supplied capture on refresh.
        """
        self._state = state
        self._registry = registry
        # The tab lifecycle previews pane drafts and tears them down on close via
        # one narrow port, without depending on the concrete sibling service.
        self._writeback = writeback
        self._cfg_resources = cfg_resources
        self._cfg_resolution = cfg_resolution
        self._captured_parameters = captured_parameters

    def get_snapshot(self, tab_id: str) -> TabSnapshot:
        """Build the immutable full render model for one tab (all live fields
        populated). The persist/restore form of ``TabSnapshot`` is produced
        elsewhere (codec / restore) with the live fields left empty."""
        tab = self._state.get_tab(tab_id)
        ctx = replace(self._state.session_env, parameters=tab.parameters)
        is_running = self._state.is_tab_running(tab_id)
        interaction = TabInteractionState(
            global_run_active=self._state.is_run_active() and not is_running,
            has_context=ctx.has_context(),
            has_active_context=ctx.is_active(),
            has_soc=ctx.has_soc(),
            is_running=is_running,
            is_analyzing=tab.is_analyzing,
            is_saving_data=tab.is_saving_data,
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
            ),
            analysis=AnalysisPaneSnapshot(
                params=tab.analysis.params,
                result=tab.analysis.result,
                figures=tab.analysis.plots,
                writeback_items=analysis_items,
                has_writeback_draft=tab.analysis.writeback_draft is not None,
                source_operation_id=tab.analysis.source_operation_id,
                result_params=tab.analysis.result_params,
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=tab.post_analysis.params,
                result=tab.post_analysis.result,
                figures=tab.post_analysis.plots,
                writeback_items=post_items,
                has_writeback_draft=tab.post_analysis.writeback_draft is not None,
                source_operation_id=tab.post_analysis.source_operation_id,
                result_params=tab.post_analysis.result_params,
            ),
            save=SavePaneSnapshot(comment=tab.save.comment),
            artifacts=self._state.get_artifact_snapshots(tab_id),
            parameters=tab.parameters,
        )

    def new_tab(
        self,
        adapter_name: str,
        from_dict: TabSnapshot | None = None,
        *,
        target: str | None = None,
    ) -> str:
        """Create a tab with roles and cfg sources captured once.

        target=None uses current GUI focus. A non-empty component name explicitly
        binds the adapter's first declared role; dependent roles still use their
        declared inference. It never changes GUI focus. Missing/wrong-kind roles
        raise RoleResolutionError before tab registration. An empty target or a
        target on a roleless adapter raises InvalidInputError.

        ``from_dict is None`` → a fresh tab with the adapter's default cfg.
        ``from_dict`` given (restore) → rebuild the tab from the snapshot's
        detached ``cfg_schema`` input memento. Path overrides are process-local and not
        persisted, so restore creates fresh panes.
        """
        if target is not None and not target.strip():
            raise InvalidInputError("target must be a non-empty component name")
        adapter = self._registry.create(adapter_name)
        tab_id = f"{_slug(adapter_name)}-{uuid.uuid4().hex[:8]}"
        logger.info(
            "new_tab: adapter=%r tab_id=%r restore=%s",
            adapter_name,
            tab_id,
            from_dict is not None,
        )
        parameters = self._captured_parameters(adapter.parameter_roles(), target)
        save_context = (
            parameters if parameters is not None else self._state.session_env.parameters
        )
        cfg = self._cfg_resources.create(
            tab_id,
            lambda: adapter.make_default_cfg(
                replace(self._state.session_env, parameters=parameters)
            ),
            resolution=lambda: self._cfg_resolution(parameters),
            initial=from_dict.cfg_schema if from_dict is not None else None,
        )
        self._state.add_tab(
            tab_id,
            Session(
                adapter_name=adapter_name,
                adapter=adapter,
                cfg=cfg,
                parameters=parameters,
                save_context=save_context,
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
        parameters = self._captured_parameters(adapter.parameter_roles(), None)
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
        """Prepare analysis without turning a committed result into a failure."""
        tab = self._state.get_tab(tab_id)
        if tab.run.result is None:
            raise RuntimeError("No run result available to build analyze params")
        if tab.adapter.capabilities.analysis is AnalysisMode.NONE:
            return AnalysisPreparation(has_params=False)
        try:
            instance = tab.adapter.get_analyze_params(
                tab.run.result,
                replace(self._state.session_env, parameters=tab.parameters),
            )
        except Exception as exc:
            logger.exception("Analysis preparation failed for tab %s", tab_id)
            return AnalysisPreparation(has_params=False, error=str(exc))
        self._state.update_tab_analyze_param_instance(tab_id, instance)
        return AnalysisPreparation(has_params=True)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._state.update_tab_analyze_param_instance(tab_id, instance)

    def initialize_tab_post_analyze_params(self, tab_id: str) -> object:
        """Build + store the post-analysis param instance once the primary analyze
        result exists. Fast-fails if
        there is no primary analyze result to seed from."""
        tab = self._state.get_tab(tab_id)
        if tab.analysis.result is None:
            raise RuntimeError(
                "No primary analyze result available to build post-analysis params"
            )
        instance = tab.adapter.get_post_analyze_params(
            tab.analysis.result,
            replace(self._state.session_env, parameters=tab.parameters),
        )
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)
        return instance

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)

    def get_tab_post_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).post_analysis.result
