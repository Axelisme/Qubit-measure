from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from zcu_tools.gui.app.measure.adapter.analyze_params import describe_analyze_params
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    SavePaneState,
    Session,
    TabInteractionState,
)
from zcu_tools.gui.cfg import CfgSchema

from .plot_lifecycle import release_retired_plots
from .ports import (
    AnalysisPaneSnapshot,
    PathResourceSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
    TabPathsSnapshot,
    TabSnapshot,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State

    from .ports import WritebackLifecyclePort


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


class TabService:
    """Tab aggregate read model, cfg state, and tab lifecycle primitives."""

    def __init__(
        self,
        state: State,
        registry: Registry,
        writeback: WritebackLifecyclePort,
    ) -> None:
        self._state = state
        self._registry = registry
        # The tab lifecycle previews pane drafts and tears them down on close via
        # one narrow port, without depending on the concrete sibling service.
        self._writeback = writeback

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
            is_saving_data=tab.is_saving_data,
            has_run_result=tab.has_run_result(),
            has_analyze_result=tab.has_analyze_result(),
            has_figure=tab.has_figure(),
            has_post_analyze_result=tab.has_post_analyze_result(),
        )
        data_path = PathResourceSnapshot(
            override=tab.save.data_path_override,
            path=tab.effective_data_path(ctx),
        )

        def image_paths(
            kind: ArtifactKind,
            pane: AnalysisPaneState[Any, Any] | PostAnalysisPaneState[Any, Any],
        ) -> Mapping[str, PathResourceSnapshot]:
            if pane.plots is None or pane.result is None:
                return MappingProxyType({})
            return MappingProxyType(
                {
                    name: PathResourceSnapshot(
                        override=pane.image_path_overrides.get(name),
                        path=tab.effective_image_path(ctx, ArtifactKey(kind, name)),
                    )
                    for name in pane.plots
                }
            )

        analysis_image_paths = image_paths(ArtifactKind.ANALYSIS, tab.analysis)
        post_image_paths = image_paths(ArtifactKind.POST_ANALYSIS, tab.post_analysis)

        # Pane-owned writeback items via opaque drafts.
        def _items_for_pane(pane) -> tuple:
            draft = pane.writeback_draft
            if draft is None:
                return ()
            return tuple(self._writeback.preview_draft(draft))

        analysis_items = _items_for_pane(tab.analysis)
        post_items = _items_for_pane(tab.post_analysis)
        return TabSnapshot(
            adapter_name=tab.adapter_name,
            cfg_schema=tab.cfg_schema,
            tab_id=tab_id,
            interaction=interaction,
            capabilities=tab.adapter.capabilities,
            run=RunPaneSnapshot(
                result=tab.run.result,
                source_path=tab.run.source_path,
            ),
            analysis=AnalysisPaneSnapshot(
                params=tab.analysis.params,
                result=tab.analysis.result,
                figures=tab.analysis.plots,
                writeback_items=analysis_items,
                image_paths=analysis_image_paths,
                has_writeback_draft=tab.analysis.writeback_draft is not None,
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=tab.post_analysis.params,
                result=tab.post_analysis.result,
                figures=tab.post_analysis.plots,
                writeback_items=post_items,
                image_paths=post_image_paths,
                has_writeback_draft=tab.post_analysis.writeback_draft is not None,
            ),
            save=SavePaneSnapshot(data_path=data_path, comment=tab.save.comment),
            paths=TabPathsSnapshot(
                data=data_path,
                analysis_images=analysis_image_paths,
                post_analysis_images=post_image_paths,
            ),
            artifacts=self._state.get_artifact_snapshots(tab_id),
        )

    def new_tab(self, adapter_name: str, from_dict: TabSnapshot | None = None) -> str:
        """Single tab-creation entry.

        ``from_dict is None`` → a fresh tab with the adapter's default cfg.
        ``from_dict`` given (restore) → rebuild the tab from the snapshot's
        live ``cfg_schema``. Path overrides are process-local and not
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
        if from_dict is None:
            cfg_schema = adapter.make_default_cfg(self._state.session_env)
        else:
            cfg_schema = from_dict.cfg_schema
        self._state.add_tab(
            tab_id,
            Session(
                adapter_name=adapter_name,
                adapter=adapter,
                cfg_schema=cfg_schema,
            ),
        )
        return tab_id

    def make_default_cfg(self, adapter_name: str) -> CfgSchema:
        """The adapter's default cfg under the current context — the base schema
        WorkspaceService needs to decode a persisted raw cfg into a live one."""
        return self._registry.create(adapter_name).make_default_cfg(
            self._state.session_env
        )

    def list_adapter_names(self) -> list[str]:
        return self._registry.list_names()

    def adapter_guide(self, adapter_name: str) -> dict:
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
        retired = self._state.remove_tab(tab_id)
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("closed-tab draft teardown failed")
        release_retired_plots(retired)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        """Store an explicit cfg replacement and advance its resource version.

        Active editor changes publish directly through their composition-injected
        State sink; the viewer does not commit model changes.
        """
        self._state.update_tab_cfg_schema(tab_id, schema)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).analysis.result

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._state.get_tab(tab_id).adapter_name

    def initialize_tab_analyze_params(self, tab_id: str) -> object:
        tab = self._state.get_tab(tab_id)
        if tab.run.result is None:
            raise RuntimeError("No run result available to build analyze params")
        instance = tab.adapter.get_analyze_params(
            tab.run.result, self._state.session_env
        )
        self._state.update_tab_analyze_param_instance(tab_id, instance)
        return instance

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._state.update_tab_analyze_param_instance(tab_id, instance)

    def initialize_tab_post_analyze_params(self, tab_id: str) -> object:
        """Build + store the post-analysis param instance once the primary analyze
        result exists (mirrors ``initialize_tab_analyze_params``). Fast-fails if
        there is no primary analyze result to seed from."""
        tab = self._state.get_tab(tab_id)
        if tab.analysis.result is None:
            raise RuntimeError(
                "No primary analyze result available to build post-analysis params"
            )
        instance = tab.adapter.get_post_analyze_params(
            tab.analysis.result, self._state.session_env
        )
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)
        return instance

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)

    def get_tab_post_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).post_analysis.result

    def get_tab_data_path(self, tab_id: str) -> str | None:
        return self._state.get_tab(tab_id).effective_data_path(self._state.session_env)

    def get_tab_image_path(self, tab_id: str, key: ArtifactKey) -> str | None:
        return self._state.get_tab(tab_id).effective_image_path(
            self._state.session_env, key
        )

    def update_tab_data_path_override(self, tab_id: str, data_path: str | None) -> None:
        self._state.update_tab_data_path_override(tab_id, data_path)

    def update_tab_image_path_override(
        self, tab_id: str, key: ArtifactKey, image_path: str | None
    ) -> None:
        self._state.update_tab_image_path_override(tab_id, key, image_path)
