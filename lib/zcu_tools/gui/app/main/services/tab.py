from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from zcu_tools.gui.app.main.adapter import (
    AnalysisMode,
    CfgSchema,
    CfgSectionSpec,
    SavePaths,
)
from zcu_tools.gui.app.main.state import Session, TabInteractionState

from .ports import TabSnapshot

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.state import State

    from .ports import WritebackQueryPort


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
    """Encapsulates tab lifecycle, and per-tab operations like analyze/writeback/save."""

    def __init__(
        self,
        state: State,
        registry: Registry,
        writeback: WritebackQueryPort,
    ) -> None:
        self._state = state
        self._registry = registry
        # Read model composes writeback proposals via a narrow query port — no
        # concrete sibling app-service dependency (ADR-0005 violation 2).
        self._writeback = writeback

    def get_snapshot(self, tab_id: str) -> TabSnapshot:
        """Build the immutable full render model for one tab (all live fields
        populated). The persist/restore form of ``TabSnapshot`` is produced
        elsewhere (codec / restore) with the live fields left empty."""
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        interaction = TabInteractionState(
            # cross-cutting facts read directly off State's aggregates (no
            # app-service dependency — ADR-0005 violation 2 / ADR-0004 Query).
            global_run_active=self._state.is_run_active() and not tab.is_running,
            has_context=ctx.has_context(),
            has_active_context=ctx.is_active(),
            has_soc=ctx.has_soc(),
            # tab-intrinsic facts are the tab aggregate's own predicates
            is_running=tab.is_running,
            is_analyzing=tab.is_analyzing,
            is_saving_data=tab.is_saving_data,
            has_run_result=tab.has_run_result(),
            has_analyze_result=tab.has_analyze_result(),
            has_figure=tab.has_figure(),
            has_post_analyze_result=tab.has_post_analyze_result(),
        )
        return TabSnapshot(
            adapter_name=tab.adapter_name,
            cfg_schema=tab.cfg_schema,
            save_paths_override=tab.save_path_overrides,
            tab_id=tab_id,
            interaction=interaction,
            capabilities=tab.adapter.capabilities,
            analyze_params=tab.analyze_param_instance,
            post_analyze_params=tab.post_analyze_param_instance,
            post_figure=tab.post_figure,
            writeback_items=tuple(self._writeback.get_tab_writeback_items(tab_id)),
            figure=tab.figure,
            save_paths=tab.effective_save_paths(ctx),
        )

    def new_tab(self, adapter_name: str, from_dict: TabSnapshot | None = None) -> str:
        """Single tab-creation entry.

        ``from_dict is None`` → a fresh tab with the adapter's default cfg.
        ``from_dict`` given (restore) → rebuild the tab in one step from the
        snapshot's live ``cfg_schema`` + ``save_paths_override`` — no
        build-default-then-overwrite. The caller (WorkspaceService) has already
        turned the on-disk raw cfg into a live ``cfg_schema`` (it holds both the
        adapter default as base and the codec), so this method never touches the
        persistence codec.
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
            cfg_schema = adapter.make_default_cfg(self._state.exp_context)
            save_path_overrides = None
        else:
            cfg_schema = from_dict.cfg_schema
            save_path_overrides = from_dict.save_paths_override
        self._state.add_tab(
            tab_id,
            Session(
                adapter_name=adapter_name,
                adapter=adapter,
                cfg_schema=cfg_schema,
                save_path_overrides=save_path_overrides,
            ),
        )
        return tab_id

    def make_default_cfg(self, adapter_name: str) -> CfgSchema:
        """The adapter's default cfg under the current context — the base schema
        WorkspaceService needs to decode a persisted raw cfg into a live one."""
        return self._registry.create(adapter_name).make_default_cfg(
            self._state.exp_context
        )

    def list_adapter_names(self) -> list[str]:
        return self._registry.list_names()

    def adapter_cfg_spec(self, adapter_name: str) -> CfgSectionSpec:
        """Static cfg spec of an adapter — no tab/context needed."""
        return self._registry.create(adapter_name).cfg_spec()

    def adapter_analyze_params(self, adapter_name: str) -> list[dict]:
        """Static analyze-params field spec, or [] when analysis unsupported."""
        from zcu_tools.gui.app.main.adapter import describe_analyze_params

        adapter = self._registry.create(adapter_name)
        if adapter.capabilities.analysis is AnalysisMode.NONE:
            return []
        return describe_analyze_params(adapter.analyze_params_cls())

    def adapter_post_analyze_params(self, adapter_name: str) -> list[dict]:
        """Static post-analysis param field spec, or [] when post-analysis is not
        supported. Mirrors ``adapter_analyze_params`` for the second layer."""
        from zcu_tools.gui.app.main.adapter import describe_analyze_params

        adapter = self._registry.create(adapter_name)
        if not adapter.capabilities.post_analysis:
            return []
        return describe_analyze_params(adapter.post_analyze_params_cls())

    def adapter_guide(self, adapter_name: str) -> dict:
        """Static human-facing orientation guide of an adapter (five fields)."""
        import dataclasses

        return dataclasses.asdict(self._registry.create(adapter_name).guide())

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        self._state.remove_tab(tab_id)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        """Commit boundary: store the latest draft as the committed cfg.

        Idempotent. Called from ``Controller.update_tab_cfg`` whenever the
        tab's CfgFormWidget reports a change.
        """
        self._state.update_tab_cfg_schema(tab_id, schema)

    def get_tab_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).run_result

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).analyze_result

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._state.get_tab(tab_id).adapter_name

    def initialize_tab_analyze_params(self, tab_id: str) -> object:
        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to build analyze params")
        instance = tab.adapter.get_analyze_params(
            tab.run_result, self._state.exp_context
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
        if tab.analyze_result is None:
            raise RuntimeError(
                "No primary analyze result available to build post-analysis params"
            )
        instance = tab.adapter.get_post_analyze_params(
            tab.analyze_result, self._state.exp_context
        )
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)
        return instance

    def update_tab_post_analyze_param_instance(
        self, tab_id: str, instance: object
    ) -> None:
        self._state.update_tab_post_analyze_param_instance(tab_id, instance)

    def get_tab_post_analyze_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).post_analyze_result

    def get_tab_save_paths(self, tab_id: str) -> SavePaths | None:
        tab = self._state.get_tab(tab_id)
        return tab.effective_save_paths(self._state.exp_context)

    def update_tab_save_path_overrides(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        if not data_path and not image_path:
            self._state.clear_tab_save_path_overrides(tab_id)
            return
        if not data_path or not image_path:
            raise RuntimeError("Save path overrides require both data and image paths")
        self._state.update_tab_save_path_overrides(
            tab_id, SavePaths(data_path=data_path, image_path=image_path)
        )
