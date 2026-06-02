from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from matplotlib.figure import Figure

from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    CfgSchema,
    CfgSectionSpec,
    SavePaths,
    WritebackItem,
)
from zcu_tools.gui.state import TabInteractionState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.state import State

    from .ports import WritebackQueryPort
from zcu_tools.gui.state import TabState


@dataclass(frozen=True)
class TabSnapshot:
    """Immutable full-state snapshot of one tab.

    Single type for two consumers (formerly ``TabViewSnapshot`` + ``PersistedTab``):

    - **render** (``TabService.get_snapshot``): every field populated, handed to
      the View to draw one tab.
    - **restore** (``TabService.new_tab(from_dict=...)``) and **persist**
      (``PersistedSession.tabs``): only the serializable head three fields carry
      meaning; the live fields below are ``None`` / empty.

    ``cfg_schema`` is always the *live* ``CfgSchema`` (resolved EvalValue), which
    the render path uses directly. The disk codec
    (``SessionPersistenceService``) converts ``cfg_schema`` ↔ ``cfg_raw`` at the
    file boundary, so the persisted form never leaks into the tab snapshot.
    """

    adapter_name: str
    cfg_schema: CfgSchema
    # The user's explicit override only (None = follow the adapter suggestion).
    # This is the serializable save-path state — persist/restore round-trip it so
    # a reload never pins an adapter-derived path.
    save_paths_override: Optional[SavePaths]
    # Live render-only fields; None / empty on the persist + restore paths.
    tab_id: Optional[str] = None
    interaction: Optional[TabInteractionState] = None
    capabilities: Optional[AdapterCapabilities] = None
    analyze_params: object | None = None
    writeback_items: tuple[WritebackItem, ...] = ()
    figure: Optional[Figure] = None
    # Render-computed effective paths (override, else adapter suggestion from
    # ctx). The View shows this; it is *not* persisted (derivable on restore).
    save_paths: Optional[SavePaths] = None


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
        state: "State",
        registry: "Registry",
        writeback: "WritebackQueryPort",
    ) -> None:
        self._state = state
        self._registry = registry
        # Read model composes writeback proposals via a narrow query port — no
        # concrete sibling app-service dependency (ADR-0008 violation 2).
        self._writeback = writeback

    def get_snapshot(self, tab_id: str) -> "TabSnapshot":
        """Build the immutable full render model for one tab (all live fields
        populated). The persist/restore form of ``TabSnapshot`` is produced
        elsewhere (codec / restore) with the live fields left empty."""
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        interaction = TabInteractionState(
            # cross-cutting facts read directly off State's aggregates (no
            # app-service dependency — ADR-0008 violation 2 / ADR-0007 Query).
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
        )
        return TabSnapshot(
            adapter_name=tab.adapter_name,
            cfg_schema=tab.cfg_schema,
            save_paths_override=tab.save_path_overrides,
            tab_id=tab_id,
            interaction=interaction,
            capabilities=tab.adapter.capabilities,
            analyze_params=tab.analyze_param_instance,
            writeback_items=tuple(self._writeback.get_tab_writeback_items(tab_id)),
            figure=tab.figure,
            save_paths=tab.effective_save_paths(ctx),
        )

    def new_tab(self, adapter_name: str) -> str:
        adapter = self._registry.create(adapter_name)
        tab_id = f"{_slug(adapter_name)}-{uuid.uuid4().hex[:8]}"
        logger.info("new_tab: adapter=%r tab_id=%r", adapter_name, tab_id)
        self._state.add_tab(
            tab_id,
            TabState(
                adapter_name=adapter_name,
                adapter=adapter,
                cfg_schema=adapter.make_default_cfg(self._state.exp_context),
            ),
        )
        return tab_id

    def restore_tab(self, adapter_name: str) -> str:
        """Create a tab for restore flow using the same lifecycle as new_tab."""
        return self.new_tab(adapter_name)

    def list_adapter_names(self) -> list[str]:
        return self._registry.list_names()

    def adapter_cfg_spec(self, adapter_name: str) -> "CfgSectionSpec":
        """Static cfg spec of an adapter — no tab/context needed."""
        return self._registry.create(adapter_name).cfg_spec()

    def adapter_analyze_params(self, adapter_name: str) -> list[dict]:
        """Static analyze-params field spec, or [] when analysis unsupported."""
        from zcu_tools.gui.adapter import describe_analyze_params

        adapter = self._registry.create(adapter_name)
        if not adapter.capabilities.supports_analysis:
            return []
        return describe_analyze_params(adapter.analyze_params_cls())

    def adapter_guide(self, adapter_name: str) -> dict:
        """Static human-facing orientation guide of an adapter (five fields)."""
        import dataclasses

        return dataclasses.asdict(self._registry.create(adapter_name).guide())

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        self._state.remove_tab(tab_id)

    def get_tab_default_cfg(self, tab_id: str) -> CfgSchema:
        return self._state.get_tab(tab_id).cfg_schema

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

    def get_tab_save_paths(self, tab_id: str) -> Optional[SavePaths]:
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
