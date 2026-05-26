from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    CfgSchema,
    SavePaths,
    WritebackItem,
)
from zcu_tools.gui.state import State, TabInteractionState

if TYPE_CHECKING:
    from .context import ContextService
    from .tab import TabService
    from .writeback import WritebackService


@dataclass(frozen=True)
class TabViewSnapshot:
    tab_id: str
    interaction: TabInteractionState
    cfg_schema: CfgSchema
    capabilities: AdapterCapabilities
    analyze_params: object | None
    writeback_items: tuple[WritebackItem, ...]
    save_paths: SavePaths | None
    figure: Figure | None


class TabViewService:
    """Build immutable read models for rendering one experiment tab."""

    def __init__(
        self,
        state: State,
        tabs: "TabService",
        writeback: "WritebackService",
        context: "ContextService",
    ) -> None:
        self._state = state
        self._tabs = tabs
        self._writeback = writeback
        self._context = context

    def get_snapshot(self, tab_id: str) -> TabViewSnapshot:
        tab = self._state.get_tab(tab_id)
        interaction = TabInteractionState(
            global_run_active=self._state.is_run_active() and not tab.is_running,
            is_running=tab.is_running,
            is_analyzing=tab.is_analyzing,
            is_saving_data=tab.is_saving_data,
            has_context=self._context.has_context(),
            has_active_context=self._context.is_active_context(),
            has_soc=self._state.exp_context.soc is not None,
            has_run_result=tab.run_result is not None,
            has_analyze_result=tab.analyze_result is not None,
            has_figure=tab.figure is not None,
        )
        return TabViewSnapshot(
            tab_id=tab_id,
            interaction=interaction,
            cfg_schema=tab.cfg_schema,
            capabilities=tab.adapter.capabilities,
            analyze_params=tab.analyze_param_instance,
            writeback_items=tuple(self._writeback.get_tab_writeback_items(tab_id)),
            save_paths=self._tabs.get_tab_save_paths(tab_id),
            figure=tab.figure,
        )
