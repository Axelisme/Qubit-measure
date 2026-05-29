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
    from .ports import WritebackQueryPort


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
        writeback: "WritebackQueryPort",
    ) -> None:
        self._state = state
        # Read model depends only on State (aggregates) + a narrow writeback
        # query port — no concrete sibling app-service (ADR-0008 violation 2).
        self._writeback = writeback

    def get_snapshot(self, tab_id: str) -> TabViewSnapshot:
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
        return TabViewSnapshot(
            tab_id=tab_id,
            interaction=interaction,
            cfg_schema=tab.cfg_schema,
            capabilities=tab.adapter.capabilities,
            analyze_params=tab.analyze_param_instance,
            writeback_items=tuple(self._writeback.get_tab_writeback_items(tab_id)),
            save_paths=tab.effective_save_paths(ctx),
            figure=tab.figure,
        )
