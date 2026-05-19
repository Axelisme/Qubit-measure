from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .adapter import AbsExpAdapter, ExpContext


@dataclass
class TabState:
    adapter: AbsExpAdapter
    last_result: Any = None
    last_analyze_result: Any = None
    last_figure: Any = None
    last_cfg: Any = None


class State:
    """Passive state container; only updated by Controller."""

    def __init__(self, ctx: ExpContext) -> None:
        self.exp_context: ExpContext = ctx
        self.tabs: dict[str, TabState] = {}
        self.active_tab_id: Optional[str] = None
        self.is_running: bool = False

    def set_context(self, ctx: ExpContext) -> None:
        self.exp_context = ctx

    def add_tab(self, tab_id: str, adapter: AbsExpAdapter) -> None:
        if tab_id in self.tabs:
            raise ValueError(f"tab_id {tab_id!r} already exists")
        self.tabs[tab_id] = TabState(adapter=adapter)

    def remove_tab(self, tab_id: str) -> None:
        del self.tabs[tab_id]
        if self.active_tab_id == tab_id:
            self.active_tab_id = None

    def get_tab(self, tab_id: str) -> TabState:
        return self.tabs[tab_id]

    def set_active_tab(self, tab_id: str) -> None:
        if tab_id not in self.tabs:
            raise KeyError(f"tab_id {tab_id!r} not found")
        self.active_tab_id = tab_id

    def update_tab_result(self, tab_id: str, result: Any, cfg: Any) -> None:
        tab = self.tabs[tab_id]
        tab.last_result = result
        tab.last_cfg = cfg

    def update_tab_analyze(self, tab_id: str, analyze_result: Any, figure: Any) -> None:
        tab = self.tabs[tab_id]
        tab.last_analyze_result = analyze_result
        tab.last_figure = figure

    def set_running(self, running: bool) -> None:
        self.is_running = running
