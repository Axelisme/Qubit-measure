from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .adapter import AbsExpAdapter, ExpContext

logger = logging.getLogger(__name__)


@dataclass
class TabState:
    adapter: AbsExpAdapter
    last_result: Any = None
    last_analyze_result: Any = None
    last_figure: Any = None
    last_cfg: Any = None


@dataclass(frozen=True)
class TabInteractionState:
    is_running: bool
    has_context: bool
    has_soc: bool
    has_run_result: bool
    has_analyze_result: bool


class State:
    """Passive state container; only updated by Controller."""

    def __init__(self, ctx: ExpContext) -> None:
        self.exp_context: ExpContext = ctx
        self.tabs: dict[str, TabState] = {}
        self.active_tab_id: Optional[str] = None
        self.is_running: bool = False
        self.has_startup_context: bool = False

    def set_context(self, ctx: ExpContext) -> None:
        self.exp_context = ctx

    def add_tab(self, tab_id: str, adapter: AbsExpAdapter, ctx: ExpContext) -> None:
        if tab_id in self.tabs:
            raise ValueError(f"tab_id {tab_id!r} already exists")
        logger.debug("add_tab: tab_id=%r adapter=%s", tab_id, type(adapter).__name__)
        tab = TabState(adapter=adapter)
        tab.last_cfg = adapter.make_default_cfg(ctx)
        self.tabs[tab_id] = tab

    def remove_tab(self, tab_id: str) -> None:
        logger.debug("remove_tab: tab_id=%r", tab_id)
        del self.tabs[tab_id]
        if self.active_tab_id == tab_id:
            self.active_tab_id = None

    def get_tab(self, tab_id: str) -> TabState:
        return self.tabs[tab_id]

    def set_active_tab(self, tab_id: str) -> None:
        if tab_id not in self.tabs:
            raise KeyError(f"tab_id {tab_id!r} not found")
        logger.debug("set_active_tab: tab_id=%r", tab_id)
        self.active_tab_id = tab_id

    def update_tab_result(self, tab_id: str, result: Any, cfg: Any) -> None:
        logger.debug(
            "update_tab_result: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        tab = self.tabs[tab_id]
        tab.last_result = result
        tab.last_cfg = cfg
        # invalidate stale analyze results from the previous run
        tab.last_analyze_result = None
        tab.last_figure = None

    def update_tab_analyze(self, tab_id: str, analyze_result: Any, figure: Any) -> None:
        logger.debug(
            "update_tab_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        tab = self.tabs[tab_id]
        tab.last_analyze_result = analyze_result
        tab.last_figure = figure

    def set_running(self, running: bool) -> None:
        logger.debug("set_running: %s", running)
        self.is_running = running
