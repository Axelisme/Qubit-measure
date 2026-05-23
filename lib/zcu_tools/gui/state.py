from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from matplotlib.figure import Figure

from .adapter import AbsExpAdapter, CfgSchema, ExpContext

logger = logging.getLogger(__name__)


@dataclass
class TabState:
    adapter: AbsExpAdapter[Any, Any]
    default_cfg: CfgSchema
    run_result: Optional[object] = None
    analyze_result: Optional[object] = None
    figure: Optional[Figure] = None
    applied_writeback_keys: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TabInteractionState:
    is_running: bool
    is_analyzing: bool
    is_saving_data: bool
    has_context: bool
    has_soc: bool
    has_run_result: bool
    has_analyze_result: bool


class State:
    """Passive GUI state container shared by Controller and domain services."""

    def __init__(self, ctx: ExpContext) -> None:
        self.exp_context: ExpContext = ctx
        self.tabs: dict[str, TabState] = {}
        self.active_tab_id: Optional[str] = None
        self.is_running: bool = False
        self.is_analyzing: bool = False
        self.is_saving_data: bool = False
        self.has_startup_context: bool = False

    def set_context(self, ctx: ExpContext) -> None:
        self.exp_context = ctx

    def add_tab(
        self, tab_id: str, adapter: AbsExpAdapter[Any, Any], ctx: ExpContext
    ) -> None:
        if tab_id in self.tabs:
            raise ValueError(f"tab_id {tab_id!r} already exists")
        logger.debug("add_tab: tab_id=%r adapter=%s", tab_id, type(adapter).__name__)
        tab = TabState(adapter=adapter, default_cfg=adapter.make_default_cfg(ctx))
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

    def update_tab_result(self, tab_id: str, result: object) -> None:
        logger.debug(
            "update_tab_result: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        tab = self.tabs[tab_id]
        tab.run_result = result
        # invalidate stale analyze results from the previous run
        tab.analyze_result = None
        tab.figure = None
        tab.applied_writeback_keys.clear()

    def update_tab_analyze(
        self, tab_id: str, analyze_result: object, figure: Optional[Figure]
    ) -> None:
        logger.debug(
            "update_tab_analyze: tab_id=%r figure=%s",
            tab_id,
            "yes" if figure is not None else "none",
        )
        tab = self.tabs[tab_id]
        tab.analyze_result = analyze_result
        tab.figure = figure
        tab.applied_writeback_keys.clear()

    def set_running(self, running: bool) -> None:
        logger.debug("set_running: %s", running)
        self.is_running = running

    def set_analyzing(self, analyzing: bool) -> None:
        logger.debug("set_analyzing: %s", analyzing)
        self.is_analyzing = analyzing

    def set_saving_data(self, saving_data: bool) -> None:
        logger.debug("set_saving_data: %s", saving_data)
        self.is_saving_data = saving_data

    @property
    def has_active_long_task(self) -> bool:
        return self.is_running or self.is_analyzing or self.is_saving_data
