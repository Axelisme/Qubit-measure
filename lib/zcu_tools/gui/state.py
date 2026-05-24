from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import Generic, TypeVar

from .adapter import (
    AbsExpAdapter,
    AnalyzeResultWithFigure,
    CfgSchema,
    ExpContext,
    SavePaths,
    T_AnalyzeParams,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

T_Result = TypeVar("T_Result")
T_AnalyzeResult = TypeVar("T_AnalyzeResult", bound=AnalyzeResultWithFigure)


@dataclass
class TabState(Generic[T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    adapter: AbsExpAdapter[T_Result, T_AnalyzeResult, T_AnalyzeParams]
    cfg_schema: CfgSchema
    run_result: Optional[T_Result] = None
    analyze_result: Optional[T_AnalyzeResult] = None
    figure: Optional["Figure"] = None
    analyze_param_instance: Optional[T_AnalyzeParams] = None
    suggested_save_paths: Optional[SavePaths] = None
    save_path_overrides: Optional[SavePaths] = None
    applied_writeback_keys: set[str] = field(default_factory=set)
    is_running: bool = False
    is_analyzing: bool = False
    is_saving_data: bool = False


@dataclass(frozen=True)
class TabInteractionState:
    global_run_active: bool
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
        self.tabs: dict[str, TabState[Any, Any, Any]] = {}
        self.active_tab_id: Optional[str] = None
        self.running_tab_id: Optional[str] = None
        self.has_startup_context: bool = False

    def set_context(self, ctx: ExpContext) -> None:
        self.exp_context = ctx

    def add_tab(
        self, tab_id: str, adapter: AbsExpAdapter[Any, Any, Any], ctx: ExpContext
    ) -> None:
        if tab_id in self.tabs:
            raise ValueError(f"tab_id {tab_id!r} already exists")
        logger.debug("add_tab: tab_id=%r adapter=%s", tab_id, type(adapter).__name__)
        tab = TabState(adapter=adapter, cfg_schema=adapter.make_default_cfg(ctx))
        self.tabs[tab_id] = tab

    def remove_tab(self, tab_id: str) -> None:
        logger.debug("remove_tab: tab_id=%r", tab_id)
        if self.is_tab_busy(tab_id):
            raise RuntimeError(f"Cannot close busy tab {tab_id!r}")
        del self.tabs[tab_id]
        if self.active_tab_id == tab_id:
            self.active_tab_id = None
        if self.running_tab_id == tab_id:
            self.running_tab_id = None

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
        tab.analyze_param_instance = None
        tab.suggested_save_paths = None
        # invalidate stale analyze results from the previous run
        tab.analyze_result = None
        tab.figure = None
        tab.applied_writeback_keys.clear()

    def update_tab_analyze(
        self, tab_id: str, analyze_result: object, figure: Optional["Figure"]
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

    def update_tab_cfg_schema(self, tab_id: str, schema: CfgSchema) -> None:
        logger.debug("update_tab_cfg_schema: tab_id=%r", tab_id)
        self.tabs[tab_id].cfg_schema = schema

    def update_tab_analyze_params(self, tab_id: str, instance: object) -> None:
        logger.debug(
            "update_tab_analyze_params: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].analyze_param_instance = instance

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        logger.debug(
            "update_tab_analyze_param_instance: tab_id=%r instance_type=%s",
            tab_id,
            type(instance).__name__,
        )
        self.tabs[tab_id].analyze_param_instance = instance

    def update_tab_suggested_save_paths(
        self, tab_id: str, paths: Optional[SavePaths]
    ) -> None:
        logger.debug("update_tab_suggested_save_paths: tab_id=%r", tab_id)
        self.tabs[tab_id].suggested_save_paths = paths

    def update_tab_save_path_overrides(
        self,
        tab_id: str,
        paths: Optional[SavePaths],
    ) -> None:
        logger.debug("update_tab_save_path_overrides: tab_id=%r", tab_id)
        self.tabs[tab_id].save_path_overrides = paths

    def clear_tab_save_path_overrides(
        self,
        tab_id: str,
    ) -> None:
        logger.debug("clear_tab_save_path_overrides: tab_id=%r", tab_id)
        self.tabs[tab_id].save_path_overrides = None

    def get_effective_save_paths(self, tab_id: str) -> Optional[SavePaths]:
        tab = self.tabs[tab_id]
        return tab.save_path_overrides or tab.suggested_save_paths

    def set_tab_running(self, tab_id: str, running: bool) -> None:
        logger.debug("set_tab_running: tab_id=%r running=%s", tab_id, running)
        tab = self.tabs[tab_id]
        if (
            running
            and self.running_tab_id is not None
            and self.running_tab_id != tab_id
        ):
            raise RuntimeError(
                f"Cannot mark tab {tab_id!r} running while "
                f"{self.running_tab_id!r} is already running"
            )
        tab.is_running = running
        if running:
            self.running_tab_id = tab_id
        elif self.running_tab_id == tab_id:
            self.running_tab_id = None

    def set_tab_analyzing(self, tab_id: str, analyzing: bool) -> None:
        logger.debug("set_tab_analyzing: tab_id=%r analyzing=%s", tab_id, analyzing)
        self.tabs[tab_id].is_analyzing = analyzing

    def set_tab_saving_data(self, tab_id: str, saving_data: bool) -> None:
        logger.debug(
            "set_tab_saving_data: tab_id=%r saving_data=%s", tab_id, saving_data
        )
        self.tabs[tab_id].is_saving_data = saving_data

    def is_run_active(self) -> bool:
        return self.running_tab_id is not None

    def is_tab_running(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_running

    def is_tab_analyzing(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_analyzing

    def is_tab_saving_data(self, tab_id: str) -> bool:
        return self.tabs[tab_id].is_saving_data

    def is_tab_busy(self, tab_id: str) -> bool:
        tab = self.tabs[tab_id]
        return tab.is_running or tab.is_analyzing or tab.is_saving_data
