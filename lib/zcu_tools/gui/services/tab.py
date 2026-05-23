from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from zcu_tools.gui.adapter import AnalyzeParam, CfgSchema
from zcu_tools.gui.event_bus import GuiEvent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.state import State


class TabService:
    """Encapsulates tab lifecycle, and per-tab operations like analyze/writeback/save."""

    def __init__(
        self,
        state: "State",
        registry: "Registry",
        bus: "EventBus",
    ) -> None:
        self._state = state
        self._registry = registry
        self._bus = bus

    def new_tab(self, adapter_name: str) -> str:
        adapter = self._registry.create(adapter_name)
        tab_id = str(uuid.uuid4())
        logger.info("new_tab: adapter=%r tab_id=%r", adapter_name, tab_id)
        self._state.add_tab(tab_id, adapter, self._state.exp_context)
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        self._state.remove_tab(tab_id)

    def get_tab_default_cfg(self, tab_id: str) -> CfgSchema:
        return self._state.get_tab(tab_id).default_cfg

    def get_tab_fresh_cfg(self, tab_id: str) -> CfgSchema:
        tab = self._state.get_tab(tab_id)
        return tab.adapter.make_default_cfg(self._state.exp_context)

    def get_tab_result(self, tab_id: str) -> object | None:
        return self._state.get_tab(tab_id).run_result

    def has_run_result(self, tab_id: str) -> bool:
        return self._state.get_tab(tab_id).run_result is not None

    def has_analyze_result(self, tab_id: str) -> bool:
        return self._state.get_tab(tab_id).analyze_result is not None

    def get_tab_figure(self, tab_id: str) -> Figure | None:
        return self._state.get_tab(tab_id).figure

    def get_tab_analyze_params(self, tab_id: str) -> list[AnalyzeParam]:
        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to build analyze params")
        return tab.adapter.get_analyze_params(tab.run_result, self._state.exp_context)

    def get_tab_save_paths(self, tab_id: str) -> Any:
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        return tab.adapter.make_save_paths(ctx)

    def analyze(self, tab_id: str, analyze_params: dict[str, object]) -> None:
        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to analyze")
        logger.info(
            "analyze: tab_id=%r analyze_params=%r", tab_id, list(analyze_params)
        )
        ctx = self._state.exp_context
        analyze_result = tab.adapter.analyze(tab.run_result, ctx, analyze_params)
        self._state.update_tab_analyze(tab_id, analyze_result, analyze_result.figure)

    def save_data(self, tab_id: str, data_path: str) -> None:
        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to save")
        logger.info("save_data: tab_id=%r path=%r", tab_id, data_path)
        ctx = self._state.exp_context
        tab.adapter.save(data_path, tab.run_result, ctx)

    def save_image(self, tab_id: str, image_path: str) -> None:
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")
        logger.info("save_image: tab_id=%r path=%r", tab_id, image_path)
        tab.figure.savefig(image_path)
