from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from matplotlib.figure import Figure

from zcu_tools.gui.adapter import AnalyzeParam, CfgSchema, SavePaths

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
        self.refresh_tab_save_paths(tab_id)
        return tab_id

    def list_adapter_names(self) -> list[str]:
        return self._registry.list_names()

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        self._state.remove_tab(tab_id)

    def get_tab_default_cfg(self, tab_id: str) -> CfgSchema:
        return self._state.get_tab(tab_id).cfg_schema

    def get_tab_fresh_cfg(self, tab_id: str) -> CfgSchema:
        tab = self._state.get_tab(tab_id)
        return tab.adapter.make_default_cfg(self._state.exp_context)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        self._state.update_tab_cfg_schema(tab_id, schema)

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
        params = tab.adapter.get_analyze_params(tab.run_result, self._state.exp_context)
        if not tab.analyze_params or [param.key for param in tab.analyze_params] != [
            param.key for param in params
        ]:
            self._state.update_tab_analyze_params(tab_id, params)
        elif not tab.analyze_param_values:
            self._state.update_tab_analyze_params(
                tab_id,
                params,
                {param.key: param.default for param in params},
            )
        return params

    def get_tab_analyze_param_values(self, tab_id: str) -> dict[str, object]:
        tab = self._state.get_tab(tab_id)
        return dict(tab.analyze_param_values)

    def update_tab_analyze_param_values(
        self, tab_id: str, values: dict[str, object]
    ) -> None:
        self._state.update_tab_analyze_param_values(tab_id, values)

    def get_tab_save_paths(self, tab_id: str) -> Optional[SavePaths]:
        self.refresh_tab_save_paths(tab_id)
        return self._state.get_effective_save_paths(tab_id)

    def refresh_tab_save_paths(self, tab_id: str) -> None:
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        if not ctx.database_path or not ctx.result_dir or not ctx.active_label:
            self._state.update_tab_suggested_save_paths(tab_id, None)
            return
        paths = tab.adapter.make_save_paths(ctx)
        self._state.update_tab_suggested_save_paths(tab_id, paths)

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._state.update_tab_save_path_overrides(
            tab_id, data_path=data_path, image_path=image_path
        )

    def save_image(self, tab_id: str, image_path: str) -> None:
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")
        logger.info("save_image: tab_id=%r path=%r", tab_id, image_path)
        tab.figure.savefig(image_path)
