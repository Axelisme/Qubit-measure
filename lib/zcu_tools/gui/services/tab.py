from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from zcu_tools.gui.adapter import CfgSchema, CfgSectionSpec, SavePaths

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.registry import Registry
    from zcu_tools.gui.state import State
from zcu_tools.gui.state import TabState


class TabService:
    """Encapsulates tab lifecycle, and per-tab operations like analyze/writeback/save."""

    def __init__(
        self,
        state: "State",
        registry: "Registry",
    ) -> None:
        self._state = state
        self._registry = registry

    def new_tab(self, adapter_name: str) -> str:
        adapter = self._registry.create(adapter_name)
        tab_id = str(uuid.uuid4())
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
