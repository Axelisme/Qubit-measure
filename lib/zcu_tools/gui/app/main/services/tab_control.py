"""App-facing tab control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.cfg import CfgSchema
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import TabSnapshot
    from .tab import TabService
    from .workspace import WorkspaceService


class TabControlPort(Protocol):
    """App-facing tab resource surface for driving adapters."""

    def new_tab(self, adapter_name: str) -> str: ...
    def close_tab(self, tab_id: str) -> None: ...
    def set_active_tab(self, tab_id: str) -> None: ...
    def reorder_tabs(self, tab_ids: Sequence[str]) -> None: ...
    def get_active_tab_id(self) -> str | None: ...
    def get_running_tab_id(self) -> str | None: ...

    def has_tab(self, tab_id: str) -> bool: ...
    def list_tab_ids(self) -> list[str]: ...
    def get_tab_adapter_name(self, tab_id: str) -> str: ...
    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot: ...

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None: ...
    def reset_tab_cfg(self, tab_id: str) -> CfgSchema: ...
    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None: ...


class TabControlFacet:
    """Composite adapter over tab lifecycle, tab read model, and tab state."""

    def __init__(
        self,
        *,
        state: State,
        tab: TabService,
        workspace: WorkspaceService,
        bus: EventBus,
    ) -> None:
        self._state = state
        self._tab = tab
        self._workspace = workspace
        self._bus = bus

    def new_tab(self, adapter_name: str) -> str:
        return self._workspace.new_tab(adapter_name)

    def close_tab(self, tab_id: str) -> None:
        self._workspace.close_tab(tab_id)

    def set_active_tab(self, tab_id: str) -> None:
        self._workspace.set_active_tab(tab_id)

    def reorder_tabs(self, tab_ids: Sequence[str]) -> None:
        self._workspace.reorder_tabs(tab_ids)

    def get_active_tab_id(self) -> str | None:
        return self._state.active_tab_id

    def get_running_tab_id(self) -> str | None:
        return self._state.running_tab_id

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def list_tab_ids(self) -> list[str]:
        return self._state.list_tab_ids()

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._tab.get_tab_adapter_name(tab_id)

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab.get_snapshot(tab_id)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        self._tab.update_tab_cfg(tab_id, schema)

    def reset_tab_cfg(self, tab_id: str) -> CfgSchema:
        if self._state.running_tab_id == tab_id:
            raise RuntimeError(
                f"tab {tab_id!r} is currently running; cancel the run before "
                "resetting cfg"
            )
        adapter_name = self._tab.get_tab_adapter_name(tab_id)
        schema = self._tab.make_default_cfg(adapter_name)
        self._tab.update_tab_cfg(tab_id, schema)
        return schema

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._tab.update_tab_save_path_overrides(tab_id, data_path, image_path)
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
