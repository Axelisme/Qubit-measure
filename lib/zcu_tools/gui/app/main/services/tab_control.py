"""App-facing tab control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

from zcu_tools.gui.app.main.catalog import ExperimentAccess
from zcu_tools.gui.expected_error import FailedPreconditionError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.cfg import CfgSchema
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .load import LoadTabResultOutcome
    from .ports import TabSnapshot
    from .tab import TabService
    from .workspace import WorkspaceService


class TabControlPort(Protocol):
    """App-facing tab resource surface for driving adapters."""

    def new_tab(self, adapter_name: str) -> str: ...
    def open_tab_from_file(
        self, adapter_name: str, data_path: str
    ) -> LoadTabResultOutcome:
        """Create/load/focus, cleaning up on failure; cfg backfill is best effort."""
        ...

    def close_tab(self, tab_id: str) -> None: ...
    def set_active_tab(self, tab_id: str) -> None: ...
    def reorder_tabs(self, tab_ids: Sequence[str]) -> None: ...
    def get_active_tab_id(self) -> str | None: ...
    def get_running_tab_id(self) -> str | None: ...

    def has_tab(self, tab_id: str) -> bool: ...
    def list_tab_ids(self) -> list[str]: ...
    def get_tab_adapter_name(self, tab_id: str) -> str: ...
    def analyze_param_definitions(
        self, adapter_name: str, *, stage: Literal["primary", "post"]
    ) -> list[dict[str, Any]]: ...
    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot: ...

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None: ...

    def reset_tab_cfg(self, tab_id: str) -> CfgSchema: ...


class TabControlFacet:
    """Composite adapter over tab lifecycle, tab read model, and tab state."""

    def __init__(
        self,
        *,
        state: State,
        tab: TabService,
        workspace: WorkspaceService,
        bus: EventBus,
        load_tab_result: Callable[[str, str], LoadTabResultOutcome],
        access: ExperimentAccess | None = None,
    ) -> None:
        self._state = state
        self._tab = tab
        self._workspace = workspace
        self._bus = bus
        self._load_tab_result = load_tab_result
        self._access = access if access is not None else ExperimentAccess()

    def new_tab(self, adapter_name: str) -> str:
        self._access.require_available()
        return self._workspace.new_tab(adapter_name)

    def open_tab_from_file(
        self, adapter_name: str, data_path: str
    ) -> LoadTabResultOutcome:
        """Compose new-tab loading; this does not roll back external side effects."""
        previous = self.get_active_tab_id()
        tab_id = self.new_tab(adapter_name)
        try:
            outcome = self._load_tab_result(tab_id, data_path)
            self.set_active_tab(tab_id)
        except Exception as open_error:
            # This boundary owns the new resource, not the loader's internals.
            try:
                self.close_tab(tab_id)
                if previous is not None:
                    self.set_active_tab(previous)
            except Exception as cleanup_error:
                raise FailedPreconditionError(
                    f"Opening tab {tab_id!r} failed: {open_error}; "
                    f"cleanup or focus restoration to {previous!r} also failed: "
                    f"{cleanup_error}; tab may remain open or focus may differ",
                    reason_code="cleanup_failed",
                ) from cleanup_error
            raise
        return outcome

    def close_tab(self, tab_id: str) -> None:
        self._access.require_available()
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

    def analyze_param_definitions(
        self, adapter_name: str, *, stage: Literal["primary", "post"]
    ) -> list[dict[str, Any]]:
        return self._tab.analyze_param_definitions(adapter_name, stage=stage)

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab.get_snapshot(tab_id)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        self._access.require_available()
        self._tab.update_tab_cfg(tab_id, schema)

    def reset_tab_cfg(self, tab_id: str) -> CfgSchema:
        self._access.require_available()
        if self._state.running_tab_id == tab_id:
            raise RuntimeError(
                f"tab {tab_id!r} is currently running; cancel the run before "
                "resetting cfg"
            )
        adapter_name = self._tab.get_tab_adapter_name(tab_id)
        schema = self._tab.make_default_cfg(adapter_name)
        self._tab.update_tab_cfg(tab_id, schema)
        return schema
