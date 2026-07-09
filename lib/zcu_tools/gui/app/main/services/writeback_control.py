"""App-facing writeback control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import WritebackItem
    from zcu_tools.gui.app.main.state import State

    from .guard import GuardService
    from .writeback import WritebackService


class WritebackControlPort(Protocol):
    """App-facing writeback surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]: ...

    def set_writeback_item(
        self, tab_id: str, session_id: str, **changes: Any
    ) -> dict[str, object]: ...

    def apply_writeback(self, tab_id: str) -> dict[str, Any]: ...

    def get_context_version(self) -> int: ...


class WritebackControlFacet:
    """Composite adapter over writeback guards, draft service, and version reads."""

    def __init__(
        self,
        *,
        state: State,
        guard: GuardService,
        writeback: WritebackService,
        resource_versions: Callable[[], Mapping[str, int]],
    ) -> None:
        self._state = state
        self._guard = guard
        self._writeback = writeback
        self._resource_versions = resource_versions

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        return list(self._writeback.get_tab_writeback_items(tab_id))

    def set_writeback_item(
        self, tab_id: str, session_id: str, **changes: Any
    ) -> dict[str, object]:
        self._guard.acquire_writeback_permit(tab_id)
        return self._writeback.set_item_field(tab_id, session_id, **changes)

    def apply_writeback(self, tab_id: str) -> dict[str, Any]:
        permit = self._guard.acquire_writeback_permit(tab_id)
        return self._writeback.apply_tab_writeback(permit)

    def get_context_version(self) -> int:
        return self._resource_versions().get("context", 0)
