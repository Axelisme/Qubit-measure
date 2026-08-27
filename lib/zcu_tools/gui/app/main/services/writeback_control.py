"""App-facing writeback control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from zcu_tools.gui.expected_error import FailedPreconditionError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import WritebackItem
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.cfg.binding import CfgDraft

    from .guard import GuardService
    from .writeback import WritebackService


class WritebackControlPort(Protocol):
    """App-facing writeback surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]: ...

    def get_writeback_item_draft(self, tab_id: str, session_id: str) -> CfgDraft: ...

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

    def get_writeback_item_draft(self, tab_id: str, session_id: str) -> CfgDraft:
        return self._writeback.get_tab_writeback_item_draft(tab_id, session_id)

    def set_writeback_item(
        self, tab_id: str, session_id: str, **changes: Any
    ) -> dict[str, object]:
        self._guard.acquire_writeback_permit(tab_id)
        self._require_tab_idle(tab_id)
        return self._writeback.set_item_field(tab_id, session_id, **changes)

    def apply_writeback(self, tab_id: str) -> dict[str, Any]:
        permit = self._guard.acquire_writeback_permit(tab_id)
        self._require_tab_idle(tab_id)
        return self._writeback.apply_tab_writeback(permit)

    def _require_tab_idle(self, tab_id: str) -> None:
        """Keep edits/apply out of a same-tab lifecycle transition."""
        busy = getattr(self._state, "is_tab_busy", None)
        if callable(busy) and busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def get_context_version(self) -> int:
        return self._resource_versions().get("context", 0)
