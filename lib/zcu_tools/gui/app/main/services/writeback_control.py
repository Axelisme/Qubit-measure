"""App-facing writeback control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

WritebackPane: TypeAlias = Literal["analysis", "post_analysis"]

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.cfg.binding import CfgDraft
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .guard import GuardService
    from .writeback import WritebackService


class WritebackControlPort(Protocol):
    """App-facing writeback surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def get_writeback_item_draft_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str
    ) -> CfgDraft: ...

    def set_writeback_item_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str, **changes: Any
    ) -> dict[str, object]: ...

    def apply_writeback_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, Any]: ...

    def get_writeback_summaries_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, tuple[str | None, str | None]]: ...

    def get_writeback_applied_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, bool]: ...

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
        bus: EventBus,
    ) -> None:
        self._state = state
        self._guard = guard
        self._writeback = writeback
        self._resource_versions = resource_versions
        self._bus = bus

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def _draft_for_pane(self, tab_id: str, pane: WritebackPane):
        tab = self._state.get_tab(tab_id)
        if pane == "analysis":
            draft = tab.analysis.writeback_draft
        elif pane == "post_analysis":
            draft = tab.post_analysis.writeback_draft
        else:
            raise InvalidInputError(f"unknown writeback pane: {pane!r}")
        if draft is None:
            raise FailedPreconditionError(
                f"No {'post ' if pane == 'post_analysis' else ''}writeback draft for tab {tab_id!r}"
            )
        if not getattr(draft, "is_active", False):  # type: ignore[attr-defined]
            raise FailedPreconditionError(
                f"Writeback draft for tab {tab_id!r} pane {pane!r} has been torn down"
            )
        return draft  # type: ignore[return-value]

    def get_writeback_item_draft_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str
    ) -> CfgDraft:
        self._guard.acquire_writeback_permit(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        return self._writeback.get_item_draft(draft, session_id)  # type: ignore[arg-type]

    def set_writeback_item_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str, **changes: Any
    ) -> dict[str, object]:
        self._guard.acquire_writeback_permit(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        result = self._writeback.edit_draft(draft, session_id, **changes)  # type: ignore[arg-type]
        self._emit_draft_changed(tab_id)
        return result

    def apply_writeback_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, Any]:
        self._guard.acquire_writeback_permit(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        result = self._writeback.apply_draft(draft)  # type: ignore[arg-type]
        self._emit_draft_changed(tab_id)
        return result

    def get_writeback_summaries_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, tuple[str | None, str | None]]:
        # Summaries are display-only, read through the service-owned draft;
        # no guard needed beyond existence check (read-only).
        try:
            draft = self._draft_for_pane(tab_id, pane)
        except FailedPreconditionError:
            return {}
        return self._writeback.get_all_summaries(draft)  # type: ignore[arg-type]

    def get_writeback_applied_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, bool]:
        try:
            draft = self._draft_for_pane(tab_id, pane)
        except FailedPreconditionError:
            return {}
        return self._writeback.get_all_applied(draft)  # type: ignore[arg-type]

    def _emit_draft_changed(self, tab_id: str) -> None:
        self._bus.emit(
            TabInteractionChangedPayload(
                tab_id=tab_id,
                fact=TabInteractionFact.WRITEBACK_DRAFT_CHANGED,
            )
        )

    def _require_tab_idle(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def get_context_version(self) -> int:
        return self._resource_versions().get("context", 0)
