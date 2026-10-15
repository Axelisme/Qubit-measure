"""App-facing writeback control facet for UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from .service import WritebackDraft

WritebackPane: TypeAlias = Literal["analysis", "post_analysis"]

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.library_entry import ModuleEntryForm

    from ..services.guard import GuardService
    from .service import (
        WritebackEdit,
        WritebackService,
        WritebackValues,
        WritebackWritten,
    )


class WritebackControlPort(Protocol):
    """App-facing writeback surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...

    def get_writeback_item_form_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str
    ) -> ModuleEntryForm:
        """Admit access to a module item's shared editing Form for this pane.

        Every call checks writeback permission, tab idle and current draft
        ownership. Call again immediately before each state.write or
        state.run_command; retaining a Form does not retain admission.
        See WritebackService.get_item_form for the Form's editing effects.
        """
        ...

    def set_writeback_item_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str, **changes: Any
    ) -> None:
        """Edit one item; changes follow WritebackService.edit_draft."""
        ...

    def apply_writeback_for_pane(
        self,
        tab_id: str,
        pane: WritebackPane,
        *,
        item_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]: ...

    def get_writeback_summaries_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, tuple[str | None, str | None]]: ...

    def get_writeback_applied_for_pane(
        self, tab_id: str, pane: WritebackPane
    ) -> dict[str, bool]: ...

    def get_writeback_values_for_pane(
        self, tab_id: str, pane: WritebackPane, *, session_id: str | None = None
    ) -> dict[str, WritebackValues]:
        """Read captured pane facts; None means strict bulk, an ID means one item.

        Unknown IDs raise KeyError. Invalid/inactive/missing panes or drafts,
        unreadable targets and requested proposal validation failures propagate.
        Item reads do not lower siblings and never change selection or focus.
        """
        ...

    def write_writeback_for_pane(
        self, tab_id: str, pane: WritebackPane, changes: tuple[WritebackEdit, ...]
    ) -> list[WritebackWritten]: ...

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
        access: ExperimentAccess | None = None,
    ) -> None:
        self._state = state
        self._guard = guard
        self._writeback = writeback
        self._resource_versions = resource_versions
        self._bus = bus
        self._access = access if access is not None else ExperimentAccess()

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

    def get_writeback_item_form_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str
    ) -> ModuleEntryForm:
        """Admit each Form access under the writeback, idle and draft checks.

        Consumers repeat this call before every state.write or state.run_command;
        a retained Form is not continuing admission. See get_item_form for
        the shared draft and its editing effects.
        """
        self._guard.check_writeback(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        if not isinstance(draft, WritebackDraft):
            raise InvalidInputError("unknown writeback draft")
        return self._writeback.get_item_form(draft, session_id)

    def set_writeback_item_for_pane(
        self, tab_id: str, pane: WritebackPane, session_id: str, **changes: Any
    ) -> None:
        self._guard.check_writeback(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        if not isinstance(draft, WritebackDraft):
            raise InvalidInputError("unknown writeback draft")
        self._writeback.edit_draft(draft, session_id, **changes)
        self._emit_draft_changed(tab_id)

    def apply_writeback_for_pane(
        self,
        tab_id: str,
        pane: WritebackPane,
        *,
        item_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        self._guard.check_writeback(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        if not isinstance(draft, WritebackDraft):
            raise InvalidInputError("unknown writeback draft")
        try:
            return self._writeback.apply_draft(draft, item_ids=item_ids)
        finally:
            # The shared flow may have committed a successful prefix.
            self._emit_draft_changed(tab_id)

    def write_writeback_for_pane(
        self, tab_id: str, pane: WritebackPane, changes: tuple[WritebackEdit, ...]
    ) -> list[WritebackWritten]:
        self._guard.check_writeback(tab_id)
        self._require_tab_idle(tab_id)
        draft = self._draft_for_pane(tab_id, pane)
        if not isinstance(draft, WritebackDraft):
            raise InvalidInputError("unknown writeback draft")
        try:
            return self._writeback.write_draft(draft, changes)
        finally:
            # Failed batches may have changed the shared draft's prefix.
            self._emit_draft_changed(tab_id)

    def get_writeback_values_for_pane(
        self, tab_id: str, pane: WritebackPane, *, session_id: str | None = None
    ) -> dict[str, WritebackValues]:
        """Read all pane facts or only session_id, without changing draft/selection.

        None keeps strict bulk validation, including unselected items. An ID
        excludes sibling lowering, and an unknown ID raises KeyError. Existing
        tab/pane/draft errors and requested target/source/I/O/validation failures
        propagate. Facts belong to this pane's creation capture, not active focus.
        """
        draft = self._draft_for_pane(tab_id, pane)
        if not isinstance(draft, WritebackDraft):
            raise InvalidInputError("unknown writeback draft")
        return self._writeback.preview_values(draft, session_id=session_id)

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
        self._access.require_available()
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def get_context_version(self) -> int:
        return self._resource_versions().get("context", 0)
