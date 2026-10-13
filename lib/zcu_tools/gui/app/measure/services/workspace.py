from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from zcu_tools.gui.app.measure.events.tab import (
    TabActivatedPayload,
    TabAddedPayload,
    TabClosedPayload,
)
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

from .persistence_types import PersistedSession, PersistedTab
from .ports import RestoreDroppedInputs, RestoreIssue, RestoreReport, TabSnapshot

if TYPE_CHECKING:
    from .ports import TabLifecyclePort


class WorkspaceService:
    """Own tab lifecycle and tab-session capture/apply workflow.

    Capture preserves authored Form input JSON. Tab resources own compatible
    restoration against a new source capture; the PersistenceCaretaker only
    reads/writes the resulting versioned PersistedSession memento.
    """

    def __init__(
        self,
        state: State,
        tabs: TabLifecyclePort,
        bus: EventBus,
    ) -> None:
        self._state = state
        self._tabs = tabs
        self._bus = bus

    def new_tab(self, adapter_name: str) -> str:
        """Create, activate and announce a tab using current global sources.

        Default-cfg failure publishes no tab fact.
        """
        tab_id = self._tabs.new_tab(adapter_name)
        self._state.set_active_tab(tab_id)
        self._bus.emit(
            TabAddedPayload(tab_id=tab_id, adapter_name=adapter_name),
        )
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError("Cannot close a busy tab")
        self._tabs.close_tab(tab_id)
        self._bus.emit(TabClosedPayload(tab_id=tab_id))

    def set_active_tab(self, tab_id: str) -> None:
        previous = self._state.active_tab_id
        self._state.set_active_tab(tab_id)
        if previous != tab_id:
            self._bus.emit(TabActivatedPayload(tab_id=tab_id))

    def reorder_tabs(self, tab_ids: Sequence[str]) -> None:
        self._state.reorder_tabs(tab_ids)

    def capture_session(self) -> PersistedSession:
        """Snapshot authored Form input, including unfinished expressions, without I/O.

        Sources, alias overrides, result models and figures are not persisted.
        """
        tabs = list(self._state.tabs.items())
        tab_ids = [tab_id for tab_id, _ in tabs]
        active_tab_index = (
            tab_ids.index(self._state.active_tab_id)
            if self._state.active_tab_id in tab_ids
            else None
        )
        payload_tabs = tuple(
            PersistedTab(
                adapter_name=tab.adapter_name,
                form_input=tab.cfg.input.data,
            )
            for _, tab in tabs
        )
        return PersistedSession(tabs=payload_tabs, active_tab_index=active_tab_index)

    def apply_session(self, session: PersistedSession) -> RestoreReport:
        """Create fresh resources and overlay saved compatible Form input JSON.

        Per-tab preparation failures are rejected_tabs; other tabs still restore.
        Successfully restored tabs report incompatible leaves through dropped_inputs.
        Ordinary unfinished/invalid input remains editable. Old memento versions
        are rejected by the caretaker codec, not migrated here.
        """
        restored_by_index: dict[int, str] = {}
        rejected: list[RestoreIssue] = []
        dropped: list[RestoreDroppedInputs] = []
        for index, persisted_tab in enumerate(session.tabs):
            snapshot = TabSnapshot(
                adapter_name=persisted_tab.adapter_name,
                cfg_input=persisted_tab.form_input,
            )
            try:
                tab_id = self._tabs.new_tab(
                    persisted_tab.adapter_name, from_dict=snapshot
                )
            except (KeyError, ValueError, TypeError) as exc:
                rejected.append(
                    RestoreIssue(
                        persisted_tab.adapter_name, f"Cannot restore form ({exc})"
                    )
                )
                continue
            paths = self._tabs.dropped_input_paths(tab_id)
            if paths:
                dropped.append(RestoreDroppedInputs(persisted_tab.adapter_name, paths))
            restored_by_index[index] = tab_id
            self._bus.emit(
                TabAddedPayload(
                    tab_id=tab_id,
                    adapter_name=persisted_tab.adapter_name,
                ),
            )

        if session.active_tab_index is not None:
            active_tab_id = restored_by_index.get(session.active_tab_index)
            if active_tab_id is not None:
                self._state.set_active_tab(active_tab_id)
        return RestoreReport(
            restored_tabs=len(restored_by_index),
            rejected_tabs=tuple(rejected),
            dropped_inputs=tuple(dropped),
        )
