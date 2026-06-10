from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.events.tab import TabAddedPayload, TabClosedPayload
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

from .persistence_types import PersistedSession, PersistedTab
from .ports import RestoreIssue, RestoreReport, TabSnapshot
from .session_codec import SessionCodecError, raw_to_schema, schema_to_raw

if TYPE_CHECKING:
    from .ports import TabLifecyclePort


class WorkspaceService:
    """Own tab lifecycle and tab-session capture/apply workflow.

    The cfg raw↔live codec (``session_codec``) is this service's internal
    implementation of capturing/applying a session; the PersistenceCaretaker
    only ever sees the resulting ``PersistedSession`` memento (opaque cfg_raw).
    """

    def __init__(
        self,
        state: State,
        tabs: "TabLifecyclePort",
        bus: EventBus,
    ) -> None:
        self._state = state
        self._tabs = tabs
        self._bus = bus

    def new_tab(self, adapter_name: str) -> str:
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
        self._state.set_active_tab(tab_id)

    def capture_session(self) -> PersistedSession:
        """Snapshot the live tabs into a serializable session memento (no disk).
        Lowers each tab's live cfg to raw via the internal codec."""
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
                cfg_raw=schema_to_raw(tab.cfg_schema),
                save_paths_override=tab.save_path_overrides,
            )
            for _, tab in tabs
        )
        return PersistedSession(tabs=payload_tabs, active_tab_index=active_tab_index)

    def apply_session(self, session: PersistedSession) -> RestoreReport:
        """Rebuild tabs from a session memento. Per-tab failures (adapter
        missing / cfg invalid) are collected into the report; good tabs still
        restore. Bridges the raw cfg back to live via the internal codec."""
        restored_by_index: dict[int, str] = {}
        rejected: list[RestoreIssue] = []
        for index, persisted_tab in enumerate(session.tabs):
            try:
                base_schema = self._tabs.make_default_cfg(persisted_tab.adapter_name)
            except KeyError as exc:
                rejected.append(
                    RestoreIssue(
                        persisted_tab.adapter_name,
                        f"adapter unavailable ({exc})",
                    )
                )
                continue
            try:
                restored_schema = raw_to_schema(base_schema, persisted_tab.cfg_raw)
            except SessionCodecError as exc:
                rejected.append(
                    RestoreIssue(
                        persisted_tab.adapter_name,
                        f"invalid saved configuration ({exc})",
                    )
                )
                continue
            snapshot = TabSnapshot(
                adapter_name=persisted_tab.adapter_name,
                cfg_schema=restored_schema,
                save_paths_override=persisted_tab.save_paths_override,
            )
            tab_id = self._tabs.new_tab(persisted_tab.adapter_name, from_dict=snapshot)
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
        )
