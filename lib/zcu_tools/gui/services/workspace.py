from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.event_bus import (
    EventBus,
    GuiEvent,
    TabAddedPayload,
    TabClosedPayload,
)
from zcu_tools.gui.state import State

from .ports import TabSnapshot
from .session_persistence import (
    SESSION_VERSION,
    PersistedSession,
    PersistedTab,
    SessionPersistenceError,
)

if TYPE_CHECKING:
    from .ports import SessionStorePort, TabLifecyclePort


@dataclass(frozen=True)
class RestoreIssue:
    subject: str
    message: str


@dataclass(frozen=True)
class RestoreReport:
    restored_tabs: int
    rejected_tabs: tuple[RestoreIssue, ...]


class WorkspaceService:
    """Own tab lifecycle and tab-session application workflow."""

    def __init__(
        self,
        state: State,
        tabs: "TabLifecyclePort",
        persistence: "SessionStorePort",
        bus: EventBus,
    ) -> None:
        self._state = state
        self._tabs = tabs
        self._persistence = persistence
        self._bus = bus

    def new_tab(self, adapter_name: str) -> str:
        tab_id = self._tabs.new_tab(adapter_name)
        self._state.set_active_tab(tab_id)
        self._bus.emit(
            GuiEvent.TAB_ADDED,
            TabAddedPayload(tab_id=tab_id, adapter_name=adapter_name),
        )
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError("Cannot close a busy tab")
        self._tabs.close_tab(tab_id)
        self._bus.emit(GuiEvent.TAB_CLOSED, TabClosedPayload(tab_id=tab_id))

    def set_active_tab(self, tab_id: str) -> None:
        self._state.set_active_tab(tab_id)

    def persist_session(self) -> None:
        tabs = list(self._state.tabs.items())
        tab_ids = [tab_id for tab_id, _ in tabs]
        active_tab_index = (
            tab_ids.index(self._state.active_tab_id)
            if self._state.active_tab_id in tab_ids
            else None
        )
        payload_tabs = [
            PersistedTab(
                adapter_name=tab.adapter_name,
                cfg_raw=self._persistence.schema_to_raw(
                    tab.cfg_schema, ml=self._state.exp_context.ml
                ),
                save_paths_override=tab.save_path_overrides,
            )
            for _, tab in tabs
        ]
        self._persistence.save_session(
            PersistedSession(
                version=SESSION_VERSION,
                tabs=payload_tabs,
                active_tab_index=active_tab_index,
            )
        )

    def restore_session(self) -> RestoreReport:
        session = self._persistence.load_session()
        if session is None:
            return RestoreReport(restored_tabs=0, rejected_tabs=())
        return self._restore_loaded_session(session)

    def _restore_loaded_session(self, session: PersistedSession) -> RestoreReport:
        restored_by_index: dict[int, str] = {}
        rejected: list[RestoreIssue] = []
        for index, persisted_tab in enumerate(session.tabs):
            # Bridge the on-disk raw form (PersistedTab) into a live TabSnapshot:
            # WorkspaceService holds both the adapter default (as decode base) and
            # the codec, so the raw→live conversion lives here (TabService stays
            # codec-free, SessionPersistenceService stays adapter-free).
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
                restored_schema = self._persistence.raw_to_schema(
                    base_schema, persisted_tab.cfg_raw
                )
            except SessionPersistenceError as exc:
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
                GuiEvent.TAB_ADDED,
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
