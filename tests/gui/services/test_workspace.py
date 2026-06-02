from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from zcu_tools.gui.event_bus import EventBus, GuiEvent, TabAddedPayload
from zcu_tools.gui.services.session_persistence import (
    SESSION_VERSION,
    PersistedSession,
    PersistedTab,
    SessionPersistenceError,
)
from zcu_tools.gui.services.workspace import WorkspaceService
from zcu_tools.gui.state import State


def _make_service() -> tuple[WorkspaceService, State, MagicMock, MagicMock, EventBus]:
    state = State(MagicMock())
    tabs = MagicMock()
    persistence = MagicMock()
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    return (
        WorkspaceService(state, tabs, persistence, bus),
        state,
        tabs,
        persistence,
        bus,
    )


def test_new_tab_owns_active_tab_and_event() -> None:
    svc, state, tabs, _, bus = _make_service()
    state.tabs["tab-1"] = MagicMock()
    tabs.new_tab.return_value = "tab-1"

    assert svc.new_tab("fake") == "tab-1"

    assert state.active_tab_id == "tab-1"
    cast(MagicMock, bus.emit).assert_called_once_with(
        GuiEvent.TAB_ADDED,
        TabAddedPayload(tab_id="tab-1", adapter_name="fake"),
    )


def test_restore_invalid_configuration_returns_typed_issue() -> None:
    svc, state, tabs, persistence, _ = _make_service()
    persistence.load_session.return_value = PersistedSession(
        version=SESSION_VERSION,
        tabs=[PersistedTab(adapter_name="fake", cfg_raw={}, save_paths_override=None)],
        active_tab_index=0,
    )
    tabs.make_default_cfg.return_value = MagicMock()
    persistence.raw_to_schema.side_effect = SessionPersistenceError("bad cfg")

    report = svc.restore_session()

    assert report.restored_tabs == 0
    assert report.rejected_tabs[0].subject == "fake"
    assert "invalid saved configuration" in report.rejected_tabs[0].message
    # New flow decodes raw→live *before* creating the tab, so a bad cfg never
    # creates a tab to close.
    tabs.new_tab.assert_not_called()
    tabs.close_tab.assert_not_called()
    assert state.active_tab_id is None
