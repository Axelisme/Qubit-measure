from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from zcu_tools.gui.app.main.events.tab import TabAddedPayload
from zcu_tools.gui.app.main.services import workspace as workspace_mod
from zcu_tools.gui.app.main.services.persistence_types import (
    PersistedSession,
    PersistedTab,
)
from zcu_tools.gui.app.main.services.session_codec import SessionCodecError
from zcu_tools.gui.app.main.services.workspace import WorkspaceService
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def _make_service() -> tuple[WorkspaceService, State, MagicMock, EventBus]:
    state = State(MagicMock())
    tabs = MagicMock()
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    return WorkspaceService(state, tabs, bus), state, tabs, bus


def test_new_tab_owns_active_tab_and_event() -> None:
    svc, state, tabs, bus = _make_service()
    state.tabs["tab-1"] = MagicMock()
    tabs.new_tab.return_value = "tab-1"

    assert svc.new_tab("fake") == "tab-1"

    assert state.active_tab_id == "tab-1"
    cast(MagicMock, bus.emit).assert_called_once_with(
        TabAddedPayload(tab_id="tab-1", adapter_name="fake"),
    )


def test_capture_session_returns_payload_without_disk(monkeypatch) -> None:
    """capture_session lowers tabs to a PersistedSession and returns it — no
    disk write (the Caretaker owns I/O)."""
    svc, state, _, _ = _make_service()
    tab = MagicMock()
    tab.adapter_name = "fake"
    tab.save_path_overrides = None
    state.tabs["tab-1"] = tab
    state.active_tab_id = "tab-1"
    monkeypatch.setattr(workspace_mod, "schema_to_raw", lambda schema: {"x": 1})

    session = svc.capture_session()

    assert isinstance(session, PersistedSession)
    assert len(session.tabs) == 1
    assert session.tabs[0].adapter_name == "fake"
    assert session.tabs[0].cfg_raw == {"x": 1}
    assert session.active_tab_index == 0


def test_apply_invalid_configuration_returns_typed_issue(monkeypatch) -> None:
    svc, state, tabs, _ = _make_service()
    session = PersistedSession(
        tabs=(PersistedTab(adapter_name="fake", cfg_raw={}, save_paths_override=None),),
        active_tab_index=0,
    )
    tabs.make_default_cfg.return_value = MagicMock()
    monkeypatch.setattr(
        workspace_mod,
        "raw_to_schema",
        MagicMock(side_effect=SessionCodecError("bad cfg")),
    )

    report = svc.apply_session(session)

    assert report.restored_tabs == 0
    assert report.rejected_tabs[0].subject == "fake"
    assert "invalid saved configuration" in report.rejected_tabs[0].message
    # Decodes raw→live *before* creating the tab, so a bad cfg never creates one.
    tabs.new_tab.assert_not_called()
    tabs.close_tab.assert_not_called()
    assert state.active_tab_id is None
