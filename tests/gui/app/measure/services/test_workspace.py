"""Workspace captures authored input and reports per-tab preparation failures."""

from unittest.mock import MagicMock

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.events.tab import TabAddedPayload
from zcu_tools.gui.app.measure.services.persistence_types import (
    PersistedSession,
    PersistedTab,
)
from zcu_tools.gui.app.measure.services.workspace import WorkspaceService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.form_adapter import Form

from tests.gui.app.measure._form_fakes import make_form


class WorkspaceCfg(ConfigBase):
    """One input leaf is enough to observe memento ordering and detachment."""

    x: int = 1


def make_service() -> tuple[WorkspaceService, State, MagicMock, EventBus]:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    tabs = MagicMock()
    bus = EventBus()
    return WorkspaceService(state, tabs, bus), state, tabs, bus


def add_tab(state: State, tab_id: str, adapter_name: str) -> None:
    state.add_tab(
        tab_id,
        Session(
            adapter_name=adapter_name,
            adapter=MagicMock(),
            cfg=make_form(Form(WorkspaceCfg, {}), state=state),
        ),
    )


def test_new_tab_owns_active_tab_and_event() -> None:
    svc, state, tabs, bus = make_service()
    events: list[TabAddedPayload] = []
    bus.subscribe(TabAddedPayload, events.append)
    add_tab(state, "tab-1", "fake")
    tabs.new_tab.return_value = "tab-1"
    assert svc.new_tab("fake") == "tab-1"
    assert state.active_tab_id == "tab-1"
    assert events == [TabAddedPayload(tab_id="tab-1", adapter_name="fake")]


def test_capture_session_returns_payload_without_disk() -> None:
    svc, state, _, _ = make_service()
    add_tab(state, "tab-1", "fake")
    state.active_tab_id = "tab-1"
    session = svc.capture_session()
    assert isinstance(session, PersistedSession)
    assert session.tabs[0].adapter_name == "fake"
    assert session.tabs[0].form_input == {"x": 1}
    assert session.active_tab_index == 0
    session.tabs[0].form_input["x"] = 99
    assert svc.capture_session().tabs[0].form_input == {"x": 1}


def test_reorder_tabs_updates_capture_session_order() -> None:
    svc, state, _, _ = make_service()
    add_tab(state, "tab-a", "fake-a")
    add_tab(state, "tab-b", "fake-b")
    state.active_tab_id = "tab-a"
    svc.reorder_tabs(["tab-b", "tab-a"])
    session = svc.capture_session()
    assert [tab.adapter_name for tab in session.tabs] == ["fake-b", "fake-a"]
    assert session.active_tab_index == 1


def test_apply_invalid_configuration_returns_typed_issue() -> None:
    svc, state, tabs, _ = make_service()
    tabs.new_tab.side_effect = ValueError("bad defaults")
    session = PersistedSession(
        tabs=(PersistedTab(adapter_name="fake", form_input={}),), active_tab_index=0
    )
    report = svc.apply_session(session)
    assert report.restored_tabs == 0
    assert report.rejected_tabs[0].subject == "fake"
    assert "bad defaults" in report.rejected_tabs[0].message
    assert state.active_tab_id is None
