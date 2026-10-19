"""Workspace captures authored input and reports per-tab preparation failures."""

import json
from typing import Literal
from unittest.mock import MagicMock

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.events.tab import TabAddedPayload
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.app.measure.services.workspace import WorkspaceService
from zcu_tools.gui.app.measure.services.workspace_types import (
    WorkspaceSnapshot,
    WorkspaceTab,
)
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.form import FormEnv, FormInput
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


class ChoiceCfg(ConfigBase):
    mode: Literal["a", "b"] = "a"


def test_workspace_roundtrip_keeps_raw_and_independent_selection() -> None:
    svc, state, _, _ = make_service()
    form = Form(ChoiceCfg, {})
    authored = FormInput(form.model, {"mode": "a"}, selections={("mode",): "b"})
    initial = json.loads(authored.to_json())
    resource = TabFormResource(
        lambda _ctx: form,
        capture=lambda _aliases: TabFormSources(FormEnv()),
        mutation_allowed=lambda: True,
        initial=initial,
    )
    state.add_tab(
        "choice", Session(adapter_name="choice", adapter=MagicMock(), cfg=resource)
    )

    session = svc.capture_session()
    restored = TabFormResource(
        lambda _ctx: form,
        capture=lambda _aliases: TabFormSources(FormEnv()),
        mutation_allowed=lambda: True,
        initial=session.tabs[0].form_input,
    )

    assert restored.input.data == {"mode": "a"}
    assert restored.input.selections == {("mode",): "b"}
    assert session.tabs[0].form_input == {
        "data": {"mode": "a"},
        "selections": [{"path": ["mode"], "value": "b"}],
    }


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
    assert isinstance(session, WorkspaceSnapshot)
    assert session.tabs[0].adapter_name == "fake"
    assert session.tabs[0].form_input == {"data": {"x": 1}, "selections": []}
    assert session.active_tab_index == 0
    session.tabs[0].form_input["data"] = {"x": 99}
    assert svc.capture_session().tabs[0].form_input == {
        "data": {"x": 1},
        "selections": [],
    }


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
    session = WorkspaceSnapshot(
        tabs=(WorkspaceTab(adapter_name="fake", form_input={}),), active_tab_index=0
    )
    report = svc.apply_session(session)
    assert report.restored_tabs == 0
    assert report.rejected_tabs[0].subject == "fake"
    assert "bad defaults" in report.rejected_tabs[0].message
    assert state.active_tab_id is None
