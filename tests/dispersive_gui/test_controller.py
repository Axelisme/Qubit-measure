"""Tests for the dispersive-fit-gui Controller façade (Phase 1: project command)."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.event_bus import EventBus, ProjectChangedPayload
from zcu_tools.gui.app.dispersive.state import (
    PROJECT_VERSION_KEY,
    DispersiveState,
    ProjectInfo,
)


def test_setup_project_writes_state_and_emits_event():
    state = DispersiveState()
    bus = EventBus()
    seen = []
    bus.subscribe(ProjectChangedPayload, lambda _p: seen.append(True))
    ctrl = Controller(state, bus)

    before = state.version.get(PROJECT_VERSION_KEY)
    ctrl.setup_project(ProjectInfo(chip_name="ChipA", qub_name="Q1"))

    assert state.project.chip_name == "ChipA"
    assert state.version.get(PROJECT_VERSION_KEY) > before
    assert seen == [True]


def test_controller_exposes_state_and_bus():
    state = DispersiveState()
    ctrl = Controller(state)
    assert ctrl.state is state
    assert isinstance(ctrl.bus, EventBus)
