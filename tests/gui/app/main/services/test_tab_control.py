"""TabControlFacet public contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.services.tab_control import TabControlFacet

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.active_tab_id: str | None = "tab-1"
        self.running_tab_id: str | None = None

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def list_tab_ids(self) -> list[str]:
        self._log.add("state", "list_tab_ids")
        return ["tab-1", "tab-2"]


class RecordingTab:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.default_schema = object()
        self.snapshot = object()

    def get_tab_adapter_name(self, tab_id: str) -> str:
        self._log.add("tab", "get_tab_adapter_name", tab_id)
        return "adapter-a"

    def get_snapshot(self, tab_id: str) -> object:
        self._log.add("tab", "get_snapshot", tab_id)
        return self.snapshot

    def update_tab_cfg(self, tab_id: str, schema: object) -> None:
        self._log.add("tab", "update_tab_cfg", tab_id, schema)

    def make_default_cfg(self, adapter_name: str) -> object:
        self._log.add("tab", "make_default_cfg", adapter_name)
        return self.default_schema

    def update_tab_save_path_overrides(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._log.add(
            "tab",
            "update_tab_save_path_overrides",
            tab_id,
            data_path,
            image_path,
        )


class RecordingWorkspace:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def new_tab(self, adapter_name: str) -> str:
        self._log.add("workspace", "new_tab", adapter_name)
        return "new-tab"

    def close_tab(self, tab_id: str) -> None:
        self._log.add("workspace", "close_tab", tab_id)

    def set_active_tab(self, tab_id: str) -> None:
        self._log.add("workspace", "set_active_tab", tab_id)

    def reorder_tabs(self, tab_ids: Sequence[str]) -> None:
        self._log.add("workspace", "reorder_tabs", tuple(tab_ids))


class RecordingBus:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.payloads: list[object] = []

    def emit(self, payload: object) -> None:
        self._log.add("bus", "emit", type(payload).__name__)
        self.payloads.append(payload)


def _facet() -> tuple[
    TabControlFacet,
    CallLog,
    RecordingState,
    RecordingTab,
    RecordingWorkspace,
    RecordingBus,
]:
    log = CallLog()
    state = RecordingState(log)
    tab = RecordingTab(log)
    workspace = RecordingWorkspace(log)
    bus = RecordingBus(log)
    return (
        TabControlFacet(
            state=cast(Any, state),
            tab=cast(Any, tab),
            workspace=cast(Any, workspace),
            bus=cast(Any, bus),
        ),
        log,
        state,
        tab,
        workspace,
        bus,
    )


def test_tab_control_routes_lifecycle_to_workspace() -> None:
    facet, log, _state, _tab, _workspace, _bus = _facet()

    assert facet.new_tab("adapter-a") == "new-tab"
    facet.close_tab("tab-1")
    facet.set_active_tab("tab-1")
    facet.reorder_tabs(["tab-2", "tab-1"])

    assert log.calls == [
        call("workspace", "new_tab", "adapter-a"),
        call("workspace", "close_tab", "tab-1"),
        call("workspace", "set_active_tab", "tab-1"),
        call("workspace", "reorder_tabs", ("tab-2", "tab-1")),
    ]


def test_tab_control_reads_tab_identity_from_state() -> None:
    facet, log, state, _tab, _workspace, _bus = _facet()
    state.running_tab_id = "tab-2"

    assert facet.get_active_tab_id() == "tab-1"
    assert facet.get_running_tab_id() == "tab-2"
    assert facet.has_tab("tab-1") is True
    assert facet.list_tab_ids() == ["tab-1", "tab-2"]

    assert log.calls == [
        call("state", "has_tab", "tab-1"),
        call("state", "list_tab_ids"),
    ]


def test_tab_control_routes_tab_read_model_to_tab_service() -> None:
    facet, log, _state, tab, _workspace, _bus = _facet()

    assert facet.get_tab_adapter_name("tab-1") == "adapter-a"
    assert facet.get_tab_snapshot("tab-1") is tab.snapshot

    assert log.calls == [
        call("tab", "get_tab_adapter_name", "tab-1"),
        call("tab", "get_snapshot", "tab-1"),
    ]


def test_tab_control_updates_cfg_via_tab_service() -> None:
    facet, log, _state, _tab, _workspace, _bus = _facet()
    schema = cast(Any, object())

    facet.update_tab_cfg("tab-1", schema)

    assert log.calls == [
        call("tab", "update_tab_cfg", "tab-1", schema),
    ]


def test_tab_control_reset_cfg_rebuilds_default_and_commits() -> None:
    facet, log, _state, tab, _workspace, _bus = _facet()

    assert facet.reset_tab_cfg("tab-1") is tab.default_schema

    assert log.calls == [
        call("tab", "get_tab_adapter_name", "tab-1"),
        call("tab", "make_default_cfg", "adapter-a"),
        call("tab", "update_tab_cfg", "tab-1", tab.default_schema),
    ]


def test_tab_control_reset_cfg_rejects_running_tab() -> None:
    facet, log, state, _tab, _workspace, _bus = _facet()
    state.running_tab_id = "tab-1"

    with pytest.raises(RuntimeError, match="currently running"):
        facet.reset_tab_cfg("tab-1")

    assert log.calls == []


def test_tab_control_update_save_paths_emits_interaction_changed() -> None:
    facet, log, _state, _tab, _workspace, bus = _facet()

    facet.update_tab_save_paths("tab-1", "data.h5", "image.png")

    assert log.calls == [
        call(
            "tab",
            "update_tab_save_path_overrides",
            "tab-1",
            "data.h5",
            "image.png",
        ),
        call("bus", "emit", "TabInteractionChangedPayload"),
    ]
    assert len(bus.payloads) == 1
    payload = bus.payloads[0]
    assert isinstance(payload, TabInteractionChangedPayload)
    assert payload.tab_id == "tab-1"
    assert payload.fact is TabInteractionFact.SAVE_PATHS_CHANGED
