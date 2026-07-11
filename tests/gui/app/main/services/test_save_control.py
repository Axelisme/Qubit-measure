"""SaveControlFacet public contract tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
from zcu_tools.gui.app.main.adapter import SavePaths
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.services.save_control import SaveControlFacet

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"


class RecordingGuard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def acquire_save_permit(self, tab_id: str) -> str:
        self._log.add("guard", "acquire_save_permit", tab_id)
        return f"permit:{tab_id}"


class RecordingTab:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.save_paths: SavePaths | None = SavePaths(
            data_path="default.h5",
            image_path="default.png",
        )

    def get_tab_save_paths(self, tab_id: str) -> SavePaths | None:
        self._log.add("tab", "get_tab_save_paths", tab_id)
        return self.save_paths

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


class RecordingSave:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def start_save_data(self, permit: object, data_path: str, comment: str = "") -> str:
        self._log.add("save", "start_save_data", permit, data_path, comment=comment)
        return f"written:{data_path}"

    def save_image_sync(self, permit: object, image_path: str) -> None:
        self._log.add("save", "save_image_sync", permit, image_path)

    def save_post_image_sync(self, permit: object, image_path: str) -> None:
        self._log.add("save", "save_post_image_sync", permit, image_path)

    def start_save_result(
        self,
        permit: object,
        data_path: str,
        image_path: str,
        comment: str = "",
    ) -> str:
        self._log.add(
            "save",
            "start_save_result",
            permit,
            data_path,
            image_path,
            comment=comment,
        )
        return f"written:{data_path}"


class RecordingBus:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.payloads: list[object] = []

    def emit(self, payload: object) -> None:
        self._log.add("bus", "emit", type(payload).__name__)
        self.payloads.append(payload)


def _facet() -> tuple[
    SaveControlFacet,
    CallLog,
    RecordingState,
    RecordingTab,
    RecordingSave,
    RecordingBus,
    list[str],
]:
    log = CallLog()
    state = RecordingState(log)
    tab = RecordingTab(log)
    save = RecordingSave(log)
    bus = RecordingBus(log)
    notifications: list[str] = []
    return (
        SaveControlFacet(
            state=cast(Any, state),
            bus=cast(Any, bus),
            guard=cast(Any, RecordingGuard(log)),
            tab=cast(Any, tab),
            save=cast(Any, save),
            notify_info=notifications.append,
        ),
        log,
        state,
        tab,
        save,
        bus,
        notifications,
    )


def test_has_tab_reads_state() -> None:
    facet, log, _state, _tab, _save, _bus, _notifications = _facet()

    assert facet.has_tab("tab-1") is True

    assert log.calls == [call("state", "has_tab", "tab-1")]


def test_save_data_uses_explicit_path_without_resolving_defaults() -> None:
    facet, log, _state, _tab, _save, _bus, _notifications = _facet()

    assert facet.save_data("tab-1", "explicit.h5", comment="note") == (
        "written:explicit.h5"
    )

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call(
            "save",
            "start_save_data",
            "permit:tab-1",
            "explicit.h5",
            comment="note",
        ),
    ]


def test_save_image_uses_default_path_and_notifies() -> None:
    facet, log, _state, _tab, _save, _bus, notifications = _facet()

    assert facet.save_image("tab-1") == "default.png"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("tab", "get_tab_save_paths", "tab-1"),
        call("save", "save_image_sync", "permit:tab-1", "default.png"),
    ]
    assert notifications == ["Image saved to default.png"]


def test_save_post_image_uses_default_path_and_notifies() -> None:
    facet, log, _state, _tab, _save, _bus, notifications = _facet()

    assert facet.save_post_image("tab-1") == "default.png"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("tab", "get_tab_save_paths", "tab-1"),
        call("save", "save_post_image_sync", "permit:tab-1", "default.png"),
    ]
    assert notifications == ["Post-analysis image saved to default.png"]


def test_save_result_resolves_paths_even_with_explicit_overrides() -> None:
    facet, log, _state, _tab, _save, _bus, _notifications = _facet()

    assert facet.save_result(
        "tab-1",
        "explicit.h5",
        "explicit.png",
        comment="bundle",
    ) == ("written:explicit.h5", "explicit.png")

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("tab", "get_tab_save_paths", "tab-1"),
        call(
            "save",
            "start_save_result",
            "permit:tab-1",
            "explicit.h5",
            "explicit.png",
            comment="bundle",
        ),
    ]


def test_missing_save_paths_fast_fails() -> None:
    facet, log, _state, tab, _save, _bus, notifications = _facet()
    tab.save_paths = None

    with pytest.raises(RuntimeError, match="no save paths configured"):
        facet.save_image("tab-1")

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("tab", "get_tab_save_paths", "tab-1"),
    ]
    assert notifications == []


def test_update_tab_save_paths_updates_override_and_emits_interaction_changed() -> None:
    facet, log, _state, _tab, _save, bus, _notifications = _facet()

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
