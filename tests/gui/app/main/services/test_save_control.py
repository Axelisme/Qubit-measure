"""SaveControlFacet public contract tests."""

from __future__ import annotations

from typing import Any, cast

import pytest
from zcu_tools.gui.app.main.services.save_control import SaveControlFacet
from zcu_tools.gui.expected_error import FailedPreconditionError

from tests.gui._control_fakes import CallLog, call


class RecordingState:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def has_tab(self, tab_id: str) -> bool:
        self._log.add("state", "has_tab", tab_id)
        return tab_id == "tab-1"

    def is_tab_busy(self, tab_id: str) -> bool:
        self._log.add("state", "is_tab_busy", tab_id)
        return False


class RecordingGuard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def acquire_save_permit(self, tab_id: str) -> str:
        self._log.add("guard", "acquire_save_permit", tab_id)
        return f"permit:{tab_id}"


class RecordingTab:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.data_path: str | None = "default.h5"
        self.analysis_image_path: str | None = "default.png"
        self.post_analysis_image_path: str | None = "default.png"

    def get_tab_data_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_data_path", tab_id)
        return self.data_path

    def get_tab_analysis_image_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_analysis_image_path", tab_id)
        return self.analysis_image_path

    def get_tab_post_analysis_image_path(self, tab_id: str) -> str | None:
        self._log.add("tab", "get_tab_post_analysis_image_path", tab_id)
        return self.post_analysis_image_path


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
        call("state", "is_tab_busy", "tab-1"),
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
        call("state", "is_tab_busy", "tab-1"),
        call("tab", "get_tab_analysis_image_path", "tab-1"),
        call("save", "save_image_sync", "permit:tab-1", "default.png"),
    ]
    assert notifications == ["Image saved to default.png"]


def test_save_post_image_uses_default_path_and_notifies() -> None:
    facet, log, _state, _tab, _save, _bus, notifications = _facet()

    assert facet.save_post_image("tab-1") == "default.png"

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("tab", "get_tab_post_analysis_image_path", "tab-1"),
        call("save", "save_post_image_sync", "permit:tab-1", "default.png"),
    ]
    assert notifications == ["Post-analysis image saved to default.png"]


def test_missing_save_paths_fast_fails() -> None:
    facet, log, _state, tab, _save, _bus, notifications = _facet()
    tab.analysis_image_path = None

    with pytest.raises(
        FailedPreconditionError, match="no analysis image path configured"
    ):
        facet.save_image("tab-1")

    assert log.calls == [
        call("guard", "acquire_save_permit", "tab-1"),
        call("state", "is_tab_busy", "tab-1"),
        call("tab", "get_tab_analysis_image_path", "tab-1"),
    ]
    assert notifications == []
