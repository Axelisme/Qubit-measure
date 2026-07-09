"""Tests for MainWindowEventCoordinator payload routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from zcu_tools.gui.app.main.events.run import RunFinishedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.app.main.ui.main_window_events import MainWindowEventCoordinator
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import ContextSwitchedPayload, MlChangedPayload

from tests.gui._control_fakes import CallLog, call


def _snapshot(
    *,
    is_running: bool = False,
    is_analyzing: bool = False,
    is_saving_data: bool = False,
    has_analyze_result: bool = False,
    has_post_analyze_result: bool = False,
    has_run_result: bool = False,
) -> object:
    return SimpleNamespace(
        interaction=SimpleNamespace(
            is_running=is_running,
            is_analyzing=is_analyzing,
            is_saving_data=is_saving_data,
            has_analyze_result=has_analyze_result,
            has_post_analyze_result=has_post_analyze_result,
            has_run_result=has_run_result,
        )
    )


class RecordingCtrl:
    def __init__(self, log: CallLog, snapshot: object | None = None) -> None:
        self._log = log
        self.snapshot = snapshot or _snapshot()
        self.running_tab_id: str | None = "running-tab"

    def get_tab_snapshot(self, tab_id: str) -> object:
        self._log.add("ctrl", "get_tab_snapshot", tab_id)
        return self.snapshot

    def get_running_tab_id(self) -> str | None:
        self._log.add("ctrl", "get_running_tab_id")
        return self.running_tab_id


class RecordingHost:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.tab_ids = ["tab-1", "tab-2"]

    def add_tab_widget(self, tab_id: str, adapter_name: str) -> None:
        self._log.add("host", "add_tab_widget", tab_id, adapter_name)

    def remove_tab_widget(self, tab_id: str) -> None:
        self._log.add("host", "remove_tab_widget", tab_id)

    def has_tab_widget(self, tab_id: str) -> bool:
        self._log.add("host", "has_tab_widget", tab_id)
        return tab_id in self.tab_ids

    def view_tab_ids(self) -> list[str]:
        self._log.add("host", "view_tab_ids")
        return list(self.tab_ids)

    def focus_run_result_panel(self, tab_id: str) -> None:
        self._log.add("host", "focus_run_result_panel", tab_id)

    def refresh_tab_analyze_form(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_analyze_form", tab_id, snapshot)

    def refresh_tab_post_analyze_form(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_post_analyze_form", tab_id, snapshot)

    def refresh_tab_writeback(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_writeback", tab_id, snapshot)

    def refresh_tab_save_paths(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_save_paths", tab_id, snapshot)

    def refresh_tab_figure(self, tab_id: str, snapshot: object | None = None) -> None:
        self._log.add("host", "refresh_tab_figure", tab_id, snapshot)

    def refresh_tab_post_figure(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_post_figure", tab_id, snapshot)

    def refresh_tab_interaction(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_interaction", tab_id, snapshot)

    def refresh_run_lock(self, running_tab_id: str | None) -> None:
        self._log.add("host", "refresh_run_lock", running_tab_id)

    def refresh_context_panel(self) -> None:
        self._log.add("host", "refresh_context_panel")

    def refresh_predictor_panel(self) -> None:
        self._log.add("host", "refresh_predictor_panel")

    def refresh_feedback_widget(self) -> None:
        self._log.add("host", "refresh_feedback_widget")


def _coordinator(
    snapshot: object | None = None,
) -> tuple[MainWindowEventCoordinator, CallLog, RecordingCtrl, RecordingHost]:
    log = CallLog()
    ctrl = RecordingCtrl(log, snapshot=snapshot)
    host = RecordingHost(log)
    return (
        MainWindowEventCoordinator(cast(Any, ctrl), host),
        log,
        ctrl,
        host,
    )


def test_tab_interaction_changed_refreshes_writeback_interaction_and_idle_figures() -> (
    None
):
    snapshot = _snapshot(has_analyze_result=True, has_post_analyze_result=True)
    coordinator, log, _ctrl, _host = _coordinator(snapshot)

    coordinator._on_tab_interaction_changed(TabInteractionChangedPayload("tab-1"))

    assert log.calls == [
        call("ctrl", "get_tab_snapshot", "tab-1"),
        call("host", "refresh_tab_writeback", "tab-1", snapshot),
        call("host", "refresh_tab_interaction", "tab-1", snapshot),
        call("host", "refresh_tab_figure", "tab-1", snapshot),
        call("host", "refresh_tab_post_figure", "tab-1", snapshot),
        call("host", "refresh_feedback_widget"),
    ]


def test_tab_interaction_changed_skips_figures_while_tab_is_busy() -> None:
    snapshot = _snapshot(
        is_analyzing=True,
        has_analyze_result=True,
        has_post_analyze_result=True,
    )
    coordinator, log, _ctrl, _host = _coordinator(snapshot)

    coordinator._on_tab_interaction_changed(TabInteractionChangedPayload("tab-1"))

    assert log.calls == [
        call("ctrl", "get_tab_snapshot", "tab-1"),
        call("host", "refresh_tab_writeback", "tab-1", snapshot),
        call("host", "refresh_tab_interaction", "tab-1", snapshot),
        call("host", "refresh_feedback_widget"),
    ]


def test_tab_content_changed_refreshes_full_tab_content_sequence() -> None:
    snapshot = _snapshot()
    coordinator, log, _ctrl, _host = _coordinator(snapshot)

    coordinator._on_tab_content_changed(TabContentChangedPayload("tab-1"))

    assert log.calls == [
        call("host", "has_tab_widget", "tab-1"),
        call("ctrl", "get_tab_snapshot", "tab-1"),
        call("host", "refresh_tab_analyze_form", "tab-1", snapshot),
        call("host", "refresh_tab_post_analyze_form", "tab-1", snapshot),
        call("host", "refresh_tab_writeback", "tab-1", snapshot),
        call("host", "refresh_tab_save_paths", "tab-1", snapshot),
        call("host", "refresh_tab_figure", "tab-1", snapshot),
        call("host", "refresh_tab_post_figure", "tab-1", snapshot),
        call("host", "refresh_tab_interaction", "tab-1", snapshot),
        call("host", "refresh_feedback_widget"),
    ]


def test_tab_content_changed_ignores_missing_view_tab() -> None:
    coordinator, log, _ctrl, host = _coordinator()
    host.tab_ids = []

    coordinator._on_tab_content_changed(TabContentChangedPayload("tab-1"))

    assert log.calls == [
        call("host", "has_tab_widget", "tab-1"),
    ]


def test_bind_routes_events_and_close_unsubscribes() -> None:
    coordinator, log, _ctrl, _host = _coordinator()
    bus = BaseEventBus()

    coordinator.bind(bus)
    bus.emit(TabAddedPayload("tab-1", "adapter-a"))
    coordinator.close()
    bus.emit(TabAddedPayload("tab-2", "adapter-b"))

    assert log.calls == [
        call("host", "add_tab_widget", "tab-1", "adapter-a"),
    ]


def test_run_finished_focuses_result_panel_only_for_finished_outcome() -> None:
    coordinator, log, _ctrl, _host = _coordinator()

    coordinator._on_run_finished(RunFinishedPayload("tab-1", outcome="cancelled"))
    coordinator._on_run_finished(RunFinishedPayload("tab-1", outcome="finished"))

    assert log.calls == [
        call("host", "refresh_run_lock", None),
        call("host", "refresh_feedback_widget"),
        call("host", "refresh_run_lock", None),
        call("host", "refresh_feedback_widget"),
        call("host", "focus_run_result_panel", "tab-1"),
    ]


def test_context_and_ml_events_refresh_current_view_tabs() -> None:
    snapshot = _snapshot()
    coordinator, log, _ctrl, _host = _coordinator(snapshot)

    coordinator._on_context_switched(
        ContextSwitchedPayload(cast(Any, None), cast(Any, None))
    )
    coordinator._on_ml_changed(MlChangedPayload(cast(Any, None)))

    assert log.calls == [
        call("host", "refresh_context_panel"),
        call("host", "view_tab_ids"),
        call("ctrl", "get_tab_snapshot", "tab-1"),
        call("host", "refresh_tab_writeback", "tab-1", snapshot),
        call("host", "refresh_tab_save_paths", "tab-1", snapshot),
        call("host", "refresh_tab_interaction", "tab-1", snapshot),
        call("ctrl", "get_tab_snapshot", "tab-2"),
        call("host", "refresh_tab_writeback", "tab-2", snapshot),
        call("host", "refresh_tab_save_paths", "tab-2", snapshot),
        call("host", "refresh_tab_interaction", "tab-2", snapshot),
        call("host", "view_tab_ids"),
        call("ctrl", "get_tab_snapshot", "tab-1"),
        call("host", "refresh_tab_writeback", "tab-1", snapshot),
        call("host", "refresh_tab_interaction", "tab-1", snapshot),
        call("ctrl", "get_tab_snapshot", "tab-2"),
        call("host", "refresh_tab_writeback", "tab-2", snapshot),
        call("host", "refresh_tab_interaction", "tab-2", snapshot),
    ]
