"""Tests for MainWindowEventCoordinator payload routing."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QWidget
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabAddedPayload,
    TabContentChangedPayload,
    TabContentFact,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.ui.main_window_events import (
    _CONTENT_REACTIONS,
    _INTERACTION_REACTIONS,
    MainWindowEventCoordinator,
    _validate_reaction_matrix,
)
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

    def select_tab_widget(self, tab_id: str) -> None:
        self._log.add("host", "select_tab_widget", tab_id)

    def has_tab_widget(self, tab_id: str) -> bool:
        self._log.add("host", "has_tab_widget", tab_id)
        return tab_id in self.tab_ids

    def view_tab_ids(self) -> list[str]:
        self._log.add("host", "view_tab_ids")
        return list(self.tab_ids)

    def focus_run_result_panel(self, tab_id: str) -> None:
        self._log.add("host", "focus_run_result_panel", tab_id)

    def refresh_tab_cfg(self, tab_id: str) -> None:
        self._log.add("host", "refresh_tab_cfg", tab_id)

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

    def refresh_tab_figure(self, tab_id: str, snapshot: object | None = None) -> None:
        self._log.add("host", "refresh_tab_figure", tab_id, snapshot)

    def refresh_tab_post_figure(
        self, tab_id: str, snapshot: object | None = None
    ) -> None:
        self._log.add("host", "refresh_tab_post_figure", tab_id, snapshot)

    def clear_tab_plot(self, tab_id: str) -> None:
        self._log.add("host", "clear_tab_plot", tab_id)

    def refresh_tab_roles(self, tab_id: str, snapshot: object) -> None:
        self._log.add("host", "refresh_tab_roles", tab_id, snapshot)

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

    def handle_save_data_finished(self, payload: object) -> None:
        self._log.add("host", "handle_save_data_finished", payload)


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


def test_reaction_matrices_cover_every_closed_fact_exactly() -> None:
    assert set(_INTERACTION_REACTIONS) == set(TabInteractionFact)
    assert set(_CONTENT_REACTIONS) == set(TabContentFact)


def test_reaction_matrix_validator_rejects_missing_and_extra_keys() -> None:
    missing = dict(_INTERACTION_REACTIONS)
    missing.pop(TabInteractionFact.SAVE_FAILED)
    with pytest.raises(
        RuntimeError,
        match=r"missing=\['save_failed'\], extra=\[\]",
    ):
        _validate_reaction_matrix("test", TabInteractionFact, missing)

    extra = cast(Any, dict(_CONTENT_REACTIONS))
    extra["unexpected"] = ()
    with pytest.raises(
        RuntimeError,
        match=r"missing=\[\], extra=\[\"'unexpected'\"\]",
    ):
        _validate_reaction_matrix("test", TabContentFact, extra)


@pytest.mark.parametrize(
    ("fact", "reaction_names"),
    [
        (
            TabInteractionFact.RUN_START_REJECTED,
            [
                "refresh_tab_analyze_form",
                "refresh_tab_post_analyze_form",
                "refresh_tab_writeback",
                "refresh_tab_interaction",
            ],
        ),
        (
            TabInteractionFact.PRIMARY_ANALYZE_STARTED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.PRIMARY_ANALYZE_FAILED,
            [
                "refresh_tab_interaction",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
            ],
        ),
        (
            TabInteractionFact.PRIMARY_ANALYZE_CANCELLED,
            [
                "refresh_tab_interaction",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
            ],
        ),
        (
            TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED,
            [
                "refresh_tab_interaction",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
            ],
        ),
        (
            TabInteractionFact.POST_ANALYZE_STARTED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.POST_ANALYZE_SUCCEEDED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.POST_ANALYZE_FAILED,
            [
                "refresh_tab_interaction",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
            ],
        ),
        (
            TabInteractionFact.POST_ANALYZE_START_REJECTED,
            [
                "refresh_tab_interaction",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
            ],
        ),
        (
            TabInteractionFact.SAVE_STARTED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.SAVE_SUCCEEDED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.SAVE_FAILED,
            ["refresh_tab_interaction"],
        ),
        (
            TabInteractionFact.WRITEBACK_DRAFT_CHANGED,
            ["refresh_tab_writeback"],
        ),
    ],
)
def test_interaction_fact_reaction_matrix(
    fact: TabInteractionFact,
    reaction_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot = _snapshot(has_analyze_result=True, has_post_analyze_result=True)
    coordinator, log, _ctrl, _host = _coordinator(snapshot)
    bus = BaseEventBus()

    try:
        coordinator.bind(bus)
        bus.emit(TabInteractionChangedPayload("tab-1", fact))

        assert log.calls[0:2] == [
            call("host", "has_tab_widget", "tab-1"),
            call("ctrl", "get_tab_snapshot", "tab-1"),
        ]
        assert [entry.method for entry in log.calls[2:]] == reaction_names
        assert sum(entry.method == "get_tab_snapshot" for entry in log.calls) == 1
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "fact",
    [
        TabInteractionFact.ANALYZE_PARAMS_CHANGED,
        TabInteractionFact.POST_ANALYZE_PARAMS_CHANGED,
        TabInteractionFact.SAVE_PATHS_CHANGED,
    ],
)
def test_local_edit_facts_have_zero_qt_reaction(
    fact: TabInteractionFact, caplog: pytest.LogCaptureFixture
) -> None:
    coordinator, log, _ctrl, _host = _coordinator()
    bus = BaseEventBus()

    try:
        coordinator.bind(bus)
        bus.emit(TabInteractionChangedPayload("tab-1", fact))

        assert log.calls == []
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    ("fact", "reaction_names"),
    [
        (
            TabContentFact.RUN_RESULT_COMMITTED,
            [
                "refresh_tab_analyze_form",
                "refresh_tab_post_analyze_form",
                "refresh_tab_writeback",
                "refresh_tab_interaction",
            ],
        ),
        (
            TabContentFact.LOADED_RESULT_COMMITTED,
            [
                "clear_tab_plot",
                "refresh_tab_analyze_form",
                "refresh_tab_post_analyze_form",
                "refresh_tab_writeback",
                "refresh_tab_interaction",
            ],
        ),
        (
            TabContentFact.PRIMARY_ANALYSIS_COMMITTED,
            [
                "refresh_tab_post_analyze_form",
                "refresh_tab_writeback",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
                "refresh_tab_interaction",
            ],
        ),
        (
            TabContentFact.POST_ANALYSIS_COMMITTED,
            [
                "refresh_tab_writeback",
                "refresh_tab_figure",
                "refresh_tab_post_figure",
                "refresh_tab_interaction",
            ],
        ),
    ],
)
def test_content_fact_reaction_matrix(
    fact: TabContentFact,
    reaction_names: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot = _snapshot()
    coordinator, log, _ctrl, _host = _coordinator(snapshot)
    bus = BaseEventBus()

    try:
        coordinator.bind(bus)
        bus.emit(TabContentChangedPayload("tab-1", fact))

        assert log.calls[0:2] == [
            call("host", "has_tab_widget", "tab-1"),
            call("ctrl", "get_tab_snapshot", "tab-1"),
        ]
        assert [entry.method for entry in log.calls[2:]] == reaction_names
        assert sum(entry.method == "get_tab_snapshot" for entry in log.calls) == 1
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        coordinator.close()


def test_tab_content_changed_ignores_missing_view_tab(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coordinator, log, _ctrl, host = _coordinator()
    host.tab_ids = []
    bus = BaseEventBus()

    try:
        coordinator.bind(bus)
        bus.emit(TabContentChangedPayload("tab-1", TabContentFact.RUN_RESULT_COMMITTED))

        assert log.calls == [
            call("host", "has_tab_widget", "tab-1"),
        ]
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        coordinator.close()


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


def test_run_finished_refreshes_without_focus_or_navigation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot = _snapshot()
    coordinator, log, ctrl, _host = _coordinator(snapshot)
    bus = BaseEventBus()

    try:
        coordinator.bind(bus)
        # Terminal events are interpreted against the State-owned running identity,
        # not against a cached/payload-derived marker value. Neither outcome may
        # trigger a pane-selection reaction.
        ctrl.running_tab_id = None
        bus.emit(RunFinishedPayload("tab-1", outcome="cancelled"))
        ctrl.running_tab_id = None
        bus.emit(RunFinishedPayload("tab-1", outcome="finished"))

        assert log.calls == [
            call("host", "has_tab_widget", "tab-1"),
            call("ctrl", "get_tab_snapshot", "tab-1"),
            call("host", "refresh_tab_interaction", "tab-1", snapshot),
            call("ctrl", "get_running_tab_id"),
            call("host", "refresh_run_lock", None),
            call("host", "has_tab_widget", "tab-1"),
            call("ctrl", "get_tab_snapshot", "tab-1"),
            call("host", "refresh_tab_interaction", "tab-1", snapshot),
            call("ctrl", "get_running_tab_id"),
            call("host", "refresh_run_lock", None),
        ]
        assert all(entry.method != "focus_run_result_panel" for entry in log.calls)
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        coordinator.close()


def test_run_finished_event_preserves_each_selected_subtab(
    qapp, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from qtpy.QtWidgets import QTabWidget
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.measure.adapter.types import AdapterGuide
    from zcu_tools.gui.app.measure.services import (
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabSnapshot,
    )
    from zcu_tools.gui.app.measure.state import TabInteractionState
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.control_identity import ControlIdentity

    from tests.gui.app.measure._form_fakes import configure_form_lookup
    from tests.gui.app.measure.artifacts._snapshots import with_artifacts

    ctrl = MagicMock()
    bus = BaseEventBus()
    control_identity = ControlIdentity()
    ctrl.control_identity = control_identity
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.active_operation_count.return_value = 0
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = with_artifacts(
        TabSnapshot(
            adapter_name="fake",
            cfg_input={},
            tab_id="tab-1",
            capabilities=AdapterCapabilities(
                analysis=AnalysisMode.FIT, post_analysis=True
            ),
            interaction=TabInteractionState(
                global_run_active=False,
                is_running=False,
                is_analyzing=False,
                has_context=True,
                has_active_context=True,
                has_soc=True,
                has_run_result=True,
                has_analyze_result=False,
                has_post_analyze_result=False,
                has_figure=False,
            ),
            run=RunPaneSnapshot(result=object(), source_path=None),
            save=SavePaneSnapshot(),
        )
    )
    configure_form_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = AdapterGuide(markdown="", resource_dir=None)
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = tab.findChild(QTabWidget)
        assert left_tabs is not None
        assert left_tabs.count() == 5
        focus_result_panel = MagicMock(wraps=tab.focus_result_panel)
        monkeypatch.setattr(tab, "focus_result_panel", focus_result_panel)

        for index in range(left_tabs.count()):
            left_tabs.setCurrentIndex(index)
            bus.emit(RunFinishedPayload("tab-1", outcome="finished"))
            assert left_tabs.currentIndex() == index

        focus_result_panel.assert_not_called()
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        try:
            window.remove_tab_widget("tab-1")
        finally:
            window.deleteLater()
            qapp.processEvents()


def test_run_started_refreshes_invalidated_content_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot = _snapshot(is_running=True)
    coordinator, log, _ctrl, _host = _coordinator(snapshot)
    bus = BaseEventBus()

    try:
        coordinator.bind(bus)
        bus.emit(RunStartedPayload("tab-1"))

        assert log.calls == [
            call("host", "has_tab_widget", "tab-1"),
            call("ctrl", "get_tab_snapshot", "tab-1"),
            call("host", "refresh_tab_analyze_form", "tab-1", snapshot),
            call("host", "refresh_tab_post_analyze_form", "tab-1", snapshot),
            call("host", "refresh_tab_writeback", "tab-1", snapshot),
            call("host", "refresh_tab_interaction", "tab-1", snapshot),
            call("ctrl", "get_running_tab_id"),
            call("host", "refresh_run_lock", "running-tab"),
        ]
        assert all(record.levelno < logging.ERROR for record in caplog.records)
    finally:
        coordinator.close()


def test_main_window_run_marker_tracks_state_and_clears(qapp) -> None:
    from qtpy.QtWidgets import QTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = BaseEventBus()
    ctrl.get_running_tab_id.return_value = "tab-a"
    ctrl.get_active_tab_id.return_value = None
    ctrl.active_operation_count.return_value = 0
    ctrl.has_tab.return_value = True

    class _TabStub(QWidget):
        def update_interaction_state(self, _snapshot: object) -> None:
            pass

    window = MainWindow(ctrl, templates=lambda: {})
    tab_a = _TabStub()
    tab_b = _TabStub()
    window._tab_widgets["tab-a"] = tab_a  # type: ignore[assignment]
    window._tab_widgets["tab-b"] = tab_b  # type: ignore[assignment]
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    tabs.addTab(tab_a, "A")
    tabs.addTab(tab_b, "B")

    window.refresh_run_lock(ctrl.get_running_tab_id.return_value)

    tab_bar = tabs.tabBar()
    assert tab_bar is not None
    assert tabs.tabText(0) == "● A"
    assert tabs.tabText(1) == "B"
    assert tab_bar.tabTextColor(0).name() == "#286ac7"
    assert tab_bar.tabToolTip(0) == "A — Run in progress"

    ctrl.get_running_tab_id.return_value = None
    window.refresh_run_lock(ctrl.get_running_tab_id.return_value)

    assert tabs.tabText(0) == "A"
    assert tabs.tabText(1) == "B"
    assert tab_bar.tabToolTip(0) == "A"


def test_context_event_refreshes_interaction_without_writeback() -> None:
    snapshot = _snapshot()
    coordinator, log, _ctrl, _host = _coordinator(snapshot)
    bus = BaseEventBus()
    coordinator.bind(bus)
    bus.emit(ContextSwitchedPayload(None, None))
    coordinator.close()

    assert log.calls == [
        call("host", "refresh_context_panel"),
        call("host", "view_tab_ids"),
        call("ctrl", "get_tab_snapshot", "tab-1"),
        call("host", "refresh_tab_interaction", "tab-1", snapshot),
        call("ctrl", "get_tab_snapshot", "tab-2"),
        call("host", "refresh_tab_interaction", "tab-2", snapshot),
    ]


def test_ml_changed_has_no_main_window_reaction() -> None:
    coordinator, log, _ctrl, _host = _coordinator()
    bus = BaseEventBus()
    coordinator.bind(bus)

    bus.emit(MlChangedPayload(binding_id=1, content_changed=True, fault=None))

    assert log.calls == []


def test_save_data_finished_payload_is_routed_with_exact_payload() -> None:
    from zcu_tools.gui.app.measure.events.completion import SaveDataFinishedPayload

    coordinator, log, _ctrl, _host = _coordinator()
    bus = BaseEventBus()
    coordinator.bind(bus)
    payload = SaveDataFinishedPayload(tab_id="tab-1", data_path="/tmp/a.h5", error=None)
    bus.emit(payload)
    assert log.calls == [call("host", "handle_save_data_finished", payload)]
    # Verify error propagation is routed unchanged
    log.calls.clear()
    payload_failed = SaveDataFinishedPayload(
        tab_id="tab-1", data_path="/tmp/b.h5", error="disk full"
    )
    bus.emit(payload_failed)
    assert log.calls == [call("host", "handle_save_data_finished", payload_failed)]
