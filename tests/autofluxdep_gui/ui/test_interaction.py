"""Headless interaction tests for the autofluxdep-gui prototype.

Drives the window through the Controller / NodeListPane (not real dialogs) and
asserts the UI reflects State and the edit↔run switch. The run uses FAKE
measurement Nodes (deterministic produce, no acquire): these tests exercise the
UI's run wiring (lock → fill → unlock, build canvases, auto-follow), not the
experiment physics — the real-acquire path is covered by the ``test_*_acquire``
integration tests. The run worker is a real QThread driven to completion via the
Qt event loop. A second ad-hoc provider (no Result → no liveplot) gives the list
two rows for reorder/remove.
"""

from __future__ import annotations

import threading
import time
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QObject, Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QCloseEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QMessageBox,
    QSizePolicy,
    QWidget,
)
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import DirectValue, EvalValue
from zcu_tools.gui.app.autofluxdep.controller import FLUX_PROGRESS_LABEL
from zcu_tools.gui.app.autofluxdep.events.run import (
    NodeEnteredPayload,
    PointDonePayload,
    RunFailedPayload,
    RunFinishedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow, _RunBridge
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    MdChangedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SessionEvent,
    SocChangedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationOutcome

from .._helpers import connect_mock, make_builder, make_measurement_builder


@pytest.fixture
def app(qapp):
    ctrl = build_core()
    ctrl.add_node(make_measurement_builder("qubit_freq"))
    # a second provider (ad-hoc, no Result → no liveplot) so the list has 2 rows
    ctrl.add_node(make_builder("probe", provides=("v",)))
    win = MainWindow(ctrl)
    yield ctrl, win
    # Quiesce the controller's BackgroundRunner before GC: device setup and
    # other session operations submit pool workers via the runner;
    # a pending queued delivery to a GC'd runner segfaults a later test's event pump.
    ctrl._background_svc.quiesce()
    win.close()
    win.deleteLater()


def _list_labels(win: MainWindow) -> list[str]:
    lst = win._list._list
    labels: list[str] = []
    for row in range(lst.count()):
        item = lst.item(row)
        assert item is not None
        widget = lst.itemWidget(item)
        assert widget is not None
        labels.append(cast(Any, widget)._label.text())
    return labels


def _node_checkbox(win: MainWindow, row: int):
    item = win._list._list.item(row)
    assert item is not None
    widget = win._list._list.itemWidget(item)
    assert widget is not None
    return cast(Any, widget)._checkbox


def test_node_list_item_text_is_owned_by_row_widget(app):
    _ctrl, win = app
    lst = win._list._list

    for row, expected in enumerate(("qubit_freq", "probe")):
        item = lst.item(row)
        assert item is not None
        widget = lst.itemWidget(item)
        assert widget is not None

        assert item.text() == ""
        assert item.toolTip() == expected
        assert item.data(Qt.ItemDataRole.UserRole) == expected  # type: ignore[attr-defined]
        assert widget.toolTip() == expected
        assert cast(Any, widget)._label.toolTip() == expected
        assert cast(Any, widget)._label.text() == expected


def _spin_until(condition, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.001)
    return False


class _RowReceiver(QObject):
    def __init__(self, bridge: _RunBridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.rows: list[tuple[str, int]] = []

    def on_row(self, name: str, idx: int, _emitted_at: float) -> None:
        self.rows.append((name, idx))
        self.bridge.row_rendered(name, idx)


def test_run_bridge_coalesces_redundant_row_updates(qapp):
    ctrl = build_core()
    bridge = _RunBridge(ctrl)
    receiver = _RowReceiver(bridge)
    bridge.row_updated.connect(receiver.on_row)

    def worker_emit() -> None:
        bridge.notify("node", 0)
        bridge.notify("node", 0)
        bridge.notify("node", 0)

    thread = threading.Thread(target=worker_emit)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    deadline = time.monotonic() + 1.0
    while len(receiver.rows) < 2 and time.monotonic() < deadline:
        qapp.processEvents()

    assert receiver.rows == [("node", 0), ("node", 0)]
    bridge.teardown()


def test_run_bridge_teardown_removes_event_bus_subscriptions(qapp):
    ctrl = build_core()
    bridge = _RunBridge(ctrl)

    for payload_type in (
        RunStartedPayload,
        NodeEnteredPayload,
        PointDonePayload,
        RunFinishedPayload,
        RunStoppedPayload,
        RunFailedPayload,
    ):
        assert ctrl.bus._subs.get(payload_type)

    bridge.teardown()

    for payload_type in (
        RunStartedPayload,
        NodeEnteredPayload,
        PointDonePayload,
        RunFinishedPayload,
        RunStoppedPayload,
        RunFailedPayload,
    ):
        assert ctrl.bus._subs.get(payload_type, []) == []
    bridge.deleteLater()


# --- workflow editing reflects in the list ---


def test_list_reflects_nodes(app):
    _ctrl, win = app
    assert _list_labels(win) == ["qubit_freq", "probe"]
    assert _node_checkbox(win, 0).isChecked()
    assert _node_checkbox(win, 1).isChecked()


def test_node_enable_checkbox_toggles_controller_state(app):
    ctrl, win = app
    checkbox = _node_checkbox(win, 1)

    checkbox.setChecked(False)

    assert ctrl.state.nodes[1].enabled is False
    assert not checkbox.isChecked()

    checkbox.setChecked(True)

    assert ctrl.state.nodes[1].enabled is True
    assert checkbox.isChecked()


def test_node_enable_checkbox_locks_while_running(app):
    _ctrl, win = app
    checkbox = _node_checkbox(win, 0)

    assert checkbox.isEnabled()
    win._list.set_running(True)
    assert not checkbox.isEnabled()
    win._list.set_running(False)
    assert checkbox.isEnabled()


def test_reorder_swaps_and_keeps_selection(app):
    ctrl, win = app
    win._list.select_index(1)  # probe
    win._list._on_move(-1)  # move up
    assert _list_labels(win) == ["probe", "qubit_freq"]
    assert ctrl.state.node_names() == ["probe", "qubit_freq"]


def test_remove_node(app):
    ctrl, win = app
    win._list.select_index(0)
    win._list._on_remove()
    assert _list_labels(win) == ["probe"]
    assert ctrl.state.node_names() == ["probe"]


def test_devices_button_opens_shared_device_dialog(app):
    # the Devices… button opens the shared session DeviceDialog, which depends
    # only on the device-control facet exposed by the autofluxdep Controller.
    from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

    ctrl, win = app
    assert hasattr(win, "_devices_btn")
    dlg = DeviceDialog(
        ctrl.device_control,
        md_provider=ctrl.context_control.get_current_md,
        parent=win,
    )
    assert dlg._dev is ctrl.device_control
    dlg.deleteLater()


def test_session_status_and_buttons_share_top_row(app):
    # Keep the global session controls dense: status labels on the left, action
    # buttons on the right, all in the first row.
    _ctrl, win = app
    central = win.centralWidget()
    assert central is not None
    main_layout = central.layout()
    assert main_layout is not None
    first_item = main_layout.itemAt(0)
    assert first_item is not None
    row = first_item.layout()
    assert row is not None

    widgets = []
    for idx in range(row.count()):
        item = row.itemAt(idx)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widgets.append(widget)

    assert win._ctx_label in widgets
    assert win._setup_label in widgets
    assert win._predictor_label in widgets
    assert win._setup_btn in widgets
    assert win._devices_btn in widgets
    assert win._predictor_btn in widgets
    assert win._inspect_btn in widgets


def test_progress_bar_is_central_full_width_row(app):
    _ctrl, win = app
    central = win.centralWidget()
    assert central is not None
    main_layout = central.layout()
    assert main_layout is not None

    progress_item = main_layout.itemAt(main_layout.count() - 1)
    assert progress_item is not None
    assert progress_item.widget() is win._progress
    assert win._progress.parentWidget() is central
    assert win._progress.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    win.resize(900, 600)
    win.show()
    QApplication.processEvents()
    assert win._progress.width() > 0
    assert abs(win._progress.width() - central.width()) <= 16


def test_run_progress_updates_flux_bar_and_round_stack(app, monkeypatch):
    from zcu_tools.gui.session.pbar_host import ProgressBarModel

    ctrl, win = app
    flux = ProgressBarModel(FLUX_PROGRESS_LABEL, 10, time.monotonic() - 10.0)
    flux.set_n(2)
    rounds = ProgressBarModel("t1 flux 1 rounds", 100, 0.0)
    captured: list[tuple[ProgressBarModel, ...]] = []

    monkeypatch.setattr(
        ctrl.progress_control,
        "progress_bars",
        lambda _owner: ((1, flux), (2, rounds)),
    )
    monkeypatch.setattr(win._round_progress, "render_models", captured.append)

    win._on_run_progress_changed()

    assert win._progress.maximum() == flux.qt_maximum()
    assert win._progress.value() == flux.qt_value()
    assert FLUX_PROGRESS_LABEL in win._progress.format()
    assert "%v/%m" in win._progress.format()
    assert "[" in win._progress.format()
    assert "<" in win._progress.format()
    assert captured == [(rounds,)]


def test_flux_source_picker_records_selection(app):
    # the flux-source combo records which connected device the sweep is applied
    # through (its unit labels the flux axis); with no devices it is just "(none)".
    ctrl, win = app
    assert hasattr(win._list, "_flux_source")
    assert win._list._flux_source.count() == 1  # only "(none)"
    assert ctrl.get_flux_device() is None
    ctrl.set_flux_device("flux_dev")
    assert ctrl.get_flux_device() == "flux_dev"
    ctrl.set_flux_device(None)  # clearing → bare flux numbers
    assert ctrl.get_flux_device() is None


def test_flux_sweep_fields_accept_numeric_expressions(app):
    ctrl, win = app
    md = ctrl.get_current_md()
    md.span = 0.004
    md.count = 2

    win._list._flux_start.setText("span / 2")
    win._list._flux_stop.setText("-span / 2")
    win._list._flux_npts.setText("2 * count + 1")

    assert isinstance(win._list._flux_start._field.get_value(), EvalValue)
    ghost = win._list._flux_start._widget._ghost
    assert ghost is not None
    assert ghost.text() == "= 0.002"

    win._list._commit_flux()

    assert ctrl.state.flux_values == pytest.approx([0.002, 0.001, 0.0, -0.001, -0.002])


def test_flux_sweep_rejects_non_integer_npts_expression(app):
    _ctrl, win = app
    win._list._flux_start.setText("0.0")
    win._list._flux_stop.setText("1.0")
    win._list._flux_npts.setText("2.5")

    with pytest.raises(RuntimeError, match="not an integer"):
        win._list._commit_flux()


def test_flux_sweep_rejects_non_positive_npts(app):
    _ctrl, win = app
    win._list._flux_start.setText("0.0")
    win._list._flux_stop.setText("1.0")
    win._list._flux_npts.setText("0")

    with pytest.raises(RuntimeError, match="at least 1"):
        win._list._commit_flux()


@pytest.mark.parametrize("expr", ["missing", ""])
def test_flux_sweep_direct_fallback_from_invalid_or_empty_expression_is_not_empty(
    app, expr: str
):
    ctrl, win = app
    win._list._flux_start.setText(expr)
    assert isinstance(win._list._flux_start._field.get_value(), EvalValue)
    ghost = win._list._flux_start._widget._ghost
    assert ghost is not None
    assert ghost.text() == "= ?"

    # Mirrors ScalarWidget's unresolved EvalValue -> direct-mode branch.
    win._list._flux_start._field.set_value(None)
    win._list._flux_stop.setText("1.0")
    win._list._flux_npts.setText("2")
    win._list._commit_flux()

    assert isinstance(win._list._flux_start._field.get_value(), DirectValue)
    assert ctrl.state.flux_values == pytest.approx([0.0, 1.0])


def test_node_list_teardown_is_idempotent_and_called_by_close_event(app):
    _ctrl, win = app

    win._list.teardown()
    win._list.teardown()

    win.closeEvent(QCloseEvent())
    QApplication.processEvents()
    win.closeEvent(QCloseEvent())

    assert win._list._torn_down is True


def test_main_window_close_flushes_persistence(app):
    ctrl, win = app
    persist_all = MagicMock()
    ctrl.persist_all = persist_all  # type: ignore[method-assign]

    win.closeEvent(QCloseEvent())
    QApplication.processEvents()

    persist_all.assert_called_once_with()


def test_main_window_close_removes_event_bus_subscriptions(app):
    ctrl, win = app

    def has_owner(payload_type: type[Any], owner: object) -> bool:
        return any(
            getattr(callback, "__self__", None) is owner
            for callback in ctrl.bus._subs.get(payload_type, [])
        )

    for payload_type in (
        SocChangedPayload,
        DeviceChangedPayload,
        PredictorChangedPayload,
        ContextSwitchedPayload,
        MdChangedPayload,
        MlChangedPayload,
    ):
        assert has_owner(payload_type, win)

    for payload_type in (
        RunStartedPayload,
        NodeEnteredPayload,
        PointDonePayload,
        RunFinishedPayload,
        RunStoppedPayload,
        RunFailedPayload,
    ):
        assert has_owner(payload_type, win._bridge)

    win.closeEvent(QCloseEvent())
    QApplication.processEvents()

    for payload_type in (
        SocChangedPayload,
        DeviceChangedPayload,
        PredictorChangedPayload,
        ContextSwitchedPayload,
        MdChangedPayload,
        MlChangedPayload,
    ):
        assert not has_owner(payload_type, win)

    for payload_type in (
        RunStartedPayload,
        NodeEnteredPayload,
        PointDonePayload,
        RunFinishedPayload,
        RunStoppedPayload,
        RunFailedPayload,
    ):
        assert not has_owner(payload_type, win._bridge)


def test_main_window_close_waits_for_live_operation(app, monkeypatch):
    ctrl, win = app
    cancelled: list[bool] = []
    token = ctrl._operation_handles.create(cancel_hook=lambda: cancelled.append(True))
    persist_all = MagicMock()
    ctrl.persist_all = persist_all  # type: ignore[method-assign]
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    event = QCloseEvent()
    win.closeEvent(event)

    assert not event.isAccepted()
    QApplication.processEvents()
    assert cancelled == [True]
    persist_all.assert_not_called()

    ctrl._operation_handles.settle(token, OperationOutcome("cancelled"))
    assert _spin_until(lambda: persist_all.called)
    assert ctrl.active_operation_count() == 0


def test_running_close_requests_terminal_stop_then_closes_after_run_done(
    app, monkeypatch
):
    ctrl, win = app
    stop_reasons: list[str] = []
    perform_close = MagicMock()
    token = ctrl._operation_handles.create(
        cancel_hook=lambda: stop_reasons.append("cancelled")
    )
    ctrl._active_run_token = token
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(win, "_perform_close", perform_close)

    event = QCloseEvent()
    win.closeEvent(event)

    assert not event.isAccepted()
    assert stop_reasons == ["cancelled"]
    assert win._close_after_run_terminal is True
    perform_close.assert_not_called()

    win._on_run_done()
    QApplication.processEvents()

    assert win._close_after_run_terminal is False
    perform_close.assert_called_once_with()
    ctrl._active_run_token = None
    ctrl._operation_handles.settle(token, OperationOutcome("cancelled"))
    win._closing = True


def test_paused_close_finalizes_stopped_then_closes(app, monkeypatch):
    from zcu_tools.gui.app.autofluxdep.controller import Controller

    ctrl, win = app
    finalize = MagicMock()
    perform_close = MagicMock()
    monkeypatch.setattr(Controller, "is_paused", property(lambda _self: True))
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(ctrl, "finalize_paused_run_as_stopped", finalize)
    monkeypatch.setattr(win, "_perform_close", perform_close)

    event = QCloseEvent()
    win.closeEvent(event)

    assert not event.isAccepted()
    finalize.assert_called_once_with()
    perform_close.assert_called_once_with()
    win._closing = True


def test_force_close_prompt_only_closes_when_force_button_clicked(app, monkeypatch):
    ctrl, win = app
    perform_close = MagicMock()
    buttons: list[object] = []
    token = ctrl._operation_handles.create(cancel_hook=lambda: None)
    ctrl._active_run_token = token
    win._close_after_run_terminal = True

    def fake_add_button(self, *args):
        del self, args
        button = object()
        buttons.append(button)
        return button

    monkeypatch.setattr(win, "_perform_close", perform_close)
    monkeypatch.setattr(QMessageBox, "addButton", fake_add_button)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: buttons[0])

    win._show_force_close_prompt()

    perform_close.assert_called_once_with()
    assert win._force_close_prompt_open is False
    ctrl._active_run_token = None
    ctrl._operation_handles.settle(token, OperationOutcome("cancelled"))
    win._closing = True


def test_predictor_button_opens_shared_predictor_dialog(app):
    # the Predictor… button opens the shared session PredictorDialog (load a
    # FluxoniumPredictor into the context) — runtime port conformance check.
    from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

    ctrl, win = app
    assert hasattr(win, "_predictor_btn")
    dlg = PredictorDialog(ctrl.predictor_control, win)  # must not raise
    assert dlg._pred is ctrl.predictor_control
    dlg.deleteLater()


# --- Inspect… mounts the shared (non-modal) context inspector ---


def test_inspect_button_opens_single_non_modal_inspector(app):
    from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase

    ctrl, win = app
    assert hasattr(win, "_inspect_btn")
    assert win._dialog_refs.get("inspect") is None

    win._inspect_btn.click()  # the toolbar button's slot
    dlg = win._dialog_refs.get("inspect")
    assert dlg is not None
    assert isinstance(dlg, InspectDialogBase)
    # Opened via ``open()`` (non-blocking, so the run worker's event loop keeps
    # pumping) rather than ``exec()``.
    assert dlg.isVisible()

    # A second request raises the same instance, never a second dialog.
    win._inspect_btn.click()
    assert win._dialog_refs.get("inspect") is dlg

    dlg.reject()  # closing drops the reference so the next request rebuilds
    assert win._dialog_refs.get("inspect") is None


def test_inspect_button_stays_enabled_during_run(app):
    # The inspector reflects the live context; it must remain reachable mid-run
    # (unlike the workflow-editing buttons, which lock while running).
    ctrl, win = app
    win._list.set_running(True)
    win._refresh_toolbar_buttons()
    assert win._inspect_btn.isEnabled()
    win._list.set_running(False)


def test_inspect_dialog_is_read_only_while_run_is_active(app):
    ctrl, win = app

    win._inspect_btn.click()
    dlg = cast(Any, win._dialog_refs.get("inspect"))
    assert dlg is not None

    dlg._edit_key.setText("r_f")
    assert dlg._edit_key.isEnabled()
    assert dlg._edit_value.isEnabled()
    assert dlg._set_btn.isEnabled()
    assert dlg._delete_btn.isEnabled()

    win._on_run_started()

    assert not dlg._edit_key.isEnabled()
    assert not dlg._edit_value.isEnabled()
    assert not dlg._set_btn.isEnabled()
    assert not dlg._delete_btn.isEnabled()

    win._on_run_done()

    assert dlg._edit_key.isEnabled()
    assert dlg._edit_value.isEnabled()
    dlg.reject()


def test_inspect_dialog_is_read_only_while_run_is_paused(app):
    _ctrl, win = app

    win._inspect_btn.click()
    dlg = cast(Any, win._dialog_refs.get("inspect"))
    assert dlg is not None
    dlg._edit_key.setText("r_f")

    win._on_run_paused(1)

    assert not dlg._set_btn.isEnabled()
    assert not dlg._delete_btn.isEnabled()

    win._on_run_done()
    assert dlg._set_btn.isEnabled()
    dlg.reject()


def test_run_start_closes_open_setup_and_device_dialogs(app):
    _ctrl, win = app

    win.open_setup_dialog(startup_mode=False)
    win._on_devices_clicked()
    assert win._dialog_refs.get("setup") is not None
    assert win._dialog_refs.get("devices") is not None

    win._on_run_started()
    QApplication.processEvents()

    assert win._dialog_refs.get("setup") is None
    assert win._dialog_refs.get("devices") is None


# --- selection drives the right pane ---


def test_selection_shows_node_form(app):
    _ctrl, win = app
    win._list.select_index(0)
    assert win._detail._title.text() == "qubit_freq"
    assert win._detail.current_form is not None
    win._list.select_index(1)
    assert win._detail._title.text() == "probe"


def test_session_events_refresh_current_cfg_editor(app, monkeypatch):
    ctrl, win = app
    form = win._detail.current_form
    assert form is not None
    seen: list[object] = []
    monkeypatch.setattr(form, "refresh_external", seen.append)

    md = ctrl.get_current_md()
    ml = ctrl.get_current_ml()
    ctrl.bus.emit(MdChangedPayload(md=md))
    ctrl.bus.emit(MlChangedPayload(ml=ml))
    ctrl.bus.emit(ContextSwitchedPayload(md=md, ml=ml))
    ctrl.bus.emit(DeviceChangedPayload(name="flux_dev"))

    assert seen == [
        SessionEvent.MD_CHANGED,
        SessionEvent.ML_CHANGED,
        SessionEvent.CONTEXT_SWITCHED,
        SessionEvent.DEVICE_CHANGED,
    ]


# --- Setup → Run enable ---


def test_run_disabled_until_setup(app):
    ctrl, win = app
    assert not win._list._run_btn.isEnabled()  # no setup yet
    connect_mock(ctrl)
    win._list.refresh_flux_sources()
    # A bare-number flux sweep is legal; Run click commits the editable fields.
    assert win._list._run_btn.isEnabled()
    win._refresh_session_status()
    assert win._setup_label.text() == "connected"


def test_run_disabled_without_nodes(qapp):
    ctrl = build_core()
    win = MainWindow(ctrl)
    connect_mock(ctrl)
    win._list.refresh_flux_sources()

    assert not win._list._run_btn.isEnabled()
    assert win._list._run_btn.toolTip() == "Add at least one node"
    ctrl._background_svc.quiesce()
    win.close()
    win.deleteLater()


def test_run_disabled_when_all_nodes_disabled(app):
    ctrl, win = app
    ctrl.set_node_enabled(0, False)
    ctrl.set_node_enabled(1, False)
    win._list.refresh_list()
    connect_mock(ctrl)
    win._list.refresh_flux_sources()

    assert not win._list._run_btn.isEnabled()
    assert win._list._run_btn.toolTip() == "Enable at least one node"


# --- run lifecycle: edit↔run lock, liveplot canvas, progress ---


def _zero_delays(ctrl):
    # the fake measurement Nodes produce instantly (no acquire / no delay), so a UI
    # run completes within the event pump without any pacing to neutralise. Kept as
    # a no-op so the run helpers read the same with or without a delay seam.
    del ctrl


def _pump_until_done(ctrl, win):
    del win
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if not ctrl.is_running:
            break
        time.sleep(0.001)
    QApplication.processEvents()
    ctrl._background_svc.quiesce()
    assert not ctrl.is_running


def _run_to_completion(ctrl, win):
    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 1.0, 2.0])
    connect_mock(ctrl)
    win._list.refresh_run_availability()
    win._start()
    _pump_until_done(ctrl, win)


def _current_form_node_name(win: MainWindow) -> str:
    form = win._detail.current_form
    assert form is not None
    return form._node.name


def _current_form_editors_enabled(win: MainWindow) -> bool:
    form = win._detail.current_form
    assert form is not None
    default_enabled = form._default_form.isEnabled()
    generation_enabled = (
        True if form._generation_form is None else form._generation_form.isEnabled()
    )
    return default_enabled and generation_enabled


def test_run_locks_then_unlocks(app):
    ctrl, win = app
    win._list.select_index(0)
    _run_to_completion(ctrl, win)
    # back in edit state after finish
    assert win._list._run_btn.text() == "▶ Run"
    assert win._list._add_btn.isEnabled()
    assert win._detail.current_form is not None
    # progress reached the end
    assert win._progress.value() == 3


def test_produce_exception_during_gui_run_does_not_crash_and_unlocks(qapp, monkeypatch):
    # a Node whose produce raises mid-run (e.g. an unconfigured real acquire) must
    # NOT abort the run worker QThread: the orchestrator catches it, the run ends on
    # RUN_FAILED, and the UI unlocks back to edit state. The warning dialog is
    # stubbed so the headless test does not block on a modal.
    from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]
    from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency

    from .._helpers import make_builder

    def boom(env, snapshot):
        del env, snapshot
        raise RuntimeError("node not configured")

    ctrl = build_core()
    ctrl.add_node(
        make_builder("broken", requires=(Dependency("predict_freq"),), produce_fn=boom)
    )
    win = MainWindow(ctrl)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )

    ctrl.set_flux_values([0.0, 1.0])
    connect_mock(ctrl)
    win._list.refresh_run_availability()
    win._start()
    _pump_until_done(ctrl, win)

    # the run ended and the UI is back in edit state
    assert not ctrl.is_running
    assert win._list._run_btn.text() == "▶ Run"
    assert win._list._add_btn.isEnabled()
    # the failure surfaced to the user
    assert any("node not configured" in m for m in shown)
    ctrl._background_svc.quiesce()
    win.close()
    win.deleteLater()


def test_run_start_exception_does_not_escape_qt_slot(app, monkeypatch):
    ctrl, win = app
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("store failed")

    ctrl.set_flux_values([0.0])
    monkeypatch.setattr(ctrl, "start_run", boom)

    win._start()

    assert not ctrl.is_running
    assert win._list._run_btn.text() == "▶ Run"
    assert any("store failed" in message for message in shown)
    assert ctrl.state.run_results == {}
    assert ctrl.state.run_predictor is None


def test_empty_flux_start_failure_warns_and_does_not_create_products(app, monkeypatch):
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor

    ctrl, win = app
    ctrl.set_flux_values([])
    ctrl.state.run_results = {"stale": object()}
    ctrl.state.run_predictor = SimplePredictor()
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )

    win._start()

    assert any("at least one flux point" in message for message in shown)
    assert not ctrl.is_running
    assert win._list._run_btn.text() == "▶ Run"
    assert win._list._add_btn.isEnabled()
    assert ctrl.state.run_results == {}
    assert ctrl.state.run_predictor is None


def test_run_build_plot_exception_clears_stale_products(app, monkeypatch):
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor

    ctrl, win = app
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )
    ctrl.set_flux_values([0.0])

    def fail_after_publish() -> None:
        ctrl.prepare_run_results()
        ctrl.state.run_predictor = SimplePredictor()
        raise RuntimeError("plot build failed")

    monkeypatch.setattr(win, "_build_plots", fail_after_publish)

    win._start()

    assert ctrl.state.run_results == {}
    assert ctrl.state.run_predictor is None
    assert any("plot build failed" in message for message in shown)


def test_row_update_plotter_exception_is_logged_and_coalescing_released(
    app, monkeypatch, caplog
):
    ctrl, win = app
    canvas = QWidget()

    class BrokenPlotter:
        def update(self, _result, _idx) -> None:
            raise RuntimeError("draw failed")

    rendered = MagicMock()
    monkeypatch.setattr(win._bridge, "row_rendered", rendered)
    ctrl.state.run_results = {"qubit_freq": object()}
    win._plots = {"qubit_freq": (canvas, BrokenPlotter())}

    with caplog.at_level("ERROR"):
        win._on_row_updated("qubit_freq", 0, time.monotonic())

    assert win._plots["qubit_freq"][1] is None
    rendered.assert_called_once_with("qubit_freq", 0)
    assert "autofluxdep plot update failed" in caplog.text
    canvas.deleteLater()


def test_run_builds_liveplot_canvas_for_measurement_node(app):
    ctrl, win = app
    win._list.select_index(0)  # qubit_freq has a Result → a canvas is built
    _run_to_completion(ctrl, win)
    # the qubit_freq provider got a sweep-lived canvas + plotter
    assert "qubit_freq" in win._plots
    canvas, plotter = win._plots["qubit_freq"]
    assert canvas is not None and plotter is not None
    # the ad-hoc "probe" provider has no Result → no canvas
    assert "probe" not in win._plots
    # the Result was filled (the worker ran produce over the sweep)
    import numpy as np

    result = ctrl.state.run_results["qubit_freq"]
    assert not np.isnan(result.signal[-1]).any()  # the last row was filled


def test_run_switches_detail_to_run_tab(app):
    ctrl, win = app
    win._list.select_index(0)
    captured = {}

    def on_started():
        captured["tab"] = win._detail.current_tab
        captured["btn"] = win._list._run_btn.text()
        captured["setup_enabled"] = win._setup_btn.isEnabled()
        captured["devices_enabled"] = win._devices_btn.isEnabled()
        captured["predictor_enabled"] = win._predictor_btn.isEnabled()
        captured["inspect_enabled"] = win._inspect_btn.isEnabled()

    win._bridge.run_started.connect(on_started)
    _run_to_completion(ctrl, win)
    # the run_started slot (main thread) switched to the run sub-tab + showed Pause
    assert captured.get("tab") == 1
    assert captured.get("btn") == "⏸ Pause"
    assert captured.get("setup_enabled") is False
    assert captured.get("devices_enabled") is False
    assert captured.get("predictor_enabled") is True
    assert captured.get("inspect_enabled") is True
    assert win._setup_btn.isEnabled()


def test_predictor_dialog_can_open_as_live_view_during_run(app):
    ctrl, win = app
    ctrl.set_flux_values([0.1, 0.2])
    win._on_run_started()

    win._on_predictor_clicked()

    dlg = win._predictor_dialog
    assert dlg is not None
    assert dlg._live_mode
    assert not dlg._apply_btn.isEnabled()
    assert dlg._predict_value_spin.value() == pytest.approx(0.1)

    win._on_node_entered("qubit_freq", 1)

    assert dlg._predict_value_spin.value() == pytest.approx(0.2)

    win._on_run_done()

    assert not dlg._live_mode
    assert dlg._apply_btn.isEnabled()


def test_pause_continue_ui_states_lock_workflow_controls(app):
    _ctrl, win = app

    win._on_run_started()
    assert win._list._run_btn.text() == "⏸ Pause"
    assert not win._list._abort_btn.isHidden()
    assert not win._list._add_btn.isEnabled()

    win._on_run_pause_requested()
    assert win._list._run_btn.text() == "Pausing..."
    assert not win._list._run_btn.isEnabled()
    assert win._list._abort_btn.isEnabled()

    win._on_run_paused(1)
    assert win._list._run_btn.text() == "▶ Continue"
    assert win._list._run_btn.isEnabled()
    assert win._list._abort_btn.isEnabled()
    assert not win._list._add_btn.isEnabled()
    assert not win._list._flux_source.isEnabled()
    assert not win._setup_btn.isEnabled()
    assert not win._devices_btn.isEnabled()
    assert not win._predictor_btn.isEnabled()
    assert win._inspect_btn.isEnabled()

    win._on_run_continued(1)
    assert win._list._run_btn.text() == "⏸ Pause"

    win._on_run_done()
    assert win._list._run_btn.text() == "▶ Run"
    assert win._list._abort_btn.isHidden()
    assert win._list._add_btn.isEnabled()


def test_auto_follow_checkbox_disables_tab_switch_and_navigation(app):
    ctrl, win = app
    win._list.select_index(0)
    win._detail._tabs.setCurrentIndex(0)

    win._list._auto_follow_tabs.setChecked(False)

    assert ctrl.get_auto_follow_tabs() is False
    win._on_run_started()
    assert win._detail.current_tab == 0
    assert win._list._run_btn.text() == "⏸ Pause"

    win._on_node_entered("probe", 0)

    assert win._list.selected_index == 0

    win._detail._tabs.setCurrentIndex(1)
    win._on_run_done()

    assert win._detail.current_tab == 1
    assert win._list._run_btn.text() == "▶ Run"


def test_auto_follow_checkbox_can_turn_off_during_run(app):
    _ctrl, win = app
    win._list.select_index(0)
    win._on_run_started()

    win._list._auto_follow_tabs.setChecked(False)
    win._on_node_entered("probe", 0)

    assert win._list.selected_index == 0
    win._on_run_done()


def test_auto_follow_selection_defers_edit_form_rebuild_while_showing_run_tab(app):
    _ctrl, win = app
    win._list.select_index(0)
    original_form = win._detail.current_form
    assert original_form is not None

    win._on_run_started()
    win._on_node_entered("probe", 0)

    assert win._list.selected_index == 1
    assert win._detail.current_tab == 1
    assert win._detail._title.text() == "probe"
    assert win._detail.current_form is original_form
    assert _current_form_node_name(win) == "qubit_freq"
    win._on_run_done()


def test_deferred_edit_form_materializes_when_user_opens_edit_tab_during_run(app):
    ctrl, win = app
    win._list.select_index(0)
    win._on_run_started()
    win._on_node_entered("probe", 0)
    assert _current_form_node_name(win) == "qubit_freq"

    win._detail._tabs.setCurrentIndex(0)

    assert _current_form_node_name(win) == "probe"
    assert win._detail.current_form is not None
    assert not _current_form_editors_enabled(win)
    assert ctrl.get_auto_follow_tabs() is False
    assert not win._list._auto_follow_tabs.isChecked()
    win._on_run_done()


def test_run_done_materializes_pending_edit_form_after_auto_follow(app):
    _ctrl, win = app
    win._list.select_index(0)
    win._on_run_started()
    win._on_node_entered("probe", 0)
    assert _current_form_node_name(win) == "qubit_freq"

    win._on_run_done()

    assert win._detail.current_tab == 0
    assert _current_form_node_name(win) == "probe"
    assert win._detail.current_form is not None
    assert _current_form_editors_enabled(win)


def test_auto_follow_checkbox_can_turn_on_during_run(app):
    ctrl, win = app
    win._list.select_index(0)
    win._detail._tabs.setCurrentIndex(0)
    win._list._auto_follow_tabs.setChecked(False)
    win._on_run_started()
    win._on_node_entered("probe", 0)

    assert ctrl.get_auto_follow_tabs() is False
    assert win._list.selected_index == 0
    assert win._detail.current_tab == 0

    win._list._auto_follow_tabs.setChecked(True)

    assert ctrl.get_auto_follow_tabs() is True
    assert win._list.selected_index == 1
    assert win._detail.current_tab == 1
    assert _current_form_node_name(win) == "qubit_freq"
    win._on_run_done()
    assert _current_form_node_name(win) == "probe"


def test_manual_node_switch_during_run_disables_auto_follow(app):
    ctrl, win = app
    win._list.select_index(0)
    win._on_run_started()

    win._list.select_index(1)

    assert ctrl.get_auto_follow_tabs() is False
    assert not win._list._auto_follow_tabs.isChecked()
    assert _current_form_node_name(win) == "probe"
    assert win._detail.current_form is not None
    assert not _current_form_editors_enabled(win)
    win._on_node_entered("qubit_freq", 0)
    assert win._list.selected_index == 1
    win._on_run_done()


def test_manual_node_switch_during_run_reuses_materialized_forms(app):
    _ctrl, win = app
    win._list.select_index(0)
    qubit_form = win._detail.current_form
    assert qubit_form is not None
    win._on_run_started()

    win._list.select_index(1)
    probe_form = win._detail.current_form
    assert probe_form is not None
    assert probe_form is not qubit_form
    assert _current_form_node_name(win) == "probe"

    win._list.select_index(0)
    assert win._detail.current_form is qubit_form
    assert _current_form_node_name(win) == "qubit_freq"

    win._list.select_index(1)
    assert win._detail.current_form is probe_form
    assert _current_form_node_name(win) == "probe"
    win._on_run_done()


def test_manual_detail_tab_switch_during_run_disables_auto_follow(app):
    ctrl, win = app
    win._list.select_index(0)
    win._on_run_started()
    assert win._detail.current_tab == 1

    win._detail._tabs.setCurrentIndex(0)

    assert ctrl.get_auto_follow_tabs() is False
    assert not win._list._auto_follow_tabs.isChecked()
    win._on_node_entered("probe", 0)
    assert win._list.selected_index == 0
    win._on_run_done()


def test_multiple_real_experiments_each_get_a_liveplot(qapp):
    # a real multi-experiment workflow: each measurement provider gets its own
    # sweep-lived canvas + plotter, and the LivePlot-backed Plotter redraws on the
    # main thread (every experiment shares the same notify→update→draw wiring).
    ctrl = build_core()
    for t in ("qubit_freq", "t1", "ro_optimize", "mist"):
        ctrl.add_node(make_measurement_builder(t))
    win = MainWindow(ctrl)
    win._list.select_index(1)  # follow t1's plot
    redraws = {"t1": 0}

    def patch_counter():
        if "t1" in win._plots:
            _canvas, plotter = win._plots["t1"]
            orig = plotter.update

            def wrapped(result, idx, _o=orig):
                redraws["t1"] += 1
                _o(result, idx)

            plotter.update = wrapped

    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5, 1.0])
    connect_mock(ctrl)
    win._list.refresh_run_availability()
    win._start()
    patch_counter()  # wrap after _build_plots created the plotter
    _pump_until_done(ctrl, win)

    # every measurement provider built a canvas + plotter (predictor Service none)
    assert set(win._plots) == {"qubit_freq", "t1", "ro_optimize", "mist"}
    for name in win._plots:
        canvas, plotter = win._plots[name]
        assert canvas is not None and plotter is not None
    # t1's Plotter redrew on the main thread as rows filled
    assert redraws["t1"] >= 1
    win.close()
    win.deleteLater()


def test_run_auto_follows_each_entered_node(qapp):
    # as the sweep enters each provider, the left list selects it + the detail
    # pane switches to its run tab (the canvas it shows follows the selection).
    ctrl = build_core()
    for t in ("qubit_freq", "t1", "mist"):
        ctrl.add_node(make_measurement_builder(t))
    win = MainWindow(ctrl)
    win._list.select_index(0)

    # record which row was selected + the sub-tab when each Node was entered
    followed = []

    def on_entered(name, _idx):
        followed.append((name, win._list.selected_index, win._detail.current_tab))

    win._bridge.node_entered.connect(on_entered)
    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5])
    connect_mock(ctrl)
    win._list.refresh_run_availability()
    win._start()
    _pump_until_done(ctrl, win)

    nav = {name: (row, tab) for name, row, tab in followed}
    # each Node, when entered, selected its own list row and showed the run tab
    assert nav["qubit_freq"] == (0, 1)
    assert nav["t1"] == (1, 1)
    assert nav["mist"] == (2, 1)
    # the predictor Service never drives navigation (filtered by the controller)
    assert "predictor" not in nav
    win.close()
    win.deleteLater()


def test_auto_follow_same_node_preserves_detail_tab(app):
    _ctrl, win = app
    win._list.select_index(0)
    win._detail._tabs.setCurrentIndex(0)
    original_form = win._detail.current_form

    win._on_node_entered("qubit_freq", 0)

    assert win._list.selected_index == 0
    assert win._detail.current_tab == 0
    assert win._detail.current_form is original_form

    win._on_node_entered("probe", 0)

    assert win._list.selected_index == 1
    assert win._detail.current_tab == 1


def test_rename_updates_list_and_keeps_canvas_key(qapp):
    # renaming two mist placements to g_mist / e_mist relabels the list and keys
    # each one's liveplot canvas under its instance name.
    ctrl = build_core()
    ctrl.add_node(make_measurement_builder("mist"))
    ctrl.add_node(make_measurement_builder("mist"))
    win = MainWindow(ctrl)
    ctrl.rename_node(0, "g_mist")
    ctrl.rename_node(1, "e_mist")
    win._list.refresh_list()
    assert _list_labels(win) == ["g_mist", "e_mist"]

    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5])
    connect_mock(ctrl)
    win._list.refresh_run_availability()
    win._start()
    _pump_until_done(ctrl, win)

    # canvases are keyed by instance name, independent per placement
    assert set(win._plots) == {"g_mist", "e_mist"}
    assert win._plots["g_mist"][0] is not win._plots["e_mist"][0]
    win.close()
    win.deleteLater()


def test_no_canvas_is_ever_a_toplevel_window(qapp):
    # every Node's Plotter redraws each run point, even off-screen ones; a
    # parentless canvas becomes a top-level window the moment it draws (the
    # "stray window flashing" bug). All canvases must stay parented — the shown
    # one in the run tab, the rest under the hidden park — so none is a window.
    ctrl = build_core()
    for t in ("qubit_freq", "t1", "mist"):
        ctrl.add_node(make_measurement_builder(t))
    win = MainWindow(ctrl)
    win.show()
    win._list.select_index(0)
    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5])
    connect_mock(ctrl)
    win._list.refresh_run_availability()
    win._start()
    _pump_until_done(ctrl, win)

    park = win._canvas_park
    for name, (canvas, _plotter) in win._plots.items():
        assert not canvas.isWindow(), f"{name} canvas is a top-level window"
        assert canvas.parent() is not None, f"{name} canvas is parentless"

    # de-selecting a Node parks its canvas (never leaves it parentless)
    win._list.select_index(1)  # switch away from whatever is shown
    QApplication.processEvents()
    for name, (canvas, _plotter) in win._plots.items():
        assert not canvas.isWindow(), f"{name} canvas became a window after switch"
        # the de-selected canvases sit under the park
        if win._detail._canvas is not canvas:
            assert canvas.parent() is park, f"{name} not parked after de-select"
    win.close()
    win.deleteLater()
