"""The real measure Qt frontend projects application-owned control authority."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTabWidget,
    QTreeWidget,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.services.caretaker import create_persistence_caretaker
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._reload_fakes import OldAdapter
from tests.gui.app.measure.remote._helpers import recv_response, send

pytestmark = [pytest.mark.requires_loopback, pytest.mark.uses_wall_clock]


class StoppableAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False
    )

    def run(
        self, req: RunRequest, raw_cfg: Mapping[str, Any], *, context: RunContext
    ) -> object:
        del req, raw_cfg
        if not context.cancel_signal.event.wait(5):
            raise RuntimeError("test run was not stopped")
        return object()


class UiController(Controller):
    def quiesce(self) -> None:
        self._background_svc.quiesce()


@dataclass
class ControlUi:
    controller: UiController
    window: MainWindow
    client: socket.socket
    dialogs: RecordingDialogPresenter
    tab_id: str

    def rpc(self, method: str, **params: object) -> dict[str, Any]:
        send(self.client, {"id": "ui-control", "method": method, "params": params})
        return recv_response(self.client, "ui-control")

    def connect(self) -> None:
        assert self.rpc("control.connect", label="measure-agent")["ok"]

    @property
    def tab(self) -> ExpTabWidget:
        return next(
            w for w in self.window.findChildren(ExpTabWidget) if w.tab_id == self.tab_id
        )


def wait_until(qapp: QApplication, condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 4
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert condition()


def button(widget, text: str) -> QPushButton:
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


@pytest.fixture
def control_ui(qapp: QApplication, tmp_path: Path) -> Iterator[ControlUi]:
    registry = Registry()
    registry.register("stoppable", StoppableAdapter)
    ctrl = UiController(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        owner_scheduler=QtOwnerScheduler(),
        project_root=str(tmp_path),
    )
    ctrl.attach_caretaker(
        create_persistence_caretaker(ctrl, cache_dir=tmp_path / "cache")
    )
    ctrl.setup_control.new_entry(EntryRequest("control"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    ctrl.add_view(window)
    tab_id = ctrl.new_tab("stoppable")
    remote = RemoteControlAdapter(
        ctrl, ControlOptions(port=0), ctrl.owner_scheduler, render_view=window
    )
    client = socket.create_connection(("127.0.0.1", remote.start()))
    window.show()
    try:
        yield ControlUi(ctrl, window, client, dialogs, tab_id)
    finally:
        for operation in ctrl.operation_control.active_operations():
            ctrl.operation_control.cancel_operation(operation.op)
        client.close()
        remote.stop()
        ctrl.quiesce()
        ctrl.abort_shutdown()
        for tid in tuple(ctrl.list_tab_ids()):
            ctrl.tab_control.close_tab(tid)
        window.deleteLater()
        qapp.processEvents()


def test_indicator_clicks_lock_reclaim_unlock_and_show_offline(control_ui, qapp):
    ui = control_ui
    indicator = ui.window.findChild(QPushButton, "control_indicator")
    assert indicator is not None, "the user needs an always-enabled control indicator"
    assert indicator.text() == "Local"
    indicator.click()
    assert indicator.text() == "Local locked"
    ui.connect()
    assert ui.controller.control_identity.status().holder == "gui"
    indicator.click()
    assert indicator.text() == "Local"
    assert ui.rpc("control.acquire")["ok"]
    assert indicator.isEnabled()
    assert (
        "Remote" in indicator.text()
        and "measure-agent" in indicator.text()
        and "online" in indicator.text()
    )
    ui.client.close()
    wait_until(qapp, lambda: not ui.controller.control_identity.status().online)
    # Disconnect does not release control; only the presence display changes.
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
    assert "offline" in indicator.text()
    indicator.click()
    assert indicator.text() == "Local locked"
    indicator.click()
    assert indicator.text() == "Local"


def test_remote_disables_writes_but_keeps_tabs_figures_inspect_and_screenshots(
    control_ui,
):
    ui = control_ui
    ui.controller.new_tab("stoppable")
    ui.connect()
    assert not button(ui.window, "New Tab ▾").isEnabled()
    assert not button(ui.window, "Reload experiments").isEnabled()
    assert not ui.tab.reset_btn.isEnabled()
    assert not ui.tab.run_btn.isEnabled()
    assert all(
        not field.isEnabled() for field in ui.tab.cfg_form.findChildren(QLineEdit)
    )
    ui.window.refresh_context_panel()
    ui.window.refresh_run_lock(None)
    assert not ui.tab.reset_btn.isEnabled(), (
        "background events must not re-enable writes"
    )
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
    tabs = ui.window.findChildren(QTabWidget)
    assert all(tab.isEnabled() for tab in tabs)
    top = next(tab for tab in tabs if tab.count() == 2)
    top.setCurrentIndex(0)
    ui.window.open_dialog(DialogName.INSPECT)
    inspect = next(
        d
        for d in ui.window.findChildren(QDialog)
        if d.windowTitle() == "Inspect Context"
    )
    assert inspect.isEnabled()
    assert button(inspect, "Refresh").isEnabled()
    assert all(tree.isEnabled() for tree in inspect.findChildren(QTreeWidget))
    assert not button(inspect, "New").isEnabled()
    assert not button(inspect, "Add").isEnabled()
    assert ui.window.take_window_screenshot().startswith(b"\x89PNG")
    assert ui.window.take_dialog_screenshot(DialogName.INSPECT).startswith(b"\x89PNG")
    indicator = ui.window.findChild(QPushButton, "control_indicator")
    assert indicator is not None
    indicator.click()
    assert ui.tab.reset_btn.isEnabled()
    assert ui.tab.run_btn.isEnabled()
    assert button(inspect, "New").isEnabled()


@pytest.mark.parametrize(
    "name, dialog_type",
    [
        (DialogName.SETUP, SetupDialog),
        (DialogName.ALIASES, AliasDialog),
        (DialogName.PREDICTOR, PredictorDialog),
    ],
)
def test_remote_gates_existing_and_new_writable_dialogs(control_ui, name, dialog_type):
    ui = control_ui
    ui.window.open_dialog(name)
    existing = ui.window.findChild(dialog_type)
    assert existing is not None and existing.isEnabled()
    ui.connect()
    assert not existing.isEnabled()
    ui.window.close_dialog(name)
    ui.window.open_dialog(name)
    current = next(d for d in ui.window.findChildren(dialog_type) if d.isVisible())
    assert not current.isEnabled()
    ui.controller.control_identity.reclaim_local()
    assert current.isEnabled()
    assert ui.controller.control_identity.status().locked


def test_remote_measurement_stop_reclaims_and_cancels(control_ui, qapp):
    ui = control_ui
    handle = ui.controller.start_run(ui.tab_id)
    ui.connect()
    assert ui.tab.run_btn.text() == "Stop" and ui.tab.run_btn.isEnabled()
    ui.tab.run_btn.click()
    assert ui.controller.control_identity.status().holder == "gui"
    assert ui.controller.control_identity.status().locked
    denied = ui.rpc("tab.new", adapter_name="stoppable")
    assert denied["error"]["reason"] == "not_holder"
    wait_until(
        qapp,
        lambda: (
            ui.rpc("operation.await", timeout=0, operation_id=handle)["result"].get(
                "status"
            )
            == "cancelled"
        ),
    )
    assert ui.controller.get_running_tab_id() is None


def test_remote_device_stop_reclaims_and_cancels(control_ui, qapp):
    ui = control_ui
    ui.connect()
    connected = ui.rpc(
        "device.connect", name="bias", type_name="FakeDevice", address="none"
    )
    assert connected["ok"]
    wait_until(
        qapp,
        lambda: (
            ui.rpc(
                "operation.await",
                timeout=0,
                operation_id=connected["result"]["operation_id"],
            )["result"].get("status")
            == "finished"
        ),
    )
    setup = ui.rpc(
        "device.setup", name="bias", updates={"value": 10.0, "rampstep": 0.0001}
    )
    assert setup["ok"]
    ui.window.open_dialog(DialogName.DEVICE)
    dialog = ui.window.findChild(DeviceDialog)
    assert dialog is not None
    listing = dialog.findChild(QListWidget)
    assert listing is not None
    listing.setCurrentRow(0)
    stop = button(dialog, "Stop")
    assert stop.isEnabled()
    assert not button(dialog, "Add Device").isEnabled()
    stop.click()
    assert ui.controller.control_identity.status().holder == "gui"
    assert ui.controller.control_identity.status().locked
    assert (
        ui.rpc("device.setup", name="bias", updates={"value": 0.0})["error"]["reason"]
        == "not_holder"
    )
    wait_until(
        qapp,
        lambda: (
            ui.rpc(
                "operation.await",
                timeout=0,
                operation_id=setup["result"]["operation_id"],
            )["result"].get("status")
            == "cancelled"
        ),
    )


def test_remote_close_requires_holder_label_confirmation(control_ui):
    ui = control_ui
    ui.connect()
    ui.window.close()
    assert ui.dialogs.calls, (
        "Remote close must ask even with no active work or unsaved data"
    )
    assert "measure-agent" in ui.dialogs.calls[-1].message
    assert ui.dialogs.calls[-1].default is False
    assert ui.window.isVisible()
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
