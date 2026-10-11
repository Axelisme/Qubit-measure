"""The real measure Qt frontend projects application-owned control authority."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QMetaObject, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QTreeWidget,
)
from zcu_tools.cfg_model import waveform_spec_from_raw
from zcu_tools.datafile import RunMetadata
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    LoadDataRequest,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.services.caretaker import create_persistence_caretaker
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.app.measure.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog
from zcu_tools.gui.cfg import DirectValue
from zcu_tools.gui.cfg.resource import CfgEdit
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.resources.qubit_params import FluxDepFit, QubitParams
from zcu_tools.resources.waveform_assets import FormulaRecipe

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._load_support import historical_metadata
from tests.gui.app.measure._reload_fakes import Loader, OldAdapter
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


class ModalRuntimeCfg(ExpCfgModel):
    knob: int = 42


@dataclass
class ModalLoadResult:
    data_path: str


class ModalLoadAdapter(OldAdapter):
    """Legal native owner contract, without file parsing or acquisition."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(load_data=True)

    def load(
        self, req: LoadDataRequest
    ) -> tuple[RunRecord[ModalRuntimeCfg, ModalLoadResult], RunMetadata]:
        return (
            RunRecord(cfg=ModalRuntimeCfg(), result=ModalLoadResult(req.data_path)),
            historical_metadata(),
        )


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
def control_ui(
    qapp: QApplication, tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[ControlUi]:
    registry = Registry()
    registry.register("stoppable", StoppableAdapter)
    registry.register("loadable", ModalLoadAdapter)
    loader = Loader()
    loader.candidate.register("stoppable", StoppableAdapter)
    ctrl = UiController(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        owner_scheduler=QtOwnerScheduler(),
        project_root=str(tmp_path),
        catalog_loader=loader,
        template_catalog=TemplateCatalog(),
    )
    ctrl.attach_caretaker(
        create_persistence_caretaker(ctrl, cache_dir=tmp_path / "cache")
    )
    ctrl.setup_control.new_entry(EntryRequest("control"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    window = MainWindow(
        ctrl,
        dialog_presenter=None if getattr(request, "param", None) == "qt" else dialogs,
    )
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
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
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
    "name, dialog_type, close_text",
    [
        (DialogName.SETUP, SetupDialog, "Close"),
        (DialogName.ALIASES, AliasDialog, "Cancel"),
        (DialogName.PREDICTOR, PredictorDialog, "Close"),
        (DialogName.ARB_WAVEFORM, ArbWaveformDialog, "Close"),
    ],
)
@pytest.mark.parametrize("close_via", ["button", "window"])
def test_remote_gates_existing_and_new_writable_dialogs(
    control_ui, qapp, name, dialog_type, close_text, close_via
):
    ui = control_ui
    ui.window.open_dialog(name)
    existing = ui.window.findChild(dialog_type)
    assert existing is not None and existing.isEnabled()
    assert existing.isModal()
    ui.connect()
    for opening in range(2):
        current = next(d for d in ui.window.findChildren(dialog_type) if d.isVisible())
        assert current.isEnabled(), "Remote must leave modal Cancel/Close available"
        assert current.isModal()
        fields = current.findChildren(QLineEdit)
        tables = current.findChildren(QTableWidget)
        assert fields or tables
        assert all(not field.isEnabled() for field in fields)
        assert all(not table.isEnabled() for table in tables)
        close = button(current, close_text)
        assert close.isEnabled()
        if close_via == "button":
            close.click()
        else:
            current.close()
            assert not current.isVisible()
        assert name not in ui.window.list_open_dialogs()
        assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
        if opening == 0:
            qapp.processEvents()
            ui.window.open_dialog(name)
    indicator = ui.window.findChild(QPushButton, "control_indicator")
    assert indicator is not None and indicator.isEnabled()
    indicator.click()
    assert ui.controller.control_identity.status().holder == "gui"
    assert ui.controller.control_identity.status().locked
    ui.window.open_dialog(name)
    local = next(d for d in ui.window.findChildren(dialog_type) if d.isVisible())
    assert local.isEnabled()
    assert all(field.isEnabled() for field in local.findChildren(QLineEdit))
    button(local, close_text).click()


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


def modal_action(
    qapp: QApplication,
    action: Callable[[], object],
    interactions: list[tuple[type[QDialog], Callable[[Any], None]]],
) -> None:
    """Deliver real nested modal decisions, with bounded cleanup on test failure."""
    pending = iter(interactions)
    errors: list[Exception] = []
    visited: list[type[QDialog]] = []
    timer = QTimer()
    timer.setSingleShot(True)

    def abort() -> None:
        errors.append(AssertionError("modal interaction timed out"))
        for widget in qapp.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible():
                widget.reject()

    def schedule() -> None:
        item = next(pending, None)
        if item is None:
            return
        expected, interact = item

        def deliver() -> None:
            dialog = qapp.activeModalWidget()
            try:
                assert isinstance(dialog, expected), (expected, dialog)
                visited.append(expected)
                schedule()
                interact(dialog)
            except AssertionError as exc:
                errors.append(exc)
                if isinstance(dialog, QDialog):
                    dialog.reject()

        QTimer.singleShot(0, deliver)

    timer.timeout.connect(abort)
    timer.start(2000)
    schedule()
    try:
        action()
        wait_until(qapp, lambda: bool(errors) or len(visited) == len(interactions))
        qapp.processEvents()
    finally:
        timer.stop()
    if errors:
        raise errors[0]
    assert len(visited) == len(interactions)


@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_setup_text_return_rechecks_holder_without_creating_entry(
    control_ui, qapp, decision
):
    ui = control_ui
    ui.window.open_dialog(DialogName.SETUP)
    setup = ui.window.findChild(SetupDialog)
    assert setup is not None
    before = ui.controller.setup_control.get_entry_info()

    def choose(dialog: QInputDialog) -> None:
        dialog.setTextValue("new-entry")
        if decision == "transfer":
            ui.connect()
        if decision == "cancel":
            dialog.reject()
        else:
            dialog.accept()

    modal_action(
        qapp, lambda: button(setup, "New entry…").click(), [(QInputDialog, choose)]
    )
    entries = ui.controller.setup_control.list_entries(refresh=True)
    if decision == "local":
        assert "new-entry" in entries
        info = ui.controller.setup_control.get_entry_info()
        assert info is not None and info.name == "new-entry"
    else:
        assert "new-entry" not in entries
        assert ui.controller.setup_control.get_entry_info() == before
    if decision == "transfer":
        assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
    button(setup, "Close").click()


@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_predictor_file_return_rechecks_holder_and_preserves_path(
    control_ui, qapp, tmp_path, decision
):
    ui = control_ui
    path = tmp_path / "params.json"
    QubitParams(str(path)).set_fluxdep_fit(FluxDepFit(5.0, 1.0, 0.5, 0.0, 1.0, 1.0))
    ui.window.open_dialog(DialogName.PREDICTOR)
    predictor = ui.window.findChild(PredictorDialog)
    assert predictor is not None
    before = ui.controller.predictor_control.get_predictor_info()

    def choose(dialog: QFileDialog) -> None:
        dialog.selectFile(str(path))
        if decision == "transfer":
            ui.connect()
        if decision == "cancel":
            dialog.reject()
        else:
            QMetaObject.invokeMethod(
                dialog, "accept", Qt.ConnectionType.DirectConnection
            )

    modal_action(
        qapp, lambda: button(predictor, "Browse").click(), [(QFileDialog, choose)]
    )
    after = ui.controller.predictor_control.get_predictor_info()
    if decision == "local":
        assert after is not None and after["EJ"] == 5.0
        assert any(
            edit.text() == str(path) for edit in predictor.findChildren(QLineEdit)
        )
    else:
        assert after == before
    button(predictor, "Close").click()


@pytest.mark.parametrize("operation", ["rename", "delete", "overwrite"])
@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_waveform_modal_return_rechecks_holder(control_ui, qapp, operation, decision):
    ui = control_ui
    recipe = FormulaRecipe.from_raw(
        {"segments": [{"duration": 1.0, "formula": "1"}], "normalize": "peak"}
    )
    ui.controller.arb_waveforms.set_formula("existing", recipe, overwrite=False)
    ui.window.open_dialog(DialogName.ARB_WAVEFORM)
    dialog = ui.window.findChild(ArbWaveformDialog)
    assert dialog is not None
    table = next(t for t in dialog.findChildren(QTableWidget) if t.columnCount() == 3)
    if operation == "overwrite":
        next(edit for edit in dialog.findChildren(QLineEdit)).setText("existing")
    else:
        table.selectRow(0)

    def choose(prompt: QDialog) -> None:
        if decision == "transfer":
            ui.connect()
        if isinstance(prompt, QInputDialog):
            prompt.setTextValue("renamed")
            if decision == "cancel":
                prompt.reject()
            else:
                prompt.accept()
        else:
            assert isinstance(prompt, QMessageBox)
            selected = prompt.button(
                QMessageBox.StandardButton.No
                if decision == "cancel"
                else QMessageBox.StandardButton.Yes
            )
            assert selected is not None and selected.isEnabled(), (
                "modal decision must remain usable after transfer"
            )
            selected.click()

    text = {"rename": "Rename", "delete": "Delete", "overwrite": "Save"}[operation]
    kind = QInputDialog if operation == "rename" else QMessageBox
    modal_action(qapp, lambda: button(dialog, text).click(), [(kind, choose)])
    keys = ui.controller.arb_waveforms.list_data_keys()
    if decision == "local" and operation == "rename":
        assert keys == ["renamed"]
    elif decision == "local" and operation == "delete":
        assert keys == []
    else:
        assert keys == ["existing"]
        data = ui.controller.arb_waveforms.load_data("existing")
        if decision == "local":
            assert data.recipe != recipe
        else:
            assert data.recipe == recipe
    button(dialog, "Close").click()


@pytest.mark.parametrize("control_ui", ["qt"], indirect=True)
@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_reset_confirmation_rechecks_holder(control_ui, qapp, decision):
    ui = control_ui
    resource = ui.controller.cfg_resources.lookup(ui.tab_id)
    resource.edit((CfgEdit(("knob",), 99),))

    def choose(prompt: QMessageBox) -> None:
        if decision == "transfer":
            ui.connect()
        selected = prompt.button(
            QMessageBox.StandardButton.No
            if decision == "cancel"
            else QMessageBox.StandardButton.Yes
        )
        assert selected is not None and selected.isEnabled(), (
            "modal decision must remain usable after transfer"
        )
        selected.click()

    modal_action(qapp, ui.tab.reset_btn.click, [(QMessageBox, choose)])
    assert resource.snapshot_inputs().value.fields["knob"] == DirectValue(
        7 if decision == "local" else 99
    )


@pytest.mark.parametrize("control_ui", ["qt"], indirect=True)
def test_reload_confirmation_return_does_not_rebuild_after_transfer(control_ui, qapp):
    ui = control_ui
    before = ui.controller.list_tab_ids()

    def choose(prompt: QMessageBox) -> None:
        ui.connect()
        selected = next(
            b for b in prompt.buttons() if b.text() == "Discard Results and Reload"
        )
        selected.click()

    modal_action(qapp, ui.window.reload_experiments, [(QMessageBox, choose)])
    assert ui.controller.list_tab_ids() == before
    assert ui.window.has_tab_widget(ui.tab_id)


@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_load_file_return_rechecks_holder_without_replacing_result(
    control_ui, qapp, tmp_path, decision
):
    ui = control_ui
    tab_id = ui.controller.new_tab("loadable")
    path = tmp_path / "selected.h5"
    path.touch()
    ui.controller.load_tab_result(tab_id, "original.hdf5")
    before = ui.controller.get_tab_snapshot(tab_id).run
    assert before is not None

    def open_file(dialog: RunLoadDialog) -> None:
        assert dialog.isModal()
        button(dialog, "File…").click()

    def choose(dialog: QFileDialog) -> None:
        dialog.selectFile(str(path))
        if decision == "transfer":
            ui.connect()
        if decision == "cancel":
            dialog.reject()
            load = ui.window.findChild(RunLoadDialog)
            assert load is not None
            button(load, "Cancel").click()
        else:
            QMetaObject.invokeMethod(
                dialog, "accept", Qt.ConnectionType.DirectConnection
            )
            # The outer dialog remains cancellable when selection cannot commit.
            load = ui.window.findChild(RunLoadDialog)
            if decision == "transfer" and load is not None and load.isVisible():
                button(load, "Cancel").click()

    modal_action(
        qapp,
        lambda: ui.window.load_tab_data_dialog(tab_id),
        [(RunLoadDialog, open_file), (QFileDialog, choose)],
    )
    after = ui.controller.get_tab_snapshot(tab_id).run
    assert after is not None
    if decision == "local":
        assert after.source_path == str(path)
        assert after.result is not before.result
    else:
        assert after.source_path == before.source_path
        assert after.result is before.result


@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_inspect_delete_confirmation_uses_live_owner_and_rechecks_holder(
    control_ui, qapp, decision
):
    ui = control_ui
    capture = ui.controller.context_control.read_context()
    ui.controller.context_control.set_waveform(
        capture.target,
        "Q1.waveforms.pulse",
        waveform_spec_from_raw({"style": "const", "length": 1.0}),
    )
    ui.window.open_dialog(DialogName.INSPECT)
    inspect = ui.window.findChild(InspectDialog)
    assert inspect is not None
    tabs = inspect.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(1)
    tree = next(
        t
        for t in inspect.findChildren(QTreeWidget)
        if t.objectName() != "component_tree"
    )
    group = tree.topLevelItem(0)
    assert group is not None
    tree.setCurrentItem(group.child(0))

    failures: list[str] = []

    def dismiss_failure() -> None:
        prompt = qapp.activeModalWidget()
        if isinstance(prompt, QMessageBox) and prompt.windowTitle() == "Delete failed":
            failures.append(prompt.text())
            prompt.accept()

    def choose(prompt: QMessageBox) -> None:
        if decision == "transfer":
            ui.connect()
        selected = prompt.button(
            QMessageBox.StandardButton.No
            if decision == "cancel"
            else QMessageBox.StandardButton.Yes
        )
        assert selected is not None and selected.isEnabled(), (
            "modal decision must remain usable after transfer"
        )
        QTimer.singleShot(0, dismiss_failure)
        selected.click()

    modal_action(
        qapp, lambda: button(inspect, "Delete").click(), [(QMessageBox, choose)]
    )
    assert failures == [], "legal deletion must use the current library owner"
    assert ui.controller.has_ml_entry("waveform", "Q1.waveforms.pulse") is (
        decision != "local"
    )
    assert inspect.close()


@pytest.mark.parametrize("control_ui", ["qt"], indirect=True)
@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_unsaved_tab_close_confirmation_rechecks_holder(control_ui, qapp, decision):
    ui = control_ui
    tab_id = ui.controller.new_tab("loadable")
    ui.controller.load_tab_result(tab_id, "unsaved.hdf5")
    tabs = next(
        t
        for t in ui.window.findChildren(QTabWidget)
        if t.indexOf(ui.window.findChild(ExpTabWidget)) >= 0
    )
    tab = next(w for w in ui.window.findChildren(ExpTabWidget) if w.tab_id == tab_id)
    index = tabs.indexOf(tab)
    assert index >= 0

    def choose(prompt: QMessageBox) -> None:
        if decision == "transfer":
            ui.connect()
        text = "Cancel" if decision == "cancel" else "Discard and Close"
        next(b for b in prompt.buttons() if b.text() == text).click()

    modal_action(
        qapp, lambda: tabs.tabCloseRequested.emit(index), [(QMessageBox, choose)]
    )
    assert ui.controller.has_tab(tab_id) is (decision != "local")


@pytest.mark.parametrize("decision", ["transfer", "local", "cancel"])
def test_inspect_dirty_save_return_rechecks_holder(control_ui, qapp, decision):
    ui = control_ui
    capture = ui.controller.context_control.read_context()
    ui.controller.context_control.set_waveform(
        capture.target,
        "Q1.waveforms.pulse",
        waveform_spec_from_raw({"style": "const", "length": 1.0}),
    )
    ui.window.open_dialog(DialogName.INSPECT)
    inspect = ui.window.findChild(InspectDialog)
    assert inspect is not None
    tree = next(
        t
        for t in inspect.findChildren(QTreeWidget)
        if t.objectName() != "component_tree"
    )
    group = tree.topLevelItem(0)
    assert group is not None
    tree.setCurrentItem(group.child(0))
    name = next(
        e for e in inspect.findChildren(QLineEdit) if e.text() == "Q1.waveforms.pulse"
    )
    name.setText("Q1.waveforms.renamed")

    def choose(prompt: QMessageBox) -> None:
        if decision == "transfer":
            ui.connect()
        selected = prompt.button(
            QMessageBox.StandardButton.Cancel
            if decision == "cancel"
            else QMessageBox.StandardButton.Save
        )
        assert selected is not None and selected.isEnabled()
        selected.click()

    modal_action(qapp, inspect.close, [(QMessageBox, choose)])
    assert ui.controller.has_ml_entry("waveform", "Q1.waveforms.pulse") is (
        decision != "local"
    )
    assert ui.controller.has_ml_entry("waveform", "Q1.waveforms.renamed") is (
        decision == "local"
    )
    if DialogName.INSPECT in ui.window.list_open_dialogs():
        name.setText("Q1.waveforms.pulse")
        assert inspect.close()


def test_remote_gates_open_template_content_but_keeps_cancel(control_ui):
    ui = control_ui
    ui.window.open_dialog(DialogName.INSPECT)
    inspect = ui.window.findChild(InspectDialog)
    assert inspect is not None
    button(inspect, "New").click()
    create = next(
        d
        for d in inspect.findChildren(QDialog)
        if d.windowTitle() == "Create ModuleLibrary entry"
    )
    assert create.isModal()
    ui.connect()
    assert create.isEnabled()
    assert all(not field.isEnabled() for field in create.findChildren(QLineEdit))
    assert not button(create, "Create").isEnabled()
    assert button(create, "Cancel").isEnabled()
    button(create, "Cancel").click()
    assert not create.isVisible()
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
    assert inspect.close()


@pytest.mark.parametrize("opening", ["existing", "new"])
def test_load_dialog_remote_content_keeps_cancel(control_ui, qapp, opening):
    ui = control_ui
    if opening == "new":
        ui.connect()

    def cancel(dialog: RunLoadDialog) -> None:
        if opening == "existing":
            ui.connect()
        assert dialog.isEnabled() and dialog.isModal()
        assert all(not field.isEnabled() for field in dialog.findChildren(QLineEdit))
        assert not button(dialog, "File…").isEnabled()
        assert not button(dialog, "Load selected").isEnabled()
        assert button(dialog, "Cancel").isEnabled()
        button(dialog, "Cancel").click()

    modal_action(
        qapp,
        lambda: ui.window.load_tab_data_dialog(ui.tab_id),
        [(RunLoadDialog, cancel)],
    )
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"


@pytest.mark.parametrize("stage", ["selection", "file-chooser"])
def test_shutdown_finishes_load_interaction_before_owner_teardown(
    control_ui, qapp, tmp_path, stage
):
    ui = control_ui
    tab_id = ui.controller.new_tab("loadable")
    ui.controller.load_tab_result(tab_id, "original.hdf5")
    before = ui.controller.get_tab_snapshot(tab_id).run
    assert before is not None
    path = tmp_path / "late-choice.h5"
    path.touch()
    finished: list[tuple[int, str | None, bool]] = []
    owner_finished: list[bool] = []
    load_dialogs: list[RunLoadDialog] = []

    def observe_finished(result: int) -> None:
        load = load_dialogs[0]
        authority = ui.controller.control_identity
        authority.connect("shutdown", "shutdown-agent", "owner")
        finished.append(
            (result, load.selected_path(), button(load, "File…").isEnabled())
        )
        # Keep Local for a late chooser return; holder refusal must not hide a
        # second acceptance of an interaction already ended by owner shutdown.
        authority.reclaim_local()
        authority.toggle_local()

    def open_load(load: RunLoadDialog) -> None:
        load_dialogs.append(load)
        load.finished.connect(observe_finished)
        if stage == "file-chooser":
            button(load, "File…").click()
        else:
            ui.window.request_shutdown()
            wait_until(qapp, lambda: not ui.window.isVisible())
            owner_finished.append(bool(finished))
            if load.isVisible():
                load.reject()

    def return_late_choice(chooser: QFileDialog) -> None:
        chooser.selectFile(str(path))
        ui.window.request_shutdown()
        wait_until(qapp, lambda: not ui.window.isVisible())
        owner_finished.append(bool(finished))
        if load_dialogs[0].isVisible():
            load_dialogs[0].reject()
        if chooser.isVisible():
            QMetaObject.invokeMethod(
                chooser, "accept", Qt.ConnectionType.DirectConnection
            )

    interactions: list[tuple[type[QDialog], Callable[[Any], None]]] = [
        (RunLoadDialog, open_load)
    ]
    if stage == "file-chooser":
        interactions.append((QFileDialog, return_late_choice))
    modal_action(qapp, lambda: ui.window.load_tab_data_dialog(tab_id), interactions)
    assert owner_finished == [True]
    assert finished == [(int(QDialog.DialogCode.Rejected), None, True)]
    after = ui.controller.get_tab_snapshot(tab_id).run
    assert after is not None
    assert after.source_path == before.source_path
    assert after.result is before.result
    assert not ui.window.isVisible()


def test_inspect_delete_confirmation_pins_the_original_saved_point(control_ui, qapp):
    ui = control_ui
    original = ui.controller.context_control.read_context()
    ui.controller.context_control.set_waveform(
        original.target,
        "Q1.waveforms.pulse",
        waveform_spec_from_raw({"style": "const", "length": 1.0}),
    )
    ui.window.open_dialog(DialogName.INSPECT)
    inspect = ui.window.findChild(InspectDialog)
    assert inspect is not None
    tree = next(
        t
        for t in inspect.findChildren(QTreeWidget)
        if t.objectName() != "component_tree"
    )
    group = tree.topLevelItem(0)
    assert group is not None
    tree.setCurrentItem(group.child(0))
    failures: list[str] = []

    def dismiss_failure() -> None:
        prompt = qapp.activeModalWidget()
        if isinstance(prompt, QMessageBox) and prompt.windowTitle() == "Delete failed":
            failures.append(prompt.text())
            prompt.accept()

    def choose(prompt: QMessageBox) -> None:
        ui.controller.setup_control.new_point(
            PointCreateRequest("other", "", CommandOrigin("gui"))
        )
        other = ui.controller.context_control.read_context()
        ui.controller.context_control.set_waveform(
            other.target,
            "Q1.waveforms.pulse",
            waveform_spec_from_raw({"style": "const", "length": 2.0}),
        )
        QTimer.singleShot(0, dismiss_failure)
        selected = prompt.button(QMessageBox.StandardButton.Yes)
        assert selected is not None
        selected.click()

    modal_action(
        qapp, lambda: button(inspect, "Delete").click(), [(QMessageBox, choose)]
    )
    assert failures == []
    assert ui.controller.has_ml_entry("waveform", "Q1.waveforms.pulse")
    assert ui.controller.get_session_env().active_label == "other"
    ui.controller.context_control.use_point(PointUseRequest("point"))
    assert not ui.controller.has_ml_entry("waveform", "Q1.waveforms.pulse")
    assert inspect.close()
