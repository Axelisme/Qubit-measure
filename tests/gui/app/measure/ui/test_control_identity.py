"""The real measure Qt frontend projects application-owned control authority."""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QMetaObject, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
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
    QWidget,
)
from zcu_tools.cfg_model import ConfigBase, waveform_spec_from_raw
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
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.events import MdChangedPayload
from zcu_tools.gui.session.ui.add_component_dialog import AddComponentDialog
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.entry import component_registry
from zcu_tools.resources.qubit_params import FluxDepFit, QubitParams
from zcu_tools.resources.waveform_assets import FormulaRecipe
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import key_click, type_text
from tests.gui.app.measure._components import ResonatorSchema
from tests.gui.app.measure._load_support import historical_metadata
from tests.gui.app.measure._reload_fakes import Loader, OldAdapter
from tests.gui.app.measure.remote._helpers import recv_response, send
from tests.gui.app.measure.ui._qt_interactions import modal_action
from tests.resources.entry.fakes import registry_state, restore_registry

pytestmark = [pytest.mark.requires_loopback, pytest.mark.uses_wall_clock]


class StoppableAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False
    )

    def run(self, req: RunRequest, cfg: ConfigBase, *, context: RunContext) -> object:
        del req, cfg
        # The test Stop/close action releases this held adapter via cancellation.
        if not context.cancel_signal.event.wait(30):
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


@dataclass
class ControlUi:
    controller: Controller
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


def form_editor(tab: ExpTabWidget) -> FormEditor[FormModel]:
    """The tab's mounted cfg editor."""
    editor = tab.findChild(FormEditor)
    assert editor is not None, "the Run panel mounts a FormEditor"
    return editor


def button(widget, text: str) -> QPushButton:
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


@pytest.fixture(scope="module", autouse=True)
def component_registry_guard() -> Iterator[None]:
    incoming = registry_state()
    yield
    assert registry_state() == incoming, (
        "Control UI module polluted component/role state"
    )


@pytest.fixture
def registered_resonator(component_registry_guard) -> Iterator[None]:
    incoming = registry_state()
    try:
        if "resonator" in component_registry.kinds():
            component_registry.unregister("resonator")
        component_registry.register("resonator", ResonatorSchema)
        registered = registry_state()
        yield
        assert registry_state() == registered, (
            "Control UI test polluted component/role state"
        )
    finally:
        restore_registry(incoming)
        assert registry_state() == incoming, "Component/role restoration failed"


@pytest.fixture
def control_ui(
    qapp: QApplication,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    registered_resonator,
) -> Iterator[ControlUi]:
    registry = Registry()
    registry.register("stoppable", StoppableAdapter)
    registry.register("loadable", ModalLoadAdapter)
    loader = Loader()
    loader.candidate.register("stoppable", StoppableAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        owner_scheduler=QtOwnerScheduler(),
        project_root=str(tmp_path),
        catalog_loader=loader,
    )
    ctrl.attach_caretaker(
        create_persistence_caretaker(ctrl, cache_dir=tmp_path / "cache")
    )
    ctrl.setup_control.new_entry(EntryRequest("control"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    window = MainWindow(
        ctrl,
        templates=lambda: {},
        dialog_presenter=None if getattr(request, "param", None) == "qt" else dialogs,
    )
    ctrl.add_view(window)
    tab_id = ctrl.new_tab("stoppable")
    remote = RemoteControlAdapter(
        ctrl, ControlOptions(port=0), ctrl.owner_scheduler, render_view=window
    )
    # remote.start() has bound the listener; only guard a stuck loopback connect.
    client = socket.create_connection(("127.0.0.1", remote.start()), timeout=30)
    window.show()
    try:
        yield ControlUi(ctrl, window, client, dialogs, tab_id)
    finally:
        for operation in ctrl.operation_control.active_operations():
            ctrl.operation_control.cancel_operation(operation.op)
        client.close()
        remote.stop()
        ctrl.close()
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
    qt_until(lambda: not ui.controller.control_identity.status().online)
    # Disconnect does not release control; only the presence display changes.
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
    assert "offline" in indicator.text()
    indicator.click()
    assert indicator.text() == "Local locked"
    indicator.click()
    assert indicator.text() == "Local"


def assert_remote_forms_readable_without_writes(
    ui: ControlUi, qapp: QApplication
) -> None:
    """Typing and committing cannot change any Remote tab's visible text or input."""
    for tab in ui.window.findChildren(ExpTabWidget):
        resource = ui.controller.cfg_resources.lookup(tab.tab_id)
        before = deepcopy(resource.input.data)
        fields = form_editor(tab).findChildren(QLineEdit)
        assert fields, "each synthetic tab has a readable knob input"
        for field in fields:
            assert field.isEnabled() and field.isReadOnly()
            text = field.text()
            field.setFocus()
            field.selectAll()
            assert field.selectedText() == text, "Remote text stays selectable"
            type_text(field, "99")
            key_click(field, Qt.Key.Key_Return)
            field.clearFocus()
            qapp.processEvents()
            assert field.text() == text
        assert resource.input.data == before, "Remote input events must not write"


def assert_local_forms_accept_input(ui: ControlUi, qapp: QApplication) -> None:
    """The same typing and Enter events write every tab's knob after Local reclaim."""
    for tab in ui.window.findChildren(ExpTabWidget):
        field = form_editor(tab).findChild(QLineEdit, "form.input.knob")
        assert field is not None, "each synthetic tab mounts its public knob input"
        assert field.isEnabled() and not field.isReadOnly()
        field.setFocus()
        field.selectAll()
        type_text(field, "99")
        key_click(field, Qt.Key.Key_Return)
        qapp.processEvents()
        assert field.text() == "99"
        assert ui.controller.cfg_resources.lookup(tab.tab_id).input.data["knob"] == 99


def test_remote_disables_writes_but_keeps_tabs_figures_inspect_and_screenshots(
    control_ui,
    qapp: QApplication,
) -> None:
    ui = control_ui
    ui.controller.new_tab("stoppable")
    ui.connect()
    created = ui.rpc("tab.new", adapter_name="stoppable")
    assert created["ok"], created
    new_tab = next(
        w
        for w in ui.window.findChildren(ExpTabWidget)
        if w.tab_id == created["result"]["tab_id"]
    )
    assert not new_tab.reset_btn.isEnabled()
    assert not button(ui.window, "New Tab ▾").isEnabled()
    assert not button(ui.window, "Reload experiments").isEnabled()
    assert not ui.tab.reset_btn.isEnabled()
    assert not ui.tab.run_btn.isEnabled()
    assert_remote_forms_readable_without_writes(ui, qapp)
    ui.window.refresh_context_panel()
    ui.window.refresh_run_lock(None)
    assert not ui.tab.reset_btn.isEnabled(), (
        "background events must not re-enable writes"
    )
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"
    tabs = ui.window.findChildren(QTabWidget)
    assert all(tab.isEnabled() for tab in tabs)
    top = next(tab for tab in tabs if tab.indexOf(ui.tab) >= 0)
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
    button(inspect, "Arb Waveforms…").click()
    qapp.processEvents()
    arb = next(d for d in ui.window.findChildren(ArbWaveformDialog) if d.isVisible())
    assert not button(arb, "Save").isEnabled()
    arb_key = arb.findChild(QLineEdit)
    assert arb_key is not None and arb_key.isReadOnly()
    arb.close()
    qapp.processEvents()
    add_component = inspect.findChild(QPushButton, "component_add")
    if add_component is None:
        raise LookupError("Inspect lacks its component Add control")
    assert not add_component.isEnabled()
    assert ui.window.take_window_screenshot().startswith(b"\x89PNG")
    assert ui.window.take_dialog_screenshot(DialogName.INSPECT).startswith(b"\x89PNG")
    indicator = ui.window.findChild(QPushButton, "control_indicator")
    assert indicator is not None
    indicator.click()
    assert ui.tab.reset_btn.isEnabled()
    assert ui.tab.run_btn.isEnabled()
    assert_local_forms_accept_input(ui, qapp)
    assert inspect.close()
    ui.window.open_dialog(DialogName.INSPECT)
    local_inspect = next(
        d for d in ui.window.findChildren(InspectDialog) if d.isVisible()
    )
    assert button(local_inspect, "New").isEnabled()


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
    # Stop cancels the admitted Run; await its real terminal, not a sampling RPC probe.
    assert (
        ui.rpc("operation.await", timeout=30, operation_id=handle)["result"]["status"]
        == "cancelled"
    )
    assert ui.controller.get_running_tab_id() is None


def test_remote_device_stop_reclaims_and_cancels(control_ui, qapp):
    ui = control_ui
    ui.connect()
    connected = ui.rpc(
        "device.connect", name="bias", type_name="FakeDevice", address="none"
    )
    assert connected["ok"]
    # The admitted fake connect publishes terminal before setup can start.
    assert (
        ui.rpc(
            "operation.await",
            timeout=30,
            operation_id=connected["result"]["operation_id"],
        )["result"]["status"]
        == "finished"
    )
    snapshot = ui.rpc("device.snapshot", name="bias")
    assert snapshot["ok"], snapshot
    cfg = snapshot["result"]["snapshot"]["info"]
    setup = ui.rpc(
        "device.setup", name="bias", cfg={**cfg, "value": 10.0, "rampstep": 0.0001}
    )
    assert setup["ok"], setup
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
        ui.rpc("device.setup", name="bias", cfg={**cfg, "value": 0.0})["error"][
            "reason"
        ]
        == "not_holder"
    )
    # Stop releases the synthetic ramp and publishes its cancelled terminal.
    assert (
        ui.rpc(
            "operation.await", timeout=30, operation_id=setup["result"]["operation_id"]
        )["result"]["status"]
        == "cancelled"
    )


@pytest.mark.parametrize(
    "name",
    [
        DialogName.SETUP,
        DialogName.PREDICTOR,
        DialogName.ARB_WAVEFORM,
        DialogName.INSPECT,
        DialogName.DEVICE,
    ],
)
def test_local_named_modal_refuses_remote_owner_and_allows_retry(control_ui, name):
    ui = control_ui
    ui.window.open_dialog(name)
    dialog = QApplication.activeModalWidget()
    assert isinstance(dialog, QDialog) and dialog.isModal()
    reply = ui.rpc("control.connect", label="measure-agent")
    assert reply["ok"] is False and reply["error"]["reason"] == "local_busy"
    assert ui.controller.control_identity.status().holder == "gui"
    assert (
        ui.rpc("tab.new", adapter_name="stoppable")["error"]["reason"] == "not_holder"
    )
    ui.window.close_dialog(name)
    assert not dialog.isVisible()
    if name == DialogName.PREDICTOR:
        ui.window.open_dialog(name)
        assert QApplication.activeModalWidget() is dialog
        reply = ui.rpc("control.connect", label="measure-agent")
        assert reply["ok"] is False and reply["error"]["reason"] == "local_busy"
        ui.window.close_dialog(name)
    ui.connect()
    assert ui.controller.control_identity.status().holder == "rpc:measure-agent"


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


@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_setup_text_return_preserves_entry_input_and_cancel(control_ui, qapp, decision):
    ui = control_ui
    ui.window.open_dialog(DialogName.SETUP)
    setup = ui.window.findChild(SetupDialog)
    assert setup is not None
    before = ui.controller.setup_control.get_entry_info()

    def choose(dialog: QInputDialog) -> None:
        dialog.setTextValue("new-entry")
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
    button(setup, "Close").click()


@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_predictor_file_return_preserves_path_and_cancel(
    control_ui, qapp, tmp_path, decision
):
    ui = control_ui
    path = tmp_path / "params.json"
    QubitParams(str(path)).set_fluxdep_fit(FluxDepFit(5.0, 1.0, 0.5, 0.0, 1.0, 1.0))
    ui.window.open_dialog(DialogName.PREDICTOR)
    predictor = ui.window.findChild(PredictorDialog)
    assert predictor is not None
    before = predictor.parameters()
    before_path = predictor.capture_state().params_path_text

    def choose(dialog: QFileDialog) -> None:
        dialog.selectFile(str(path))
        if decision == "cancel":
            dialog.reject()
        else:
            QMetaObject.invokeMethod(
                dialog, "accept", Qt.ConnectionType.DirectConnection
            )

    modal_action(
        qapp, lambda: button(predictor, "Browse").click(), [(QFileDialog, choose)]
    )
    after = predictor.parameters()
    if decision == "local":
        assert isinstance(after, FluxoniumParams) and after.EJ == 5.0
        assert any(
            edit.text() == str(path) for edit in predictor.findChildren(QLineEdit)
        )
    else:
        assert after == before
        assert predictor.capture_state().params_path_text == before_path
    button(predictor, "Close").click()


@pytest.mark.parametrize("operation", ["rename", "delete", "overwrite"])
@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_waveform_modal_input_and_cancel(control_ui, qapp, operation, decision):
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
                "modal decision must remain usable"
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
@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_reset_confirmation_accept_and_cancel(control_ui, qapp, decision):
    ui = control_ui
    resource = ui.controller.cfg_resources.lookup(ui.tab_id)
    resource.write({("knob",): 99})

    def choose(prompt: QMessageBox) -> None:
        selected = prompt.button(
            QMessageBox.StandardButton.No
            if decision == "cancel"
            else QMessageBox.StandardButton.Yes
        )
        assert selected is not None and selected.isEnabled(), (
            "modal decision must remain usable"
        )
        selected.click()

    modal_action(qapp, ui.tab.reset_btn.click, [(QMessageBox, choose)])
    assert resource.input.data["knob"] == (7 if decision == "local" else 99)


@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_load_file_return_accept_and_cancel_preserve_result_contract(
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


# Inspect asks through the window presenter; "qt" keeps real message boxes.
@pytest.mark.parametrize("control_ui", ["qt"], indirect=True)
@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_inspect_delete_confirmation_uses_live_library_owner(
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
        selected = prompt.button(
            QMessageBox.StandardButton.No
            if decision == "cancel"
            else QMessageBox.StandardButton.Yes
        )
        assert selected is not None and selected.isEnabled(), (
            "modal decision must remain usable"
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
@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_unsaved_tab_close_confirmation_accept_and_cancel(control_ui, qapp, decision):
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
        text = "Cancel" if decision == "cancel" else "Discard and Close"
        next(b for b in prompt.buttons() if b.text() == text).click()

    modal_action(
        qapp, lambda: tabs.tabCloseRequested.emit(index), [(QMessageBox, choose)]
    )
    assert ui.controller.has_tab(tab_id) is (decision != "local")


@pytest.mark.parametrize("control_ui", ["qt"], indirect=True)
@pytest.mark.parametrize("decision", ["local", "cancel"])
def test_inspect_dirty_save_and_cancel(control_ui, qapp, decision):
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
        finished.append(
            (result, load.selected_path(), button(load, "File…").isEnabled())
        )

    def open_load(load: RunLoadDialog) -> None:
        load_dialogs.append(load)
        load.finished.connect(observe_finished)
        if stage == "file-chooser":
            button(load, "File…").click()
        else:
            ui.window.request_shutdown()
            qt_until(lambda: not ui.window.isVisible())
            owner_finished.append(bool(finished))
            if load.isVisible():
                load.reject()

    def return_late_choice(chooser: QFileDialog) -> None:
        chooser.selectFile(str(path))
        ui.window.request_shutdown()
        qt_until(lambda: not ui.window.isVisible())
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


@pytest.mark.parametrize("control_ui", ["qt"], indirect=True)
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


def add_control[T: QWidget](dialog: AddComponentDialog, kind: type[T], name: str) -> T:
    """Locate one declared public Add dialog control by its object name."""
    control = dialog.findChild(kind, name)
    assert control is not None, f"Add dialog lacks {name!r}"
    return control


def open_resonator_add(tree: ComponentTreeWidget, name: str) -> AddComponentDialog:
    """Open the real tree's public Add dialog with a valid resonator candidate."""
    dialog = tree.open_add_dialog()
    add_control(dialog, QComboBox, "add_kind").setCurrentText("resonator")
    add_control(dialog, QLineEdit, "add_name").setText(name)
    add_control(dialog, QLineEdit, "field:freq").setText("6500")
    return dialog


@pytest.mark.parametrize("close_via", ["cancel", "close"])
def test_inspect_add_submit_after_close_has_no_owner_effects(
    control_ui: ControlUi, qapp: QApplication, close_via: str
) -> None:
    ui = control_ui
    ui.window.open_dialog(DialogName.INSPECT)
    inspect = ui.window.findChild(InspectDialog)
    assert inspect is not None, "Inspect did not open"
    tree = inspect.findChild(ComponentTreeWidget)
    assert tree is not None, "Inspect lacks its component tree"
    dialog = open_resonator_add(tree, "retired_resonator")
    submit = add_control(dialog, QPushButton, "add_submit")
    baseline = ui.controller.context_control.inspect_document()
    changes: list[MdChangedPayload] = []
    subscription = ui.controller.get_bus().subscribe(MdChangedPayload, changes.append)
    delivered: list[bool] = []

    def deliver_queued_submit() -> None:
        submit.clicked.emit()
        delivered.append(True)

    try:
        QTimer.singleShot(0, deliver_queued_submit)
        if close_via == "cancel":
            add_control(dialog, QPushButton, "add_cancel").click()
        else:
            assert dialog.close()
        assert not dialog.isVisible()
        qapp.processEvents()
        assert delivered == [True]
        assert ui.controller.control_identity.status().holder == "gui"
        assert ui.controller.context_control.inspect_document() == baseline
        assert changes == []
    finally:
        subscription.unsubscribe()
