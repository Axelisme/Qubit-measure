"""Public EMPTY startup, source capture and composition lifecycle seams."""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)
from zcu_tools.device.manager import DeviceManager
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.runtime import GuiAssembly
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.resources.entry.fakes import registry_state, restore_registry
from zcu_lab.components import QubitSchema
from zcu_lab.workflows.gui.app import WorkflowsGuiBehavior
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.gui.remote import T1RemoteControlAdapter
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.fixture
def app(qapp: QApplication, tmp_path: Path) -> Iterator[WorkflowsGuiBehavior]:
    before = registry_state()
    component_registry.register("qubit/workflow-app-test", QubitSchema)
    try:
        role_registry.get("qubit")
    except ValueError:
        role_registry.register("qubit", RoleSpec("qubit/*"))
    installed = registry_state()
    behavior = WorkflowsGuiBehavior(project_root=str(tmp_path))
    try:
        yield behavior
    finally:
        # Every test assembles before reaching assertions; retain workers through
        # public shutdown before allowing Qt objects to be released.
        if hasattr(behavior, "window"):
            behavior.window.close()
            behavior.shutdown()
            behavior.window.deleteLater()
            qapp.processEvents()
        try:
            assert registry_state() == installed, "app test polluted definitions"
        finally:
            restore_registry(before)


def button(window: T1WorkflowWindow, name: str) -> QPushButton:
    found = window.findChild(QPushButton, name)
    assert found is not None
    return found


def setup_button(window: T1WorkflowWindow, text: str) -> QPushButton:
    return next(
        b for b in window.setup_dialog.findChildren(QPushButton) if b.text() == text
    )


def open_point(
    app: WorkflowsGuiBehavior, tmp_path: Path, qapp: QApplication
) -> ResultEntry:
    entry = ResultEntry.create(
        "sample", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.setup.add_component("Q1", kind="qubit/workflow-app-test")
    entry.new_point("source", description="source point")
    app.window.open_setup()
    combo = app.window.setup_dialog.findChild(QComboBox, "result_entry")
    assert combo is not None
    combo.setCurrentText("sample")
    setup_button(app.window, "Open").click()
    points = app.window.setup_dialog.findChild(QListWidget, "working_points")
    assert points is not None
    assert points.count() == 1
    points.setCurrentRow(0)
    setup_button(app.window, "Switch to selected").click()
    info = app.services.context_control.get_entry_info()
    assert info is not None and info.name == "sample"
    assert app.services.context_control.get_component_focus().target is not None
    app.window.setup_dialog.close()
    qapp.processEvents()
    qubit = app.window.findChild(QComboBox, "workflow.source.qubit")
    assert qubit is not None and qubit.findText("Q1") >= 0
    qubit.setCurrentText("Q1")
    assert app.services.context_control.get_component_focus().name == "Q1"
    return entry


def test_empty_assembly_uses_one_native_owner_without_source(
    app: WorkflowsGuiBehavior,
) -> None:
    assembly = app.assemble(None)
    assert assembly.window is app.window
    assert assembly.controller is app.window.controller
    assert assembly.control_adapter is None
    assert app.state.session_env.parameters is None
    assert app.services.context_control.get_entry_info() is None
    assert app.window.plan_panel.env == FormEnv()
    assert app.window.tunables_panel.env == FormEnv()
    assert (
        app.window.plan_panel.state.input.data
        == T1PlanForm.default_input(FormEnv()).data
    )
    assert isinstance(app.window.plan_panel.state.result(FormEnv()), FormErrors)
    assert isinstance(app.window.tunables_panel.state.result(FormEnv()), T1TunablesForm)
    with pytest.raises(FailedPreconditionError):
        app.window.start("not-ready")
    assert app.window.controller.operation_id is None


def test_after_show_reuses_nonmodal_setup_without_selecting_entry(
    app: WorkflowsGuiBehavior,
) -> None:
    assembly = app.assemble(None)
    app.window.show()
    app.after_show(assembly)
    setup = app.window.setup_dialog
    assert setup.isVisible() and not setup.isModal()
    assert app.services.context_control.get_entry_info() is None
    setup.close()
    button(app.window, "workflow.source.setup").click()
    assert app.window.setup_dialog is setup and setup.isVisible()
    with pytest.raises(ValueError, match="Window"):
        app.after_show(GuiAssembly(controller=object(), window=QWidget()))


def test_setup_focus_and_explicit_capture_replace_drafts_not_live_link(
    app: WorkflowsGuiBehavior, tmp_path: Path, qapp: QApplication
) -> None:
    app.assemble(None)
    window = app.window
    old = window.plan_panel
    open_point(app, tmp_path, qapp)
    context = app.services.context
    target = context.read_context().target
    context.set_parameter(target, "general.ext.guess_us", 31.0)
    window.tunables_panel.state.write({("guess_us",): 7.0}, FormEnv())
    # Source selection itself has not replaced input/capture.
    assert window.plan_panel is old and window.plan_panel.env.parameters is None
    button(window, "workflow.source.use").click()
    env = window.tunables_panel.env
    assert window.plan_panel is not old and env.parameters is not None
    assert window.plan_panel.env is env
    assert (
        window.tunables_panel.state.input.data == T1TunablesForm.default_input(env).data
    )
    assert (
        window.tunables_panel.state.base.data == window.tunables_panel.state.input.data
    )
    assert window.plan_panel.state.base.data == T1PlanForm.default_input(env).data
    knobs = window.tunables_panel.state.result(env)
    assert isinstance(knobs, T1TunablesForm) and knobs.guess_us == 31.0
    context.set_parameter(target, "general.ext.guess_us", 45.0)
    knobs = window.tunables_panel.state.result(env)
    assert isinstance(knobs, T1TunablesForm) and knobs.guess_us == 31.0
    window.tunables_panel.state.write({("guess_us",): 8.0}, env)
    edit = window.tunables_panel.findChild(QLineEdit, "form.input.guess_us")
    assert edit is not None
    edit.setText("1+")
    edit.editingFinished.emit()
    assert window.tunables_panel.editor.local_parse_errors() is not None
    window.prepare_forms()
    assert window.tunables_panel.editor.local_parse_errors() is None
    knobs = window.tunables_panel.state.result(window.tunables_panel.env)
    assert isinstance(knobs, T1TunablesForm) and knobs.guess_us == 45.0
    assert (
        window.tunables_panel.state.base.data == window.tunables_panel.state.input.data
    )


def test_capture_failure_preserves_both_panels_and_local_input(
    app: WorkflowsGuiBehavior, tmp_path: Path, qapp: QApplication
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    app.window.prepare_forms()
    plan, knobs = app.window.plan_panel, app.window.tunables_panel
    knobs.state.write({("guess_us",): 17.0}, knobs.env)
    edit = knobs.findChild(QLineEdit, "form.input.guess_us")
    assert edit is not None
    edit.setText("1+")
    edit.editingFinished.emit()
    before = knobs.state.input.data
    app.services.settings.open_entry(EntryRequest("sample"))
    with pytest.raises(FailedPreconditionError):
        app.window.prepare_forms()
    assert app.window.plan_panel is plan and app.window.tunables_panel is knobs
    assert knobs.state.input.data == before
    assert edit.text() == "1+" and knobs.editor.local_parse_errors() is not None


def test_composition_rpc_reads_same_draft_without_socket(
    app: WorkflowsGuiBehavior, qapp: QApplication
) -> None:
    assembly = app.assemble(ControlOptions(port=0))
    adapter = assembly.control_adapter
    assert isinstance(adapter, T1RemoteControlAdapter)
    app.window.tunables_panel.state.write({("guess_us",): 73.0}, FormEnv())
    link = ClientLink("app-contract", token_required=False)
    adapter.on_client_open(link)
    try:
        with ThreadPoolExecutor(max_workers=1) as io:
            request = Request(
                id="app-read", method="workflow.form", params={"form": "tunables"}
            )
            future = io.submit(adapter.route, link, request)
            deadline = monotonic() + 3
            while not future.done():
                assert monotonic() < deadline, "owner RPC did not settle"
                qapp.processEvents()
            future.result(timeout=1)
            reply = json.loads(link.outbound.get(timeout=1))
            assert reply["result"]["input"]["guess_us"] == 73.0
    finally:
        adapter.on_client_close(link, on_owner_thread=True)


def test_window_close_drains_and_closes_its_session_registry(
    app: WorkflowsGuiBehavior, qapp: QApplication
) -> None:
    app.assemble(None)
    token = app.services.device.start_connect_device(
        ConnectDeviceRequest("FakeDevice", "flux", "none")
    )
    deadline = monotonic() + 5
    while not app.services.device.get_connected_devices():
        assert monotonic() < deadline, f"FakeDevice connect {token} did not settle"
        qapp.processEvents()
    assert app.window.close()
    assert not app.services.device.get_connected_devices()
    app.shutdown()


def test_shutdown_failure_rejects_close_and_keeps_forms(
    app: WorkflowsGuiBehavior, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.assemble(None)
    window = app.window
    before = window.plan_panel
    with monkeypatch.context() as patch:
        patch.setattr(BackgroundRunner, "quiesce", lambda self: False)
        assert not window.close()
        message = window.findChild(QLabel, "workflow.run.message")
        assert message is not None and "drain" in message.text().lower()
        assert window.plan_panel is before
    assert window.close()


def test_registry_close_failure_rejects_window_close(
    app: WorkflowsGuiBehavior, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.assemble(None)

    def reject_close(self: DeviceManager) -> None:
        raise RuntimeError("driver close failed")

    with monkeypatch.context() as patch:
        patch.setattr(DeviceManager, "close_all_devices", reject_close)
        assert not app.window.close()
        message = app.window.findChild(QLabel, "workflow.run.message")
        assert message is not None and "driver close failed" in message.text()
    assert app.window.close()
