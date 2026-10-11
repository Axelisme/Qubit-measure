"""Public EMPTY startup, source capture and composition lifecycle seams."""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from zcu_tools.device import FakeDevice
from zcu_tools.device.manager import DeviceManager
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.runtime import GuiAssembly
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleResolutionError,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.gui._completion_helpers import on_device_connected
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.resources.entry.fakes import registry_state, restore_registry
from zcu_lab.components import QubitSchema, ResonatorSchema
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
    component_registry.register("resonator/workflow-app-test", ResonatorSchema)
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
    app.services.context_control.set_alias_rows((AliasRow("qubit", "Q1"),))
    app.window.setup_dialog.close()
    qapp.processEvents()
    captured = app.services.context.read_context()
    assert captured.resolve_path("$qubit.kind") == "Q1.kind"
    return entry


@pytest.mark.parametrize("embedded", [False, True])
def test_window_parent_and_close_callback_retry(
    qapp: QApplication, embedded: bool
) -> None:
    harness = StartHarness()
    parent = QWidget() if embedded else None
    attempts: list[int] = []

    def on_close() -> None:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise RuntimeError("session close refused")

    window = T1WorkflowWindow(
        harness.state,
        harness.services,
        harness.owner,
        parent=parent,
        on_close=on_close,
    )
    try:
        assert window.parentWidget() is parent
        assert window.isWindow() is (not embedded)
        if parent is not None:
            parent.show()
        window.show()
        assert not window.close()
        assert window.isVisible()
        message = window.findChild(QLabel, "workflow.run.message")
        assert message is not None and message.text() == "session close refused"
        assert attempts == [1]
        assert window.close()
        assert not window.isVisible()
        assert attempts == [1, 2]
    finally:
        window.close()
        window.deleteLater()
        if parent is not None:
            parent.deleteLater()
        qapp.processEvents()


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


def test_setup_alias_and_explicit_capture_replace_drafts_not_live_link(
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


def edit_alias_entry(
    window: T1WorkflowWindow, value_text: str
) -> tuple[AliasDialog, QDialogButtonBox, QTableWidgetItem]:
    entry = window.findChild(QPushButton, "workflow.source.aliases")
    assert entry is not None, "Window must expose the shared alias editor entry"
    entry.click()
    dialog = next((d for d in window.findChildren(AliasDialog) if d.isVisible()), None)
    assert dialog is not None, "Alias entry must show the shared AliasDialog"
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    value = table.item(0, 1)
    assert value is not None and value.text() == "Q1"
    value.setText(value_text)
    return dialog, buttons, value


@pytest.mark.parametrize("outcome", ["apply", "cancel", "invalid"])
def test_alias_entry_edits_same_session_without_replacing_captured_forms(
    app: WorkflowsGuiBehavior, tmp_path: Path, qapp: QApplication, outcome: str
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    control = app.services.context_control
    control.add_component("Q2", "qubit/workflow-app-test", {})
    window = app.window
    window.prepare_forms()
    plan, knobs = window.plan_panel, window.tunables_panel
    env = plan.env
    assert env.parameters is not None
    assert env.parameters.resolve_path("$qubit.kind") == "Q1.kind"
    plan.state.write({("points",): 9}, env)
    knobs.state.write({("guess_us",): 17.0}, env)
    edit = knobs.findChild(QLineEdit, "form.input.guess_us")
    assert edit is not None
    edit.setText("1+")
    edit.editingFinished.emit()
    before_plan, before_knobs = plan.state.input.data, knobs.state.input.data
    base_plan, base_knobs = plan.state.base.data, knobs.state.base.data

    dialog, buttons, value = edit_alias_entry(
        window, "$qubit" if outcome == "invalid" else "Q2"
    )
    if outcome != "cancel":
        apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
        assert apply is not None
        apply.click()
        assert dialog.isVisible()
        assert value.text() == ("$qubit" if outcome == "invalid" else "Q2")
    expected = "Q2" if outcome == "apply" else "Q1"
    assert control.get_alias_rows() == (AliasRow("qubit", expected),)
    cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
    assert cancel is not None
    cancel.click()
    qapp.processEvents()
    assert control.get_alias_rows() == (AliasRow("qubit", expected),)
    status = window.findChild(QLabel, "workflow.source.status")
    assert status is not None and status.text() == f"Qubit: {expected}.kind"

    # Alias edits change the global source, not the fixed capture or local drafts.
    assert window.plan_panel is plan and window.tunables_panel is knobs
    assert env.parameters.resolve_path("$qubit.kind") == "Q1.kind"
    assert plan.state.input.data == before_plan and plan.state.base.data == base_plan
    assert (
        knobs.state.input.data == before_knobs and knobs.state.base.data == base_knobs
    )
    assert edit.text() == "1+" and knobs.editor.local_parse_errors() is not None
    if outcome == "apply":
        button(window, "workflow.source.use").click()
        new_env = window.plan_panel.env
        assert new_env.parameters is not None
        assert new_env.parameters.resolve_path("$qubit.kind") == "Q2.kind"
        assert window.plan_panel is not plan and window.tunables_panel is not knobs
        assert window.tunables_panel.env is new_env
        for panel, form in (
            (window.plan_panel, T1PlanForm),
            (window.tunables_panel, T1TunablesForm),
        ):
            assert panel.state.input.data == form.default_input(new_env).data
            assert panel.state.base.data == panel.state.input.data
        assert window.tunables_panel.editor.local_parse_errors() is None


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
    with pytest.raises(RoleResolutionError, match="qubit"):
        app.window.prepare_forms()
    assert app.window.plan_panel is plan and app.window.tunables_panel is knobs
    assert knobs.state.input.data == before
    assert edit.text() == "1+" and knobs.editor.local_parse_errors() is not None


@pytest.mark.parametrize("rows", [(), (AliasRow("qubit", ""),)])
def test_prepare_missing_or_unbound_qubit_names_alias_and_preserves_panels(
    app: WorkflowsGuiBehavior,
    tmp_path: Path,
    qapp: QApplication,
    rows: tuple[AliasRow, ...],
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    window = app.window
    window.prepare_forms()
    plan, knobs = window.plan_panel, window.tunables_panel
    knobs.state.write({("guess_us",): 17.0}, knobs.env)
    before = knobs.state.input.data
    app.services.context_control.set_alias_rows(rows)
    with pytest.raises(RoleResolutionError, match="qubit"):
        window.prepare_forms()
    assert window.plan_panel is plan and window.tunables_panel is knobs
    assert knobs.state.input.data == before


def test_prepare_accepts_other_component_kind_and_reports_consuming_field_error(
    app: WorkflowsGuiBehavior, tmp_path: Path, qapp: QApplication
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    control = app.services.context_control
    control.add_component("R1", "resonator/workflow-app-test", {})
    control.set_alias_rows((AliasRow("qubit", "R1"),))
    window = app.window
    old_plan = window.plan_panel
    button(window, "workflow.source.use").click()
    assert window.plan_panel is not old_plan
    assert "pi_length_us" not in window.plan_panel.state.input.data
    reply = window.inspect_form(
        "plan", batch={("pi_length_us",): {"expr": "$qubit.resonator"}}
    )
    assert not isinstance(reply, FormErrors)
    error = next(e for e in reply["errors"] if e["path"] == ["pi_length_us"])
    assert "$qubit.resonator" in error["message"]
    assert "missing parameter" in error["message"]


def test_prepare_captures_all_alias_rows_without_live_rebinding(
    app: WorkflowsGuiBehavior, tmp_path: Path, qapp: QApplication
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    context = app.services.context
    context.add_component("Q2", "qubit/workflow-app-test", {})
    target = context.read_context().target
    context.set_parameter(target, "Q1.ext.pi", 0.12)
    context.set_parameter(target, "Q2.ext.pi", 0.24)
    context.set_alias_rows((AliasRow("qubit", "Q1"), AliasRow("probe", "Q1")))
    app.window.prepare_forms()
    context.set_alias_rows((AliasRow("qubit", "Q2"), AliasRow("probe", "Q2")))
    context.set_parameter(target, "Q1.ext.pi", 0.99)
    reply = app.window.inspect_form(
        "plan", batch={("pi_length_us",): {"expr": "$probe.ext.pi"}}
    )
    assert not isinstance(reply, FormErrors)
    assert {tuple(v["path"]): v["value"] for v in reply["values"]}[
        ("pi_length_us",)
    ] == 0.12
    app.window.prepare_forms()
    reply = app.window.inspect_form(
        "plan", batch={("pi_length_us",): {"expr": "$probe.ext.pi"}}
    )
    assert not isinstance(reply, FormErrors)
    assert {tuple(v["path"]): v["value"] for v in reply["values"]}[
        ("pi_length_us",)
    ] == 0.24


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


def test_window_close_rejects_queued_connect_before_driver_registration(
    app: WorkflowsGuiBehavior, qapp: QApplication
) -> None:
    app.assemble(None)
    rejections: list[Exception] = []
    accepted: list[int] = []

    def queued_connect() -> None:
        try:
            accepted.append(
                app.services.device.start_connect_device(
                    ConnectDeviceRequest("FakeDevice", "late", "none")
                )
            )
        except RuntimeError as error:
            rejections.append(error)

    QTimer.singleShot(0, queued_connect)
    assert app.window.close()
    assert accepted == [], "queued connect escaped terminal admission"
    assert [type(error).__name__ for error in rejections] == [
        "BackgroundAdmissionClosedError"
    ]
    app.shutdown()
    qapp.processEvents()
    assert app.services.device.get_connected_devices() == {}
    assert app.services.device.get_active_device_operations() == ()


def test_window_close_drains_accepted_connect_and_delivery_before_driver_close(
    app: WorkflowsGuiBehavior, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.assemble(None)
    started = Event()
    release = Event()
    events: list[str] = []
    original_read = FakeDevice.get_info
    original_close = FakeDevice.close
    original_drain = BackgroundRunner.quiesce

    def read(driver: FakeDevice):
        events.append("worker")
        started.set()
        assert release.wait(2), "shutdown did not reach drain"
        info = original_read(driver)
        events.append("worker done")
        return info

    def close(driver: FakeDevice) -> None:
        events.append("driver close")
        original_close(driver)

    def drain(background: BackgroundRunner, timeout_ms: int = 5000) -> bool:
        release.set()
        return original_drain(background, timeout_ms)

    with monkeypatch.context() as patch:
        patch.setattr(FakeDevice, "get_info", read)
        patch.setattr(FakeDevice, "close", close)
        patch.setattr(BackgroundRunner, "quiesce", drain)
        on_device_connected(
            app.services.device, lambda _name: events.append("delivery")
        )
        app.services.device.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "flux", "none")
        )
        try:
            assert started.wait(2)
            assert app.window.close()
            assert events == ["worker", "worker done", "delivery", "driver close"]
            assert app.services.device.get_connected_devices() == {}
        finally:
            release.set()


@pytest.mark.parametrize("failure", ["drain", "driver"])
def test_failed_window_close_keeps_terminal_admission_closed_and_can_retry(
    app: WorkflowsGuiBehavior, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    app.assemble(None)
    original_panel = app.window.plan_panel
    with monkeypatch.context() as patch:
        if failure == "drain":
            patch.setattr(BackgroundRunner, "quiesce", lambda self: False)
        else:

            def reject_close(registry: DeviceManager) -> None:
                raise RuntimeError("driver close failed")

            patch.setattr(DeviceManager, "close_all_devices", reject_close)
        assert not app.window.close()
        message = app.window.findChild(QLabel, "workflow.run.message")
        assert message is not None and failure in message.text().lower()
        assert app.window.plan_panel is original_panel
        caught: Exception | None = None
        try:
            app.services.device.start_connect_device(
                ConnectDeviceRequest("FakeDevice", "late", "none")
            )
        except RuntimeError as error:
            caught = error
        assert caught is not None, "shutdown failure must not reopen admission"
        assert type(caught).__name__ == "BackgroundAdmissionClosedError"
        assert app.services.device.get_active_device_operations() == ()
    assert app.window.close()
    with pytest.raises(RuntimeError, match="background") as rejected:
        app.services.device.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "after", "none")
        )
    assert type(rejected.value).__name__ == "BackgroundAdmissionClosedError"
