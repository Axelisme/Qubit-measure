"""Public EMPTY startup, source capture and composition lifecycle seams."""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)
from zcu_tools.device import FakeDevice
from zcu_tools.device.manager import DeviceManager
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.remote import WorkflowRemoteControlAdapter
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.runtime import GuiAssembly
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleResolutionError,
    RoleSpec,
    component_registry,
    role_registry,
)
from zcu_tools.resources.entry.aliases import EntryAliases

from tests.gui._completion_helpers import on_device_connected
from tests.gui._qt_waits import qt_until
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.app.workflows._synthetic import PlanForm, TunablesForm
from tests.resources.entry.fakes import registry_state, restore_registry

from ._sample_input import sample_input
from ._synthetic import ALIAS_PRESET, Component, Context, OtherComponent, build_registry


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.fixture
def app(qapp: QApplication, tmp_path: Path) -> Iterator[WorkflowsGuiBehavior[Context]]:
    before = registry_state()
    try:
        component_registry.register("qubit/workflow-app-test", Component)
        component_registry.register("resonator/workflow-app-test", OtherComponent)
        try:
            role_registry.get("qubit")
        except ValueError:
            role_registry.register("qubit", RoleSpec("qubit/*"))
        installed = registry_state()
        behavior = WorkflowsGuiBehavior(
            templates={},
            registrations=build_registry(),
            alias_presets={"default": ALIAS_PRESET},
            project_root=str(tmp_path),
            driver_factory=lambda type_name, address: FakeDevice(address=address),
        )
        try:
            yield behavior
        finally:
            # Retain workers through public shutdown before releasing Qt objects.
            # Cleanup failures propagate; the outer finally still restores registries.
            if hasattr(behavior, "window"):
                behavior.window.close()
                behavior.shutdown()
                behavior.window.deleteLater()
                qapp.processEvents()
            assert registry_state() == installed, "app test polluted definitions"
    finally:
        restore_registry(before)


@pytest.mark.parametrize("saved", [None, (), (AliasRow("saved", "R1"),)])
def test_named_presets_reach_workflows_composition_without_overwriting_saved_rows(
    saved: tuple[AliasRow, ...] | None, tmp_path: Path, qapp: QApplication
) -> None:
    presets = {
        "default": (AliasRow("drive", "Q7"), AliasRow("sense", "$drive.resonator")),
        "probe": (AliasRow("probe", "Q2"), AliasRow("spare", "")),
    }
    entry = ResultEntry.create(
        "sample", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    if saved is not None:
        entry.save_aliases(
            EntryAliases(aliases={row.alias: row.value for row in saved})
        )
    try:
        behavior = WorkflowsGuiBehavior(
            templates={},
            registrations=build_registry(),
            alias_presets=presets,
            project_root=str(tmp_path),
            cache_dir=tmp_path / "cache",
            clean=True,
        )
    except TypeError as error:
        pytest.fail(f"Public composition must accept named alias presets: {error}")
    try:
        behavior.assemble(None)
        assert behavior.alias_presets == presets
        context = behavior.services.context
        assert context.get_alias_rows() == presets["default"]
        context.install_entry(entry)
        assert context.get_alias_rows() == (
            presets["default"] if saved is None else saved
        )
        # Edits to the installed table do not edit caller preset definitions.
        context.set_alias_rows((AliasRow("replacement", "R2"),))
        assert behavior.alias_presets == presets
    finally:
        behavior.shutdown()
        behavior.window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("presets", [None, {"default": ()}])
def test_workflows_empty_initial_table_is_explicit_or_has_no_catalog(
    presets: dict[str, tuple[AliasRow, ...]] | None, tmp_path: Path, qapp: QApplication
) -> None:
    try:
        behavior = WorkflowsGuiBehavior(
            templates={},
            registrations=build_registry(),
            alias_presets=presets,
            cache_dir=tmp_path,
            clean=True,
        )
    except TypeError as error:
        pytest.fail(f"Public composition must accept named alias presets: {error}")
    try:
        behavior.assemble(None)
        assert behavior.services.context.get_alias_rows() == ()
        assert behavior.alias_presets == presets
    finally:
        behavior.shutdown()
        behavior.window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("presets", [{}, {"probe": (AliasRow("probe", "Q2"),)}])
def test_workflows_behavior_rejects_catalog_without_default(
    presets: dict[str, tuple[AliasRow, ...]],
) -> None:
    try:
        with pytest.raises(ValueError, match="default"):
            WorkflowsGuiBehavior(
                templates={}, registrations=build_registry(), alias_presets=presets
            )
    except TypeError as error:
        pytest.fail(f"Named alias preset admission is missing: {error}")


def button(window: WorkflowWindow[Context], name: str) -> QPushButton:
    found = window.findChild(QPushButton, name)
    assert found is not None
    return found


def setup_button(window: WorkflowWindow[Context], text: str) -> QPushButton:
    return next(
        b for b in window.setup_dialog.findChildren(QPushButton) if b.text() == text
    )


def open_point(
    app: WorkflowsGuiBehavior[Context], tmp_path: Path, qapp: QApplication
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

    window = WorkflowWindow(
        harness.state,
        harness.services,
        harness.owner,
        templates={},
        alias_presets=None,
        registrations=build_registry(),
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
    app: WorkflowsGuiBehavior[Context],
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
        app.window.plan_panel.state.input.data == PlanForm.default_input(FormEnv()).data
    )
    assert isinstance(app.window.plan_panel.state.result(FormEnv()), FormErrors)
    assert isinstance(app.window.tunables_panel.state.result(FormEnv()), TunablesForm)
    with pytest.raises(FailedPreconditionError):
        app.window.start("not-ready")
    assert app.window.controller.operation_id is None


def test_after_show_reuses_nonmodal_setup_without_selecting_entry(
    app: WorkflowsGuiBehavior[Context],
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
    app: WorkflowsGuiBehavior[Context], tmp_path: Path, qapp: QApplication
) -> None:
    app.assemble(None)
    window = app.window
    old = window.plan_panel
    open_point(app, tmp_path, qapp)
    context = app.services.context
    target = context.read_context().target
    context.set_parameter(target, "general.ext.value", 31.0)
    window.tunables_panel.state.write({("sample", "value"): 7.0}, FormEnv())
    # Source selection itself has not replaced input/capture.
    assert window.plan_panel is old and window.plan_panel.env.parameters is None
    button(window, "workflow.source.use").click()
    env = window.tunables_panel.env
    assert window.plan_panel is not old and env.parameters is not None
    assert window.plan_panel.env is env
    assert (
        window.tunables_panel.state.input.data == TunablesForm.default_input(env).data
    )
    assert (
        window.tunables_panel.state.base.data == window.tunables_panel.state.input.data
    )
    assert window.plan_panel.state.base.data == PlanForm.default_input(env).data
    knobs = window.tunables_panel.state.result(env)
    assert isinstance(knobs, TunablesForm) and knobs.sample.value == 31.0
    context.set_parameter(target, "general.ext.value", 45.0)
    knobs = window.tunables_panel.state.result(env)
    assert isinstance(knobs, TunablesForm) and knobs.sample.value == 31.0
    window.tunables_panel.state.write({("sample", "value"): 8.0}, env)
    edit = window.tunables_panel.findChild(QLineEdit, "form.input.sample.value")
    assert edit is not None
    edit.setText("1+")
    edit.editingFinished.emit()
    assert window.tunables_panel.editor.local_parse_errors() is not None
    window.prepare_forms()
    assert window.tunables_panel.editor.local_parse_errors() is None
    knobs = window.tunables_panel.state.result(window.tunables_panel.env)
    assert isinstance(knobs, TunablesForm) and knobs.sample.value == 45.0
    assert (
        window.tunables_panel.state.base.data == window.tunables_panel.state.input.data
    )


def test_capture_failure_preserves_both_panels_and_local_input(
    app: WorkflowsGuiBehavior[Context], tmp_path: Path, qapp: QApplication
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    app.window.prepare_forms()
    plan, knobs = app.window.plan_panel, app.window.tunables_panel
    knobs.state.write({("sample", "value"): 17.0}, knobs.env)
    edit = knobs.findChild(QLineEdit, "form.input.sample.value")
    assert edit is not None
    edit.setText("1+")
    edit.editingFinished.emit()
    before = knobs.state.input.data
    app.services.settings.open_entry(EntryRequest("sample"))
    app.services.context.new_point(
        PointCreateRequest("empty", "", CommandOrigin("gui"))
    )
    with pytest.raises(RoleResolutionError, match="qubit"):
        app.window.prepare_forms()
    assert app.window.plan_panel is plan and app.window.tunables_panel is knobs
    assert knobs.state.input.data == before
    assert edit.text() == "1+" and knobs.editor.local_parse_errors() is not None


@pytest.mark.parametrize("rows", [(), (AliasRow("qubit", ""),)])
def test_prepare_missing_or_unbound_qubit_names_alias_and_preserves_panels(
    app: WorkflowsGuiBehavior[Context],
    tmp_path: Path,
    qapp: QApplication,
    rows: tuple[AliasRow, ...],
) -> None:
    app.assemble(None)
    open_point(app, tmp_path, qapp)
    window = app.window
    window.prepare_forms()
    plan, knobs = window.plan_panel, window.tunables_panel
    knobs.state.write({("sample", "value"): 17.0}, knobs.env)
    before = knobs.state.input.data
    app.services.context_control.set_alias_rows(rows)
    with pytest.raises(RoleResolutionError, match="qubit"):
        window.prepare_forms()
    assert window.plan_panel is plan and window.tunables_panel is knobs
    assert knobs.state.input.data == before


def test_prepare_accepts_other_component_kind_and_reports_consuming_field_error(
    app: WorkflowsGuiBehavior[Context], tmp_path: Path, qapp: QApplication
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
    assert "scale" not in window.plan_panel.state.input.data
    reply = window.inspect_form(
        "plan", batch={("scale",): {"expr": "$qubit.resonator"}}
    )
    assert not isinstance(reply, FormErrors)
    error = next(e for e in reply["errors"] if e["path"] == ["scale"])
    assert "$qubit.resonator" in error["message"]
    assert "missing parameter" in error["message"]


def test_prepare_captures_all_alias_rows_without_live_rebinding(
    app: WorkflowsGuiBehavior[Context], tmp_path: Path, qapp: QApplication
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
        "plan", batch={("scale",): {"expr": "$probe.ext.pi"}}
    )
    assert not isinstance(reply, FormErrors)
    assert {tuple(v["path"]): v["value"] for v in reply["values"]}[("scale",)] == 0.12
    app.window.prepare_forms()
    reply = app.window.inspect_form(
        "plan", batch={("scale",): {"expr": "$probe.ext.pi"}}
    )
    assert not isinstance(reply, FormErrors)
    assert {tuple(v["path"]): v["value"] for v in reply["values"]}[("scale",)] == 0.24


def test_composition_rpc_reads_same_draft_without_socket(
    app: WorkflowsGuiBehavior[Context], qapp: QApplication
) -> None:
    assembly = app.assemble(ControlOptions(port=0))
    adapter = assembly.control_adapter
    assert isinstance(adapter, WorkflowRemoteControlAdapter)
    app.window.tunables_panel.state.write({("sample", "value"): 73.0}, FormEnv())
    link = ClientLink("app-contract", token_required=False)
    adapter.on_client_open(link)
    try:
        with ThreadPoolExecutor(max_workers=1) as io:
            request = Request(
                id="app-read", method="workflow.form", params={"form": "tunables"}
            )
            future = io.submit(adapter.route, link, request)
            # The submitted route marshals to this real Qt owner before it returns.
            qt_until(future.done, label="composition RPC")
            future.result()
            reply = json.loads(link.outbound.get_nowait())
            assert sample_input(reply["result"]["input"])["value"] == 73.0
    finally:
        adapter.on_client_close(link, on_owner_thread=True)


def test_window_close_drains_and_closes_its_session_registry(
    app: WorkflowsGuiBehavior[Context], qapp: QApplication
) -> None:
    app.assemble(None)
    token = app.services.device.start_connect_device(
        ConnectDeviceRequest("YOKOGS200", "flux", "none")
    )
    # start_connect_device accepted a FakeDevice worker and queued owner delivery.
    qt_until(
        lambda: app.services.device.get_connected_devices(),
        label=f"FakeDevice connect {token}",
    )
    assert app.window.close()
    assert not app.services.device.get_connected_devices()
    app.shutdown()


def test_shutdown_failure_rejects_close_and_keeps_forms(
    app: WorkflowsGuiBehavior[Context], monkeypatch: pytest.MonkeyPatch
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
    app: WorkflowsGuiBehavior[Context], monkeypatch: pytest.MonkeyPatch
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
    app: WorkflowsGuiBehavior[Context], qapp: QApplication
) -> None:
    app.assemble(None)
    rejections: list[Exception] = []
    accepted: list[int] = []

    def queued_connect() -> None:
        try:
            accepted.append(
                app.services.device.start_connect_device(
                    ConnectDeviceRequest("YOKOGS200", "late", "none")
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
    app: WorkflowsGuiBehavior[Context], monkeypatch: pytest.MonkeyPatch
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
        # The test closes the Window after arrival; its drain wrapper releases us.
        assert release.wait(30), "shutdown did not reach drain"
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
            ConnectDeviceRequest("YOKOGS200", "flux", "none")
        )
        try:
            # start_connect_device accepted the worker that calls the driver above.
            assert started.wait(30)
            assert app.window.close()
            assert events == ["worker", "worker done", "delivery", "driver close"]
            assert app.services.device.get_connected_devices() == {}
        finally:
            release.set()


@pytest.mark.parametrize("failure", ["drain", "driver"])
def test_failed_window_close_keeps_terminal_admission_closed_and_can_retry(
    app: WorkflowsGuiBehavior[Context], monkeypatch: pytest.MonkeyPatch, failure: str
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
                ConnectDeviceRequest("YOKOGS200", "late", "none")
            )
        except RuntimeError as error:
            caught = error
        assert caught is not None, "shutdown failure must not reopen admission"
        assert type(caught).__name__ == "BackgroundAdmissionClosedError"
        assert app.services.device.get_active_device_operations() == ()
    assert app.window.close()
    with pytest.raises(RuntimeError, match="background") as rejected:
        app.services.device.start_connect_device(
            ConnectDeviceRequest("YOKOGS200", "after", "none")
        )
    assert type(rejected.value).__name__ == "BackgroundAdmissionClosedError"
