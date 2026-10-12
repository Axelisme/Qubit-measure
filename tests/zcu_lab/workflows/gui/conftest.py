"""Real Qt/native simulator ownership for workflow integration contracts."""

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import monotonic
from types import TracebackType

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.device import DeviceInfo
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.remote import WorkflowRemoteControlAdapter
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.workflows.server import build_workflow_tools
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.resources.entry.fakes import registry_state, restore_registry
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab import ALIAS_PRESET
from zcu_lab.components import QubitSchema
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.coherence import CoherencePlan, CoherenceTunables
from zcu_lab.workflows.registry import build_registry

from ._plan_inputs import inline_input, library_env
from ._recording_transport import WindowTransport, WireRoute
from ._reload_source import InjectedCatalog
from ._sim_session import SimSession, connect_tools, wait_owner
from ._window_support import RpcRequest


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.fixture
def native_registry() -> Iterator[str]:
    """Restore shared component/role definitions even when app teardown fails."""
    before = registry_state()
    kind = "qubit/native-workflow-test"
    component_registry.register(kind, QubitSchema)
    try:
        role_registry.get("qubit")
    except ValueError:
        role_registry.register("qubit", RoleSpec("qubit/*"))
    installed = registry_state()
    try:
        yield kind
        assert registry_state() == installed, "native simulator polluted definitions"
    finally:
        restore_registry(before)


@pytest.fixture
def native_app(
    qapp: QApplication, tmp_path: Path, native_registry: str
) -> Iterator[
    tuple[WorkflowsGuiBehavior[Calibrated], WorkflowRemoteControlAdapter[Calibrated]]
]:
    """Keep real Qt workers alive through cancellation, delivery and drain."""
    del native_registry
    (tmp_path / "result").mkdir()
    app = WorkflowsGuiBehavior(
        registrations=build_registry(),
        alias_rows=ALIAS_PRESET,
        project_root=str(tmp_path),
        cache_dir=tmp_path / "cache",
        clean=True,
    )
    assembly = app.assemble(ControlOptions(port=0))
    adapter = assembly.control_adapter
    assert isinstance(adapter, WorkflowRemoteControlAdapter)
    try:
        yield app, adapter
    finally:
        controller = app.window.controller
        if controller.operation_id is not None:
            assert controller.run_id is not None
            app.window.stop(controller.run_id)
            wait_owner(qapp, lambda: controller.operation_id is None)
        assert app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


@pytest.fixture
def sim_session(
    native_app: tuple[
        WorkflowsGuiBehavior[Calibrated], WorkflowRemoteControlAdapter[Calibrated]
    ],
    native_registry: str,
    qapp: QApplication,
    inputs: Inputs,
    tmp_path: Path,
) -> Iterator[SimSession]:
    """Bind native source/modules/simboard and MCP, replacing only TCP."""
    app, adapter = native_app
    entry = ResultEntry.create(
        "sample", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    entry.setup.add_component("Q1", kind=native_registry)
    entry.new_point("source", description="native simulator")
    context = app.services.context
    context.install_entry(entry)
    context.use_point(PointUseRequest("source"))
    context.set_alias_rows((AliasRow("qubit", "Q1"),))
    library = library_env(inputs).library
    assert library is not None
    target = context.read_context().target
    for key in ("Q1.pulses.pi", "Q1.readout.ro"):
        context.set_module(target, key, library.module_spec(key))
    app.state.set_context(
        replace(app.state.session_env, soc=inputs.soc, soccfg=inputs.soccfg)
    )
    app.services.device.start_connect_device(
        ConnectDeviceRequest("FakeDevice", "flux", "none")
    )
    wait_owner(
        qapp,
        lambda: (
            "flux" in app.services.device.get_connected_devices()
            and app.services.device.active_operation_token("flux") is None
        ),
    )
    driver = app.services.device.get_connected_devices()["flux"]
    assert isinstance(driver, FakeDevice)
    driver.setup(
        FakeDeviceInfo(address="none", value=inputs.flux_value, rampstep=1.0),
        progress=False,
    )
    inputs.soc.set_flux_source(driver.get_value)
    app.window.prepare_forms()
    app.window.plan_panel.state.reset(inline_input(inputs))
    try:
        with ThreadPoolExecutor(max_workers=1) as io:
            tools, transport, bridge, link = connect_tools(adapter, qapp, tmp_path, io)
            try:
                yield SimSession(app, entry, tools, transport)
            finally:
                bridge.disconnect()
                adapter.on_client_close(link, on_owner_thread=True)
    finally:
        inputs.soc.set_flux_source(None)
        adapter.stop()


@pytest.fixture
def catalog_source(tmp_path: Path) -> Iterator[InjectedCatalog]:
    """Keep public catalog/loader injection isolated from other test modules."""
    source = InjectedCatalog(tmp_path)
    try:
        yield source
    finally:
        source.close()


@pytest.fixture
def attachment(
    qapp: QApplication,
    tmp_path: Path,
    inputs: Inputs,
    catalog_source: InjectedCatalog,
) -> Iterator[tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry]]:
    before = registry_state()
    kind = "qubit/workflow-window-test"
    component_registry.register(kind, QubitSchema)
    try:
        role_registry.get("qubit")
    except ValueError:
        role_registry.register("qubit", RoleSpec("qubit/*"))
    installed = registry_state()
    harness = StartHarness()
    window = None
    try:
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind=kind)
        entry.new_point("source", description="captured description")
        context = harness.services.context
        context.install_entry(entry)
        context.use_point(PointUseRequest("source"))
        context.set_alias_rows((AliasRow("qubit", "Q1"),))
        context.set_parameter(
            context.read_context().target, "general.ext.guess_us", 20.0
        )
        harness.state.set_context(
            replace(
                harness.state.session_env,
                soc=inputs.soc,
                soccfg=inputs.soccfg,
            )
        )
        harness.services.device.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "flux", "none")
        )
        harness.background.execute()
        harness.background.deliver()
        window = WorkflowWindow(
            harness.state,
            harness.services,
            harness.owner,
            registrations=catalog_source.catalog.attachments,
            loader=catalog_source.loader,
        )
        window.plan_panel.state.reset(inline_input(inputs))
        yield window, harness, entry
    finally:
        if window is not None:
            if harness.background.work is not None:
                controller = window.controller
                assert controller is not None and controller.operation_id is not None
                harness.handles.cancel(controller.operation_id)
                harness.background.execute()
                harness.background.deliver()
            window.close()
            window.deleteLater()
            qapp.processEvents()
        try:
            assert registry_state() == installed, (
                "workflow window test polluted registries"
            )
        finally:
            restore_registry(before)


@pytest.fixture
def run_attachment(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry]:
    """Supply typed cached native form info for Window/Controller ownership.

    The substitute does not replace Runner, Controller or native Engine.
    """
    _window, harness, _entry = attachment
    native: dict[str, DeviceInfo] = {"flux": FakeDeviceInfo(address="none", value=0.0)}
    monkeypatch.setattr(
        harness.services.device, "capture_native_device_info", lambda: native
    )
    return attachment


@pytest.fixture
def escaped_qt_errors(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Observe Qt's public exception hook without aborting the test process."""
    errors: list[BaseException] = []

    def record(
        _kind: type[BaseException],
        error: BaseException,
        _traceback: TracebackType | None,
    ) -> None:
        errors.append(error)

    monkeypatch.setattr(sys, "excepthook", record)
    return errors


@pytest.fixture
def settlements(
    run_attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Record real public settlement calls without replacing their behavior."""
    _window, harness, _entry = run_attachment
    tokens: list[int] = []
    settle = harness.handles.settle

    def record(token: int, outcome: OperationOutcome) -> None:
        tokens.append(token)
        settle(token, outcome)

    monkeypatch.setattr(harness.handles, "settle", record)
    return tokens


@pytest.fixture
def wire_route(
    attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    inputs: Inputs,
) -> Iterator[WireRoute]:
    """Route original requests off-owner and pump the real owner for RPC and MCP."""
    window, harness, _entry = attachment
    context = harness.services.context
    library = library_env(inputs).library
    assert library is not None
    target = context.read_context().target
    for key in ("Q1.pulses.pi", "Q1.readout.ro"):
        context.set_module(target, key, library.module_spec(key))
    service = window.create_remote_service(ControlOptions(port=0), harness.bus)
    link = ClientLink("workflow-contract", token_required=False)
    service.on_client_open(link)
    try:
        with ThreadPoolExecutor(max_workers=1) as io:

            def request(original: Request) -> dict[str, object]:
                pending = io.submit(service.route, link, original)
                deadline = monotonic() + 3
                while not pending.done():
                    assert monotonic() < deadline, "RPC did not settle on the owner"
                    harness.owner.pump_once(block=True, timeout=0.01)
                pending.result(timeout=1)
                return json.loads(link.outbound.get(timeout=1))

            yield request
    finally:
        service.on_client_close(link, on_owner_thread=True)
        service.stop()


@pytest.fixture
def rpc(
    run_attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    wire_route: WireRoute,
) -> RpcRequest:
    """Keep existing RPC cases on their controlled live-device seam."""
    return lambda method, params: wire_route(Request("request", method, params))


@pytest.fixture
def mcp_transport(wire_route: WireRoute) -> WindowTransport:
    return WindowTransport(wire_route)


@pytest.fixture
def mcp_tools(mcp_transport: WindowTransport, tmp_path: Path) -> Iterator[ToolTable]:
    """Bind real MCP tools to this test's Window without launching any process."""
    config = MCPBridgeConfig(
        tool_prefix="workflows_",
        server_display_name="workflows",
        server_instructions="Workflow RPC contract",
        app_name="workflows",
        default_port=8768,
        mcp_version=1,
        wire_version=1,
        pid_file=tmp_path / "gui.pid",
        log_file=tmp_path / "gui.log",
        run_script_name="run_workflows_gui.py",
        app_slug="workflows",
    )
    bridge = McpBridge(config, transport=mcp_transport)
    try:
        yield build_workflow_tools(config, bridge, tmp_path)
    finally:
        bridge.disconnect()


@pytest.fixture
def paused_tuning(
    run_attachment: tuple[WorkflowWindow[Calibrated], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> list[tuple[int, float, int]]:
    """One committed step, followed by an owner-delivered paused step."""
    window, harness, _entry = run_attachment
    attempts: list[tuple[int, float, int]] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> int:
        return 0

    @workflow(
        "tuning-paused", plan=CoherencePlan, tunables=CoherenceTunables, init=init
    )
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: int,
    ) -> Step[int, float]:
        del plan
        if state == 2:
            return Done()
        attempts.append((state, tun.t1.guess_us, tun.t1.reps))
        if state == 1 and len(attempts) == 2:
            window.pause("tuning")
            yield from env.wait_until(env.started_at)
        return Next(tun.t1.guess_us, state + 1)

    catalog_source.install(step, window)
    token = window.start("tuning")
    assert isinstance(token, int)
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "paused"
    assert window.tuning_panel is not None
    assert window.controller.operation_id == token
    assert not window.controller.has_pending_segment
    assert harness.gate.snapshot() and harness.handles.live_count() == 1
    assert harness.handles.known_outcome(token) is None
    return attempts
