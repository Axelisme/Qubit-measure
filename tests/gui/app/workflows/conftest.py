"""Isolated registrations, session owners and transports for workflow host tests."""

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
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.services.progress import (
    BoundProgressFactory,
    ProgressService,
)
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.workflows.server import build_workflow_tools
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.resources.entry.fakes import Driver, registry_state, restore_registry

from ._recording_transport import WindowTransport, WireRoute
from ._reload_source import InjectedCatalog
from ._start_fakes import StartHarness
from ._synthetic import Component, Context, Plan, Tunables, inline_input
from ._window_support import QUBIT_KIND, RpcRequest, WindowSession


@pytest.fixture
def progress_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], list[int]]:
    """Observe the run-lifetime factory/discard contract using the real service."""
    created: list[int] = []
    discarded: list[int] = []
    make_factory = ProgressService.make_factory
    discard_operation = ProgressService.discard_operation

    def make(service: ProgressService, token: int, owner: str) -> BoundProgressFactory:
        created.append(token)
        return make_factory(service, token, owner)

    def discard(service: ProgressService, token: int) -> None:
        discarded.append(token)
        discard_operation(service, token)

    monkeypatch.setattr(ProgressService, "make_factory", make)
    monkeypatch.setattr(ProgressService, "discard_operation", discard)
    return created, discarded


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


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
    catalog_source: InjectedCatalog,
) -> Iterator[tuple[WorkflowWindow[Context], StartHarness, ResultEntry]]:
    before = registry_state()
    kind = "qubit/workflow-window-test"
    component_registry.register(kind, Component)
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
        context.set_parameter(context.read_context().target, "general.ext.value", 20.0)
        harness.state.set_context(
            replace(harness.state.session_env, soc=object(), soccfg=make_mock_soccfg())
        )
        window = WorkflowWindow(
            harness.state,
            harness.services,
            harness.owner,
            registrations=catalog_source.catalog.attachments,
            loader=catalog_source.loader,
        )
        window.plan_panel.state.reset(inline_input())
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
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkflowWindow[Context], StartHarness, ResultEntry]:
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
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> Iterator[WireRoute]:
    """Route original requests off-owner and pump the real owner for RPC and MCP."""
    window, harness, _entry = attachment
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
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> list[tuple[int, float, int]]:
    """One committed step, followed by an owner-delivered paused step."""
    window, harness, _entry = run_attachment
    attempts: list[tuple[int, float, int]] = []

    def init(_env: InitEnv[Context], _plan: Plan) -> int:
        return 0

    @workflow("tuning-paused", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[Context],
        plan: Plan,
        tun: Tunables,
        state: int,
    ) -> Step[int, float]:
        del plan
        if state == 2:
            return Done()
        attempts.append((state, tun.sample.value, tun.sample.count))
        if state == 1 and len(attempts) == 2:
            window.pause("tuning")
            yield from env.wait_until(env.started_at)
        return Next(tun.sample.value, state + 1)

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


@pytest.fixture
def window_session(qapp: QApplication, tmp_path: Path) -> Iterator[WindowSession]:
    """Build a Window over synthetic workflows; restore the component registry."""
    before = registry_state()
    component_registry.register(QUBIT_KIND, Driver)
    installed = registry_state()
    session = None
    try:
        session = WindowSession(tmp_path)
        yield session
    finally:
        try:
            if session is not None:
                session.close(qapp)
            assert registry_state() == installed, "window test polluted registries"
        finally:
            restore_registry(before)
