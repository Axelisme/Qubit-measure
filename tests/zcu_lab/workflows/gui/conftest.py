"""Real Qt/native simulator ownership for workflow integration contracts."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.resources.entry.fakes import registry_state, restore_registry
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.components import QubitSchema
from zcu_lab.workflows.gui.app import WorkflowsGuiBehavior
from zcu_lab.workflows.gui.remote import T1RemoteControlAdapter

from ._plan_inputs import inline_input, library_env
from ._sim_session import SimSession, connect_tools, wait_owner


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
) -> Iterator[tuple[WorkflowsGuiBehavior, T1RemoteControlAdapter]]:
    """Keep real Qt workers alive through cancellation, delivery and drain."""
    del native_registry
    (tmp_path / "result").mkdir()
    app = WorkflowsGuiBehavior(project_root=str(tmp_path))
    assembly = app.assemble(ControlOptions(port=0))
    adapter = assembly.control_adapter
    assert isinstance(adapter, T1RemoteControlAdapter)
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
    native_app: tuple[WorkflowsGuiBehavior, T1RemoteControlAdapter],
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
