"""Fixtures shared only by real point and context-snapshot socket contracts."""

from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

from ._form_rpc import FormRpc, ImmediateScheduler
from ._helpers import Fixture, open_client
from ._interactive_support import InteractiveFixture
from ._point_application import PointAdapter, PointApplication


@pytest.fixture(scope="module")
def point_models() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed, "point RPC module polluted registries"
    finally:
        restore_registry(before)


@pytest.fixture
def point_registry_guard(request: pytest.FixtureRequest) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def application(
    qapp, tmp_path: Path, point_models, point_registry_guard
) -> Generator[PointApplication]:
    registry = Registry()
    registry.register("pointed", PointAdapter)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = Controller(
        state, registry, None, BaseEventBus(), project_root=str(tmp_path)
    )
    remote = RemoteControlAdapter(
        controller, ControlOptions(port=0, app_slug=""), controller.owner_scheduler
    )
    client = open_client(remote.start())
    try:
        yield PointApplication(state, controller, remote, client)
    finally:
        client.close()
        remote.stop()
        controller.close()
        for tab_id in tuple(state.list_tab_ids()):
            controller.tab_control.close_tab(tab_id)
        qapp.processEvents()


@pytest.fixture
def fx(qapp, tmp_path):
    fixture = InteractiveFixture(project_root=str(tmp_path))
    fixture.start()
    yield fixture
    fixture.ctrl.close()
    for widget in fixture.widgets:
        widget.teardown()
        widget.deleteLater()
    fixture.stop()
    qapp.processEvents()


@pytest.fixture
def form_rpc(tmp_path, qapp) -> Iterator[tuple[Fixture, str, FormRpc]]:
    fixture = Fixture(project_root=str(tmp_path / "project"))
    tab_id = fixture.ctrl.new_tab("fake")
    remote = RemoteControlAdapter(
        fixture.ctrl,
        ControlOptions(port=0),
        ImmediateScheduler(),
        render_view=fixture.view,
    )
    rpc = FormRpc(remote)
    try:
        yield fixture, tab_id, rpc
    finally:
        fixture.ctrl.close()
        fixture.ctrl.tab_control.close_tab(tab_id)
        remote.on_client_close(rpc.link, on_owner_thread=True)
