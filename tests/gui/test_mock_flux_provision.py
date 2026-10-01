"""Measure GUI composition exposes explicit simulated-environment setup."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.services.device import (
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.gui.session.services.predictor_from_sim import (
    build_predictor_from_simparams,
)
from zcu_tools.gui.session.services.simulated_environment import (
    FAKE_FLUX_DEVICE_NAME,
    FAKE_FLUX_INITIAL_VALUE,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.program.v2.mocksoc import MockQickSoc
from zcu_tools.resources.context import MetaDict, ModuleLibrary


@pytest.fixture
def state() -> State:
    return State(SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None))


@pytest.fixture
def make_controller(qapp: object) -> Iterator[Callable[[State], Controller]]:
    controllers: list[Controller] = []

    def make(state: State) -> Controller:
        view = MagicMock()
        view.make_run_container.return_value = None
        controller = Controller(
            state=state,
            registry=Registry(),
            io_manager=IOManager(),
            view=view,
            bus=BaseEventBus(),
        )
        controllers.append(controller)
        return controller

    yield make
    for controller in controllers:
        controller._background_svc.quiesce()


@pytest.fixture
def ctrl(make_controller: Callable[[State], Controller], state: State) -> Controller:
    return make_controller(state)


def _pump_until(condition: Callable[[], bool]) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    pytest.fail("Simulated environment did not reach its terminal state")


def _start(ctrl: Controller) -> None:
    finished: list[bool] = []
    errors: list[str] = []
    ctrl.setup_control.bind_connection_outcome(
        lambda: finished.append(True), errors.append
    )
    ctrl.setup_control.start_simulated_environment()
    _pump_until(lambda: bool(finished or errors))
    assert not errors


def _reader(state: State) -> Callable[[], float]:
    soc = state.session_env.soc
    assert isinstance(soc, MockQickSoc) and soc.flux_source is not None
    return soc.flux_source


def _set_value(ctrl: Controller, value: float) -> None:
    ctrl.device_control.start_setup_device(
        SetupDeviceRequest(
            FAKE_FLUX_DEVICE_NAME, FakeDeviceInfo(address="none", value=value)
        )
    )
    _pump_until(
        lambda: (
            ctrl.device_control.get_cached_device_value(FAKE_FLUX_DEVICE_NAME) == value
        )
    )


def test_environment_publishes_bound_source_and_matching_predictor(
    ctrl: Controller, state: State
) -> None:
    _start(ctrl)
    assert _reader(state)() == FAKE_FLUX_INITIAL_VALUE
    snapshot = ctrl.device_control.get_device_snapshot(FAKE_FLUX_DEVICE_NAME)
    assert snapshot is not None and snapshot.status == DeviceStatus.CONNECTED
    assert ctrl.get_device_unit(FAKE_FLUX_DEVICE_NAME) == "none"
    soc = state.session_env.soc
    assert isinstance(soc, MockQickSoc) and soc.sim_params is not None
    predictor = state.session_env.predictor
    assert predictor is not None
    reference = build_predictor_from_simparams(soc.sim_params)
    assert predictor.predict_freq(FAKE_FLUX_INITIAL_VALUE) == pytest.approx(
        reference.predict_freq(FAKE_FLUX_INITIAL_VALUE), abs=1e-6
    )


def test_repeated_entry_preserves_resources_and_current_value(
    ctrl: Controller, state: State
) -> None:
    _start(ctrl)
    reader = _reader(state)
    _set_value(ctrl, 0.123)
    soc = state.session_env.soc
    _start(ctrl)
    assert _reader(state) == reader
    assert reader() == 0.123
    assert state.session_env.soc is soc


def test_disconnected_source_is_recreated_and_rebound(
    ctrl: Controller, state: State
) -> None:
    _start(ctrl)
    previous = _reader(state)
    ctrl.device_control.start_disconnect_device(
        DisconnectDeviceRequest(FAKE_FLUX_DEVICE_NAME)
    )
    _pump_until(
        lambda: (
            (dev := state.get_device(FAKE_FLUX_DEVICE_NAME)) is not None
            and dev.status == DeviceStatus.MEMORY_ONLY
        )
    )
    _start(ctrl)
    assert _reader(state) != previous
    assert _reader(state)() == FAKE_FLUX_INITIAL_VALUE


def test_separate_controllers_do_not_share_device_registry(
    ctrl: Controller, state: State, make_controller: Callable[[State], Controller]
) -> None:
    _start(ctrl)
    other_state = State(
        SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    )
    other = make_controller(other_state)
    _start(other)
    _set_value(ctrl, 0.25)
    assert _reader(state)() == 0.25
    assert _reader(other_state)() == FAKE_FLUX_INITIAL_VALUE
