"""Autofluxdep uses the shared explicit simulated-environment entry."""

import time
from collections.abc import Iterator

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.session.services.device import SetupDeviceRequest
from zcu_tools.gui.session.services.simulated_environment import (
    FAKE_FLUX_DEVICE_NAME,
    FAKE_FLUX_INITIAL_VALUE,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.program.v2.mocksoc import MockQickSoc

from tests.gui.app.autofluxdep._helpers import build_test_core as build_core
from tests.gui.app.autofluxdep._helpers import connect_mock


@pytest.fixture
def ctrl(qapp: object) -> Iterator[Controller]:
    controller = build_core()
    yield controller
    controller._background_svc.quiesce()


def test_environment_registers_fake_flux_device(ctrl: Controller) -> None:
    connect_mock(ctrl)
    dev = ctrl.state.get_device(FAKE_FLUX_DEVICE_NAME)
    assert dev is not None
    assert dev.type_name == "FakeDevice"
    assert dev.status is DeviceStatus.CONNECTED
    assert ctrl.get_device_unit(FAKE_FLUX_DEVICE_NAME) == "none"
    soc = ctrl.state.session_env.soc
    assert isinstance(soc, MockQickSoc)
    assert soc.flux_source is not None
    assert soc.flux_source() == FAKE_FLUX_INITIAL_VALUE


def test_reentry_preserves_bound_source_value(ctrl: Controller) -> None:
    connect_mock(ctrl)
    soc = ctrl.state.session_env.soc
    assert isinstance(soc, MockQickSoc)
    source = soc.flux_source
    assert source is not None
    ctrl.device_control.start_setup_device(
        SetupDeviceRequest(
            FAKE_FLUX_DEVICE_NAME, FakeDeviceInfo(address="none", value=0.123)
        )
    )
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + 3.0
    while (
        time.monotonic() < deadline
        and ctrl.device_control.get_active_device_operations()
    ):
        app.processEvents()
        time.sleep(0.005)
    assert not ctrl.device_control.get_active_device_operations()
    finished: list[bool] = []
    ctrl.setup_control.bind_connection_outcome(
        lambda: finished.append(True), lambda _error: None
    )
    ctrl.setup_control.start_simulated_environment()
    assert finished == [True]
    assert ctrl.state.session_env.soc is soc
    assert source() == 0.123
    assert soc.flux_source == source
