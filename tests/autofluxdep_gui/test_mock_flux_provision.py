"""FLUX-AWARE-MOCK (autofluxdep): mock-connect auto-provisions a fake_flux
FakeDevice and binds it as the MockSoc's SimEngine flux source.

autofluxdep reuses the shared session-layer MockFluxProvisioner (built by
build_session_services), so a mock connect through its Controller provisions and
binds fake_flux exactly like measure-gui — no provisioning code in the autofluxdep
controller. These tests drive the *real* Controller connect path (real EventBus +
real ConnectionService / DeviceService), the same path "Use MockSoc" takes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDevice
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.session.services.connection import ConnectMockRequest
from zcu_tools.gui.session.services.mock_flux import (
    FAKE_FLUX_DEVICE_NAME,
    FAKE_FLUX_INITIAL_VALUE,
)
from zcu_tools.gui.session.state import DeviceStatus


@pytest.fixture()
def ctrl(qapp) -> Iterator[Controller]:  # noqa: ARG001
    controller = build_core()
    yield controller
    controller._background_svc.quiesce()
    # fake_flux is registered in the process-global GlobalDeviceManager; drop it so
    # tests stay independent (a second connect would otherwise see it "already
    # registered" at the driver layer).
    GlobalDeviceManager.drop_device(FAKE_FLUX_DEVICE_NAME, ignore_error=True)


def _pump_until(condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _connect_mock(ctrl: Controller) -> None:
    ctrl.start_connect(ConnectMockRequest())
    assert _pump_until(
        lambda: (
            (dev := ctrl.state.get_device(FAKE_FLUX_DEVICE_NAME)) is not None
            and dev.status is DeviceStatus.CONNECTED
            and dev.info is not None
            and getattr(dev.info, "value", None) == FAKE_FLUX_INITIAL_VALUE
        )
    ), "fake_flux was not provisioned to the default operating value"


def test_mock_connect_registers_fake_flux_device(ctrl):
    _connect_mock(ctrl)

    dev = ctrl.state.get_device(FAKE_FLUX_DEVICE_NAME)
    assert dev is not None
    assert dev.type_name == "FakeDevice"
    assert dev.status is DeviceStatus.CONNECTED
    assert ctrl.get_device_unit(FAKE_FLUX_DEVICE_NAME) == "none"


def test_mock_connect_sets_soc_flux_device(ctrl):
    _connect_mock(ctrl)

    soc = ctrl.state.exp_context.soc
    assert soc is not None
    assert getattr(soc, "_sim_params").flux_device == FAKE_FLUX_DEVICE_NAME


def test_reconnect_does_not_double_register_fake_flux(ctrl):
    _connect_mock(ctrl)
    dev = GlobalDeviceManager.get_device(FAKE_FLUX_DEVICE_NAME)
    assert isinstance(dev, FakeDevice)
    dev.set_value(0.123)

    ctrl.start_connect(ConnectMockRequest())
    assert _pump_until(lambda: not ctrl._conn_svc.is_connect_active())
    ctrl._background_svc.quiesce()
    QCoreApplication.instance().processEvents()  # type: ignore[union-attr]

    assert ctrl.state.get_device(FAKE_FLUX_DEVICE_NAME) is not None
    assert dev.get_value() == 0.123
    soc = ctrl.state.exp_context.soc
    assert getattr(soc, "_sim_params").flux_device == FAKE_FLUX_DEVICE_NAME
