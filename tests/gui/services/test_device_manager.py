"""Tests for the device registry, setup lifecycle, and progress events.

Two driver styles coexist deliberately: the setup / progress tests use a
``MagicMock`` driver (they assert on call interactions), while the registry CRUD
tests use a real ``FakeDevice`` (they read/write real values through the
GlobalDeviceManager registry).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEventLoop
from zcu_tools.device import FakeDevice, FakeDeviceInfo, GlobalDeviceManager
from zcu_tools.gui.app.main.services.operation_gate import OperationGate
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.events import (
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceRegistrationError,
    DeviceService,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui.services._device_fakes import FakeDeviceRegistry

# See tests/gui/services/test_device.py for why test-created BackgroundRunners must
# be quiesced before GC: a queued main-thread delivery to a GC'd runner segfaults.
_LIVE_BG: list[BackgroundRunner] = []


def _bg() -> BackgroundRunner:
    bg = BackgroundRunner()
    _LIVE_BG.append(bg)
    return bg


@pytest.fixture(autouse=True)
def _quiesce_services():
    yield
    for bg in _LIVE_BG:
        bg.quiesce()
    _LIVE_BG.clear()


@pytest.fixture(autouse=True)
def _clean_devices():
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


def _drain_until(
    condition: Callable[[], bool], *, timeout: float = 3.0, label: str = "condition"
) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {label}")


def _make_svc(driver: MagicMock | None = None) -> tuple[DeviceService, MagicMock]:
    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="none")
    gate = OperationGate()
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(gate, handles, progress, bg, bus)
    svc = DeviceService(
        bus,
        State(MagicMock()),
        gate,
        bg,
        runner,
        handles,
        driver_factory=lambda _type, _address: device,  # type: ignore[arg-type]
        device_registry=FakeDeviceRegistry(),
    )
    return svc, device


def _connect(svc: DeviceService) -> None:
    connected: list[object] = []
    errors: list[str] = []
    svc.device_connected.connect(connected.append)
    svc.operation_failed.connect(lambda _name, error: errors.append(error))
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name="test_dev", address="none")
    )
    _drain_until(lambda: bool(connected or errors), label="connect test_dev")
    assert not errors


def test_device_setup_failure_emits_setup_failed(qapp):
    # A setup whose driver raises is reported via setup_failed (bg on_error path).
    svc, device = _make_svc()
    _connect(svc)
    device.setup.side_effect = RuntimeError("setup failed")
    loop = QEventLoop()
    errors: list[str] = []
    svc.setup_failed.connect(lambda _name, error: errors.append(error) or loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    loop.exec()

    assert errors == ["setup failed"]


def test_device_setup_cancel_emits_setup_cancelled(qapp):
    # cancel_device_operation sets the stop_event via the handle; the driver's
    # setup returns normally, and DeviceService relabels it 'cancelled' because
    # the stop_event is set (the _on_setup_done cancel interpretation, ADR-0019).
    svc, device = _make_svc()
    _connect(svc)
    device.get_info.return_value = FakeDeviceInfo(address="none")
    device.setup.side_effect = lambda _info, stop_event: stop_event.wait(1.0)
    loop = QEventLoop()
    cancelled: list[str] = []
    svc.setup_cancelled.connect(lambda name: cancelled.append(name) or loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    svc.cancel_device_operation("test_dev")
    loop.exec()

    assert cancelled == ["test_dev"]


def test_device_service_emits_started_and_finished_events(qapp):
    svc, _device = _make_svc()
    _connect(svc)
    started: list[DeviceSetupStartedPayload] = []
    finished: list[DeviceSetupFinishedPayload] = []
    svc._bus.subscribe(DeviceSetupStartedPayload, started.append)
    svc._bus.subscribe(DeviceSetupFinishedPayload, finished.append)
    loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    loop.exec()

    assert [p.name for p in started] == ["test_dev"]
    assert finished[-1].name == "test_dev"
    assert finished[-1].outcome == "finished"


# ---------------------------------------------------------------------------
# Registry CRUD (real FakeDevice through GlobalDeviceManager)
# ---------------------------------------------------------------------------


def _make_real_svc(driver: object | None = None) -> tuple[DeviceService, object]:
    fake_device = driver if driver is not None else FakeDevice()
    gate = OperationGate()
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(gate, handles, progress, bg, bus)
    svc = DeviceService(
        bus,
        State(MagicMock()),
        gate,
        bg,
        runner,
        handles,
        driver_factory=lambda _type, _address: fake_device,  # type: ignore[arg-type]
    )
    return svc, fake_device


def _register(svc: DeviceService, name: str = "flux") -> None:
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name=name, address="")
    )
    loop.exec()


def _disconnect(svc: DeviceService, name: str = "flux") -> None:
    loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name=name))
    loop.exec()


def _set_value(svc: DeviceService, name: str, value: float) -> None:
    # Setting a value goes through setup (FakeDevice.setup ramps to info.value).
    # Build the setup info from the device's current info with value updated, so
    # the address matches (mirrors the device.setup RPC's with_updates path);
    # a mismatched address makes BaseDevice.setup raise.
    info = svc.get_device_info(name)
    assert info is not None
    loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: loop.quit())
    svc.setup_failed.connect(lambda _name, _err: loop.quit())
    svc.start_setup_device(
        SetupDeviceRequest(name=name, info=info.with_updates(value=value))
    )
    loop.exec()


def test_devicemanager_register_and_list(qapp):
    dev = FakeDevice()
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    entries = svc.list_devices()
    entry = next((e for e in entries if e.name == "flux"), None)
    assert entry is not None
    assert entry.type_name == "FakeDevice"
    # DeviceEntry now projects the fine-grained status string (FC7): a live driver
    # reads "connected" rather than the retired is_connected bool.
    assert entry.status == "connected"


def test_devicemanager_drop_device(qapp):
    svc, _ = _make_real_svc()
    _register(svc, "flux")
    _disconnect(svc)
    # drop moves device to memory-only; it still appears but disconnected
    entries = svc.list_devices()
    flux_entry = next((e for e in entries if e.name == "flux"), None)
    assert flux_entry is not None
    # A dropped device becomes memory-only (the retired is_connected==False).
    assert flux_entry.status == "memory_only"


def test_devicemanager_get_value_and_set_via_setup(qapp):
    dev = FakeDevice()
    dev.set_value(3.14)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")

    # The live value is read via get_device_value_for_new_context (the sole
    # prod reader; the bare get_device_value was an orphan and was removed).
    assert svc.get_device_value_for_new_context("flux") == pytest.approx(3.14)
    _set_value(svc, "flux", 2.71)
    assert svc.get_device_value_for_new_context("flux") == pytest.approx(2.71)


def test_get_device_unit_strict_whitelisted_device(qapp):
    # FakeDevice is on the bind whitelist -> unit "none" (no raise).
    dev = FakeDevice()
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    assert svc.get_device_unit_strict("flux") == "none"


def test_get_device_unit_strict_unknown_device_raises(qapp):
    # No device by that name -> Fast-Fail (binding must reference a known device).
    svc, _ = _make_real_svc()
    with pytest.raises(DeviceRegistrationError):
        svc.get_device_unit_strict("does_not_exist")


def test_devicemanager_get_all_info(qapp):
    dev = FakeDevice()
    dev.set_value(1.0)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    info = GlobalDeviceManager.get_all_info()
    assert "flux" in info
    flux_info = info["flux"]
    assert isinstance(flux_info, FakeDeviceInfo)
    assert flux_info.value == pytest.approx(1.0)
