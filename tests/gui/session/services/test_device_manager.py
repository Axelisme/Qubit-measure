"""Tests for the device registry, setup lifecycle, and progress events.

Two driver styles coexist deliberately: the setup / progress tests use a
``MagicMock`` driver (they assert on call interactions), while the registry CRUD
tests use a real ``FakeDevice`` (they read/write real values through the
DeviceManager registry).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.events import (
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import DeviceMemoryInfo
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceRegistrationError,
    DeviceService,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import DeviceStatus

from tests.gui._completion_helpers import (
    on_device_connected,
    on_device_disconnected,
    on_device_operation_failed,
    on_setup_cancelled,
    on_setup_failed,
    on_setup_finished,
)
from tests.gui._qt_waits import qt_until
from tests.gui.session.services._device_fakes import FakeDeviceRegistry
from tests.gui.session.services._device_waits import DeviceWorkers

_workers = DeviceWorkers()
_bg = _workers.background


@pytest.fixture(autouse=True, name="_quiesce_services")
def fixture_quiesce_services():
    """Close tracked service workers after each test."""
    try:
        yield
    finally:
        _workers.close()


def _make_svc(
    driver: MagicMock | None = None, bus: EventBus | None = None
) -> tuple[DeviceService, MagicMock]:
    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="none")
    gate = OperationGate(EventBus())
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = bus or EventBus()
    _workers.watch(bus)
    runner = OperationRunner(gate, handles, progress, bg, bus)
    svc = DeviceService(
        bus,
        State(MagicMock()),
        gate,
        bg,
        runner,
        handles,
        driver_factory=lambda type_name, address: device,
        device_registry=FakeDeviceRegistry(),
    )
    return svc, device


def _connect(svc: DeviceService) -> None:
    connected: list[object] = []
    errors: list[str] = []
    on_device_connected(svc, connected.append)
    on_device_operation_failed(svc, lambda _name, error: errors.append(error))
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="YOKOGS200", name="test_dev", address="none")
    )
    qt_until(lambda: bool(connected or errors), label="connect test_dev")
    assert not errors


@pytest.mark.parametrize("origin", ["register", "remember"])
def test_script_rows_reconnect_after_rename_without_manual_admission(qapp, origin):
    bus = EventBus()
    gate = OperationGate(bus)
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    registry = DeviceManager()
    service = DeviceService(
        bus,
        State(MagicMock()),
        gate,
        bg,
        OperationRunner(gate, handles, progress, bg, bus),
        handles,
        device_registry=registry,
    )
    driver = FakeDevice(address="script-bias", fast_mode=True)
    driver.set_value(1.25)
    _workers.watch(bus)
    if origin == "register":
        service.register_device("bias", driver)
        done: list[object] = []
        on_device_disconnected(service, done.append)
        service.start_disconnect_device(DisconnectDeviceRequest("bias"))
        # The no-op fake close must produce a terminal owner delivery.
        qt_until(lambda: bool(done or _workers.errors), label="script disconnect")
        assert not _workers.errors
    else:
        service.register_remembered_devices(
            [DeviceMemoryInfo("FakeDevice", "bias", "script-bias")]
        )
    service.rename_device("bias", "renamed")
    with pytest.raises(FailedPreconditionError):
        service.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "renamed", "script-bias")
        )
    connected: list[object] = []
    on_device_connected(service, connected.append)
    try:
        admission_errors: list[str] = []
        try:
            service.start_reconnect_device("renamed")
        except FailedPreconditionError as exc:
            admission_errors.append(str(exc))
        assert admission_errors == [], "Stored script rows must admit Reconnect"
        # Known adapter creates/retrieves the addressed fake, then owner settles.
        qt_until(lambda: bool(connected or _workers.errors), label="script reconnect")
        assert not _workers.errors
        row = service.get_device_snapshot("renamed")
        assert row is not None and row.status is DeviceStatus.CONNECTED
        assert row.address == "script-bias"
        assert service.get_cached_device_value("renamed") == 1.25
        assert registry.get_device("renamed") is driver
    finally:
        _workers.close()
        registry.close_all_devices()


def test_device_setup_failure_emits_setup_failed(qapp):
    # A setup whose driver raises is reported via setup_failed (bg on_error path).
    svc, device = _make_svc()
    _connect(svc)
    device.setup.side_effect = RuntimeError("setup failed")
    loop = _workers.completion(allow_failure=True)
    errors: list[str] = []
    on_setup_failed(svc, lambda _name, error: errors.append(error) or loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    loop.exec()

    assert errors == ["setup failed"]


def test_device_setup_cancel_emits_setup_cancelled(qapp):
    # cancel_device_operation sets the stop_event via the handle; the driver's
    # setup returns normally, and DeviceService relabels it 'cancelled' because
    # the stop_event is set (the _on_setup_done cancel interpretation, ADR-0066).
    svc, device = _make_svc()
    _connect(svc)
    device.get_info.return_value = FakeDeviceInfo(address="none")
    block = _workers.setup()
    device.setup.side_effect = block
    loop = _workers.completion()
    cancelled: list[str] = []
    on_setup_cancelled(svc, lambda name: cancelled.append(name) or loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    qt_until(block.entered.is_set, label="setup driver entered")
    svc.cancel_device_operation("test_dev")
    assert block.stop_event is not None and block.stop_event.is_set()
    block.release.set()
    loop.exec()

    assert cancelled == ["test_dev"]


def test_device_service_emits_started_and_finished_events(qapp):
    bus = EventBus()
    svc, _device = _make_svc(bus=bus)
    _connect(svc)
    started: list[DeviceSetupStartedPayload] = []
    finished: list[DeviceSetupFinishedPayload] = []
    bus.subscribe(DeviceSetupStartedPayload, started.append)
    bus.subscribe(DeviceSetupFinishedPayload, finished.append)
    loop = _workers.completion()
    on_setup_finished(svc, lambda _name: loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    loop.exec()

    assert [p.name for p in started] == ["test_dev"]
    assert finished[-1].name == "test_dev"
    assert finished[-1].outcome == "finished"


# ---------------------------------------------------------------------------
# Registry CRUD (real FakeDevice through DeviceManager)
# ---------------------------------------------------------------------------


def _make_real_svc(driver: FakeDevice | None = None) -> tuple[DeviceService, object]:
    fake_device = driver if driver is not None else FakeDevice(fast_mode=True)
    manager = DeviceManager()
    manager.register_device("flux", fake_device)
    gate = OperationGate(EventBus())
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    _workers.watch(bus)
    runner = OperationRunner(gate, handles, progress, bg, bus)
    svc = DeviceService(
        bus,
        State(MagicMock()),
        gate,
        bg,
        runner,
        handles,
        driver_factory=lambda type_name, address: fake_device,
        device_registry=manager,
    )
    return svc, fake_device


def _register(svc: DeviceService, name: str = "flux") -> None:
    svc.publish_initial_devices()


def _disconnect(svc: DeviceService, name: str = "flux") -> None:
    loop = _workers.completion()
    on_device_disconnected(svc, lambda _request: loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name=name))
    loop.exec()


def _set_value(svc: DeviceService, name: str, value: float) -> None:
    # Setting a value goes through setup (FakeDevice.setup ramps to info.value).
    # Build the setup info from the device's current info with value updated, so
    # the address matches (mirrors the device.setup RPC's with_updates path);
    # a mismatched address makes BaseDevice.setup raise.
    info = svc.get_device_info(name)
    assert info is not None
    loop = _workers.completion()
    on_setup_finished(svc, lambda _name: loop.quit())
    on_setup_failed(svc, lambda _name, _err: loop.quit())
    svc.start_setup_device(
        SetupDeviceRequest(name=name, info=info.with_updates(value=value))
    )
    loop.exec()


def test_devicemanager_register_and_list(qapp):
    dev = FakeDevice(fast_mode=True)
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
    dev = FakeDevice(fast_mode=True)
    dev.set_value(3.14)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")

    # The live value is read via get_device_value_for_new_context (the sole
    # prod reader; the bare get_device_value was an orphan and was removed).
    assert svc.get_device_value_for_new_context("flux") == pytest.approx(3.14)
    _set_value(svc, "flux", 2.71)
    assert svc.get_device_value_for_new_context("flux") == pytest.approx(2.71)


def test_get_device_unit_strict_whitelisted_device(qapp):
    # Connected FakeDevice Info declares the native coordinate (no raise).
    dev = FakeDevice(fast_mode=True)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    assert svc.get_device_unit_strict("flux") == "native"


def test_get_device_unit_strict_unknown_device_raises(qapp):
    # No device by that name -> Fast-Fail (binding must reference a known device).
    svc, _ = _make_real_svc()
    with pytest.raises(DeviceRegistrationError):
        svc.get_device_unit_strict("does_not_exist")


def test_devicemanager_get_all_info(qapp):
    dev = FakeDevice(fast_mode=True)
    dev.set_value(1.0)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    info = {
        name: driver.get_info() for name, driver in svc.get_connected_devices().items()
    }
    assert "flux" in info
    flux_info = info["flux"]
    assert isinstance(flux_info, FakeDeviceInfo)
    assert flux_info.value == pytest.approx(1.0)


def test_direct_poll_skips_terminal_admission_without_reading_driver(qapp) -> None:
    from zcu_tools.gui.session.adapters.manual_owner_scheduler import (
        ManualOwnerScheduler,
    )
    from zcu_tools.gui.session.adapters.thread_pool_background import (
        ThreadPoolBackgroundExecutor,
    )

    scheduler = ManualOwnerScheduler()
    bg = ThreadPoolBackgroundExecutor(scheduler)
    driver = FakeDevice(fast_mode=True)
    bus = EventBus()
    state = State(MagicMock())
    gate = OperationGate(bus)
    handles = OperationHandles()
    runner = OperationRunner(gate, handles, None, bg, bus)
    svc = DeviceService(
        bus,
        state,
        gate,
        bg,
        runner,
        handles,
        driver_factory=lambda type_name, address: driver,
        device_registry=FakeDeviceRegistry(),
    )
    try:
        svc.start_connect_device(ConnectDeviceRequest("YOKOGS200", "test_dev", "none"))
        # The accepted connect worker posts its terminal to this manual owner.
        assert scheduler.pump_once(block=True, timeout=30)
        assert svc.get_connected_devices()
        # Connect was delivered; its worker and owner acknowledgement must drain.
        assert bg.quiesce(timeout=30)
        before = svc.get_device_snapshot("test_dev")
        error: Exception | None = None
        try:
            svc.poll_device_info("test_dev")
        except RuntimeError as exc:
            error = exc
        assert error is None, "best-effort poll must skip closed admission"
        assert svc.get_device_snapshot("test_dev") == before
        assert handles.live_count() == 0
    finally:
        scheduler.pump_all()
        # All accepted connect deliveries are acknowledged before driver disposal.
        assert bg.quiesce(timeout=30)
        driver.close()
